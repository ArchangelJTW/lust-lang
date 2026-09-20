//! Where the JIT may reach into `Value` payloads directly.
//!
//! Every heap value is one thin `Rc`: an array is an
//! `Rc<RefCell<Vec<Value>>>`, a struct an `Rc<StructObject>`, an enum an
//! `Rc<EnumObject>`. Neither `Rc`'s allocation, `RefCell`, `Vec` nor the
//! objects have a layout the language guarantees, so the offsets native
//! code uses are not assumed: they are measured once at runtime from real
//! values, and only if every measurement is unambiguous does the backend
//! emit inline access. Otherwise it keeps calling the helpers.

use crate::bytecode::Value;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

/// Byte offsets into the allocations behind arrays and structs. "From the
/// allocation" means from the `RcInner` start, which is the pointer a
/// `Value` carries.
#[derive(Debug, Clone, Copy)]
pub struct RcVecLayout {
    /// Offset of the `RefCell<Vec<Value>>` from an array's allocation.
    pub value_offset: usize,
    /// Offsets, from an array's allocation, of the `RefCell` borrow flag
    /// (0 = not borrowed, > 0 shared borrows, < 0 a mutable borrow) and
    /// of the `Vec`'s element pointer and length.
    pub borrow_offset: usize,
    pub ptr_offset: usize,
    pub len_offset: usize,
    /// Offset of the allocation pointer within a `Value::Array`.
    pub array_rc_offset: usize,
    /// Offset of the allocation pointer within a `Value::Struct`.
    pub struct_fields_offset: usize,
    /// Offsets, from a struct's allocation, of its fields' borrow flag,
    /// element pointer and length.
    pub struct_borrow_offset: usize,
    pub struct_ptr_offset: usize,
    pub struct_len_offset: usize,
    /// Offsets, from a struct's allocation, of the `layout` `Rc` pointer
    /// (its allocation's start) and of the `name` allocation pointer.
    pub struct_layout_offset: usize,
    pub struct_name_offset: usize,
}

/// Byte offsets into the allocation behind a `Value::Enum`.
#[derive(Debug, Clone, Copy)]
pub struct EnumLayout {
    /// The discriminant byte of a `Value::Enum` (`ValueTag` numbers the
    /// variants differently).
    pub tag: u8,
    /// Offset of the allocation pointer within the `Value`.
    pub object_offset: usize,
    /// Offsets, from the allocation, of the `Name` allocation pointers of
    /// the enum's type name and variant name.
    pub enum_name_offset: usize,
    pub variant_offset: usize,
    /// Offsets, from the allocation, of the payload `Vec`'s element pointer
    /// and length.
    pub values_ptr_offset: usize,
    pub values_len_offset: usize,
    /// A unit variant (`values: None`) is recognisable by one word of the
    /// `Option<Vec>` holding a value no payload can have: this word, at
    /// this offset from the allocation.
    pub unit_word_offset: usize,
    pub unit_word_value: usize,
}

/// What generated code needs to copy a `Value` and adjust the reference
/// count it owns: every variant not listed in `plain_tags` owns one `Rc`
/// allocation pointer at `single_rc_offset`, except `weak_tag` (a weak
/// reference, which the runtime handles).
#[derive(Debug, Clone, Copy)]
pub struct OwnershipLayout {
    /// Offset of the allocation pointer within the `Value`.
    pub single_rc_offset: usize,
    pub struct_tag: u8,
    pub enum_tag: u8,
    pub weak_tag: u8,
    pub native_tag: u8,
    /// Discriminants that own nothing: Nil, Bool, Int, Float, Function,
    /// Task.
    pub plain_tags: [u8; 6],
}

#[cfg(feature = "std")]
static OWNERSHIP: std::sync::OnceLock<Option<OwnershipLayout>> = std::sync::OnceLock::new();

#[cfg(feature = "std")]
pub fn ownership_layout() -> Option<OwnershipLayout> {
    *OWNERSHIP.get_or_init(probe_ownership)
}

#[cfg(not(feature = "std"))]
pub fn ownership_layout() -> Option<OwnershipLayout> {
    None
}

fn tag_of(value: &Value) -> u8 {
    // SAFETY: reading the first byte of a live `Value`.
    unsafe { *(value as *const Value as *const u8) }
}

/// The allocation (`RcInner { strong, weak, value }`) behind an `Rc`.
fn inner_of<T: ?Sized>(rc: &Rc<T>) -> usize {
    (Rc::as_ptr(rc) as *const u8).wrapping_sub(16) as usize
}

fn value_words(value: &Value) -> Vec<usize> {
    words(
        value as *const Value as *const u8,
        core::mem::size_of::<Value>() / 8,
    )
}

fn probe_ownership() -> Option<OwnershipLayout> {
    // Every owning variant carries its allocation pointer at one word.
    let string = Rc::new(alloc::string::String::from("probe"));
    let single_rc_offset = unique_position(
        &value_words(&Value::String(Rc::clone(&string))),
        inner_of(&string),
    )? * 8;
    let same_word = |value: &Value, inner: usize| -> Option<u8> {
        (unique_position(&value_words(value), inner)? * 8 == single_rc_offset)
            .then(|| tag_of(value))
    };
    let array = Rc::new(RefCell::new(alloc::vec![Value::Int(1)]));
    same_word(&Value::Array(Rc::clone(&array)), inner_of(&array))?;
    let tuple = Rc::new(alloc::vec![Value::Int(1)]);
    same_word(&Value::Tuple(Rc::clone(&tuple)), inner_of(&tuple))?;
    let map = Rc::new(RefCell::new(crate::bytecode::LustMap::default()));
    same_word(&Value::Map(Rc::clone(&map)), inner_of(&map))?;
    let native = crate::bytecode::native_fn(|_args: &[Value]| {
        Ok(crate::bytecode::value::NativeCallResult::Return(Value::Nil))
    });
    let native_tag = same_word(
        &Value::NativeFunction(Rc::clone(&native)),
        inner_of(&native),
    )?;
    let layout = Rc::new(crate::bytecode::StructLayout::new(
        alloc::string::String::from("probe"),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ));
    let object = crate::bytecode::StructObject::new("probe", layout, Vec::new());
    let struct_tag = same_word(&Value::Struct(Rc::clone(&object)), inner_of(&object))?;
    let weak_tag = tag_of(&Value::WeakStruct(crate::bytecode::WeakStructRef::new(
        &object,
    )));
    let enum_object = crate::bytecode::EnumObject::new("Probe", "Unit", None);
    let enum_tag = same_word(
        &Value::Enum(Rc::clone(&enum_object)),
        inner_of(&enum_object),
    )?;
    let closure = Rc::new(crate::bytecode::ClosureObject {
        function_idx: 0,
        upvalues: Vec::new(),
    });
    same_word(&Value::Closure(Rc::clone(&closure)), inner_of(&closure))?;
    let iterator = Rc::new(RefCell::new(crate::bytecode::value::IteratorState::Array {
        items: Vec::new(),
        index: 0,
    }));
    same_word(&Value::Iterator(Rc::clone(&iterator)), inner_of(&iterator))?;
    let plain_tags = [
        tag_of(&Value::Nil),
        tag_of(&Value::Bool(true)),
        tag_of(&Value::Int(1)),
        tag_of(&Value::Float(1.0)),
        tag_of(&Value::Function(0)),
        tag_of(&Value::Task(crate::bytecode::TaskHandle(0))),
    ];
    Some(OwnershipLayout {
        single_rc_offset,
        struct_tag,
        enum_tag,
        weak_tag,
        native_tag,
        plain_tags,
    })
}

#[cfg(feature = "std")]
static LAYOUT: std::sync::OnceLock<Option<RcVecLayout>> = std::sync::OnceLock::new();
#[cfg(feature = "std")]
static ENUM_LAYOUT: std::sync::OnceLock<Option<EnumLayout>> = std::sync::OnceLock::new();

/// The measured enum layout, or `None` when the probe could not pin it down.
#[cfg(feature = "std")]
pub fn enum_layout() -> Option<EnumLayout> {
    *ENUM_LAYOUT.get_or_init(probe_enum)
}

#[cfg(not(feature = "std"))]
pub fn enum_layout() -> Option<EnumLayout> {
    None
}

/// Words of an `EnumObject` or `StructObject` allocation: the two counts,
/// then the object (at most `count` words).
fn probe_enum() -> Option<EnumLayout> {
    use crate::bytecode::value::Name;
    let enum_name = Name::from("ProbeEnum");
    let variant = Name::from("ProbeVariant");
    let mut payload: Vec<Value> = Vec::with_capacity(7);
    payload.push(Value::Int(0x6161));
    payload.push(Value::Int(0x6262));
    payload.push(Value::Int(0x6363));
    let elements = payload.as_ptr() as usize;
    let object =
        crate::bytecode::EnumObject::new(enum_name.clone(), variant.clone(), Some(payload));
    let value = Value::Enum(Rc::clone(&object));
    let tag = tag_of(&value);
    let inner = inner_of(&object);
    let object_offset = unique_position(&value_words(&value), inner)? * 8;
    // `RcInner { strong, weak, EnumObject }`: the counts, then the two
    // names (two words each) and the payload `Vec` (three words).
    let object_words = words(
        inner as *const u8,
        2 + core::mem::size_of::<crate::bytecode::EnumObject>() / 8,
    );
    if object_words[0] != 2 || object_words[1] != 1 {
        return None;
    }
    let enum_name_offset = unique_position(&object_words, enum_name.inner_ptr() as usize)? * 8;
    let variant_offset = unique_position(&object_words, variant.inner_ptr() as usize)? * 8;
    let values_ptr_offset = unique_position(&object_words, elements)? * 8;
    let values_len_offset = unique_position(&object_words, 3)? * 8;
    if unique_position(&object_words, 7).is_none() {
        return None;
    }
    // A unit variant has no payload. `Option<Vec>` encodes `None` in a
    // niche — a null element pointer, or a capacity above `isize::MAX` —
    // and which one is measured, not assumed: the word of the `Vec` that
    // holds an impossible value in a `None`.
    let unit = crate::bytecode::EnumObject::new(enum_name, variant, None);
    let unit_words = words(inner_of(&unit) as *const u8, object_words.len());
    // The capacity niche is definitive when present; a null element
    // pointer is the other encoding (the words a `None` does not use hold
    // whatever was there, so a zero pointer next to a capacity niche means
    // nothing).
    let cap_word = unique_position(&object_words, 7)?;
    let ptr_word = values_ptr_offset / 8;
    let (unit_word_offset, unit_word_value) = if unit_words[cap_word] > isize::MAX as usize {
        (cap_word * 8, unit_words[cap_word])
    } else if unit_words[ptr_word] == 0 {
        (ptr_word * 8, 0)
    } else {
        return None;
    };
    drop(value);
    drop(unit);
    Some(EnumLayout {
        tag,
        object_offset,
        enum_name_offset,
        variant_offset,
        values_ptr_offset,
        values_len_offset,
        unit_word_offset,
        unit_word_value,
    })
}

/// The measured layout, or `None` when the probe could not pin it down.
#[cfg(feature = "std")]
pub fn rc_vec_layout() -> Option<RcVecLayout> {
    *LAYOUT.get_or_init(probe)
}

#[cfg(not(feature = "std"))]
pub fn rc_vec_layout() -> Option<RcVecLayout> {
    None
}

fn words(ptr: *const u8, count: usize) -> Vec<usize> {
    // SAFETY: callers pass a pointer into a live object at least
    // `count * 8` bytes long.
    (0..count)
        .map(|i| unsafe { core::ptr::read_unaligned(ptr.add(i * 8) as *const usize) })
        .collect()
}

fn unique_position(words: &[usize], value: usize) -> Option<usize> {
    let positions: Vec<usize> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| **w == value)
        .map(|(i, _)| i)
        .collect();
    match positions.as_slice() {
        [one] => Some(*one),
        _ => None,
    }
}

/// Within a `RefCell<Vec<Value>>` holding two of five elements: the words
/// of the borrow flag, the element pointer and the length, given the
/// cell's words at rest and a way to take a shared borrow.
fn cell_words(
    cell_ptr: *const u8,
    elements: usize,
    borrow: impl FnOnce() -> Vec<usize>,
) -> Option<(usize, usize, usize)> {
    let at_rest = words(cell_ptr, 4);
    let ptr_word = unique_position(&at_rest, elements)?;
    let len_word = unique_position(&at_rest, 2)?;
    if unique_position(&at_rest, 5).is_none() {
        return None;
    }
    let borrowed = borrow();
    let changed: Vec<usize> = (0..4).filter(|i| borrowed[*i] != at_rest[*i]).collect();
    let borrow_word = match changed.as_slice() {
        [one] if at_rest[*one] == 0 && borrowed[*one] == 1 => *one,
        _ => return None,
    };
    Some((borrow_word, ptr_word, len_word))
}

fn probe() -> Option<RcVecLayout> {
    let mut vec: Vec<Value> = Vec::with_capacity(5);
    vec.push(Value::Int(0x5151));
    vec.push(Value::Int(0x5252));
    let elements = vec.as_ptr() as usize;
    let rc = Rc::new(RefCell::new(vec));
    let cell_ptr = Rc::as_ptr(&rc) as *const u8;

    // `RcInner` is `{ strong, weak, value }`: the value sits 16 bytes in.
    // Confirm by reading the counts back and watching `strong` follow a
    // clone.
    let inner = cell_ptr.wrapping_sub(16);
    let counts_before = words(inner, 2);
    let extra = Rc::clone(&rc);
    let counts_after = words(inner, 2);
    drop(extra);
    if counts_before != [1, 1] || counts_after != [2, 1] {
        return None;
    }
    let value_offset = 16;
    let (borrow_word, ptr_word, len_word) = cell_words(cell_ptr, elements, || {
        let shared = rc.borrow();
        let borrowed = words(cell_ptr, 4);
        drop(shared);
        borrowed
    })?;
    if !matches!(rc.borrow().get(1), Some(Value::Int(0x5252))) {
        return None;
    }
    let array_value = Value::Array(Rc::clone(&rc));
    let array_rc_offset = unique_position(&value_words(&array_value), inner as usize)? * 8;
    drop(array_value);

    // A struct: its allocation holds the name, the layout pointer and the
    // fields cell.
    let layout = Rc::new(crate::bytecode::StructLayout::new(
        alloc::string::String::from("probe"),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ));
    let layout_inner = inner_of(&layout);
    let name = crate::bytecode::value::Name::from("probe");
    let mut fields: Vec<Value> = Vec::with_capacity(5);
    fields.push(Value::Int(0x5353));
    fields.push(Value::Int(0x5454));
    let field_elements = fields.as_ptr() as usize;
    let object = crate::bytecode::StructObject::new(name.clone(), Rc::clone(&layout), fields);
    let struct_value = Value::Struct(Rc::clone(&object));
    let object_inner = inner_of(&object);
    let struct_fields_offset = unique_position(&value_words(&struct_value), object_inner)? * 8;
    let object_words = words(
        object_inner as *const u8,
        2 + core::mem::size_of::<crate::bytecode::StructObject>() / 8,
    );
    let struct_layout_offset = unique_position(&object_words, layout_inner)? * 8;
    let struct_name_offset = unique_position(&object_words, name.inner_ptr() as usize)? * 8;
    let fields_cell = &object.fields as *const RefCell<Vec<Value>> as *const u8;
    let (fields_borrow_word, fields_ptr_word, fields_len_word) =
        cell_words(fields_cell, field_elements, || {
            let shared = object.fields.borrow();
            let borrowed = words(fields_cell, 4);
            drop(shared);
            borrowed
        })?;
    let fields_cell_offset = fields_cell as usize - object_inner;
    drop(struct_value);

    Some(RcVecLayout {
        value_offset,
        borrow_offset: value_offset + borrow_word * 8,
        ptr_offset: value_offset + ptr_word * 8,
        len_offset: value_offset + len_word * 8,
        array_rc_offset,
        struct_fields_offset,
        struct_borrow_offset: fields_cell_offset + fields_borrow_word * 8,
        struct_ptr_offset: fields_cell_offset + fields_ptr_word * 8,
        struct_len_offset: fields_cell_offset + fields_len_word * 8,
        struct_layout_offset,
        struct_name_offset,
    })
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn ownership_layout_is_measured() {
        let layout = ownership_layout().expect("ownership probe");
        assert_eq!(layout.single_rc_offset, 8);
        assert_eq!(layout.plain_tags[..4], [0, 1, 2, 3]);
        assert_ne!(layout.struct_tag, layout.enum_tag);
        assert_ne!(layout.weak_tag, layout.struct_tag);
    }

    #[test]
    fn enum_layout_is_measured() {
        let layout = enum_layout().expect("enum layout probe");
        assert_ne!(layout.enum_name_offset, layout.variant_offset);
        let value = Value::some(Value::Int(5));
        let base = &value as *const Value as *const u8;
        let object = unsafe { core::ptr::read(base.add(layout.object_offset) as *const usize) };
        let variant = unsafe { core::ptr::read((object + layout.variant_offset) as *const usize) };
        assert_eq!(
            variant,
            crate::bytecode::value::Name::from("Some").inner_ptr() as usize
        );
        let len = unsafe { core::ptr::read((object + layout.values_len_offset) as *const usize) };
        assert_eq!(len, 1);
    }

    #[test]
    fn layout_is_measured_and_consistent() {
        let layout = rc_vec_layout().expect("layout probe");
        assert_eq!(layout.value_offset, 16);
        assert_eq!(layout.array_rc_offset, 8);
        assert_eq!(layout.struct_fields_offset, 8);
        assert!(layout.struct_ptr_offset >= 16 && layout.struct_ptr_offset < 80);
        // Read an element through the measured offsets.
        let vec = alloc::vec![Value::Int(7), Value::Int(9)];
        let elements = vec.as_ptr() as usize;
        let value = Value::Array(Rc::new(RefCell::new(vec)));
        let base = &value as *const Value as *const u8;
        let rc = unsafe { core::ptr::read(base.add(layout.array_rc_offset) as *const usize) };
        let ptr = unsafe { core::ptr::read((rc + layout.ptr_offset) as *const usize) };
        let len = unsafe { core::ptr::read((rc + layout.len_offset) as *const usize) };
        let borrow = unsafe { core::ptr::read((rc + layout.borrow_offset) as *const isize) };
        assert_eq!(ptr, elements);
        assert_eq!(len, 2);
        assert_eq!(borrow, 0);
    }
}

/// Generated code adjusts reference counts itself for the values whose
/// layout the probes cover; these run compiled functions that copy structs
/// and enums around and check the counts they leave behind.
#[cfg(all(test, feature = "std", any(target_arch = "aarch64", target_arch = "x86_64")))]
mod refcount_tests {
    use crate::bytecode::Value;
    use crate::embed::EmbeddedProgram;
    use alloc::rc::Rc;

    fn strong_count_of(value: &Value) -> usize {
        match value {
            Value::Struct(object) => Rc::strong_count(object),
            Value::Enum(object) => Rc::strong_count(object),
            other => panic!("not a container: {other:?}"),
        }
    }

    #[test]
    fn compiled_field_reads_and_moves_balance_reference_counts() {
        let source = r#"
            struct Node
                value: int
                left: Option<Node>
                right: Option<Node>
            end

            function walk(n: Node): int
                local s: int = n.value
                local copy: Node = n
                local again: Node = copy
                if n.left is Some(l) then
                    s = s + walk(l)
                    local held: Node = l
                    s = s + held.value
                end
                if n.right is Some(r) then
                    s = s + walk(r)
                end
                return s + again.value
            end

            function churn(n: Node, count: int): int
                local total: int = 0
                local i: int = 0
                while i < count do
                    total = total + walk(n)
                    i = i + 1
                end
                return total
            end

            local leaf_a: Node = Node { value = 1, left = Option.None, right = Option.None }
            local leaf_b: Node = Node { value = 2, left = Option.None, right = Option.None }
            local root: Node = Node { value = 3, left = Option.Some(leaf_a), right = Option.Some(leaf_b) }
        "#;
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()
            .expect("compile");
        program.run_entry_script().expect("run entry script");
        let root = program.get_global_value("main.root").expect("root");
        let leaf_a = program.get_global_value("main.leaf_a").expect("leaf_a");
        let before_root = strong_count_of(&root);
        let before_leaf = strong_count_of(&leaf_a);
        // Enough direct calls for `walk` to be compiled whole and run
        // natively (recursively), then a traced loop calling it.
        for _ in 0..80 {
            let total: i64 = program
                .call_typed("main.walk", root.clone())
                .expect("walk");
            assert_eq!(total, 3 + 1 + 1 + 2 + 3 + 1 + 2);
        }
        for _ in 0..3 {
            let total: i64 = program
                .call_typed("main.churn", (root.clone(), 500i64))
                .expect("churn");
            assert_eq!(total, 500 * (3 + 1 + 1 + 2 + 3 + 1 + 2));
        }
        let stats = program.jit_stats();
        assert!(stats.functions_compiled >= 1, "{stats:?}");
        assert_eq!(strong_count_of(&root), before_root);
        assert_eq!(strong_count_of(&leaf_a), before_leaf);
        drop(root);
        drop(leaf_a);
    }
}
