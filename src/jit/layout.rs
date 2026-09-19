//! Where the JIT may reach into `Value` payloads directly.
//!
//! Arrays and struct fields are `Rc<RefCell<Vec<Value>>>`. Neither `Rc`'s
//! allocation, `RefCell`, nor `Vec` has a layout the language guarantees,
//! so the offsets native code uses are not assumed: they are measured once
//! at runtime from real values, and only if every measurement is
//! unambiguous does the backend emit inline element access. Otherwise it
//! keeps calling the helpers.

use crate::bytecode::Value;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

/// Byte offsets, relative to the `Rc` pointer's allocation (`RcInner`) and
/// to the `Value` holding it.
#[derive(Debug, Clone, Copy)]
pub struct RcVecLayout {
    /// Offset of the `RefCell<Vec<Value>>` from the `RcInner` start; the
    /// `Rc` pointer stored in a `Value` is the `RcInner` start.
    pub value_offset: usize,
    /// Offset of the `RefCell` borrow flag from the `RcInner` start
    /// (0 = not borrowed, > 0 shared borrows, < 0 a mutable borrow).
    pub borrow_offset: usize,
    /// Offsets of the `Vec`'s element pointer and length from the
    /// `RcInner` start.
    pub ptr_offset: usize,
    pub len_offset: usize,
    /// Offset of the `Rc` pointer within a `Value::Array`.
    pub array_rc_offset: usize,
    /// Offset of the `fields` `Rc` pointer within a `Value::Struct`.
    pub struct_fields_offset: usize,
    /// Offset of the `layout` `Rc` pointer (its allocation's start) within
    /// a `Value::Struct`.
    pub struct_layout_offset: usize,
    /// Offset of the `name` allocation pointer within a `Value::Struct`.
    pub struct_name_offset: usize,
}

/// Byte offsets within a `Value::Enum` and its payload allocation.
#[derive(Debug, Clone, Copy)]
pub struct EnumLayout {
    /// The discriminant byte of a `Value::Enum` (`ValueTag` numbers the
    /// variants differently).
    pub tag: u8,
    /// Offsets, within the `Value`, of the `Name` allocation pointers of
    /// the enum's type name and variant name.
    pub enum_name_offset: usize,
    pub variant_offset: usize,
    /// Offset, within the `Value`, of the payload `Rc<Vec<Value>>` pointer
    /// (null for a unit variant).
    pub values_offset: usize,
    /// Offsets of the payload `Vec`'s element pointer and length from that
    /// `Rc` allocation's start.
    pub values_ptr_offset: usize,
    pub values_len_offset: usize,
}

/// What generated code needs to copy a `Value` and adjust the reference
/// counts it owns: the discriminant of each variant that owns exactly one
/// `Rc` (all at `single_rc_offset`), plus the struct and enum variants
/// (their offsets are in `RcVecLayout` and `EnumLayout`). Every other
/// variant (weak references, closures, tasks) goes through the runtime.
#[derive(Debug, Clone, Copy)]
pub struct OwnershipLayout {
    /// Discriminants whose payload is a single `Rc` allocation pointer.
    pub single_rc_tags: [u8; 5],
    /// Offset of that pointer within the `Value`.
    pub single_rc_offset: usize,
    pub struct_tag: u8,
    pub enum_tag: u8,
    /// Discriminants that own nothing: Nil, Bool, Int, Float, Function.
    pub plain_tags: [u8; 5],
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

fn probe_ownership() -> Option<OwnershipLayout> {
    let words_of = |value: &Value| words(value as *const Value as *const u8, core::mem::size_of::<Value>() / 8);
    // A value owning one `Rc`: its allocation pointer must sit at one word.
    let string = Rc::new(alloc::string::String::from("probe"));
    let string_inner = (Rc::as_ptr(&string) as *const u8).wrapping_sub(16) as usize;
    let string_value = Value::String(Rc::clone(&string));
    let single_rc_offset = unique_position(&words_of(&string_value), string_inner)? * 8;
    let check = |value: &Value, inner: usize| -> Option<u8> {
        (unique_position(&words_of(value), inner)? * 8 == single_rc_offset).then(|| tag_of(value))
    };
    let array = Rc::new(RefCell::new(alloc::vec![Value::Int(1)]));
    let array_inner = (Rc::as_ptr(&array) as *const u8).wrapping_sub(16) as usize;
    let array_tag = check(&Value::Array(Rc::clone(&array)), array_inner)?;
    let tuple = Rc::new(alloc::vec![Value::Int(1)]);
    let tuple_inner = (Rc::as_ptr(&tuple) as *const u8).wrapping_sub(16) as usize;
    let tuple_tag = check(&Value::Tuple(Rc::clone(&tuple)), tuple_inner)?;
    let map = Rc::new(RefCell::new(crate::bytecode::LustMap::default()));
    let map_inner = (Rc::as_ptr(&map) as *const u8).wrapping_sub(16) as usize;
    let map_tag = check(&Value::Map(Rc::clone(&map)), map_inner)?;
    let native: crate::bytecode::value::NativeFn =
        Rc::new(|_args: &[Value]| Ok(crate::bytecode::value::NativeCallResult::Return(Value::Nil)));
    let native_inner = (Rc::as_ptr(&native) as *const u8).wrapping_sub(16) as usize;
    let native_tag = check(&Value::NativeFunction(Rc::clone(&native)), native_inner)?;
    let struct_tag = {
        let layout = Rc::new(crate::bytecode::StructLayout::new(
            alloc::string::String::from("probe"),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
        tag_of(&Value::Struct {
            name: "probe".into(),
            layout,
            fields: Rc::clone(&array),
        })
    };
    let enum_tag = enum_layout()?.tag;
    let plain_tags = [
        tag_of(&Value::Nil),
        tag_of(&Value::Bool(true)),
        tag_of(&Value::Int(1)),
        tag_of(&Value::Float(1.0)),
        tag_of(&Value::Function(0)),
    ];
    Some(OwnershipLayout {
        single_rc_tags: [tag_of(&string_value), array_tag, tuple_tag, map_tag, native_tag],
        single_rc_offset,
        struct_tag,
        enum_tag,
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

fn probe_enum() -> Option<EnumLayout> {
    use crate::bytecode::value::Name;
    let enum_name = Name::from("ProbeEnum");
    let variant = Name::from("ProbeVariant");
    let mut payload: Vec<Value> = Vec::with_capacity(7);
    payload.push(Value::Int(0x6161));
    payload.push(Value::Int(0x6262));
    payload.push(Value::Int(0x6363));
    let elements = payload.as_ptr() as usize;
    let rc = Rc::new(payload);
    let inner = (Rc::as_ptr(&rc) as *const u8).wrapping_sub(16);
    let value = Value::Enum {
        enum_name: enum_name.clone(),
        variant: variant.clone(),
        values: Some(Rc::clone(&rc)),
    };
    // SAFETY: reading the first byte of a live `Value`.
    let tag = unsafe { *(&value as *const Value as *const u8) };
    let value_words = words(&value as *const Value as *const u8, core::mem::size_of::<Value>() / 8);
    let enum_name_offset = unique_position(&value_words, enum_name.inner_ptr() as usize)? * 8;
    let variant_offset = unique_position(&value_words, variant.inner_ptr() as usize)? * 8;
    let values_offset = unique_position(&value_words, inner as usize)? * 8;
    // `RcInner { strong, weak, Vec { .. } }`: three words of Vec after the counts.
    let inner_words = words(inner, 5);
    if inner_words[0] != 2 || inner_words[1] != 1 {
        return None;
    }
    let ptr_word = unique_position(&inner_words, elements)?;
    let len_word = unique_position(&inner_words, 3)?;
    if unique_position(&inner_words, 7).is_none() {
        return None;
    }
    let unit = Value::Enum {
        enum_name,
        variant,
        values: None,
    };
    let unit_words = words(&unit as *const Value as *const u8, core::mem::size_of::<Value>() / 8);
    if unit_words[values_offset / 8] != 0 {
        return None;
    }
    drop(value);
    drop(unit);
    Some(EnumLayout {
        tag,
        enum_name_offset,
        variant_offset,
        values_offset,
        values_ptr_offset: ptr_word * 8,
        values_len_offset: len_word * 8,
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

    // Within the RefCell<Vec>: 4 words (borrow flag, ptr, cap, len).
    let cell_words = words(cell_ptr, 4);
    let ptr_word = unique_position(&cell_words, elements)?;
    let len_word = unique_position(&cell_words, 2)?;
    if unique_position(&cell_words, 5).is_none() {
        return None;
    }
    let borrow_word = {
        let shared = rc.borrow();
        let borrowed_words = words(cell_ptr, 4);
        drop(shared);
        let changed: Vec<usize> = (0..4)
            .filter(|i| borrowed_words[*i] != cell_words[*i])
            .collect();
        match changed.as_slice() {
            [one] if cell_words[*one] == 0 && borrowed_words[*one] == 1 => *one,
            _ => return None,
        }
    };
    if !matches!(rc.borrow().get(1), Some(Value::Int(0x5252))) {
        return None;
    }

    // Where the Rc pointer lives inside a Value::Array and a Value::Struct.
    let array_value = Value::Array(Rc::clone(&rc));
    let array_words = words(
        &array_value as *const Value as *const u8,
        core::mem::size_of::<Value>() / 8,
    );
    let array_rc_offset = unique_position(&array_words, inner as usize)? * 8;
    let layout = Rc::new(crate::bytecode::StructLayout::new(
        alloc::string::String::from("probe"),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ));
    let layout_inner = (Rc::as_ptr(&layout) as *const u8).wrapping_sub(16) as usize;
    let name = crate::bytecode::value::Name::from("probe");
    let struct_value = Value::Struct {
        name: name.clone(),
        layout: Rc::clone(&layout),
        fields: Rc::clone(&rc),
    };
    let struct_words = words(
        &struct_value as *const Value as *const u8,
        core::mem::size_of::<Value>() / 8,
    );
    let struct_fields_offset = unique_position(&struct_words, inner as usize)? * 8;
    let struct_layout_offset = unique_position(&struct_words, layout_inner)? * 8;
    let struct_name_offset = unique_position(&struct_words, name.inner_ptr() as usize)? * 8;
    drop(struct_value);
    drop(array_value);

    Some(RcVecLayout {
        value_offset,
        borrow_offset: value_offset + borrow_word * 8,
        ptr_offset: value_offset + ptr_word * 8,
        len_offset: value_offset + len_word * 8,
        array_rc_offset,
        struct_fields_offset,
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
        assert_eq!(layout.plain_tags, [0, 1, 2, 3, layout.plain_tags[4]]);
        assert!(layout.single_rc_tags.contains(&4));
        assert_ne!(layout.struct_tag, layout.enum_tag);
    }

    #[test]
    fn enum_layout_is_measured() {
        let layout = enum_layout().expect("enum layout probe");
        assert_ne!(layout.enum_name_offset, layout.variant_offset);
        let value = Value::some(Value::Int(5));
        let base = &value as *const Value as *const u8;
        let variant = unsafe { core::ptr::read(base.add(layout.variant_offset) as *const usize) };
        assert_eq!(
            variant,
            crate::bytecode::value::Name::from("Some").inner_ptr() as usize
        );
        let rc = unsafe { core::ptr::read(base.add(layout.values_offset) as *const usize) };
        let len = unsafe { core::ptr::read((rc + layout.values_len_offset) as *const usize) };
        assert_eq!(len, 1);
    }

    #[test]
    fn layout_is_measured_and_consistent() {
        let layout = rc_vec_layout().expect("layout probe");
        assert_eq!(layout.value_offset, 16);
        assert_eq!(layout.array_rc_offset, 8);
        assert!(layout.struct_fields_offset >= 8 && layout.struct_fields_offset < 64);
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
            Value::Struct { fields, .. } => Rc::strong_count(fields),
            Value::Enum {
                values: Some(values),
                ..
            } => Rc::strong_count(values),
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
