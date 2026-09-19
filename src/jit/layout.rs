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
}

#[cfg(feature = "std")]
static LAYOUT: std::sync::OnceLock<Option<RcVecLayout>> = std::sync::OnceLock::new();

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
    let array_words = words(&array_value as *const Value as *const u8, 8);
    let array_rc_offset = unique_position(&array_words, inner as usize)? * 8;
    let struct_value = Value::Struct {
        name: alloc::string::String::from("probe"),
        layout: Rc::new(crate::bytecode::StructLayout::new(
            alloc::string::String::from("probe"),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )),
        fields: Rc::clone(&rc),
    };
    let struct_words = words(&struct_value as *const Value as *const u8, 8);
    let struct_fields_offset = unique_position(&struct_words, inner as usize)? * 8;
    drop(struct_value);
    drop(array_value);

    Some(RcVecLayout {
        value_offset,
        borrow_offset: value_offset + borrow_word * 8,
        ptr_offset: value_offset + ptr_word * 8,
        len_offset: value_offset + len_word * 8,
        array_rc_offset,
        struct_fields_offset,
    })
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

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
