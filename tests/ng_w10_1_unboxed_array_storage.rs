#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W10-1: Isolated unboxed storage (`UnboxedArrayData`) inside `LustArray`
//! with `has_flat_cache`, `invalidate_flat_cache`, non-panicking `try_borrow`,
//! and `PartialEq`/`Debug`/`Display` invariants.

use std::rc::Rc;

use lust::bytecode::value::UnboxedArrayData;
use lust::bytecode::{FlatArrayBuffer, FlatArrayElemTag, LustArray, Value};

#[test]
fn test_ng_w10_1_one_million_bools_stay_unboxed_one_byte_per_elem() {
    let arr = LustArray::new(Vec::new());
    for i in 0..1_000_000usize {
        arr.push_value_fast(Value::Bool(i % 2 == 0));
    }

    assert_eq!(arr.len_fast(), 1_000_000);
    assert!(!arr.is_empty_fast());

    // Backing `elements: Vec<Value>` must never have been materialized or allocated.
    {
        let raw_elems = arr.borrow_elements_without_sync();
        assert!(
            raw_elems.is_empty(),
            "elements must remain empty while data lives in UnboxedArrayData::Bool"
        );
        assert_eq!(
            raw_elems.capacity(),
            0,
            "elements must have capacity 0 (no 16MB/160MB Vec<Value> allocation)"
        );
    }

    arr.with_unboxed_data(|data| match data {
        UnboxedArrayData::Bool(bytes) => {
            assert_eq!(bytes.len(), 1_000_000);
            assert_eq!(core::mem::size_of_val(bytes.as_slice()), 1_000_000);
            assert_eq!(bytes[0], 1);
            assert_eq!(bytes[1], 0);
            assert_eq!(bytes[999_998], 1);
            assert_eq!(bytes[999_999], 0);
        }
        other => panic!("expected UnboxedArrayData::Bool, got {other:?}"),
    });

    assert_eq!(arr.get_value_fast(0), Some(Value::Bool(true)));
    assert_eq!(arr.get_value_fast(1), Some(Value::Bool(false)));
    assert_eq!(arr.get_value_fast(999_999), Some(Value::Bool(false)));
    assert_eq!(arr.get_value_fast(1_000_000), None);
}

#[test]
fn test_ng_w10_1_f05_partial_eq_unboxed_and_materialized() {
    let arr1 = LustArray::new(Vec::new());
    arr1.push_value_fast(Value::Int(1));
    arr1.push_value_fast(Value::Int(2));

    let arr2 = LustArray::new(Vec::new());
    arr2.push_value_fast(Value::Int(3));
    arr2.push_value_fast(Value::Int(4));

    let arr1_same = LustArray::new(Vec::new());
    arr1_same.push_value_fast(Value::Int(1));
    arr1_same.push_value_fast(Value::Int(2));

    let materialized = LustArray::new(vec![Value::Int(1), Value::Int(2)]);
    let materialized_diff = LustArray::new(vec![Value::Int(1), Value::Int(99)]);

    assert_ne!(arr1, arr2, "distinct unboxed int arrays must not be equal");
    assert_eq!(arr1, arr1_same, "equal unboxed int arrays must be equal");
    assert_eq!(
        arr1, materialized,
        "unboxed array must equal materialized array with same elements"
    );
    assert_eq!(
        materialized, arr1,
        "materialized array must equal unboxed array with same elements"
    );
    assert_ne!(arr1, materialized_diff);

    // Value::Array PartialEq parity (and must NOT materialize unboxed storage)
    let v_arr1 = Value::Array(Rc::new(arr1));
    let v_arr2 = Value::Array(Rc::new(arr2));
    let v_arr1_same = Value::Array(Rc::new(arr1_same));
    let v_mat = Value::Array(Rc::new(materialized));
    assert_ne!(v_arr1, v_arr2);
    assert_eq!(v_arr1, v_arr1_same);
    assert_eq!(v_arr1, v_mat);
    assert_eq!(v_mat, v_arr1);

    if let Value::Array(rc) = &v_arr1 {
        assert!(
            rc.borrow_elements_without_sync().is_empty(),
            "PartialEq must not materialize unboxed storage"
        );
    }

    // Cross-type unboxed comparisons: Int([1]) != Float([1.0]) != Bool([true])
    let u_int = LustArray::new(Vec::new());
    u_int.push_int(1);
    let u_float = LustArray::new(Vec::new());
    u_float.push_float(1.0);
    let u_bool = LustArray::new(Vec::new());
    u_bool.push_bool(true);

    assert_ne!(u_int, u_float);
    assert_ne!(u_int, u_bool);
    assert_ne!(u_float, u_bool);

    // Float & Bool parity with materialized arrays + NaN semantics
    let mat_float = LustArray::new(vec![Value::Float(1.0)]);
    let mat_bool = LustArray::new(vec![Value::Bool(true)]);
    assert_eq!(u_float, mat_float);
    assert_eq!(mat_float, u_float);
    assert_eq!(u_bool, mat_bool);
    assert_eq!(mat_bool, u_bool);

    let nan_a = LustArray::new(Vec::new());
    nan_a.push_float(f64::NAN);
    let nan_b = LustArray::new(Vec::new());
    nan_b.push_float(f64::NAN);
    assert_ne!(nan_a, nan_b, "IEEE-754 NaN array elements must not compare equal");
}

#[test]
fn test_ng_w10_1_f05_f21_try_borrow_reentrancy_no_panic_under_mut_and_shared() {
    // Case 1: Active mutable borrow via `borrow_mut()`
    let arr_mut = LustArray::new(Vec::new());
    arr_mut.push_value_fast(Value::Int(10));
    arr_mut.push_value_fast(Value::Int(20));
    {
        let _g = arr_mut.borrow_mut();
        assert!(
            arr_mut.try_borrow().is_err(),
            "try_borrow must return Err(BorrowError) while borrow_mut is active"
        );
    }
    assert_eq!(*arr_mut.try_borrow().expect("borrow after drop"), vec![Value::Int(10), Value::Int(20)]);

    // Case 2: Active shared borrow on `elements` while array holds unboxed data
    let arr_shared = LustArray::new(Vec::new());
    arr_shared.push_value_fast(Value::Bool(true));
    arr_shared.push_value_fast(Value::Bool(false));
    {
        let _shared_elems = arr_shared.borrow_elements_without_sync();
        assert!(
            arr_shared.try_borrow().is_err(),
            "try_borrow on unboxed array with active shared elements borrow must return Err(BorrowError) without panicking"
        );
    }
    // After dropping shared borrow, try_borrow lazily materializes and succeeds
    {
        let b1 = arr_shared.try_borrow().expect("try_borrow should succeed once shared guard drops");
        let b2 = arr_shared.try_borrow().expect("subsequent concurrent shared try_borrow must succeed");
        assert_eq!(*b1, vec![Value::Bool(true), Value::Bool(false)]);
        assert_eq!(*b2, *b1);
    }

    // Case 3: Active shared borrow obtained via `borrow()` on empty array before `push_value_fast`
    let arr_empty_then_push = LustArray::new(Vec::new());
    {
        let _shared_empty = arr_empty_then_push.borrow();
        arr_empty_then_push.push_value_fast(Value::Int(42));
        assert!(
            arr_empty_then_push.try_borrow().is_err(),
            "try_borrow must return Err(BorrowError) via synthetic_borrow_error() without panicking"
        );
    }
    assert_eq!(
        *arr_empty_then_push.try_borrow().expect("borrow after empty shared guard drops"),
        vec![Value::Int(42)]
    );

    // Case 4: Active shared borrow on `unboxed` via `with_unboxed_data`
    let arr_unboxed_borrow = LustArray::new(Vec::new());
    arr_unboxed_borrow.push_float(3.5);
    arr_unboxed_borrow.with_unboxed_data(|data| {
        assert!(matches!(data, UnboxedArrayData::Float(_)));
        assert!(
            arr_unboxed_borrow.try_borrow().is_err(),
            "try_borrow while unboxed is shared-borrowed must return Err without panicking"
        );
        assert!(
            arr_unboxed_borrow.try_borrow_mut().is_err(),
            "try_borrow_mut while unboxed is shared-borrowed must return Err without panicking"
        );
    });
    assert_eq!(
        *arr_unboxed_borrow.try_borrow().expect("borrow after with_unboxed_data"),
        vec![Value::Float(3.5)]
    );
}

#[test]
fn test_ng_w10_1_f05_f21_flat_cache_invalidation_and_clear_cached_flat() {
    let dummy_buf = || Rc::new(FlatArrayBuffer::new(vec![1, 1, 42]));

    // 1. push_value_fast invalidates flat cache & bumps generation
    let arr = LustArray::new(Vec::new());
    arr.push_value_fast(Value::Int(1));
    arr.set_cached_flat(FlatArrayElemTag::Int, dummy_buf());
    assert!(arr.has_cached_flat());
    assert!(arr.cached_flat(FlatArrayElemTag::Int).is_some());
    let g0 = arr.generation();
    arr.push_value_fast(Value::Int(2));
    assert!(!arr.has_cached_flat(), "push_value_fast must clear has_flat_cache");
    assert!(arr.cached_flat(FlatArrayElemTag::Int).is_none());
    assert!(arr.generation() > g0);

    // 2. set_value_fast invalidates flat cache & bumps generation
    arr.set_cached_flat(FlatArrayElemTag::Int, dummy_buf());
    assert!(arr.has_cached_flat());
    let g1 = arr.generation();
    assert_eq!(arr.set_value_fast(0, Value::Int(99)), Ok(()));
    assert!(!arr.has_cached_flat(), "set_value_fast must clear has_flat_cache");
    assert!(arr.cached_flat(FlatArrayElemTag::Int).is_none());
    assert!(arr.generation() > g1);

    // 3. clear_cached_flat resets has_cached_flat() to false
    arr.set_cached_flat(FlatArrayElemTag::Int, dummy_buf());
    assert!(arr.has_cached_flat());
    arr.clear_cached_flat();
    assert!(!arr.has_cached_flat(), "clear_cached_flat must reset has_flat_cache to false");
    assert!(arr.cached_flat(FlatArrayElemTag::Int).is_none());

    // 4. All typed unboxed mutators (`push_int`, `set_int`, `push_float`, `set_float`,
    //    `push_bool`, `set_bool`, `clear`) MUST call `invalidate_flat_cache()`
    arr.set_cached_flat(FlatArrayElemTag::Int, dummy_buf());
    arr.push_int(3);
    assert!(!arr.has_cached_flat());

    arr.set_cached_flat(FlatArrayElemTag::Int, dummy_buf());
    assert_eq!(arr.set_int(0, 7), Ok(()));
    assert!(!arr.has_cached_flat());

    arr.set_cached_flat(FlatArrayElemTag::Int, dummy_buf());
    arr.clear();
    assert!(!arr.has_cached_flat());
    assert!(arr.is_empty_fast());

    let arr_f = LustArray::new(Vec::new());
    arr_f.set_cached_flat(FlatArrayElemTag::Float, dummy_buf());
    arr_f.push_float(1.25);
    assert!(!arr_f.has_cached_flat());
    arr_f.set_cached_flat(FlatArrayElemTag::Float, dummy_buf());
    assert_eq!(arr_f.set_float(0, 2.5), Ok(()));
    assert!(!arr_f.has_cached_flat());

    let arr_b = LustArray::new(Vec::new());
    arr_b.set_cached_flat(FlatArrayElemTag::Bool, dummy_buf());
    arr_b.push_bool(true);
    assert!(!arr_b.has_cached_flat());
    arr_b.set_cached_flat(FlatArrayElemTag::Bool, dummy_buf());
    assert_eq!(arr_b.set_bool(0, false), Ok(()));
    assert!(!arr_b.has_cached_flat());
}

#[test]
fn test_ng_w10_1_heterogeneous_transition_oob_and_formatting_parity() {
    // Formatting parity (Debug & Display) without materializing unboxed arrays
    let u_int = Rc::new(LustArray::new(Vec::new()));
    u_int.push_int(10);
    u_int.push_int(-20);
    let m_int = Rc::new(LustArray::new(vec![Value::Int(10), Value::Int(-20)]));
    let v_u_int = Value::Array(Rc::clone(&u_int));
    let v_m_int = Value::Array(m_int);
    assert_eq!(format!("{v_u_int:?}"), format!("{v_m_int:?}"));
    assert_eq!(format!("{v_u_int}"), format!("{v_m_int}"));
    assert!(
        u_int.borrow_elements_without_sync().is_empty(),
        "Debug and Display must not materialize unboxed arrays"
    );

    let u_bool = Rc::new(LustArray::new(Vec::new()));
    u_bool.push_bool(true);
    u_bool.push_bool(false);
    let m_bool = Rc::new(LustArray::new(vec![Value::Bool(true), Value::Bool(false)]));
    assert_eq!(format!("{:?}", Value::Array(Rc::clone(&u_bool))), format!("{:?}", Value::Array(Rc::clone(&m_bool))));
    assert_eq!(format!("{}", Value::Array(Rc::clone(&u_bool))), format!("{}", Value::Array(m_bool)));
    assert!(u_bool.borrow_elements_without_sync().is_empty());

    // Out-of-bounds set must return Err(len) and must NOT materialize unboxed storage
    assert_eq!(u_bool.set_value_fast(2, Value::Int(999)), Err(2));
    assert_eq!(u_bool.set_int(5, 123), Err(2));
    assert!(
        u_bool.borrow_elements_without_sync().is_empty(),
        "OOB mismatched write must not materialize unboxed array"
    );

    // Heterogeneous write materializes cleanly into elements
    assert_eq!(u_bool.set_value_fast(1, Value::Int(999)), Ok(()));
    u_bool.with_unboxed_data(|data| assert_eq!(*data, UnboxedArrayData::None));
    assert_eq!(u_bool.len_fast(), 2);
    assert_eq!(u_bool.get_value_fast(0), Some(Value::Bool(true)));
    assert_eq!(u_bool.get_value_fast(1), Some(Value::Int(999)));

    // Heterogeneous push materializes cleanly into elements
    let u_mix = LustArray::new(Vec::new());
    u_mix.push_int(1);
    u_mix.push_value_fast(Value::string("two"));
    u_mix.with_unboxed_data(|data| assert_eq!(*data, UnboxedArrayData::None));
    assert_eq!(*u_mix.borrow(), vec![Value::Int(1), Value::string("two")]);

    // Layout probe invariant: elements stays at offset 0 of #[repr(C)] LustArray
    assert!(
        lust::jit::layout::rc_vec_layout().is_some(),
        "jit::layout::rc_vec_layout must still succeed with elements at offset 0"
    );
}
