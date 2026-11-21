//! PERF-T1-6: Fast-path gate and exact active-entry accounting for thread-local
//! `SPECIALIZED_{BOOL,INT,FLOAT}_CACHE` tables in `src/bytecode/value.rs`.

use lust::bytecode::value::{
    ArraySequenceId, JitVecSlot, VmId, get_specialized_cached_element, get_specialized_cached_len,
    invalidate_specialized_array_cache, invalidate_specialized_array_cache_by_key,
    jit_drop_vec_bool, jit_drop_vec_float, jit_drop_vec_int, jit_rebox_array_bool,
    jit_rebox_array_float, jit_rebox_array_int, jit_unbox_array_bool, jit_unbox_array_float,
    jit_unbox_array_int, set_specialized_cached_element, specialized_cache_active_entries,
    sync_all_specialized_arrays, try_sync_specialized_array,
};
use lust::bytecode::Value;
use lust::vm::{CurrentVmGuard, VM};
use std::cell::RefCell;
use std::rc::Rc;

fn empty_slot() -> JitVecSlot {
    JitVecSlot {
        ptr: core::ptr::null_mut(),
        len: 0,
        cap: 0,
        array: core::ptr::null(),
    }
}

#[test]
fn test_perf_t1_6_empty_cache_default_and_noops() {
    sync_all_specialized_arrays();
    assert_eq!(specialized_cache_active_entries(), 0);

    let arr = Rc::new(RefCell::new(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));

    assert_eq!(get_specialized_cached_element(&arr, 0), None);
    assert_eq!(get_specialized_cached_len(&arr), None);
    assert!(set_specialized_cached_element(&arr, 0, &Value::Int(9)).is_none());
    invalidate_specialized_array_cache(&arr);
    invalidate_specialized_array_cache(arr.as_ptr() as usize);
    invalidate_specialized_array_cache_by_key((VmId(0), ArraySequenceId(12345)));
    assert_eq!(try_sync_specialized_array(&arr), Ok(()));
    sync_all_specialized_arrays();

    // Underlying array untouched and active entries remain strictly 0.
    assert_eq!(specialized_cache_active_entries(), 0);
    assert_eq!(arr.borrow().len(), 3);
    assert_eq!(arr.borrow()[0], Value::Int(1));
}

#[test]
fn test_perf_t1_6_exact_accounting_across_bool_int_float_and_reinsert() {
    sync_all_specialized_arrays();
    assert_eq!(specialized_cache_active_entries(), 0);

    let arr_int = Rc::new(RefCell::new(vec![Value::Int(10), Value::Int(20)]));
    let arr_bool = Rc::new(RefCell::new(vec![Value::Bool(true), Value::Bool(false)]));
    let arr_float = Rc::new(RefCell::new(vec![Value::Float(1.25), Value::Float(2.5)]));

    let val_int = Value::Array(Rc::clone(&arr_int));
    let val_bool = Value::Array(Rc::clone(&arr_bool));
    let val_float = Value::Array(Rc::clone(&arr_float));

    let mut slot = empty_slot();

    // Insert int
    unsafe {
        assert_eq!(jit_unbox_array_int(&val_int, &mut slot), 1);
        *slot.ptr = 42;
        assert_eq!(jit_rebox_array_int(&mut slot), 1);
    }
    assert_eq!(specialized_cache_active_entries(), 1);

    // Unbox consumes cached int entry (count -> 0), rebox puts it back (count -> 1).
    unsafe {
        assert_eq!(jit_unbox_array_int(&val_int, &mut slot), 1);
        assert_eq!(specialized_cache_active_entries(), 0);
        *slot.ptr.add(1) = 84;
        assert_eq!(jit_rebox_array_int(&mut slot), 1);
    }
    assert_eq!(specialized_cache_active_entries(), 1);

    // Insert bool & float
    unsafe {
        assert_eq!(jit_unbox_array_bool(&val_bool, &mut slot), 1);
        *(slot.ptr as *mut u8) = 0;
        assert_eq!(jit_rebox_array_bool(&mut slot), 1);

        assert_eq!(jit_unbox_array_float(&val_float, &mut slot), 1);
        *(slot.ptr as *mut f64) = 7.5;
        assert_eq!(jit_rebox_array_float(&mut slot), 1);
    }
    assert_eq!(specialized_cache_active_entries(), 3);

    // Verify reads, OOB error, in-place write, and sync.
    assert_eq!(get_specialized_cached_element(&arr_int, 0), Some(Value::Int(42)));
    assert_eq!(get_specialized_cached_element(&arr_int, 1), Some(Value::Int(84)));
    assert_eq!(get_specialized_cached_len(&arr_int), Some(2));
    assert!(matches!(
        set_specialized_cached_element(&arr_int, 5, &Value::Int(1)),
        Some(Err(_))
    ));
    assert_eq!(
        set_specialized_cached_element(&arr_bool, 1, &Value::Bool(true)),
        Some(Ok(()))
    );
    assert_eq!(
        set_specialized_cached_element(&arr_float, 1, &Value::Float(9.5)),
        Some(Ok(()))
    );
    assert_eq!(specialized_cache_active_entries(), 3);

    sync_all_specialized_arrays();
    assert_eq!(arr_int.borrow()[0], Value::Int(42));
    assert_eq!(arr_int.borrow()[1], Value::Int(84));
    assert_eq!(arr_bool.borrow()[0], Value::Bool(false));
    assert_eq!(arr_bool.borrow()[1], Value::Bool(true));
    assert_eq!(arr_float.borrow()[0], Value::Float(7.5));
    assert_eq!(arr_float.borrow()[1], Value::Float(9.5));
    assert_eq!(specialized_cache_active_entries(), 3);

    // Repeated invalidation of already-invalidated key must not underflow.
    invalidate_specialized_array_cache(&arr_int);
    assert_eq!(specialized_cache_active_entries(), 2);
    invalidate_specialized_array_cache(&arr_int);
    assert_eq!(specialized_cache_active_entries(), 2);

    // Unbox + jit_drop_vec removes and invalidates cleanly.
    unsafe {
        assert_eq!(jit_unbox_array_bool(&val_bool, &mut slot), 1);
        assert_eq!(specialized_cache_active_entries(), 1);
        jit_drop_vec_bool(&mut slot);
        assert_eq!(specialized_cache_active_entries(), 1);

        assert_eq!(jit_unbox_array_float(&val_float, &mut slot), 1);
        assert_eq!(specialized_cache_active_entries(), 0);
        jit_drop_vec_float(&mut slot);
        assert_eq!(specialized_cache_active_entries(), 0);

        // Also test jit_drop_vec_int on empty cache.
        assert_eq!(jit_unbox_array_int(&val_int, &mut slot), 1);
        jit_drop_vec_int(&mut slot);
        assert_eq!(specialized_cache_active_entries(), 0);
    }
}

#[test]
fn test_perf_t1_6_dead_weak_pruning_and_vm_guard_transition() {
    sync_all_specialized_arrays();
    assert_eq!(specialized_cache_active_entries(), 0);

    // 1. Dropped Rc entries pruned by sync_all_specialized_arrays must decrement active count back to 0.
    {
        let ephemeral_int = Value::array(vec![Value::Int(1), Value::Int(2)]);
        let ephemeral_bool = Value::array(vec![Value::Bool(true)]);
        let ephemeral_float = Value::array(vec![Value::Float(3.5)]);
        let mut slot = empty_slot();
        unsafe {
            assert_eq!(jit_unbox_array_int(&ephemeral_int, &mut slot), 1);
            assert_eq!(jit_rebox_array_int(&mut slot), 1);
            assert_eq!(jit_unbox_array_bool(&ephemeral_bool, &mut slot), 1);
            assert_eq!(jit_rebox_array_bool(&mut slot), 1);
            assert_eq!(jit_unbox_array_float(&ephemeral_float, &mut slot), 1);
            assert_eq!(jit_rebox_array_float(&mut slot), 1);
        }
        assert_eq!(specialized_cache_active_entries(), 3);
    }
    // All 3 arrays are dropped; weak pointers are dead.
    sync_all_specialized_arrays();
    assert_eq!(
        specialized_cache_active_entries(),
        0,
        "sync_all_specialized_arrays must decrement active count when pruning dead Weak entries"
    );

    // 2. Fallback VmId(0) entry synced inside an active CurrentVmGuard must remove stale non-primary key and decrement count.
    let mut vm = VM::new();
    let arr = Rc::new(RefCell::new(vec![Value::Int(5), Value::Int(6)]));
    let val = Value::Array(Rc::clone(&arr));
    let mut slot = empty_slot();
    unsafe {
        assert_eq!(jit_unbox_array_int(&val, &mut slot), 1);
        *slot.ptr = 50;
        assert_eq!(jit_rebox_array_int(&mut slot), 1);
    }
    assert_eq!(specialized_cache_active_entries(), 1);

    {
        let _guard = CurrentVmGuard::from_vm(&mut vm);
        // Active VM has no primary key for `arr`, so non-primary VmId(0) entry is synced and removed.
        assert_eq!(try_sync_specialized_array(&arr), Ok(()));
        assert_eq!(arr.borrow()[0], Value::Int(50));
        assert_eq!(
            specialized_cache_active_entries(),
            0,
            "Removing non-primary fallback entry during sync must decrement active count to 0"
        );
    }

    // 3. Rebox under VM guard, then invalidate by SpecializedCacheKey directly.
    let vm_key = {
        let _guard = CurrentVmGuard::from_vm(&mut vm);
        unsafe {
            assert_eq!(jit_unbox_array_int(&val, &mut slot), 1);
            *slot.ptr = 77;
            assert_eq!(jit_rebox_array_int(&mut slot), 1);
        }
        assert_eq!(specialized_cache_active_entries(), 1);
        vm.get_or_register_array_id(arr.as_ptr() as usize)
    };
    assert_eq!(get_specialized_cached_element(&arr, 0), Some(Value::Int(77)));
    invalidate_specialized_array_cache(vm_key);
    assert_eq!(specialized_cache_active_entries(), 0);
    assert_eq!(get_specialized_cached_element(&arr, 0), None);
}

#[test]
fn test_perf_t1_6_rebox_overwrite_stale_key_and_length_mismatch_eviction() {
    sync_all_specialized_arrays();
    assert_eq!(specialized_cache_active_entries(), 0);

    // 1. Overwriting an existing key in jit_rebox_array_{int,bool,float} must not double-count.
    let arr_int = Rc::new(RefCell::new(vec![Value::Int(1), Value::Int(2)]));
    let val_int = Value::Array(Rc::clone(&arr_int));
    let mut slot1 = empty_slot();
    let mut slot2 = empty_slot();
    unsafe {
        assert_eq!(jit_unbox_array_int(&val_int, &mut slot1), 1);
        assert_eq!(jit_unbox_array_int(&val_int, &mut slot2), 1);
        *slot1.ptr = 10;
        *slot2.ptr = 20;
        assert_eq!(jit_rebox_array_int(&mut slot1), 1);
        assert_eq!(specialized_cache_active_entries(), 1);
        // Same-key overwrite must keep active count == 1.
        assert_eq!(jit_rebox_array_int(&mut slot2), 1);
        assert_eq!(specialized_cache_active_entries(), 1);
    }
    assert_eq!(get_specialized_cached_element(&arr_int, 0), Some(Value::Int(20)));

    // 2. Reboxing under a VM guard while a stale fallback VmId(0) entry exists must replace cleanly (count stays 1).
    let mut vm = VM::new();
    let mut slot3 = empty_slot();
    unsafe {
        let arr_clone = Value::Array(Rc::clone(&arr_int));
        // Unbox outside cache by first reading from arr_int into a separate slot?
        // Actually unbox removes from cache; so populate slot3 first while empty, then slot1 outside VM, then slot3 inside VM.
        invalidate_specialized_array_cache(&arr_int);
        assert_eq!(specialized_cache_active_entries(), 0);

        assert_eq!(jit_unbox_array_int(&arr_clone, &mut slot1), 1);
        assert_eq!(jit_unbox_array_int(&arr_clone, &mut slot3), 1);
        *slot1.ptr = 30;
        *slot3.ptr = 40;
        // Rebox slot1 outside VM -> inserts (VmId(0), seq)
        assert_eq!(jit_rebox_array_int(&mut slot1), 1);
        assert_eq!(specialized_cache_active_entries(), 1);
        {
            let _guard = CurrentVmGuard::from_vm(&mut vm);
            // Rebox slot3 inside VM -> removes stale (VmId(0), seq) and inserts (vm.id, seq)
            assert_eq!(jit_rebox_array_int(&mut slot3), 1);
            assert_eq!(specialized_cache_active_entries(), 1);
            assert_eq!(get_specialized_cached_element(&arr_int, 0), Some(Value::Int(40)));

            // 3. Error during try_sync_specialized_array (array currently borrowed) must not drop entry or decrement count.
            {
                let _borrowed = arr_int.borrow();
                assert!(try_sync_specialized_array(&arr_int).is_err());
                assert_eq!(specialized_cache_active_entries(), 1);
            }
            // Sync now succeeds (primary key matches active VM -> dirty cleared, entry retained).
            assert_eq!(try_sync_specialized_array(&arr_int), Ok(()));
            assert_eq!(arr_int.borrow()[0], Value::Int(40));
            assert_eq!(specialized_cache_active_entries(), 1);

            // 4. Underlying array length change while clean triggers eviction on try_sync_specialized_array -> count drops to 0.
            arr_int.borrow_mut().push(Value::Int(99));
            assert_eq!(try_sync_specialized_array(&arr_int), Ok(()));
            assert_eq!(
                specialized_cache_active_entries(),
                0,
                "Evicting length-mismatched clean entry during sync must decrement active count to 0"
            );
            assert_eq!(get_specialized_cached_len(&arr_int), None);
        }
    }
}
