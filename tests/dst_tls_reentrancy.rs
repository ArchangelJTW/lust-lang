//! Gate 1: TLS Reentrancy & Lua Registry / Specialization Cache Isolation Test (D4, D5)
//!
//! Validates:
//! 1. LuaCompatContext isolates function registries and IDs across independent runtime instances.
//! 2. Reentrant execution of Lua contexts does not cause cross-talk.
//! 3. Array specialization caches use composite (VmId, ArraySequenceId) keys, preventing collisions
//!    even when memory addresses are reused or shared across VMs on the same thread.

use lust::bytecode::value::{
    ArraySequenceId, invalidate_specialized_array_cache, resolve_array_key_from_ptr,
    set_specialized_cached_element,
};
use lust::bytecode::Value;
use lust::lua_compat::{
    LuaCompatContext, enter_lua_compat_context, lookup_lust_function, register_lust_function,
};
use lust::vm::{self, VM};
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn test_lua_registry_isolation_and_reentrancy() {
    let ctx1 = LuaCompatContext::new();
    let ctx2 = LuaCompatContext::new();

    // In ctx1, register functions
    {
        let _g1 = enter_lua_compat_context(&ctx1);
        let id1 = register_lust_function(Value::Int(101));
        let id2 = register_lust_function(Value::Int(102));

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(lookup_lust_function(1), Some(Value::Int(101)));
        assert_eq!(lookup_lust_function(2), Some(Value::Int(102)));

        // Reentrant nested entry into ctx2
        {
            let _g2 = enter_lua_compat_context(&ctx2);
            // ctx2 starts with its own ID sequence starting at 1
            let id2_1 = register_lust_function(Value::Int(201));
            assert_eq!(id2_1, 1, "ctx2 must allocate its own independent function IDs");
            assert_eq!(lookup_lust_function(1), Some(Value::Int(201)), "Active ctx2 must see its own value");
            // id2 should be absent in ctx2
            assert_eq!(lookup_lust_function(2), None);

            // Directly inspect ctx1 to ensure zero leakage
            assert_eq!(ctx1.lookup(1), Some(Value::Int(101)));
        } // ctx2 popped

        // Returning to ctx1
        assert_eq!(lookup_lust_function(1), Some(Value::Int(101)));
        assert_eq!(lookup_lust_function(2), Some(Value::Int(102)));

        let id3 = register_lust_function(Value::Int(103));
        assert_eq!(id3, 3);
        assert_eq!(lookup_lust_function(3), Some(Value::Int(103)));
    }
}

#[test]
fn test_specialized_cache_vmid_qualification_no_collisions() {
    let mut vm1 = VM::with_seed(1);
    let mut vm2 = VM::with_seed(2);

    assert_ne!(vm1.id, vm2.id, "VMs must have distinct VmId");

    let arr1 = Rc::new(RefCell::new(vec![Value::Int(10), Value::Int(20), Value::Int(30)]));
    let arr2 = Rc::new(RefCell::new(vec![Value::Int(100), Value::Int(200), Value::Int(300)]));

    let ptr1 = arr1.as_ptr() as usize;
    let ptr2 = arr2.as_ptr() as usize;

    // Simulate key allocation under vm1
    let key1 = vm1.get_or_register_array_id(ptr1);
    assert_eq!(key1.0, vm1.id);
    assert_eq!(key1.1, ArraySequenceId(1));

    // Under vm2, even if we register the exact same memory address ptr1:
    let key2_simulated = vm2.get_or_register_array_id(ptr1);
    assert_eq!(key2_simulated.0, vm2.id);
    assert_eq!(key2_simulated.1, ArraySequenceId(1));

    // The composite keys are completely distinct despite identical pointer addresses!
    assert_ne!(key1, key2_simulated, "Composite keys must never collide across different VMs");

    // Under vm2, real arr2 gets:
    let key2 = vm2.get_or_register_array_id(ptr2);
    assert_eq!(key2.0, vm2.id);
    assert_eq!(key2.1, ArraySequenceId(2));

    // Test caching operations with active VM context switching on the single thread
    // Activate vm1
    {
        let _g1 = vm::CurrentVmGuard::from_vm(&mut vm1);
        assert_eq!(resolve_array_key_from_ptr(ptr1), key1);

        // Populate cache for arr1 in vm1
        let set_res = set_specialized_cached_element(&arr1, 0, &Value::Int(11));
        assert!(set_res.is_none()); // not yet cached, but registers dirty or triggers sync
    }

    // Activate vm2
    {
        let _g2 = vm::CurrentVmGuard::from_vm(&mut vm2);
        assert_eq!(resolve_array_key_from_ptr(ptr2), key2);

        // In vm2, ptr1 maps to vm2's key, not vm1's key!
        assert_eq!(resolve_array_key_from_ptr(ptr1), key2_simulated);
    }

    // Invalidation: invalidating array in vm1 does not invalidate vm2
    invalidate_specialized_array_cache(key1);
    // vm2 still holds key2
    assert_eq!(vm2.array_id_map.borrow().get(&ptr2), Some(&ArraySequenceId(2)));
}
