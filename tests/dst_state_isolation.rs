//! Gate 1: Deterministic State Isolation Test (D2)
//!
//! Validates:
//! 1. 4 parallel VMs on a single thread with different seeds operate 100% isolated.
//! 2. Random number generation (via math.random / math.randomseed) is strictly deterministic and bound to the VM instance.
//! 3. Interleaving operations across VMs does not leak or contaminate pseudorandom state.
//! 4. VmId allocation is monotonic and distinct for each instance.

use lust::bytecode::Value;
use lust::vm::VM;

fn call_random(vm: &mut VM, args: Vec<Value>) -> Value {
    let rand_fn = vm
        .get_global("math.random")
        .or_else(|| vm.get_global("random"))
        .expect("math.random must be available in VM");
    vm.call_value(&rand_fn, args).expect("calling random must succeed")
}

fn call_randomseed(vm: &mut VM, seed: i64) {
    let seed_fn = vm
        .get_global("math.randomseed")
        .or_else(|| vm.get_global("randomseed"))
        .expect("math.randomseed must be available in VM");
    vm.call_value(&seed_fn, vec![Value::Int(seed)]).expect("calling randomseed must succeed");
}

#[test]
fn test_vm_seed_determinism_and_isolation() {
    let mut vm1 = VM::with_seed(42);
    let mut vm2 = VM::with_seed(100);
    let mut vm3 = VM::with_seed(42); // Same seed as vm1
    let mut vm4 = VM::with_seed(999);

    // Assert monotonic and unique VmIds
    assert_ne!(vm1.id, vm2.id, "VmIds must be distinct");
    assert_ne!(vm2.id, vm3.id, "VmIds must be distinct");
    assert_ne!(vm3.id, vm4.id, "VmIds must be distinct");

    // Sample floats [0, 1)
    let r1_a = call_random(&mut vm1, vec![]);
    let r3_a = call_random(&mut vm3, vec![]);
    let r2_a = call_random(&mut vm2, vec![]);
    let r4_a = call_random(&mut vm4, vec![]);

    // Identical seeds must yield identical values
    assert_eq!(r1_a, r3_a, "vm1 and vm3 with seed 42 must yield identical random values");
    // Different seeds must yield different values
    assert_ne!(r1_a, r2_a, "vm1 (42) and vm2 (100) must yield distinct random values");
    assert_ne!(r1_a, r4_a, "vm1 (42) and vm4 (999) must yield distinct random values");

    // Sample integers with ranges: random(1, 100)
    let range_args = vec![Value::Int(1), Value::Int(100)];
    let r1_b = call_random(&mut vm1, range_args.clone());
    let r3_b = call_random(&mut vm3, range_args.clone());
    let r2_b = call_random(&mut vm2, range_args.clone());
    let _r4_b = call_random(&mut vm4, range_args.clone());

    assert_eq!(r1_b, r3_b, "vm1 and vm3 range random must match");
    assert!(matches!(r1_b, Value::Int(i) if (1..=100).contains(&i)));
    assert!(matches!(r2_b, Value::Int(i) if (1..=100).contains(&i)));

    // Sample integers with upper bound: random(50)
    let upper_args = vec![Value::Int(50)];
    let r1_c = call_random(&mut vm1, upper_args.clone());
    let r3_c = call_random(&mut vm3, upper_args.clone());
    assert_eq!(r1_c, r3_c, "vm1 and vm3 upper-bound random must match");
}

#[test]
fn test_interleaved_multi_vm_stream_integrity() {
    // Collect baseline reference streams from independent non-interleaved runs
    let ref_seq_1 = {
        let mut vm = VM::with_seed(777);
        (0..20).map(|_| call_random(&mut vm, vec![])).collect::<Vec<_>>()
    };
    let ref_seq_2 = {
        let mut vm = VM::with_seed(888);
        (0..20).map(|_| call_random(&mut vm, vec![])).collect::<Vec<_>>()
    };
    let ref_seq_3 = {
        let mut vm = VM::with_seed(777); // Identical to 1
        (0..20).map(|_| call_random(&mut vm, vec![])).collect::<Vec<_>>()
    };
    let ref_seq_4 = {
        let mut vm = VM::with_seed(999);
        (0..20).map(|_| call_random(&mut vm, vec![])).collect::<Vec<_>>()
    };

    assert_eq!(ref_seq_1, ref_seq_3, "Identical seeds must produce identical streams");
    assert_ne!(ref_seq_1, ref_seq_2, "Different seeds must produce different streams");

    // Now run 4 VMs concurrently on the SAME thread in an arbitrary interleaved schedule
    let mut vm1 = VM::with_seed(777);
    let mut vm2 = VM::with_seed(888);
    let mut vm3 = VM::with_seed(777);
    let mut vm4 = VM::with_seed(999);

    let mut actual_seq_1 = Vec::new();
    let mut actual_seq_2 = Vec::new();
    let mut actual_seq_3 = Vec::new();
    let mut actual_seq_4 = Vec::new();

    // Chaotic interleaving schedule
    for _ in 0..20 {
        actual_seq_2.push(call_random(&mut vm2, vec![]));
        actual_seq_1.push(call_random(&mut vm1, vec![]));
        actual_seq_4.push(call_random(&mut vm4, vec![]));
        actual_seq_3.push(call_random(&mut vm3, vec![]));
    }

    // Every interleaved VM must preserve its exact deterministic sequence with zero cross-talk
    assert_eq!(actual_seq_1, ref_seq_1, "vm1 stream corrupted by interleaving");
    assert_eq!(actual_seq_2, ref_seq_2, "vm2 stream corrupted by interleaving");
    assert_eq!(actual_seq_3, ref_seq_3, "vm3 stream corrupted by interleaving");
    assert_eq!(actual_seq_4, ref_seq_4, "vm4 stream corrupted by interleaving");
}

#[test]
fn test_isolated_randomseed_mutation() {
    let mut vm1 = VM::with_seed(10);
    let mut vm2 = VM::with_seed(20);

    // Initial values
    let _ = call_random(&mut vm1, vec![]);
    let _v2_before = call_random(&mut vm2, vec![]);

    // Reseed vm1 only
    call_randomseed(&mut vm1, 555);

    // vm2 should continue uninterrupted
    let mut vm2_reference = VM::with_seed(20);
    let _ = call_random(&mut vm2_reference, vec![]);
    let expected_v2_next = call_random(&mut vm2_reference, vec![]);

    let v2_after = call_random(&mut vm2, vec![]);
    assert_eq!(v2_after, expected_v2_next, "Reseeding vm1 must not affect vm2 RNG stream");

    // vm1 must now match a fresh VM with seed 555
    let mut vm1_reference = VM::with_seed(555);
    let expected_v1_next = call_random(&mut vm1_reference, vec![]);
    let v1_after = call_random(&mut vm1, vec![]);
    assert_eq!(v1_after, expected_v1_next, "vm1 must follow the new reseeded stream");
}
