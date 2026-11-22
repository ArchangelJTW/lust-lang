#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Mutex;

use lust::bytecode::Value;
use lust::mir::cranelift_backend::{
    safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::vfs::VirtualConsole;
use lust::vm::cranelift_dispatch::{
    cranelift_writeback_commit_count, reset_cranelift_writeback_commit_count,
};
use lust::vm::{VM, reset_effect_handler_state};
use lust::{EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
    reset_cranelift_writeback_commit_count();
}

fn compile_embedded_vm(source: &str, cranelift: bool) -> VM {
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load embedded program");
    let mut cfg = LustConfig::default().with_cranelift_jit(cranelift);
    cfg.set_use_mir(false);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile embedded program");
    vm.console = VirtualConsole::captured();
    vm
}

fn find_fn_name(vm: &VM, suffix: &str) -> String {
    vm.mir_functions
        .keys()
        .find(|k| *k == suffix || k.ends_with(&format!(".{suffix}")))
        .cloned()
        .unwrap_or_else(|| panic!("MIR function '{suffix}' not found in {:?}", vm.mir_functions.keys().collect::<Vec<_>>()))
}

#[test]
fn test_perf_t2_4_readonly_array_param_skips_writeback() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
struct Point
    x: int
    y: int
end

function count_even(arr: Array<int>): int
    local n: int = array.len(arr)
    local c: int = 0
    local i: int = 0
    while i < n do
        if arr[i] is Ok(v) then
            if v % 2 == 0 then
                c = c + 1
            end
        end
        i = i + 1
    end
    return c
end

function inc_first(arr: Array<int>): int
    if arr[0] is Ok(v) then
        arr[0] = v + 1
        return v + 1
    end
    return -1
end

function identity_arr(arr: Array<int>): Array<int>
    return arr
end

function sum_point(p: Point): int
    return p.x + p.y
end

function bump_point(p: Point): int
    p.x = p.x + 10
    return p.x
end

function make_point(x: int, y: int): Point
    return Point { x = x, y = y }
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let count_even_name = find_fn_name(&vm, "count_even");
    let inc_first_name = find_fn_name(&vm, "inc_first");
    let identity_arr_name = find_fn_name(&vm, "identity_arr");
    let sum_point_name = find_fn_name(&vm, "sum_point");
    let bump_point_name = find_fn_name(&vm, "bump_point");
    let make_point_name = find_fn_name(&vm, "make_point");

    let arr_rc = Rc::new(RefCell::new(vec![
        Value::Int(2),
        Value::Int(3),
        Value::Int(4),
        Value::Int(5),
        Value::Int(6),
    ]));

    // 1. Read-only `count_even(arr)` called 100 times MUST execute in Cranelift and commit 0 writebacks!
    reset_cranelift_writeback_commit_count();
    let entries_before = vm.cranelift_entry_count;
    for _ in 0..100 {
        let res = vm
            .call(&count_even_name, vec![Value::Array(arr_rc.clone())])
            .expect("count_even call");
        assert_eq!(res, Value::Int(3));
    }
    assert_eq!(
        vm.cranelift_entry_count - entries_before,
        100,
        "count_even must enter Cranelift on all 100 calls"
    );
    assert_eq!(
        vm.cranelift_writeback_commit_count(),
        0,
        "read-only count_even(arr) must skip writeback commits across all 100 calls"
    );
    assert_eq!(
        vm.cranelift_readonly_params(&count_even_name),
        Some(vec![true]),
        "count_even param 0 must be proven read-only"
    );

    // 2. Read-only `identity_arr(arr)` skips writeback AND preserves exact caller Rc identity!
    reset_cranelift_writeback_commit_count();
    let ret_val = vm
        .call(&identity_arr_name, vec![Value::Array(arr_rc.clone())])
        .expect("identity_arr call");
    match ret_val {
        Value::Array(ret_rc) => {
            assert!(
                Rc::ptr_eq(&ret_rc, &arr_rc),
                "identity_arr must preserve exact caller Rc::ptr_eq identity"
            );
        }
        other => panic!("expected Value::Array, got {other:?}"),
    }
    assert_eq!(
        vm.cranelift_readonly_params(&identity_arr_name),
        Some(vec![true])
    );
    assert_eq!(
        cranelift_writeback_commit_count(),
        0,
        "read-only identity_arr must skip writeback commits"
    );

    // 3. Mutating `inc_first(arr)` MUST commit writebacks and mutate caller's array!
    reset_cranelift_writeback_commit_count();
    let inc_res = vm
        .call(&inc_first_name, vec![Value::Array(arr_rc.clone())])
        .expect("inc_first call");
    assert_eq!(inc_res, Value::Int(3));
    assert_eq!(
        vm.cranelift_readonly_params(&inc_first_name),
        Some(vec![false]),
        "inc_first param 0 must be marked mutated (readonly=false)"
    );
    assert!(
        cranelift_writeback_commit_count() >= 1,
        "mutating inc_first(arr) must commit at least 1 writeback"
    );
    assert_eq!(
        arr_rc.borrow()[0],
        Value::Int(3),
        "caller array element 0 must reflect writeback mutation"
    );

    // 4. Read-only `sum_point(p)` skips struct writeback while mutating `bump_point(p)` commits it!
    let pt_val = vm
        .call(&make_point_name, vec![Value::Int(7), Value::Int(5)])
        .expect("make_point");
    let pt_rc = match &pt_val {
        Value::Struct(rc) => rc.clone(),
        other => panic!("expected Value::Struct, got {other:?}"),
    };
    reset_cranelift_writeback_commit_count();
    for _ in 0..50 {
        let s = vm
            .call(&sum_point_name, vec![Value::Struct(pt_rc.clone())])
            .expect("sum_point");
        assert_eq!(s, Value::Int(12));
    }
    assert_eq!(
        vm.cranelift_readonly_params(&sum_point_name),
        Some(vec![true]),
        "sum_point param 0 must be proven read-only"
    );
    assert_eq!(
        cranelift_writeback_commit_count(),
        0,
        "read-only sum_point(p) must skip struct writeback commits"
    );

    reset_cranelift_writeback_commit_count();
    let bumped = vm
        .call(&bump_point_name, vec![Value::Struct(pt_rc.clone())])
        .expect("bump_point");
    assert_eq!(bumped, Value::Int(17));
    assert_eq!(
        vm.cranelift_readonly_params(&bump_point_name),
        Some(vec![false]),
        "bump_point param 0 must be marked mutated"
    );
    assert!(
        cranelift_writeback_commit_count() >= 1,
        "mutating bump_point(p) must commit struct writeback"
    );
    assert_eq!(
        pt_rc.fields.borrow()[0],
        Value::Int(17),
        "caller Point.x must reflect writeback mutation"
    );
}

#[test]
fn test_perf_t2_4_aliased_array_and_struct_params_still_writeback() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
struct Point
    x: int
    y: int
end

function read_first_mutate_second_arr(ro: Array<int>, rw: Array<int>): int
    local before: int = 0
    if ro[0] is Ok(v) then
        before = v
    end
    rw[0] = 42
    local after: int = 0
    if ro[0] is Ok(v2) then
        after = v2
    end
    return before * 100 + after
end

function read_first_mutate_second_struct(ro: Point, rw: Point): int
    local before: int = ro.x
    rw.x = 42
    return before * 100 + ro.x
end

function make_point(x: int, y: int): Point
    return Point { x = x, y = y }
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let arr_fn = find_fn_name(&vm, "read_first_mutate_second_arr");
    let struct_fn = find_fn_name(&vm, "read_first_mutate_second_struct");
    let make_point_fn = find_fn_name(&vm, "make_point");

    // 1. Aliased array f(a, a): param 0 is read-only, param 1 mutates a[0] = 42 -> MUST write back 42!
    let a_rc = Rc::new(RefCell::new(vec![Value::Int(5), Value::Int(9)]));
    reset_cranelift_writeback_commit_count();
    let res = vm
        .call(&arr_fn, vec![Value::Array(a_rc.clone()), Value::Array(a_rc.clone())])
        .expect("aliased f(a, a)");
    assert_eq!(res, Value::Int(542));
    assert_eq!(
        vm.cranelift_readonly_params(&arr_fn),
        Some(vec![true, false]),
        "param 0 must be readonly and param 1 mutated"
    );
    assert!(
        cranelift_writeback_commit_count() >= 1,
        "aliased f(a, a) must still commit writeback because param 1 mutates the shared Rc"
    );
    assert_eq!(
        *a_rc.borrow(),
        vec![Value::Int(42), Value::Int(9)],
        "aliased f(a, a) must write back a[0] = 42"
    );

    // Non-aliased f(a2, b2): only b2 is mutated; a2 stays untouched.
    let a2_rc = Rc::new(RefCell::new(vec![Value::Int(3), Value::Int(8)]));
    let b2_rc = Rc::new(RefCell::new(vec![Value::Int(7), Value::Int(1)]));
    let res_distinct = vm
        .call(&arr_fn, vec![Value::Array(a2_rc.clone()), Value::Array(b2_rc.clone())])
        .expect("distinct f(a2, b2)");
    assert_eq!(res_distinct, Value::Int(303));
    assert_eq!(*a2_rc.borrow(), vec![Value::Int(3), Value::Int(8)]);
    assert_eq!(*b2_rc.borrow(), vec![Value::Int(42), Value::Int(1)]);

    // 2. Aliased struct g(p, p): param 0 is read-only, param 1 mutates p.x = 42 -> MUST write back 42!
    let pt_val = vm
        .call(&make_point_fn, vec![Value::Int(6), Value::Int(11)])
        .expect("make_point");
    let pt_rc = match pt_val {
        Value::Struct(rc) => rc,
        other => panic!("expected Value::Struct, got {other:?}"),
    };
    reset_cranelift_writeback_commit_count();
    let sres = vm
        .call(&struct_fn, vec![Value::Struct(pt_rc.clone()), Value::Struct(pt_rc.clone())])
        .expect("aliased g(p, p)");
    assert_eq!(sres, Value::Int(642));
    assert_eq!(
        vm.cranelift_readonly_params(&struct_fn),
        Some(vec![true, false]),
        "struct param 0 must be readonly and param 1 mutated"
    );
    assert!(
        cranelift_writeback_commit_count() >= 1,
        "aliased g(p, p) must commit struct writeback"
    );
    assert_eq!(
        *pt_rc.fields.borrow(),
        vec![Value::Int(42), Value::Int(11)],
        "aliased g(p, p) must write back p.x = 42"
    );
}

#[test]
fn test_perf_t2_4_transitive_helper_mutation_writes_back() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // `mutate_loop_helper` contains a loop so MirInliner will NOT inline it,
    // exercising true interprocedural fixed-point propagation in compute_readonly_params!
    // `id_loop` also contains a loop (not inlined) and returns its param without mutating it,
    // exercising return-value alias propagation in `mutate_via_returned_alias`!
    let src = r#"
function mutate_loop_helper(x: Array<int>): int
    local i: int = 0
    while i < 1 do
        x[0] = 99
        i = i + 1
    end
    return 1
end

function caller_transitive(a: Array<int>): int
    return mutate_loop_helper(a)
end

function id_loop(x: Array<int>): Array<int>
    local i: int = 0
    while i < 1 do
        i = i + 1
    end
    return x
end

function mutate_via_returned_alias(a: Array<int>): int
    local b: Array<int> = id_loop(a)
    b[0] = 77
    return 1
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let caller_fn = find_fn_name(&vm, "caller_transitive");
    let id_fn = find_fn_name(&vm, "id_loop");
    let alias_ret_fn = find_fn_name(&vm, "mutate_via_returned_alias");

    let a_rc = Rc::new(RefCell::new(vec![Value::Int(1), Value::Int(2)]));
    reset_cranelift_writeback_commit_count();
    let r1 = vm
        .call(&caller_fn, vec![Value::Array(a_rc.clone())])
        .expect("caller_transitive");
    assert_eq!(r1, Value::Int(1));
    assert_eq!(
        vm.cranelift_readonly_params(&caller_fn),
        Some(vec![false]),
        "caller_transitive param 0 must propagate mutation transitively from mutate_loop_helper"
    );
    assert!(cranelift_writeback_commit_count() >= 1);
    assert_eq!(
        *a_rc.borrow(),
        vec![Value::Int(99), Value::Int(2)],
        "transitive helper mutation must write back to caller array"
    );

    let b_rc = Rc::new(RefCell::new(vec![Value::Int(10), Value::Int(20)]));
    reset_cranelift_writeback_commit_count();
    let r2 = vm
        .call(&alias_ret_fn, vec![Value::Array(b_rc.clone())])
        .expect("mutate_via_returned_alias");
    assert_eq!(r2, Value::Int(1));
    assert_eq!(
        vm.cranelift_readonly_params(&id_fn),
        Some(vec![true]),
        "id_loop itself does not mutate its parameter"
    );
    assert_eq!(
        vm.cranelift_readonly_params(&alias_ret_fn),
        Some(vec![false]),
        "mutate_via_returned_alias must detect mutation through alias returned by id_loop"
    );
    assert!(cranelift_writeback_commit_count() >= 1);
    assert_eq!(
        *b_rc.borrow(),
        vec![Value::Int(77), Value::Int(20)],
        "mutation through returned alias must write back to caller array"
    );
}

#[test]
fn test_perf_t2_4_array_push_marks_param_mutated_and_writes_back() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function push_many(a: Array<int>, count: int): int
    local i: int = 1
    while i <= count do
        array.push(a, i * 10)
        i = i + 1
    end
    return array.len(a)
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let push_fn = find_fn_name(&vm, "push_many");

    let arr_rc = Rc::new(RefCell::new(vec![Value::Int(1)]));
    reset_cranelift_writeback_commit_count();
    let new_len = vm
        .call(&push_fn, vec![Value::Array(arr_rc.clone()), Value::Int(8)])
        .expect("push_many");
    assert_eq!(new_len, Value::Int(9));
    assert_eq!(
        vm.cranelift_readonly_params(&push_fn),
        Some(vec![false, true]),
        "array.push(a, ...) must mark param 0 mutated (readonly=false)"
    );
    assert!(
        cranelift_writeback_commit_count() >= 1,
        "array.push must commit writeback of grown array"
    );
    assert_eq!(
        *arr_rc.borrow(),
        vec![
            Value::Int(1),
            Value::Int(10),
            Value::Int(20),
            Value::Int(30),
            Value::Int(40),
            Value::Int(50),
            Value::Int(60),
            Value::Int(70),
            Value::Int(80),
        ],
        "grown array elements from array.push must write back completely"
    );
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_perf_t2_4_readonly_array_deopt_skips_writeback_and_preserves_rc() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function sum_ro_loop(arr: Array<int>): int
    local n: int = array.len(arr)
    local acc: int = 0
    local i: int = 0
    while i < n do
        if arr[i] is Ok(v) then
            acc = acc + v
        end
        i = i + 1
    end
    return acc
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let sum_fn = find_fn_name(&vm, "sum_ro_loop");
    let arr_rc = Rc::new(RefCell::new(vec![
        Value::Int(10),
        Value::Int(20),
        Value::Int(30),
        Value::Int(40),
        Value::Int(50),
    ]));

    // Trigger native deopt at loop poll 2
    safepoint_request_deopt_on_poll(Some(sum_fn.clone()), 2);
    reset_cranelift_writeback_commit_count();
    let res = vm
        .call(&sum_fn, vec![Value::Array(arr_rc.clone())])
        .expect("sum_ro_loop with poll-2 deopt");
    assert_eq!(res, Value::Int(150));
    assert_eq!(
        vm.cranelift_readonly_params(&sum_fn),
        Some(vec![true]),
        "sum_ro_loop param 0 must be proven read-only"
    );
    assert_eq!(
        cranelift_writeback_commit_count(),
        0,
        "deopt inside read-only array loop must skip writeback commits while resuming cleanly"
    );
    assert_eq!(
        *arr_rc.borrow(),
        vec![
            Value::Int(10),
            Value::Int(20),
            Value::Int(30),
            Value::Int(40),
            Value::Int(50),
        ]
    );
}

#[test]
fn test_perf_t2_4_compute_readonly_params_mir_conditions_a_through_e() {
    use lust::ast::Span;
    use lust::mir::ir::{
        FunctionSignature, MirConstant, MirFunction, MirModule, MirType, Operand, Place, Rvalue,
        Statement, Terminator,
    };
    use lust::vm::cranelift_dispatch::compute_readonly_params;

    let span = Span::dummy();
    let arr_ty = MirType::Array(Box::new(MirType::Int));
    let mut module = MirModule::new("ro_unit_mod".into());

    // 1. Condition (a): projected Call destination mutates base local (via alias `alias = p0`)
    {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: "cond_a_call_dest".into(),
                param_types: vec![arr_ty.clone()],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let p0 = f.params[0];
        let alias = f.add_local(Some("alias".into()), arr_ty.clone(), true, span);
        let idx = f.add_local(Some("idx".into()), MirType::Int, true, span);
        let bb1 = f.new_block();
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(alias),
            Rvalue::Use(Operand::Copy(Place::Local(p0))),
            span,
        ));
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(idx),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            span,
        ));
        f.blocks[0].terminator = Some(Terminator::Call {
            callee: "abs".into(),
            args: vec![Operand::Constant(MirConstant::Int(-5))],
            destination: Place::Index(Box::new(Place::Local(alias)), idx),
            success: bb1,
            error: None,
            span,
        });
        f.blocks[bb1.0].terminator =
            Some(Terminator::Return(Some(Operand::Constant(MirConstant::Int(0))), span));
        module.add_function(f);
    }

    // 2. Condition (b): map.set / table.push mark param mutated
    {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: "cond_b_table_push".into(),
                param_types: vec![arr_ty.clone()],
                return_type: MirType::Unit,
                is_public: true,
                effects: None,
            },
            span,
        );
        let p0 = f.params[0];
        let dst = f.add_local(None, MirType::Unit, true, span);
        let bb1 = f.new_block();
        f.blocks[0].terminator = Some(Terminator::Call {
            callee: "table.push".into(),
            args: vec![
                Operand::Copy(Place::Local(p0)),
                Operand::Constant(MirConstant::Int(1)),
            ],
            destination: Place::Local(dst),
            success: bb1,
            error: None,
            span,
        });
        f.blocks[bb1.0].terminator = Some(Terminator::Return(None, span));
        module.add_function(f);
    }

    // 3. Condition (c): escape via Rvalue::Aggregate marks param mutated
    {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: "cond_c_escape_aggregate".into(),
                param_types: vec![arr_ty.clone()],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let p0 = f.params[0];
        let wrapper = f.add_local(
            Some("w".into()),
            MirType::Tuple(vec![arr_ty.clone()]),
            true,
            span,
        );
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(wrapper),
            Rvalue::Aggregate(
                MirType::Tuple(vec![arr_ty.clone()]),
                vec![Operand::Copy(Place::Local(p0))],
            ),
            span,
        ));
        f.blocks[0].terminator =
            Some(Terminator::Return(Some(Operand::Constant(MirConstant::Int(0))), span));
        module.add_function(f);
    }

    // 4. Condition (d): 3-hop transitive chain h3 -> h2 -> h1 (mutates param 1 only)
    {
        let mut h1 = MirFunction::new(
            FunctionSignature {
                name: "h1_mut_p1".into(),
                param_types: vec![arr_ty.clone(), arr_ty.clone()],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let p1 = h1.params[1];
        let idx = h1.add_local(None, MirType::Int, true, span);
        h1.blocks[0].statements.push(Statement::Assign(
            Place::Local(idx),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            span,
        ));
        h1.blocks[0].statements.push(Statement::Assign(
            Place::Index(Box::new(Place::Local(p1)), idx),
            Rvalue::Use(Operand::Constant(MirConstant::Int(7))),
            span,
        ));
        h1.blocks[0].terminator =
            Some(Terminator::Return(Some(Operand::Constant(MirConstant::Int(0))), span));
        module.add_function(h1);

        let mut h2 = MirFunction::new(
            FunctionSignature {
                name: "h2_forward".into(),
                param_types: vec![arr_ty.clone(), arr_ty.clone()],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let dst = h2.add_local(None, MirType::Int, true, span);
        let bb1 = h2.new_block();
        h2.blocks[0].terminator = Some(Terminator::Call {
            callee: "h1_mut_p1".into(),
            args: vec![
                Operand::Copy(Place::Local(h2.params[1])),
                Operand::Copy(Place::Local(h2.params[0])),
            ],
            destination: Place::Local(dst),
            success: bb1,
            error: None,
            span,
        });
        h2.blocks[bb1.0].terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(dst))), span));
        module.add_function(h2);

        let mut h3 = MirFunction::new(
            FunctionSignature {
                name: "h3_entry".into(),
                param_types: vec![arr_ty.clone(), arr_ty.clone()],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let dst = h3.add_local(None, MirType::Int, true, span);
        let bb1 = h3.new_block();
        h3.blocks[0].terminator = Some(Terminator::Call {
            callee: "h2_forward".into(),
            args: vec![
                Operand::Copy(Place::Local(h3.params[0])),
                Operand::Copy(Place::Local(h3.params[1])),
            ],
            destination: Place::Local(dst),
            success: bb1,
            error: None,
            span,
        });
        h3.blocks[bb1.0].terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(dst))), span));
        module.add_function(h3);
    }

    // 5. Condition (e): unknown external callee and effect callee conservatively mark param mutated
    {
        let mut f_ext = MirFunction::new(
            FunctionSignature {
                name: "cond_e_external".into(),
                param_types: vec![arr_ty.clone()],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let dst = f_ext.add_local(None, MirType::Int, true, span);
        let bb1 = f_ext.new_block();
        f_ext.blocks[0].terminator = Some(Terminator::Call {
            callee: "some_unknown_extern_fn".into(),
            args: vec![Operand::Copy(Place::Local(f_ext.params[0]))],
            destination: Place::Local(dst),
            success: bb1,
            error: None,
            span,
        });
        f_ext.blocks[bb1.0].terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(dst))), span));
        module.add_function(f_ext);

        let mut f_eff = MirFunction::new(
            FunctionSignature {
                name: "cond_e_effect".into(),
                param_types: vec![arr_ty.clone()],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let dst = f_eff.add_local(None, MirType::Int, true, span);
        let bb1 = f_eff.new_block();
        f_eff.blocks[0].terminator = Some(Terminator::Call {
            callee: "__lust_perform".into(),
            args: vec![Operand::Copy(Place::Local(f_eff.params[0]))],
            destination: Place::Local(dst),
            success: bb1,
            error: None,
            span,
        });
        f_eff.blocks[bb1.0].terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(dst))), span));
        module.add_function(f_eff);
    }

    // 6. Pure read-only builtin (`array.len`) keeps param read-only
    {
        let mut f_ro = MirFunction::new(
            FunctionSignature {
                name: "pure_ro_len".into(),
                param_types: vec![arr_ty.clone()],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let dst = f_ro.add_local(None, MirType::Int, true, span);
        let bb1 = f_ro.new_block();
        f_ro.blocks[0].terminator = Some(Terminator::Call {
            callee: "array.len".into(),
            args: vec![Operand::Copy(Place::Local(f_ro.params[0]))],
            destination: Place::Local(dst),
            success: bb1,
            error: None,
            span,
        });
        f_ro.blocks[bb1.0].terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(dst))), span));
        module.add_function(f_ro);
    }

    let ro_map = compute_readonly_params(&module);
    assert_eq!(ro_map.get("cond_a_call_dest"), Some(&vec![false]));
    assert_eq!(ro_map.get("cond_b_table_push"), Some(&vec![false]));
    assert_eq!(ro_map.get("cond_c_escape_aggregate"), Some(&vec![false]));
    assert_eq!(ro_map.get("h1_mut_p1"), Some(&vec![true, false]));
    // h2 swaps args when calling h1_mut_p1(p1, p0), so h2's p0 is mutated and p1 is readonly!
    assert_eq!(ro_map.get("h2_forward"), Some(&vec![false, true]));
    assert_eq!(ro_map.get("h3_entry"), Some(&vec![false, true]));
    assert_eq!(ro_map.get("cond_e_external"), Some(&vec![false]));
    assert_eq!(ro_map.get("cond_e_effect"), Some(&vec![false]));
    assert_eq!(ro_map.get("pure_ro_len"), Some(&vec![true]));
}
