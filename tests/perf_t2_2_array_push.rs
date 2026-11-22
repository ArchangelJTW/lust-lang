#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use lust::ast::Span;
use lust::mir::cranelift_backend::{
    lust_mir_alloc_array, lust_mir_alloc_array_typed, lust_mir_alloc_struct, lust_mir_array_get,
    lust_mir_array_len, lust_mir_array_len_checked, lust_mir_array_push, lust_mir_array_set,
    lust_mir_get_status, lust_mir_set_status, CraneliftJitCompiler, MirArrayHeader,
    MIR_ARRAY_ELEM_TAG_BOOL, MIR_ARRAY_ELEM_TAG_INT,
    MIR_STATUS_ERR_OOM, MIR_STATUS_ERR_OUT_OF_BOUNDS, MIR_STATUS_OK,
};
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType, MirUnaryOp, Operand, Place,
    Rvalue, Statement, Terminator,
};
use lust::mir::runtime_helpers::{enter_mir_context, MirAllocationKind, MirContext};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

struct ClifIrBlocks {
    blocks: HashMap<String, Vec<String>>,
    sym_map: HashMap<String, String>,
    block_order: Vec<String>,
}

impl ClifIrBlocks {
    fn parse(ir: &str) -> Self {
        let mut sym_map: HashMap<String, String> = HashMap::new();
        let mut blocks: HashMap<String, Vec<String>> = HashMap::new();
        let mut block_order: Vec<String> = Vec::new();
        let mut cur_block: Option<String> = None;

        for raw_line in ir.lines() {
            let line = raw_line.trim();
            if let Some(rest) = line.strip_prefix("; sym ") {
                if let Some((fref, sym)) = rest.split_once(" = ") {
                    sym_map.insert(fref.trim().to_string(), sym.trim().to_string());
                }
                continue;
            }
            if line.starts_with("block") && line.contains(':') {
                let bname = line
                    .split(|c: char| c == '(' || c == ':' || c.is_whitespace())
                    .next()
                    .unwrap()
                    .to_string();
                block_order.push(bname.clone());
                blocks.entry(bname.clone()).or_default();
                cur_block = Some(bname);
                continue;
            }
            if line == "}" {
                cur_block = None;
                continue;
            }
            if let Some(ref bname) = cur_block {
                if !line.is_empty() && !line.starts_with(';') {
                    blocks
                        .entry(bname.clone())
                        .or_default()
                        .push(line.to_string());
                }
            }
        }

        Self {
            blocks,
            sym_map,
            block_order,
        }
    }

    fn extract_call_sym(&self, line: &str) -> Option<String> {
        let idx = line.find("call ")?;
        let after = &line[idx + 5..];
        let callee_ref = after
            .split(|c: char| c == '(' || c.is_whitespace())
            .next()?
            .trim();
        Some(
            self.sym_map
                .get(callee_ref)
                .cloned()
                .unwrap_or_else(|| callee_ref.to_string()),
        )
    }

    fn calls_in_block(&self, block: &str) -> Vec<String> {
        self.blocks
            .get(block)
            .map(|lines| {
                lines
                    .iter()
                    .filter_map(|l| self.extract_call_sym(l))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn successors(&self, block: &str) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(lines) = self.blocks.get(block) {
            for line in lines {
                if line.starts_with("jump ") || line.starts_with("brif ") {
                    for tok in line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                        if tok.starts_with("block")
                            && tok.len() > 5
                            && tok[5..].chars().all(|c| c.is_ascii_digit())
                        {
                            out.push(tok.to_string());
                        }
                    }
                }
            }
        }
        out
    }

    /// Returns all blocks that belong to at least one strongly-connected cycle (loop blocks).
    fn loop_blocks(&self) -> HashSet<String> {
        let mut in_cycle = HashSet::new();
        for start in &self.block_order {
            let mut visited = HashSet::new();
            let mut q = VecDeque::new();
            for s in self.successors(start) {
                if visited.insert(s.clone()) {
                    q.push_back(s);
                }
            }
            while let Some(cur) = q.pop_front() {
                if &cur == start {
                    in_cycle.insert(start.clone());
                    break;
                }
                for s in self.successors(&cur) {
                    if visited.insert(s.clone()) {
                        q.push_back(s);
                    }
                }
            }
        }
        in_cycle
    }
}

/// Builds MIR:
/// ```text
/// fn push_alias_and_sum(n: int) -> int {
///     let a: Array<int> = [];
///     let b: Array<int> = a; // alias
///     let i: int = 0;
///     while i < n {
///         array.push(a, i * 10 + 1);
///         i = i + 1;
///     }
///     // Verify alias `b` sees the grown length and elements
///     let la = array.len(a);
///     let lb = array.len(b);
///     if la != lb { return -1; }
///     let sum = 0;
///     let j = 0;
///     while j < lb {
///         let v = b[j];
///         sum = sum + v;
///         j = j + 1;
///     }
///     return sum;
/// }
/// ```
fn build_push_alias_and_sum_fn() -> MirFunction {
    let sig = FunctionSignature {
        name: "push_alias_and_sum".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let n = f.params[0];
    let a = f.add_local(
        Some("a".into()),
        MirType::Array(Box::new(MirType::Int)),
        true,
        Span::dummy(),
    );
    let b = f.add_local(
        Some("b".into()),
        MirType::Array(Box::new(MirType::Int)),
        true,
        Span::dummy(),
    );
    let i = f.add_local(Some("i".into()), MirType::Int, true, Span::dummy());
    let cond1 = f.add_local(Some("cond1".into()), MirType::Bool, true, Span::dummy());
    let val = f.add_local(Some("val".into()), MirType::Int, true, Span::dummy());
    let tmp_mul = f.add_local(Some("tmp_mul".into()), MirType::Int, true, Span::dummy());
    let push_dst = f.add_local(Some("push_dst".into()), MirType::Unit, true, Span::dummy());
    let la = f.add_local(Some("la".into()), MirType::Int, true, Span::dummy());
    let lb = f.add_local(Some("lb".into()), MirType::Int, true, Span::dummy());
    let len_eq = f.add_local(Some("len_eq".into()), MirType::Bool, true, Span::dummy());
    let sum = f.add_local(Some("sum".into()), MirType::Int, true, Span::dummy());
    let j = f.add_local(Some("j".into()), MirType::Int, true, Span::dummy());
    let cond2 = f.add_local(Some("cond2".into()), MirType::Bool, true, Span::dummy());
    let elem = f.add_local(Some("elem".into()), MirType::Int, true, Span::dummy());

    let bb_hdr1 = f.new_block(); // 1
    let bb_body1 = f.new_block(); // 2
    let bb_after_push = f.new_block(); // 3
    let bb_after_loop1 = f.new_block(); // 4
    let bb_after_la = f.new_block(); // 5
    let bb_after_lb = f.new_block(); // 6
    let bb_mismatch = f.new_block(); // 7
    let bb_init_loop2 = f.new_block(); // 8
    let bb_hdr2 = f.new_block(); // 9
    let bb_body2 = f.new_block(); // 10
    let bb_exit = f.new_block(); // 11

    // bb0: a = []; b = a; i = 0; goto bb_hdr1
    // Intentionally use Array(Unknown) aggregate like AstLowering produces for `[]`!
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(a),
        Rvalue::Aggregate(MirType::Array(Box::new(MirType::Unknown)), vec![]),
        Span::dummy(),
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(b),
        Rvalue::Use(Operand::Copy(Place::Local(a))),
        Span::dummy(),
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[0].terminator = Some(Terminator::Goto(bb_hdr1, Span::dummy()));

    // bb_hdr1: cond1 = i < n; brif cond1 -> bb_body1 else bb_after_loop1
    f.blocks[bb_hdr1.0].statements.push(Statement::Assign(
        Place::Local(cond1),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(n)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_hdr1.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond1)),
        then_block: bb_body1,
        else_block: bb_after_loop1,
        span: Span::dummy(),
    });

    // bb_body1: val = i * 10 + 1; array.push(a, val) -> bb_after_push
    f.blocks[bb_body1.0].statements.push(Statement::Assign(
        Place::Local(tmp_mul),
        Rvalue::BinaryOp(
            MirBinaryOp::Mul,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(10)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body1.0].statements.push(Statement::Assign(
        Place::Local(val),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(tmp_mul)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body1.0].terminator = Some(Terminator::Call {
        callee: "array.push".into(),
        args: vec![
            Operand::Copy(Place::Local(a)),
            Operand::Copy(Place::Local(val)),
        ],
        destination: Place::Local(push_dst),
        success: bb_after_push,
        error: None,
        span: Span::dummy(),
    });

    // bb_after_push: i = i + 1; goto bb_hdr1
    f.blocks[bb_after_push.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_after_push.0].terminator = Some(Terminator::Goto(bb_hdr1, Span::dummy()));

    // bb_after_loop1: la = array.len(a) -> bb_after_la
    f.blocks[bb_after_loop1.0].terminator = Some(Terminator::Call {
        callee: "array.len".into(),
        args: vec![Operand::Copy(Place::Local(a))],
        destination: Place::Local(la),
        success: bb_after_la,
        error: None,
        span: Span::dummy(),
    });

    // bb_after_la: lb = array.len(b) -> bb_after_lb
    f.blocks[bb_after_la.0].terminator = Some(Terminator::Call {
        callee: "array.len".into(),
        args: vec![Operand::Copy(Place::Local(b))],
        destination: Place::Local(lb),
        success: bb_after_lb,
        error: None,
        span: Span::dummy(),
    });

    // bb_after_lb: if la == lb -> bb_init_loop2 else bb_mismatch
    f.blocks[bb_after_lb.0].statements.push(Statement::Assign(
        Place::Local(len_eq),
        Rvalue::BinaryOp(
            MirBinaryOp::Eq,
            Operand::Copy(Place::Local(la)),
            Operand::Copy(Place::Local(lb)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_after_lb.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(len_eq)),
        then_block: bb_init_loop2,
        else_block: bb_mismatch,
        span: Span::dummy(),
    });

    f.blocks[bb_mismatch.0].terminator = Some(Terminator::Return(
        Some(Operand::Constant(MirConstant::Int(-1))),
        Span::dummy(),
    ));

    // bb_init_loop2: sum = 0; j = 0; goto bb_hdr2
    f.blocks[bb_init_loop2.0].statements.push(Statement::Assign(
        Place::Local(sum),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb_init_loop2.0].statements.push(Statement::Assign(
        Place::Local(j),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb_init_loop2.0].terminator = Some(Terminator::Goto(bb_hdr2, Span::dummy()));

    // bb_hdr2: cond2 = j < lb; brif cond2 -> bb_body2 else bb_exit
    f.blocks[bb_hdr2.0].statements.push(Statement::Assign(
        Place::Local(cond2),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(j)),
            Operand::Copy(Place::Local(lb)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_hdr2.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond2)),
        then_block: bb_body2,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    // bb_body2: elem = b[j]; sum += elem; j += 1; goto bb_hdr2
    f.blocks[bb_body2.0].statements.push(Statement::Assign(
        Place::Local(elem),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(b)), j))),
        Span::dummy(),
    ));
    f.blocks[bb_body2.0].statements.push(Statement::Assign(
        Place::Local(sum),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(sum)),
            Operand::Copy(Place::Local(elem)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body2.0].statements.push(Statement::Assign(
        Place::Local(j),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(j)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body2.0].terminator = Some(Terminator::Goto(bb_hdr2, Span::dummy()));

    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(sum))),
        Span::dummy(),
    ));
    f
}

/// Builds MIR:
/// ```text
/// fn loop_param_array_len(arr: Array<int>, n: int) -> int {
///     let i = 0;
///     let acc = 0;
///     while i < n {
///         let l = array.len(arr);
///         acc = acc + l;
///         i = i + 1;
///     }
///     return acc;
/// }
/// ```
fn build_loop_param_array_len_fn() -> MirFunction {
    let sig = FunctionSignature {
        name: "loop_param_array_len".into(),
        param_types: vec![MirType::Array(Box::new(MirType::Int)), MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let arr = f.params[0];
    let n = f.params[1];
    let i = f.add_local(Some("i".into()), MirType::Int, true, Span::dummy());
    let acc = f.add_local(Some("acc".into()), MirType::Int, true, Span::dummy());
    let cond = f.add_local(Some("cond".into()), MirType::Bool, true, Span::dummy());
    let len_v = f.add_local(Some("len_v".into()), MirType::Int, true, Span::dummy());

    let bb_hdr = f.new_block(); // 1
    let bb_body = f.new_block(); // 2
    let bb_after_len = f.new_block(); // 3
    let bb_exit = f.new_block(); // 4

    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(n)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    f.blocks[bb_body.0].terminator = Some(Terminator::Call {
        callee: "array.len".into(),
        args: vec![Operand::Copy(Place::Local(arr))],
        destination: Place::Local(len_v),
        success: bb_after_len,
        error: None,
        span: Span::dummy(),
    });

    f.blocks[bb_after_len.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Copy(Place::Local(len_v)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_after_len.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_after_len.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        Span::dummy(),
    ));
    f
}

/// Builds MIR using `MirUnaryOp::Len` in a loop header (`while i < len(arr)`).
fn build_loop_unary_len_fn() -> MirFunction {
    let sig = FunctionSignature {
        name: "loop_unary_len".into(),
        param_types: vec![MirType::Array(Box::new(MirType::Int))],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let arr = f.params[0];
    let i = f.add_local(Some("i".into()), MirType::Int, true, Span::dummy());
    let acc = f.add_local(Some("acc".into()), MirType::Int, true, Span::dummy());
    let l = f.add_local(Some("l".into()), MirType::Int, true, Span::dummy());
    let cond = f.add_local(Some("cond".into()), MirType::Bool, true, Span::dummy());
    let elem = f.add_local(Some("elem".into()), MirType::Int, true, Span::dummy());

    let bb_hdr = f.new_block(); // 1
    let bb_body = f.new_block(); // 2
    let bb_exit = f.new_block(); // 3

    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(l),
        Rvalue::UnaryOp(MirUnaryOp::Len, Operand::Copy(Place::Local(arr))),
        Span::dummy(),
    ));
    f.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(l)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(elem),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(arr)), i))),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Copy(Place::Local(elem)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        Span::dummy(),
    ));
    f
}

#[test]
fn test_perf_t2_2_array_push_growth_aliases_and_oom() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Verify MirArrayHeader ABI layout via offset_of!
    assert_eq!(core::mem::offset_of!(MirArrayHeader, len), 0);
    assert_eq!(core::mem::offset_of!(MirArrayHeader, cap), 8);
    assert_eq!(core::mem::size_of::<MirArrayHeader>(), 16);

    // 1. Cranelift-compiled push across multiple capacity doublings (0 -> 4 -> 8 -> 16 -> 32 -> 64)
    //    with aliased local `b = a` observing identical len and elements via Place::Index.
    {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        let func = build_push_alias_and_sum_fn();
        let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
        let code = compiler
            .compile_function(&func)
            .expect("Compile push_alias_and_sum");
        let f_native: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code) };

        for n in [0i64, 1, 4, 5, 8, 9, 16, 17, 32, 40] {
            lust_mir_set_status(MIR_STATUS_OK);
            let got = f_native(n);
            assert_eq!(lust_mir_get_status(), MIR_STATUS_OK, "status ok for n={n}");
            let expected: i64 = (0..n).map(|i| i * 10 + 1).sum();
            assert_eq!(got, expected, "alias sum mismatch for n={n}");
        }
        ctx.clear_to_baselines(0, 0);
        assert_eq!(ctx.buffer_count(), 0);
    }

    // 2. Float array.push bit-exact preservation (including -0.0) via Cranelift
    {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);

        let sig = FunctionSignature {
            name: "push_float_neg_zero".into(),
            param_types: vec![MirType::Float, MirType::Float],
            return_type: MirType::Float,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, Span::dummy());
        let x = f.params[0];
        let y = f.params[1];
        let arr = f.add_local(
            Some("arr".into()),
            MirType::Array(Box::new(MirType::Float)),
            true,
            Span::dummy(),
        );
        let push_dst = f.add_local(None, MirType::Unit, true, Span::dummy());
        let idx1 = f.add_local(None, MirType::Int, true, Span::dummy());
        let out = f.add_local(None, MirType::Float, true, Span::dummy());
        let bb1 = f.new_block();
        let bb2 = f.new_block();

        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(arr),
            Rvalue::Aggregate(MirType::Array(Box::new(MirType::Float)), vec![]),
            Span::dummy(),
        ));
        f.blocks[0].terminator = Some(Terminator::Call {
            callee: "array.push".into(),
            args: vec![
                Operand::Copy(Place::Local(arr)),
                Operand::Copy(Place::Local(x)),
            ],
            destination: Place::Local(push_dst),
            success: bb1,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb1.0].terminator = Some(Terminator::Call {
            callee: "array.push".into(),
            args: vec![
                Operand::Copy(Place::Local(arr)),
                Operand::Copy(Place::Local(y)),
            ],
            destination: Place::Local(push_dst),
            success: bb2,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb2.0].statements.push(Statement::Assign(
            Place::Local(idx1),
            Rvalue::Use(Operand::Constant(MirConstant::Int(1))),
            Span::dummy(),
        ));
        f.blocks[bb2.0].statements.push(Statement::Assign(
            Place::Local(out),
            Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(arr)), idx1))),
            Span::dummy(),
        ));
        f.blocks[bb2.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(out))),
            Span::dummy(),
        ));

        let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
        let code = compiler.compile_function(&f).expect("Compile float push");
        let f_native: extern "C" fn(f64, f64) -> f64 = unsafe { std::mem::transmute(code) };
        lust_mir_set_status(MIR_STATUS_OK);
        let res = f_native(3.25, -0.0);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
        assert_eq!(res.to_bits(), (-0.0f64).to_bits());
        ctx.clear_to_baselines(0, 0);
    }

    // 3. Direct runtime helper verification: growth from 0 -> 4 -> 8 -> 16 -> 32,
    //    alias invariance (pointer never moves), resolve_validated_array_data, and OOM rollback.
    {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);

        let a = lust_mir_alloc_array_typed(0, MIR_ARRAY_ELEM_TAG_INT);
        assert!(!a.is_null());
        let b = a; // alias
        // Flat empty array has len=0, cap=0, word_capacity >= 3
        let (len0, cap0, data0) = ctx
            .resolve_validated_array_data(a)
            .expect("empty flat array must validate");
        assert_eq!(len0, 0);
        assert_eq!(cap0, 0);
        assert_eq!(data0, unsafe { a.add(2) as *const i64 });

        for i in 0..32i64 {
            assert_eq!(lust_mir_array_push(a, i * 3 + 7), MIR_STATUS_OK);
        }
        assert_eq!(lust_mir_array_len(a), 32);
        assert_eq!(lust_mir_array_len(b), 32);
        let (len32, cap32, data32) = ctx
            .resolve_validated_array_data(b)
            .expect("redirected array must validate");
        assert_eq!(len32, 32);
        assert_eq!(cap32, 32);
        assert_ne!(data32, unsafe { b.add(2) as *const i64 });
        assert!(unsafe { *b.add(1) } < 0, "redirected array must have cap bit 63 set");

        for i in 0..32i64 {
            assert_eq!(lust_mir_array_get(b, i), i * 3 + 7);
            assert_eq!(unsafe { *data32.add(i as usize) }, i * 3 + 7);
        }

        // OOM rollback test: next push (33rd element) needs cap=64 (512 bytes)
        let mem_before_oom = ctx.memory_used();
        let cap_raw_before_oom = unsafe { *a.add(1) };
        let data_word_before_oom = unsafe { *a.add(2) };
        ctx.set_max_memory_limit(mem_before_oom + 64); // far below 512 bytes needed for 64-word growth

        lust_mir_set_status(MIR_STATUS_OK);
        let rc = lust_mir_array_push(a, 99999);
        assert_eq!(rc, MIR_STATUS_ERR_OOM);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OOM);

        // Header, data pointer, memory_used, and all 32 elements MUST be untouched!
        assert_eq!(ctx.memory_used(), mem_before_oom);
        assert_eq!(unsafe { *a }, 32);
        assert_eq!(unsafe { *a.add(1) }, cap_raw_before_oom);
        assert_eq!(unsafe { *a.add(2) }, data_word_before_oom);
        for i in 0..32i64 {
            assert_eq!(lust_mir_array_get(a, i), i * 3 + 7);
        }

        // Clean teardown without leak or double-free
        ctx.clear_to_baselines(0, 0);
        assert_eq!(ctx.buffer_count(), 0);
        assert!(ctx.get_active_allocation_record(a as usize).is_none());
        assert!(ctx.get_active_allocation_record(data_word_before_oom as usize).is_none());
    }
}

#[test]
fn test_perf_t2_2_hoisted_inline_array_len_ir() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // 1. Parameter array used in `array.len(arr)` inside a loop:
    //    zero `lust_mir_array_len_checked` / `lust_mir_array_len` calls inside loop blocks,
    //    exactly 1 dominating provenance validation before loop entry.
    {
        let func = build_loop_param_array_len_fn();
        let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
        let ir = compiler
            .dump_function_ir(&func)
            .expect("dump_function_ir for loop_param_array_len");
        let parsed = ClifIrBlocks::parse(&ir);
        let loop_bbs = parsed.loop_blocks();
        assert!(!loop_bbs.is_empty(), "Expected loop blocks in IR:\n{ir}");

        let mut loop_len_calls = Vec::new();
        let mut non_loop_len_calls = Vec::new();
        for b in &parsed.block_order {
            for sym in parsed.calls_in_block(b) {
                if sym == "lust_mir_array_len_checked" || sym == "lust_mir_array_len" {
                    if loop_bbs.contains(b) {
                        loop_len_calls.push((b.clone(), sym));
                    } else {
                        non_loop_len_calls.push((b.clone(), sym));
                    }
                }
            }
        }
        assert!(
            loop_len_calls.is_empty(),
            "Expected ZERO array len helper calls inside loop blocks, found {loop_len_calls:?}\nIR:\n{ir}"
        );
        assert_eq!(
            non_loop_len_calls.len(),
            1,
            "Expected exactly 1 dominating lust_mir_array_len_checked call before loop, found {non_loop_len_calls:?}\nIR:\n{ir}"
        );
        assert_eq!(non_loop_len_calls[0].1, "lust_mir_array_len_checked");

        // Behavioral execution: valid array + forged pointer rejection!
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        let mut exec_compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
        let code = exec_compiler
            .compile_function(&func)
            .expect("Compile loop_param_array_len");
        let f_native: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };

        let arr = lust_mir_alloc_array_typed(5, MIR_ARRAY_ELEM_TAG_INT);
        lust_mir_set_status(MIR_STATUS_OK);
        assert_eq!(f_native(arr as i64, 10), 50);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);

        // Forged pointer MUST still be rejected fail-closed without SIGSEGV!
        lust_mir_set_status(MIR_STATUS_OK);
        let _ = f_native(0x1000, 10);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

        ctx.clear_to_baselines(0, 0);
    }

    // 2. Parameter array used in `MirUnaryOp::Len` in loop header (`while i < len(arr)`):
    //    zero helper calls in loop blocks, and works on both flat and redirected arrays!
    {
        let func = build_loop_unary_len_fn();
        let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
        let ir = compiler
            .dump_function_ir(&func)
            .expect("dump_function_ir for loop_unary_len");
        let parsed = ClifIrBlocks::parse(&ir);
        let loop_bbs = parsed.loop_blocks();
        assert!(!loop_bbs.is_empty(), "Expected loop blocks in IR:\n{ir}");

        for b in &loop_bbs {
            let calls = parsed.calls_in_block(b);
            assert!(
                !calls
                    .iter()
                    .any(|s| s == "lust_mir_array_len_checked" || s == "lust_mir_array_len"),
                "Unexpected array len helper call in loop block {b}: {calls:?}\nIR:\n{ir}"
            );
        }

        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        let mut exec_compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
        let code = exec_compiler
            .compile_function(&func)
            .expect("Compile loop_unary_len");
        let f_native: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code) };

        // Flat array
        let flat = lust_mir_alloc_array_typed(3, MIR_ARRAY_ELEM_TAG_INT);
        lust_mir_array_set(flat, 0, 10);
        lust_mir_array_set(flat, 1, 20);
        lust_mir_array_set(flat, 2, 30);
        lust_mir_set_status(MIR_STATUS_OK);
        assert_eq!(f_native(flat as i64), 60);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);

        // Redirected array (after pushes)
        for v in 1..=10i64 {
            assert_eq!(lust_mir_array_push(flat, v), MIR_STATUS_OK);
        }
        lust_mir_set_status(MIR_STATUS_OK);
        assert_eq!(f_native(flat as i64), 60 + 55);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);

        // Forged pointer rejected
        lust_mir_set_status(MIR_STATUS_OK);
        let _ = f_native(0x2000);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

        ctx.clear_to_baselines(0, 0);
    }
}

#[test]
fn test_perf_t2_2_forged_redirect_and_unowned_rejected_fail_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // 1. Unowned / forged pointer with cap < 0 rejected fail-closed
    let mut forged_buf: [i64; 3] = [1, 4 | i64::MIN, 0xdead_beefusize as i64];
    let forged_ptr = forged_buf.as_mut_ptr();
    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(lust_mir_array_len_checked(forged_ptr), -1);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(lust_mir_array_push(forged_ptr, 123), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert!(ctx.resolve_validated_array_data(forged_ptr).is_none());

    // 2. Owned array tampered with cap < 0 and forged unowned data_ptr at word[2]
    let arr = lust_mir_alloc_array_typed(1, MIR_ARRAY_ELEM_TAG_INT);
    assert!(!arr.is_null());
    unsafe {
        *arr = 1;
        *arr.add(1) = 4 | i64::MIN;
        *arr.add(2) = 0x1000; // unowned bogus data_ptr
    }
    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(lust_mir_array_len_checked(arr), -1);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(lust_mir_array_get(arr, 0), 0);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(lust_mir_array_push(arr, 42), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert!(ctx.resolve_validated_array_data(arr).is_none());

    // 3. Owned array tampered with cap < 0 pointing to another owned non-ArrayBacking allocation (Struct)
    let other_struct = lust_mir_alloc_struct(4);
    assert!(!other_struct.is_null());
    unsafe {
        *arr = 1;
        *arr.add(1) = 4 | i64::MIN;
        *arr.add(2) = other_struct as i64;
    }
    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(lust_mir_array_len_checked(arr), -1);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert!(ctx.resolve_validated_array_data(arr).is_none());

    // 4. Owned array tampered with cap < 0 pointing to itself
    unsafe {
        *arr = 1;
        *arr.add(1) = 2 | i64::MIN;
        *arr.add(2) = arr as i64;
    }
    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(lust_mir_array_len_checked(arr), -1);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert!(ctx.resolve_validated_array_data(arr).is_none());

    // 5. External 2-word flat buffer [0, 0] rejected by lust_mir_array_push (word_capacity < 3)
    //    and 2-word buffer with forged cap < 0 rejected without reading word[2]
    let mut two_word: [i64; 2] = [0, 0];
    let two_ptr = two_word.as_mut_ptr();
    ctx.register_allocation(two_ptr, 2, MirAllocationKind::Array(MirType::Int));
    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(lust_mir_array_len_checked(two_ptr), 0);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);

    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(lust_mir_array_push(two_ptr, 7), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

    unsafe {
        *two_ptr.add(1) = 4 | i64::MIN;
    }
    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(lust_mir_array_len_checked(two_ptr), -1);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert!(ctx.resolve_validated_array_data(two_ptr).is_none());

    // 6. Bool & RawWords arrays also support growth cleanly
    let bool_arr = lust_mir_alloc_array_typed(0, MIR_ARRAY_ELEM_TAG_BOOL);
    let raw_arr = lust_mir_alloc_array(0);
    for i in 0..10i64 {
        assert_eq!(lust_mir_array_push(bool_arr, i % 2), MIR_STATUS_OK);
        assert_eq!(lust_mir_array_push(raw_arr, i * 2), MIR_STATUS_OK);
    }
    assert_eq!(lust_mir_array_len(bool_arr), 10);
    assert_eq!(lust_mir_array_len(raw_arr), 10);
    assert_eq!(lust_mir_array_get(bool_arr, 9), 1);
    assert_eq!(lust_mir_array_get(raw_arr, 9), 18);

    ctx.clear_to_baselines(0, 0);
    assert_eq!(ctx.buffer_count(), 0);
}

#[test]
fn test_perf_t2_2_reassignment_invalidates_provenance_in_loop() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Function starts with freshly allocated valid `arr = []`, then inside a loop
    // reassigns `arr = ext_arr` (an unvalidated parameter) and calls `array.len(arr)`.
    // Provenance MUST be invalidated on reassignment so forged `ext_arr = 0x3000` is caught!
    let sig = FunctionSignature {
        name: "reassign_in_loop".into(),
        param_types: vec![MirType::Array(Box::new(MirType::Int)), MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let ext_arr = f.params[0];
    let n = f.params[1];
    let arr = f.add_local(
        Some("arr".into()),
        MirType::Array(Box::new(MirType::Int)),
        true,
        Span::dummy(),
    );
    let i = f.add_local(Some("i".into()), MirType::Int, true, Span::dummy());
    let acc = f.add_local(Some("acc".into()), MirType::Int, true, Span::dummy());
    let cond = f.add_local(Some("cond".into()), MirType::Bool, true, Span::dummy());
    let len_v = f.add_local(Some("len_v".into()), MirType::Int, true, Span::dummy());

    let bb_hdr = f.new_block(); // 1
    let bb_body = f.new_block(); // 2
    let bb_after_len = f.new_block(); // 3
    let bb_exit = f.new_block(); // 4

    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(arr),
        Rvalue::Aggregate(MirType::Array(Box::new(MirType::Int)), vec![]),
        Span::dummy(),
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(n)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    // Reassign arr = ext_arr before calling array.len(arr)
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(arr),
        Rvalue::Use(Operand::Copy(Place::Local(ext_arr))),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].terminator = Some(Terminator::Call {
        callee: "array.len".into(),
        args: vec![Operand::Copy(Place::Local(arr))],
        destination: Place::Local(len_v),
        success: bb_after_len,
        error: None,
        span: Span::dummy(),
    });

    f.blocks[bb_after_len.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Copy(Place::Local(len_v)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_after_len.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_after_len.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        Span::dummy(),
    ));

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
    let code = compiler
        .compile_function(&f)
        .expect("Compile reassign_in_loop");
    let f_native: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };

    // Valid ext_arr succeeds
    let valid = lust_mir_alloc_array_typed(3, MIR_ARRAY_ELEM_TAG_INT);
    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(f_native(valid as i64, 4), 12);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);

    // Forged ext_arr MUST be rejected fail-closed without SIGSEGV!
    lust_mir_set_status(MIR_STATUS_OK);
    let _ = f_native(0x3000, 4);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

    ctx.clear_to_baselines(0, 0);
}

#[test]
fn test_perf_t2_2_simd_loop_diverts_redirected_array_to_scalar() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let sig = FunctionSignature {
        name: "simd_sum_redirected".into(),
        param_types: vec![MirType::Array(Box::new(MirType::Int)), MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let a_loc = func.params[0];
    let n_loc = func.params[1];
    let i_loc = func.add_local(Some("i".into()), MirType::Int, true, Span::dummy());
    let acc_loc = func.add_local(Some("acc".into()), MirType::Int, true, Span::dummy());
    let cond_loc = func.add_local(Some("cond".into()), MirType::Bool, true, Span::dummy());
    let elem_loc = func.add_local(Some("elem".into()), MirType::Int, true, Span::dummy());

    let bb_hdr = func.new_block();
    let bb_body = func.new_block();
    let bb_exit = func.new_block();

    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(i_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    func.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Copy(Place::Local(n_loc)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond_loc)),
        then_block: bb_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(elem_loc),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(a_loc)), i_loc))),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc_loc)),
            Operand::Copy(Place::Local(elem_loc)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(i_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));
    func.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc_loc))),
        Span::dummy(),
    ));

    let changed = lust::mir::opt::simd::SimdOptimizer::run_on_function(&mut func);
    assert!(changed, "Sum loop must vectorize");

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
    let code = compiler
        .compile_function(&func)
        .expect("Compile vectorized sum");
    let f_native: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };

    // Build redirected (cap < 0) array with 20 elements via lust_mir_array_push
    let arr = lust_mir_alloc_array_typed(0, MIR_ARRAY_ELEM_TAG_INT);
    for v in 1..=20i64 {
        assert_eq!(lust_mir_array_push(arr, v), MIR_STATUS_OK);
    }
    assert!(unsafe { *arr.add(1) } < 0);

    lust_mir_set_status(MIR_STATUS_OK);
    let got = f_native(arr as i64, 20);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(got, 210, "SIMD loop must divert cap < 0 array to scalar path and sum accurately");

    ctx.clear_to_baselines(0, 0);
}
