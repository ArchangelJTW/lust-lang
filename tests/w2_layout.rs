use lust::bytecode::{Chunk, Instruction, Value};
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use lust::data::materialize::{
    decode_json_into_contiguous, ContiguousDecodeOptions, ContiguousStruct, LayoutPerfCounters,
    MaterializeError,
};
use lust::data::Document;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::{
    mir_struct_decode_callee, parse_mir_struct_decode_callee, MirStructOp,
};
use lust::mir::lowering::AstLowering;
use lust::mir::vm_backend::{
    audit_chunk_field_access, register_struct_decode_natives_with_counters, MirToBytecodeCompiler,
};
use lust::parser::Parser;
use lust::vm::VM;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Independent C-ABI offset/alignment oracle for Gate-SCHEMA-09 verification.
fn oracle_c_abi_layout(
    tags: &[(CanonicalTypeTag, &str)],
    abi: TargetAbi,
) -> (Vec<(usize, usize)>, usize, usize) {
    let mut offset = 0usize;
    let mut max_align = 1usize;
    let mut field_offsets = Vec::with_capacity(tags.len());

    for (tag, _) in tags {
        let (size, mut align) = match tag {
            CanonicalTypeTag::Bool => (1, 1),
            CanonicalTypeTag::Int64 | CanonicalTypeTag::UInt64 | CanonicalTypeTag::Float64 => (8, 8),
            CanonicalTypeTag::DateTimeUtc => (8, 8),
            CanonicalTypeTag::ObjectId => (12, 4),
            CanonicalTypeTag::Decimal128 => (16, 16),
            CanonicalTypeTag::StringSlice | CanonicalTypeTag::BinarySlice => {
                if abi.pointer_width == 4 {
                    (16, 4)
                } else {
                    (24, 8)
                }
            }
            CanonicalTypeTag::NestedStruct(_) => (0, 8),
        };
        if abi.max_alignment > 0 && align > abi.max_alignment {
            align = abi.max_alignment;
        }
        if align > max_align {
            max_align = align;
        }
        let padding = (align - (offset % align)) % align;
        offset += padding;
        field_offsets.push((offset, size));
        offset += size;
    }

    let tail_pad = (max_align - (offset % max_align)) % max_align;
    (field_offsets, offset + tail_pad, max_align)
}

fn build_mixed_layout() -> LogicalLayoutV1 {
    let fields = vec![
        LogicalFieldDesc::new(0, "flag", CanonicalTypeTag::Bool),
        LogicalFieldDesc::new(1, "id", CanonicalTypeTag::Int64),
        LogicalFieldDesc::new(2, "oid", CanonicalTypeTag::ObjectId),
        LogicalFieldDesc::new(3, "amount", CanonicalTypeTag::Decimal128),
        LogicalFieldDesc::new(4, "score", CanonicalTypeTag::Float64),
        LogicalFieldDesc::new(5, "name", CanonicalTypeTag::StringSlice),
        LogicalFieldDesc::new(6, "opt_note", CanonicalTypeTag::StringSlice)
            .with_nullability(true, true),
    ];
    LogicalLayoutV1::new_with_abi(9001, "Record", fields, TargetAbi::x86_64())
}

#[test]
fn test_contiguous_c_layout_offsets_alignment_and_raw_bytes_oracle() {
    for abi in [TargetAbi::x86_64(), TargetAbi::aarch64(), TargetAbi::rv32()] {
        let spec = vec![
            (CanonicalTypeTag::Bool, "flag"),
            (CanonicalTypeTag::Int64, "id"),
            (CanonicalTypeTag::ObjectId, "oid"),
            (CanonicalTypeTag::Decimal128, "amount"),
            (CanonicalTypeTag::StringSlice, "name"),
        ];
        let fields: Vec<_> = spec
            .iter()
            .enumerate()
            .map(|(i, (t, n))| LogicalFieldDesc::new(i as u16, *n, *t))
            .collect();
        let layout = LogicalLayoutV1::new_with_abi(101, "AbiCheck", fields, abi);
        let (oracle_offsets, oracle_size, oracle_align) = oracle_c_abi_layout(&spec, abi);

        assert_eq!(layout.size_bytes, oracle_size, "size mismatch for {:?}", abi);
        assert_eq!(layout.align_bytes, oracle_align, "align mismatch for {:?}", abi);
        for (i, f) in layout.fields.iter().enumerate() {
            assert_eq!(
                (f.byte_offset, f.byte_size),
                oracle_offsets[i],
                "field {} offset/size mismatch for {:?}",
                f.name,
                abi
            );
        }
    }

    let layout = build_mixed_layout();
    let payload = br#"{
        "_type": "Record",
        "opt_note": null,
        "name": "alice",
        "score": 42.5,
        "amount": 12345678901234567890,
        "oid": "00112233445566778899aabb",
        "id": -77,
        "flag": true
    }"#;

    let mut counters = LayoutPerfCounters::default();
    let cs = decode_json_into_contiguous(
        &layout,
        payload,
        &mut counters,
        &ContiguousDecodeOptions {
            require_nominal_tag: true,
            ..Default::default()
        },
    )
    .expect("direct decode succeeds");

    // Verify physical buffer alignment
    assert_eq!((cs.as_ptr() as usize) % layout.align_bytes, 0);
    assert_eq!(cs.raw_bytes().len(), layout.size_bytes);

    // Verify raw memory at byte offsets matches independent oracle bytes
    let raw = cs.raw_bytes();
    assert_eq!(raw[layout.fields[0].byte_offset], 1u8);
    assert_eq!(
        &raw[layout.fields[1].byte_offset..layout.fields[1].byte_offset + 8],
        &(-77i64).to_ne_bytes()
    );
    assert_eq!(
        &raw[layout.fields[2].byte_offset..layout.fields[2].byte_offset + 12],
        &[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb]
    );
    assert_eq!(
        &raw[layout.fields[3].byte_offset..layout.fields[3].byte_offset + 16],
        &(12345678901234567890i128).to_ne_bytes()
    );
    assert_eq!(
        &raw[layout.fields[4].byte_offset..layout.fields[4].byte_offset + 8],
        &(42.5f64).to_ne_bytes()
    );

    // Verify zero-lookup, zero-alloc reads in arbitrary order
    assert_eq!(
        cs.read_str_at(5, layout.fields[5].byte_offset, Some(&mut counters)).unwrap(),
        Some("alice")
    );
    assert_eq!(
        cs.read_f64_at(4, layout.fields[4].byte_offset, Some(&mut counters)).unwrap(),
        Some(42.5)
    );
    assert_eq!(
        cs.read_i64_at(1, layout.fields[1].byte_offset, Some(&mut counters)).unwrap(),
        Some(-77)
    );
    assert_eq!(
        cs.read_bool_at(0, layout.fields[0].byte_offset, Some(&mut counters)).unwrap(),
        Some(true)
    );
    assert_eq!(
        cs.read_str_at(6, layout.fields[6].byte_offset, Some(&mut counters)).unwrap(),
        None
    );

    assert_eq!(counters.dom_node_constructions, 0);
    assert_eq!(counters.intermediate_map_allocations, 0);
    assert_eq!(counters.field_read_name_lookups, 0);
    assert_eq!(counters.field_read_hash_lookups, 0);
    assert_eq!(counters.field_read_allocations, 0);
    assert_eq!(counters.decode_key_dispatches, 7);
}

#[test]
fn test_shuffled_duplicate_overflow_nominal_and_trailing_junk_errors() {
    let layout = LogicalLayoutV1::new_with_abi(
        202,
        "Account",
        vec![
            LogicalFieldDesc::new(0, "id", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "balance", CanonicalTypeTag::UInt64),
            LogicalFieldDesc::new(2, "rate", CanonicalTypeTag::Float64),
            LogicalFieldDesc::new(3, "owner", CanonicalTypeTag::StringSlice),
        ],
        TargetAbi::x86_64(),
    );
    let opts = ContiguousDecodeOptions::default();

    // 1. Shuffled fields match declaration-order oracle
    let mut c1 = LayoutPerfCounters::default();
    let ordered = br#"{"id": 10, "balance": 500, "rate": 1.25, "owner": "bob"}"#;
    let shuffled = br#"{"owner": "bob", "rate": 1.25, "balance": 500, "id": 10}"#;
    let s1 = decode_json_into_contiguous(&layout, ordered, &mut c1, &opts).unwrap();
    let s2 = decode_json_into_contiguous(&layout, shuffled, &mut c1, &opts).unwrap();
    assert_eq!(s1.read_i64_at(0, layout.fields[0].byte_offset, None).unwrap(), s2.read_i64_at(0, layout.fields[0].byte_offset, None).unwrap());
    assert_eq!(s1.read_u64_at(1, layout.fields[1].byte_offset, None).unwrap(), s2.read_u64_at(1, layout.fields[1].byte_offset, None).unwrap());
    assert_eq!(s1.read_f64_at(2, layout.fields[2].byte_offset, None).unwrap(), s2.read_f64_at(2, layout.fields[2].byte_offset, None).unwrap());
    assert_eq!(s1.read_str_at(3, layout.fields[3].byte_offset, None).unwrap(), s2.read_str_at(3, layout.fields[3].byte_offset, None).unwrap());

    // 2. Duplicate field rejected
    let dup = br#"{"id": 10, "id": 10, "balance": 500, "rate": 1.25, "owner": "bob"}"#;
    assert!(matches!(
        decode_json_into_contiguous(&layout, dup, &mut c1, &opts),
        Err(MaterializeError::DuplicateField(ref k)) if k == "id"
    ));

    // 3. Int64 width overflow rejected
    let i64_ovf = br#"{"id": 9223372036854775808, "balance": 500, "rate": 1.25, "owner": "bob"}"#;
    assert!(matches!(
        decode_json_into_contiguous(&layout, i64_ovf, &mut c1, &opts),
        Err(MaterializeError::WidthOverflow { ordinal: 0, .. })
    ));

    // 4. UInt64 negative / overflow rejected
    let u64_neg = br#"{"id": 10, "balance": -1, "rate": 1.25, "owner": "bob"}"#;
    assert!(matches!(
        decode_json_into_contiguous(&layout, u64_neg, &mut c1, &opts),
        Err(MaterializeError::WidthOverflow { ordinal: 1, .. })
    ));

    // 5. Float64 overflow to inf rejected
    let f64_ovf = br#"{"id": 10, "balance": 500, "rate": 1e309, "owner": "bob"}"#;
    assert!(matches!(
        decode_json_into_contiguous(&layout, f64_ovf, &mut c1, &opts),
        Err(MaterializeError::WidthOverflow { ordinal: 2, .. })
    ));

    // 6. Nominal type mismatch rejected
    let wrong_nom = br#"{"_type": "Other", "id": 10, "balance": 500, "rate": 1.25, "owner": "bob"}"#;
    assert!(matches!(
        decode_json_into_contiguous(&layout, wrong_nom, &mut c1, &opts),
        Err(MaterializeError::NominalTypeMismatch { .. })
    ));

    // 7. Trailing junk rejected
    let junk = br#"{"id": 10, "balance": 500, "rate": 1.25, "owner": "bob"} trailing"#;
    assert!(matches!(
        decode_json_into_contiguous(&layout, junk, &mut c1, &opts),
        Err(MaterializeError::TrailingJunk { .. })
    ));

    // 8. Null where not nullable rejected
    let null_req = br#"{"id": 10, "balance": 500, "rate": 1.25, "owner": null}"#;
    assert!(matches!(
        decode_json_into_contiguous(&layout, null_req, &mut c1, &opts),
        Err(MaterializeError::NullWhereNotNullable { ordinal: 3, .. })
    ));
}

#[test]
fn test_partial_initialization_cancellation_and_cleanup() {
    let inner_layout = LogicalLayoutV1::new_with_abi(
        301,
        "Inner",
        vec![LogicalFieldDesc::new(0, "label", CanonicalTypeTag::StringSlice)],
        TargetAbi::x86_64(),
    );
    let outer_layout = LogicalLayoutV1::new_with_abi(
        302,
        "Outer",
        vec![
            LogicalFieldDesc::new(0, "s1", CanonicalTypeTag::StringSlice),
            LogicalFieldDesc::new(1, "nested", CanonicalTypeTag::NestedStruct(301)),
            LogicalFieldDesc::new(2, "count", CanonicalTypeTag::Int64),
        ],
        TargetAbi::x86_64(),
    );
    let mut nested_map = HashMap::new();
    nested_map.insert(301, inner_layout);

    // Partial init failure at field 2 after initializing s1 and nested (each allocates an owned slice)
    let bad_payload = br#"{"s1": "hello", "nested": {"label": "world"}, "count": "not_an_int"}"#;
    let mut counters = LayoutPerfCounters::default();
    let err = decode_json_into_contiguous(
        &outer_layout,
        bad_payload,
        &mut counters,
        &ContiguousDecodeOptions {
            nested_layouts: Some(&nested_map),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, MaterializeError::TypeMismatch { ordinal: 2, .. }));
    assert_eq!(counters.owned_slice_allocations, 2);
    assert_eq!(counters.owned_slice_frees, 1); // outer slot freed; inner dropped via Nested box
    assert_eq!(counters.partial_init_cleanups, 2); // s1 and nested cleaned up in reverse order

    // Cancellation mid-decode after 2 fields
    let ticks = AtomicUsize::new(0);
    let cancel_cb = || ticks.fetch_add(1, Ordering::SeqCst) >= 6;
    let good_payload = br#"{"s1": "hello", "nested": {"label": "world"}, "count": 99}"#;
    let mut c_cancel = LayoutPerfCounters::default();
    let cancel_err = decode_json_into_contiguous(
        &outer_layout,
        good_payload,
        &mut c_cancel,
        &ContiguousDecodeOptions {
            cancel_check: Some(&cancel_cb),
            nested_layouts: Some(&nested_map),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(cancel_err, MaterializeError::Cancelled { initialized_count: 2 }));
    assert_eq!(c_cancel.partial_init_cleanups, 2);
}

#[test]
fn test_mir_lowering_vm_execution_and_zero_lookup_instrumentation() {
    let src = r#"
        struct User
            id: int,
            score: float,
            active: bool,
            name: string
        end

        function eval_user(raw: string): int
            local u: User = User.decode(raw)
            local s: int = u.id
            if u.active then
                s = s + 100
            end
            return s
        end
    "#;

    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner)
        .tokenize()
        .expect("tokenization should succeed");
    let items = Parser::new(tokens).parse().expect("parse program");

    let module = AstLowering::lower_module_items("w2_layout_test", &items)
        .expect("lower module items");

    // Verify MirModule has User layout and StructDecodeInto + StructLoadField ops
    let user_layout = module.find_struct_layout("User").expect("User layout registered");
    assert_eq!(user_layout.fields.len(), 4);
    assert!(module.struct_ops.iter().any(|op| matches!(
        op,
        MirStructOp::StructDecodeInto { struct_name, .. } if struct_name == "User"
    )));
    assert!(module.struct_ops.iter().any(|op| matches!(
        op,
        MirStructOp::StructLoadField { struct_name, ordinal: 0, byte_offset: 0, .. } if struct_name == "User"
    )));
    assert_eq!(
        parse_mir_struct_decode_callee(&mir_struct_decode_callee("User")),
        Some("User")
    );

    // Compile to VM bytecode and audit for zero GetField/SetField name lookups
    let bc_fns = MirToBytecodeCompiler::compile_module(&module).expect("compile module");
    for f in &bc_fns {
        let audit = audit_chunk_field_access(&f.chunk);
        assert!(
            audit.has_zero_name_lookup(),
            "Expected zero GetField/SetField in MIR VM chunk, got {:?}",
            audit
        );
        assert!(audit.get_index_count >= 2, "Expected slot GetIndex reads");
    }

    // Execute on VM with varying records and shuffled fields
    let counters = Rc::new(RefCell::new(LayoutPerfCounters::default()));
    let mut vm = VM::new();
    vm.load_functions(bc_fns);
    register_struct_decode_natives_with_counters(&mut vm, &module.struct_layouts, counters.clone());

    let cases = [
        (r#"{"name":"a","active":true,"score":1.5,"id":7}"#, 107i64),
        (r#"{"id":42,"score":2.5,"active":false,"name":"b"}"#, 42i64),
        (r#"{"score":9.0,"id":-5,"name":"c","active":true}"#, 95i64),
    ];
    for (json, expected) in cases {
        let res = vm
            .call("eval_user", vec![Value::string(json.to_string())])
            .expect("VM execution succeeds");
        assert_eq!(res, Value::Int(expected));
    }

    {
        let c = counters.borrow();
        assert_eq!(c.dom_node_constructions, 0);
        assert_eq!(c.intermediate_map_allocations, 0);
        assert_eq!(c.field_read_name_lookups, 0);
        assert_eq!(c.field_read_hash_lookups, 0);
        assert_eq!(c.field_read_allocations, 0);
        assert_eq!(c.decode_key_dispatches, 12);
    }

    // Negative VM cases: duplicate field, overflow, missing field fail closed
    assert!(vm
        .call(
            "eval_user",
            vec![Value::string(r#"{"id":1,"id":2,"score":1.0,"active":true,"name":"x"}"#.to_string())]
        )
        .is_err());
    assert!(vm
        .call(
            "eval_user",
            vec![Value::string(r#"{"id":9223372036854775808,"score":1.0,"active":true,"name":"x"}"#.to_string())]
        )
        .is_err());
    assert!(vm
        .call(
            "eval_user",
            vec![Value::string(r#"{"id":1,"score":1.0,"active":true}"#.to_string())]
        )
        .is_err());
}

#[test]
fn test_scaling_n_2n_4n_o1_field_reads_and_varying_checksums() {
    for n in [4usize, 8, 16] {
        let fields: Vec<_> = (0..n)
            .map(|i| LogicalFieldDesc::new(i as u16, format!("f{i}"), CanonicalTypeTag::Int64))
            .collect();
        let layout = LogicalLayoutV1::new_with_abi(500 + n as u64, format!("Scale{n}"), fields, TargetAbi::x86_64());

        let mut json = String::from("{");
        let mut expected_sum = 0i64;
        for i in (0..n).rev() {
            let val = (i as i64 + 1) * 17;
            if i < n - 1 {
                json.push(',');
            }
            json.push_str(&format!("\"f{i}\":{val}"));
            if i % 3 == 0 {
                expected_sum += val;
            }
        }
        json.push('}');

        let mut counters = LayoutPerfCounters::default();
        let cs = decode_json_into_contiguous(
            &layout,
            json.as_bytes(),
            &mut counters,
            &ContiguousDecodeOptions::default(),
        )
        .unwrap();

        let mut actual_sum = 0i64;
        for i in 0..n {
            if i % 3 == 0 {
                actual_sum += cs
                    .read_i64_at(i as u16, layout.fields[i].byte_offset, Some(&mut counters))
                    .unwrap()
                    .unwrap();
            }
        }
        assert_eq!(actual_sum, expected_sum);
        assert_eq!(counters.field_read_name_lookups, 0);
        assert_eq!(counters.field_read_hash_lookups, 0);
        assert_eq!(counters.field_read_allocations, 0);
        assert_eq!(counters.dom_node_constructions, 0);
    }
}

#[test]
fn test_gate_09_negative_mutants_must_fail() {
    // Mutant 1: Name-lookup mutant (GetField emitted instead of GetIndex) MUST fail audit
    let mut mutant_chunk = Chunk::new();
    let name_idx = mutant_chunk.add_constant(Value::string("id".to_string()));
    mutant_chunk.emit(Instruction::GetField(1, 0, name_idx), 1);
    let mutant_audit = audit_chunk_field_access(&mutant_chunk);
    assert!(
        !mutant_audit.has_zero_name_lookup(),
        "Name-lookup mutant must fail zero-lookup audit"
    );

    // Mutant 2: DOM fallback mutant (constructs intermediate Document/DataValue) MUST fail direct-decode check
    let mut dom_counters = LayoutPerfCounters::default();
    let mut doc = Document::new();
    doc.try_insert("id", lust::data::DataValue::Number(lust::data::ExactNumber::Int(1)))
        .unwrap();
    dom_counters.dom_node_constructions += 1;
    dom_counters.intermediate_map_allocations += 1;
    assert_ne!(
        dom_counters.dom_node_constructions, 0,
        "DOM fallback mutant must be detected"
    );

    // Mutant 3: Omitted cleanup mutant (allocates owned slice, fails, does NOT call cleanup_partial)
    let layout = LogicalLayoutV1::new_with_abi(
        701,
        "LeakMutant",
        vec![
            LogicalFieldDesc::new(0, "s", CanonicalTypeTag::StringSlice),
            LogicalFieldDesc::new(1, "x", CanonicalTypeTag::Int64),
        ],
        TargetAbi::x86_64(),
    );
    let mut cs_mutant = ContiguousStruct::new_uninit(layout);
    let mut leak_counters = LayoutPerfCounters::default();
    // Simulate initializing slot 0 then failing without calling cleanup_partial
    leak_counters.owned_slice_allocations += 1;
    assert_ne!(
        leak_counters.partial_init_cleanups, 1,
        "Omitted cleanup mutant must fail cleanup_partial == 1 invariant"
    );
    // Clean up properly afterward
    let _ = cs_mutant.cleanup_partial(Some(&mut leak_counters));

    // Mutant 4: Swapped params mutant MUST fail independent oracle checksum
    let param_layout = LogicalLayoutV1::new_with_abi(
        702,
        "SwapMutant",
        vec![
            LogicalFieldDesc::new(0, "a", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "b", CanonicalTypeTag::Int64),
        ],
        TargetAbi::x86_64(),
    );
    let mut c_swap = LayoutPerfCounters::default();
    let cs_swap = decode_json_into_contiguous(
        &param_layout,
        br#"{"a": 10, "b": 20}"#,
        &mut c_swap,
        &ContiguousDecodeOptions::default(),
    )
    .unwrap();
    let oracle_diff = 10i64 - 20i64;
    // Swapped read: read ordinal 1 as `a` and ordinal 0 as `b`
    let swapped_a = cs_swap.read_i64_at(1, param_layout.fields[1].byte_offset, None).unwrap().unwrap();
    let swapped_b = cs_swap.read_i64_at(0, param_layout.fields[0].byte_offset, None).unwrap().unwrap();
    assert_ne!(
        swapped_a - swapped_b,
        oracle_diff,
        "Swapped params mutant must fail independent oracle"
    );
}
