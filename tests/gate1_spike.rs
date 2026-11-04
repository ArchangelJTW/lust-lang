// Empirical feasibility spike for Issue #11 Gate 1:
// Assessing pinned serde_json 1.0.151 capabilities, token class fidelity,
// duplicate key rejection, quota enforcement, and caller-borrowed input exclusion.

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use std::collections::HashSet;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ParsedNumber {
    Int(i64),
    Float(f64),
    UnsignedAboveI64(u64),
    Overflow,
    UnderflowToZero,
    InvalidTokenClass,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpikeValue<'a> {
    Nil,
    Bool(bool),
    Number(ParsedNumber),
    BorrowedStr(&'a str),
    OwnedStr(String),
    Array(Vec<SpikeValue<'a>>),
    Map(Vec<(String, SpikeValue<'a>)>),
}

// ---------------------------------------------------------------------------
// 1. Raw serde_json Visitor to see exact callbacks emitted by serde_json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum RawToken {
    Unit,
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    BorrowedStr(String),
    OwnedStr(String),
}

struct RawVisitor;

impl<'de> Visitor<'de> for RawVisitor {
    type Value = RawToken;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("any JSON scalar")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(RawToken::Unit)
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
        Ok(RawToken::Bool(v))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
        Ok(RawToken::I64(v))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
        Ok(RawToken::U64(v))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
        Ok(RawToken::F64(v))
    }

    fn visit_borrowed_str<E: de::Error>(self, v: &'de str) -> Result<Self::Value, E> {
        Ok(RawToken::BorrowedStr(v.to_string()))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
        Ok(RawToken::OwnedStr(v.to_string()))
    }

    fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
        Ok(RawToken::OwnedStr(v))
    }
}

fn raw_parse(json: &str) -> Result<RawToken, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_str(json);
    de.deserialize_any(RawVisitor)
}

#[test]
fn test_serde_json_raw_token_fidelity() {
    println!("\n=== 1. RAW SERDE_JSON TOKEN BEHAVIOR & FIDELITY ASSERTIONS ===");

    // Integers vs Floats
    assert_eq!(raw_parse("1").unwrap(), RawToken::U64(1));
    assert_eq!(raw_parse("0").unwrap(), RawToken::U64(0));
    assert_eq!(raw_parse("-1").unwrap(), RawToken::I64(-1));
    assert_eq!(raw_parse("1.0").unwrap(), RawToken::F64(1.0));
    assert_eq!(raw_parse("1e0").unwrap(), RawToken::F64(1.0));
    assert_eq!(raw_parse("1e5").unwrap(), RawToken::F64(100000.0));

    // Hard defect 1: Negative zero - integer `-0` vs float `-0.0`
    // RFC 04 requires: integer token without decimal/exponent -> Int(0).
    // decimal/exponent token -> Float(-0.0).
    // Serde_json yields RawToken::F64(-0.0) for BOTH, making them completely indistinguishable!
    let neg_zero_int = raw_parse("-0").unwrap();
    let neg_zero_float = raw_parse("-0.0").unwrap();
    assert_eq!(neg_zero_int, RawToken::F64(-0.0));
    assert_eq!(neg_zero_float, RawToken::F64(-0.0));
    assert_eq!(neg_zero_int, neg_zero_float, "Serde_json collapses integer -0 and float -0.0 into identical F64(-0.0)");

    // i64 boundaries
    assert_eq!(raw_parse("9223372036854775807").unwrap(), RawToken::U64(9223372036854775807));
    assert_eq!(raw_parse("-9223372036854775808").unwrap(), RawToken::I64(-9223372036854775808));
    assert_eq!(raw_parse("9223372036854775808").unwrap(), RawToken::U64(9223372036854775808)); // > i64::MAX, fits in u64
    assert_eq!(raw_parse("18446744073709551615").unwrap(), RawToken::U64(18446744073709551615)); // u64::MAX

    // Hard defect 2: Integer overflow > u64::MAX silently falls back to F64!
    // RFC 04 requires: "Reject unsigned-only integers above i64::MAX and below i64::MIN; never silently fall back to f64."
    // 18446744073709551616 has no '.' or 'e', but serde_json yields F64!
    let u64_overflow = raw_parse("18446744073709551616").unwrap();
    assert!(matches!(u64_overflow, RawToken::F64(_)), "Serde_json silently converts u64 overflow to F64!");

    // Hard defect 3: Integer underflow < i64::MIN silently falls back to F64!
    // -9223372036854775809 has no '.' or 'e', but serde_json yields F64!
    let i64_underflow = raw_parse("-9223372036854775809").unwrap();
    assert!(matches!(i64_underflow, RawToken::F64(_)), "Serde_json silently converts i64 underflow to F64!");

    // Hard defect 4: Float underflow to zero
    // RFC 04 requires: "reject overflow, non-finite results, and nonzero values rounded to zero by underflow."
    // 1e-500 is a nonzero float, but serde_json silently returns 0.0!
    let float_underflow = raw_parse("1e-500").unwrap();
    assert_eq!(float_underflow, RawToken::F64(0.0), "Serde_json silently rounds 1e-500 to 0.0 without underflow notification!");

    // Float overflow: correctly errors
    let float_overflow = raw_parse("1e500");
    assert!(float_overflow.is_err(), "Serde_json correctly rejects 1e500 as number out of range");
}

// ---------------------------------------------------------------------------
// 2. Duplicate decoded & escape-equivalent key rejection test
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct DuplicateKeyError(pub String);

struct BoundedMapVisitor {
    seen_keys: HashSet<String>,
}

impl<'de> Visitor<'de> for BoundedMapVisitor {
    type Value = Vec<(String, String)>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a JSON map with unique decoded keys")
    }

    fn visit_map<A: MapAccess<'de>>(mut self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut entries = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            if !self.seen_keys.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate object key: {:?}", key)));
            }
            let val = map.next_value::<String>()?;
            entries.push((key, val));
        }
        Ok(entries)
    }
}

fn parse_map_with_duplicate_check(json: &str) -> Result<Vec<(String, String)>, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_str(json);
    de.deserialize_map(BoundedMapVisitor {
        seen_keys: HashSet::new(),
    })
}

#[test]
fn test_duplicate_and_escape_equivalent_keys() {
    println!("\n=== 2. DUPLICATE AND ESCAPE-EQUIVALENT KEY REJECTION ===");

    // Plain duplicate
    let plain_dup = r#"{"a": "1", "a": "2"}"#;
    let res1 = parse_map_with_duplicate_check(plain_dup);
    println!("Plain duplicate 'a' -> {:?}", res1);
    assert!(res1.is_err());
    assert!(res1.unwrap_err().to_string().contains("duplicate object key: \"a\""));

    // Escape-equivalent duplicate: "foo" vs "\u0066oo"
    let escape_dup = r#"{"foo": "1", "\u0066oo": "2"}"#;
    let res2 = parse_map_with_duplicate_check(escape_dup);
    println!("Escape-equivalent 'foo' vs '\\u0066oo' -> {:?}", res2);
    assert!(res2.is_err());
    assert!(res2.unwrap_err().to_string().contains("duplicate object key: \"foo\""));

    // Escaped surrogate / non-ascii duplicate: "café" vs "caf\u00e9"
    let unicode_dup = r#"{"café": "1", "caf\u00e9": "2"}"#;
    let res3 = parse_map_with_duplicate_check(unicode_dup);
    println!("Unicode duplicate 'café' vs 'caf\\u00e9' -> {:?}", res3);
    assert!(res3.is_err());

    // Distinct keys
    let distinct = r#"{"a": "1", "b": "2"}"#;
    let res4 = parse_map_with_duplicate_check(distinct);
    println!("Distinct keys 'a', 'b' -> {:?}", res4);
    assert!(res4.is_ok());
}

// ---------------------------------------------------------------------------
// 3. Quotas and Capacity Enforcement Before Allocation/Growth
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Quotas {
    pub max_input_bytes: usize,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_bridge_bytes: usize,
}

#[derive(Debug, Default)]
pub struct AccountingLedger {
    pub current_bridge_bytes: usize,
    pub peak_bridge_bytes: usize,
    pub node_count: usize,
    pub current_depth: usize,
}

impl AccountingLedger {
    pub fn charge(&mut self, bytes: usize, quotas: &Quotas) -> Result<(), &'static str> {
        let new_bytes = self.current_bridge_bytes.checked_add(bytes).ok_or("integer overflow in ledger")?;
        if new_bytes > quotas.max_bridge_bytes {
            return Err("LIMIT: max_bridge_bytes exceeded");
        }
        self.current_bridge_bytes = new_bytes;
        if self.current_bridge_bytes > self.peak_bridge_bytes {
            self.peak_bridge_bytes = self.current_bridge_bytes;
        }
        Ok(())
    }

    pub fn release(&mut self, bytes: usize) {
        self.current_bridge_bytes = self.current_bridge_bytes.saturating_sub(bytes);
    }

    pub fn enter_depth(&mut self, quotas: &Quotas) -> Result<(), &'static str> {
        if self.current_depth >= quotas.max_depth {
            return Err("LIMIT: max_depth exceeded");
        }
        self.current_depth += 1;
        Ok(())
    }

    pub fn exit_depth(&mut self) {
        self.current_depth = self.current_depth.saturating_sub(1);
    }

    pub fn charge_node(&mut self, quotas: &Quotas) -> Result<(), &'static str> {
        if self.node_count >= quotas.max_nodes {
            return Err("LIMIT: max_nodes exceeded");
        }
        self.node_count += 1;
        Ok(())
    }
}

#[test]
fn test_quota_limits_empirical() {
    println!("\n=== 3. QUOTA AND CAPACITY ENFORCEMENT ===");
    let quotas = Quotas {
        max_input_bytes: 30,
        max_depth: 3,
        max_nodes: 4,
        max_bridge_bytes: 64,
    };

    // 1. max_input_bytes check (verified prior to any parsing/allocation)
    let oversized_input = "{\"a\": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}";
    assert!(oversized_input.len() > quotas.max_input_bytes);
    let check_input = |input: &str, q: &Quotas| -> Result<(), &'static str> {
        if input.len() > q.max_input_bytes {
            Err("LIMIT: max_input_bytes exceeded")
        } else {
            Ok(())
        }
    };
    assert_eq!(check_input(oversized_input, &quotas), Err("LIMIT: max_input_bytes exceeded"));

    // 2. Depth limit test
    let deep_json = "[[[[[1]]]]]";
    let mut ledger = AccountingLedger::default();
    let mut de = serde_json::Deserializer::from_str(deep_json);

    struct DepthGuardedVisitor<'a> {
        ledger: &'a mut AccountingLedger,
        quotas: &'a Quotas,
    }

    impl<'de, 'a> de::DeserializeSeed<'de> for DepthGuardedVisitor<'a> {
        type Value = ();

        fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
            deserializer.deserialize_any(self)
        }
    }

    impl<'de, 'a> Visitor<'de> for DepthGuardedVisitor<'a> {
        type Value = ();

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("any JSON")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            self.ledger.enter_depth(self.quotas).map_err(de::Error::custom)?;
            self.ledger.charge_node(self.quotas).map_err(de::Error::custom)?;
            while let Some(()) = seq.next_element_seed(DepthGuardedVisitor {
                ledger: self.ledger,
                quotas: self.quotas,
            })? {}
            self.ledger.exit_depth();
            Ok(())
        }

        fn visit_u64<E: de::Error>(self, _v: u64) -> Result<Self::Value, E> {
            self.ledger.charge_node(self.quotas).map_err(de::Error::custom)?;
            Ok(())
        }
    }

    let res_depth = de.deserialize_any(DepthGuardedVisitor {
        ledger: &mut ledger,
        quotas: &quotas,
    });
    println!("Deep nesting (depth > 3) -> {:?}", res_depth);
    assert!(res_depth.is_err());
    assert!(res_depth.unwrap_err().to_string().contains("LIMIT: max_depth exceeded"));

    // 3. Node count limit test
    let many_nodes_json = "[1, 2, 3, 4, 5, 6]"; // 1 array node + 6 integer nodes = 7 nodes > max_nodes (4)
    let mut node_ledger = AccountingLedger::default();
    let mut de_nodes = serde_json::Deserializer::from_str(many_nodes_json);
    let res_nodes = de_nodes.deserialize_any(DepthGuardedVisitor {
        ledger: &mut node_ledger,
        quotas: &quotas,
    });
    println!("Excess nodes (> 4) -> {:?}", res_nodes);
    assert!(res_nodes.is_err());
    assert!(res_nodes.unwrap_err().to_string().contains("LIMIT: max_nodes exceeded"));

    // 4. Hard limitation of serde_json: Parser scratch buffer is unmetered prior to allocation
    // When parsing escaped strings like "\u0041\u0042\u0043...", serde_json's internal `scratch: Vec<u8>`
    // reallocates without any callback or capacity check hook.
    // The visitor only receives the parsed string *after* scratch has already allocated on heap.
    // Therefore, pre-growth capacity enforcement for parser scratch cannot be enforced via serde_json.
}

// ---------------------------------------------------------------------------
// 4. Caller-borrowed input exclusion test
// ---------------------------------------------------------------------------

#[test]
fn test_caller_borrowed_input_exclusion() {
    println!("\n=== 4. CALLER-BORROWED INPUT EXCLUSION ===");
    let caller_json = r#"["hello", "world", "unescaped string that is quite long to ensure no accidental allocation occurs without escapes"]"#;

    // We demonstrate that serde_json can deserialize &'a str directly from the input buffer
    // without allocating a String!
    let mut de = serde_json::Deserializer::from_str(caller_json);

    struct BorrowedSeqVisitor;
    impl<'de> Visitor<'de> for BorrowedSeqVisitor {
        type Value = Vec<&'de str>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("array of borrowed strings")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut list = Vec::new();
            while let Some(s) = seq.next_element::<&'de str>()? {
                list.push(s);
            }
            Ok(list)
        }
    }

    let borrowed_slices = de.deserialize_seq(BorrowedSeqVisitor).unwrap();
    println!("Borrowed strings count: {}", borrowed_slices.len());
    for s in &borrowed_slices {
        // Assert that the pointer points directly inside caller_json's memory range!
        let caller_start = caller_json.as_ptr() as usize;
        let caller_end = caller_start + caller_json.len();
        let s_ptr = s.as_ptr() as usize;
        println!("Slice: {:?} (ptr={}, caller=[{}, {}])", s, s_ptr, caller_start, caller_end);
        assert!(s_ptr >= caller_start && s_ptr + s.len() <= caller_end);
    }
    println!("Verified: borrowed strings point directly into caller buffer with zero bridge heap allocation!");

    // What happens if an escaped string is encountered with &'de str?
    let escaped_json = r#"["hello\nworld"]"#;
    let mut de_escaped = serde_json::Deserializer::from_str(escaped_json);
    let res_escaped = de_escaped.deserialize_seq(BorrowedSeqVisitor);
    println!("Escaped string deserialized into &'de str -> {:?}", res_escaped);
    assert!(res_escaped.is_err(), "serde_json cannot borrow an escaped string without scratch buffer!");
}
