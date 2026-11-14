//! Micro-benchmarking utilities (`T-BENCH`).

use std::time::Instant;
use crate::bytecode::{LustMap, Value, ValueKey};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchOptions {
    pub warmup_iters: usize,
    pub sample_count: usize,
    pub iters_per_sample: usize,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            warmup_iters: 5,
            sample_count: 20,
            iters_per_sample: 10,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchResult {
    pub name: String,
    pub samples: usize,
    pub iters_per_sample: usize,
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: u64,
    pub median_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub mad_ns: u64,
    pub mean_gas: u64,
    pub total_gas: u64,
    pub mean_alloc_bytes: usize,
}

impl BenchResult {
    pub fn to_lust_map(&self) -> LustMap {
        let mut m = LustMap::default();
        m.insert(
            ValueKey::from(Value::string("name")),
            Value::string(self.name.clone()),
        );
        m.insert(
            ValueKey::from(Value::string("samples")),
            Value::Int(self.samples as i64),
        );
        m.insert(
            ValueKey::from(Value::string("iters_per_sample")),
            Value::Int(self.iters_per_sample as i64),
        );
        m.insert(
            ValueKey::from(Value::string("min_ns")),
            Value::Int(self.min_ns as i64),
        );
        m.insert(
            ValueKey::from(Value::string("max_ns")),
            Value::Int(self.max_ns as i64),
        );
        m.insert(
            ValueKey::from(Value::string("mean_ns")),
            Value::Int(self.mean_ns as i64),
        );
        m.insert(
            ValueKey::from(Value::string("median_ns")),
            Value::Int(self.median_ns as i64),
        );
        m.insert(
            ValueKey::from(Value::string("p95_ns")),
            Value::Int(self.p95_ns as i64),
        );
        m.insert(
            ValueKey::from(Value::string("p99_ns")),
            Value::Int(self.p99_ns as i64),
        );
        m.insert(
            ValueKey::from(Value::string("mad_ns")),
            Value::Int(self.mad_ns as i64),
        );
        m.insert(
            ValueKey::from(Value::string("mean_gas")),
            Value::Int(self.mean_gas as i64),
        );
        m.insert(
            ValueKey::from(Value::string("total_gas")),
            Value::Int(self.total_gas as i64),
        );
        m.insert(
            ValueKey::from(Value::string("mean_alloc_bytes")),
            Value::Int(self.mean_alloc_bytes as i64),
        );
        m
    }

    pub fn report_line(&self) -> String {
        format!(
            "{:<32} {:>6} samples | median: {:>8.2} µs | p95: {:>8.2} µs | mad: {:>8.2} µs",
            self.name,
            self.samples,
            self.median_ns as f64 / 1_000.0,
            self.p95_ns as f64 / 1_000.0,
            self.mad_ns as f64 / 1_000.0,
        )
    }
}

pub fn bench<F: FnMut()>(name: impl Into<String>, mut f: F, options: BenchOptions) -> BenchResult {
    let n = options.sample_count.max(1);
    for _ in 0..options.warmup_iters {
        f();
    }
    let iters = options.iters_per_sample.max(1);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let per_iter = (t.elapsed().as_nanos() / iters as u128).min(u64::MAX as u128) as u64;
        v.push(per_iter);
    }
    v.sort_unstable();
    let mean = v.iter().map(|&x| x as u128).sum::<u128>() / v.len() as u128;
    let median = v[v.len() / 2];
    let p = |q: usize| v[((v.len() - 1) * q / 100).min(v.len() - 1)];
    let mut d: Vec<u64> = v.iter().map(|x| x.abs_diff(median)).collect();
    d.sort_unstable();
    let mad = d[d.len() / 2];

    BenchResult {
        name: name.into(),
        samples: n,
        iters_per_sample: iters,
        min_ns: v[0],
        max_ns: *v.last().unwrap(),
        mean_ns: mean as u64,
        median_ns: median,
        p95_ns: p(95),
        p99_ns: p(99),
        mad_ns: mad,
        mean_gas: 0,
        total_gas: 0,
        mean_alloc_bytes: 0,
    }
}
