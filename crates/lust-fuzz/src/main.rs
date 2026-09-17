//! The differential runner: write a program, run it through the interpreter
//! and through the JIT, compare what they said. One worker per core, one
//! shared seed counter, everything reproducible from a seed.
//!
//!   lust-fuzz run    --cases N [--seed S] [--size K] [--jobs J] [--keep-going] [--no-shrink]
//!   lust-fuzz one    --seed S [--size K]        # print the program for a seed
//!   lust-fuzz replay --seed S [--size K]        # run one case, verbose
//!
//! The interpreter is the ground truth. A disagreement is a JIT bug until
//! shown otherwise; a program either engine refuses to compile is a
//! generator bug. Both are findings, both stop the run unless --keep-going.

mod engine;
mod writer;
mod rng;

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use engine::{Answer, Outcome};
use writer::Program;

struct Settings {
    cases: u64,
    first_seed: u64,
    size: u32,
    jobs: usize,
    keep_going: bool,
    no_shrink: bool,
}

#[derive(Debug, Clone)]
enum Kind {
    /// The generator wrote a program the compiler refused.
    Broke { message: String },
    /// Interpreter and JIT disagreed.
    Disagree { interp: Answer, jit: Answer },
}

#[derive(Debug, Clone)]
struct Finding {
    seed: u64,
    program: String,
    kind: Kind,
}

#[derive(Default, Clone, Copy)]
struct Stats {
    cases: u64,
    agreed: u64,
    /// Cases where the JIT actually ran native code.
    native: u64,
    root_traces: u64,
    side_traces: u64,
    native_entries: u64,
    guard_exits: u64,
    runtime_errors: u64,
}

impl Stats {
    fn add(&mut self, o: &Stats) {
        self.cases += o.cases;
        self.agreed += o.agreed;
        self.native += o.native;
        self.root_traces += o.root_traces;
        self.side_traces += o.side_traces;
        self.native_entries += o.native_entries;
        self.guard_exits += o.guard_exits;
        self.runtime_errors += o.runtime_errors;
    }
}

fn usage() -> ! {
    eprintln!("usage: lust-fuzz run --cases N [--seed S] [--size K] [--jobs J] [--keep-going] [--no-shrink]");
    eprintln!("       lust-fuzz one --seed S [--size K]");
    eprintln!("       lust-fuzz replay --seed S [--size K]");
    std::process::exit(64)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let mut s = Settings {
        cases: 1000,
        first_seed: 1,
        size: 3,
        jobs: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        keep_going: false,
        no_shrink: false,
    };
    let cmd = args[0].clone();
    let mut i = 1;
    let val = |i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| usage())
    };
    while i < args.len() {
        match args[i].as_str() {
            "--cases" => s.cases = val(&mut i).parse().unwrap_or_else(|_| usage()),
            "--seed" => s.first_seed = val(&mut i).parse().unwrap_or_else(|_| usage()),
            "--size" => s.size = val(&mut i).parse().unwrap_or_else(|_| usage()),
            "--jobs" => s.jobs = val(&mut i).parse().unwrap_or_else(|_| usage()),
            "--keep-going" => s.keep_going = true,
            "--no-shrink" => s.no_shrink = true,
            _ => usage(),
        }
        i += 1;
    }
    match cmd.as_str() {
        "one" => {
            print!("{}", writer::render(&writer::program(s.first_seed, s.size)));
        }
        "replay" => {
            let program = writer::program(s.first_seed, s.size);
            let source = writer::render(&program);
            print!("{source}");
            let mut stats = Stats::default();
            match run_case(&program, s.first_seed, &mut stats) {
                Some(f) => {
                    print_finding(&f);
                    std::process::exit(1);
                }
                None => {
                    let jit = engine::run(&source, true).unwrap();
                    println!("--- seed {}: interpreter and JIT agree", s.first_seed);
                    println!("--- jit: {}", jit.summary());
                }
            }
        }
        "run" => run(&s),
        _ => usage(),
    }
}

fn run(s: &Settings) {
    let next = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let done = AtomicU64::new(0);
    let findings: Mutex<Vec<Finding>> = Mutex::new(Vec::new());
    let totals: Mutex<Stats> = Mutex::new(Stats::default());
    let start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..s.jobs {
            let (next, stop, done, findings, totals) = (&next, &stop, &done, &findings, &totals);
            scope.spawn(move || {
                let mut stats = Stats::default();
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= s.cases {
                        break;
                    }
                    let seed = s.first_seed + k;
                    let program = writer::program(seed, s.size);
                    let started = Instant::now();
                    let result = run_case(&program, seed, &mut stats);
                    let took = started.elapsed();
                    if took.as_secs_f64() > 5.0 {
                        eprintln!("\nslow: seed {seed} took {:.1}s", took.as_secs_f64());
                    }
                    if let Some(f) = result {
                        let f = if s.no_shrink { f } else { shrink(program, f) };
                        findings.lock().unwrap().push(f);
                        if !s.keep_going {
                            stop.store(true, Ordering::Relaxed);
                        }
                    }
                    done.fetch_add(1, Ordering::Relaxed);
                }
                totals.lock().unwrap().add(&stats);
            });
        }
        scope.spawn(|| {
            let mut last = 0;
            while !stop.load(Ordering::Relaxed) && done.load(Ordering::Relaxed) < s.cases {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let d = done.load(Ordering::Relaxed);
                if d != last {
                    let secs = start.elapsed().as_secs_f64();
                    eprint!(
                        "\r{d}/{} cases, {:.0}/s, {} finding(s)   ",
                        s.cases,
                        d as f64 / secs,
                        findings.lock().unwrap().len()
                    );
                    last = d;
                }
            }
        });
    });
    eprintln!();
    let t = *totals.lock().unwrap();
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{} cases in {secs:.1}s on {} worker(s) ({:.0} cases/s); {} agreed, {} ran native code, {} raised runtime errors",
        t.cases,
        s.jobs,
        t.cases as f64 / secs.max(0.001),
        t.agreed,
        t.native,
        t.runtime_errors
    );
    println!(
        "jit: {} root traces, {} side traces, {} native entries, {} guard exits",
        t.root_traces, t.side_traces, t.native_entries, t.guard_exits
    );
    let fs = findings.lock().unwrap();
    for f in fs.iter() {
        println!();
        print_finding(f);
    }
    if !fs.is_empty() {
        std::process::exit(1);
    }
}

fn print_finding(f: &Finding) {
    println!("=== seed {} ===", f.seed);
    print!("{}", f.program);
    match &f.kind {
        Kind::Broke { message } => println!("--- the compiler refused the program: {message}"),
        Kind::Disagree { interp, jit } => {
            println!("--- interpreter and JIT disagree:");
            println!("  interpreter {}", interp.summary());
            println!("  jit         {}", jit.summary());
        }
    }
}

/// Run one program through both engines. `None` means they agreed.
fn run_case(program: &Program, seed: u64, stats: &mut Stats) -> Option<Finding> {
    let source = writer::render(program);
    stats.cases += 1;
    let finding = |kind: Kind| {
        Some(Finding {
            seed,
            program: source.clone(),
            kind,
        })
    };
    let interp = match engine::run(&source, false) {
        Ok(a) => a,
        Err(message) => return finding(Kind::Broke { message }),
    };
    let jit = match engine::run(&source, true) {
        Ok(a) => a,
        Err(message) => return finding(Kind::Broke { message }),
    };
    stats.root_traces += jit.root_traces;
    stats.side_traces += jit.side_traces;
    stats.native_entries += jit.native_entries;
    stats.guard_exits += jit.guard_exits;
    if jit.native_entries > 0 {
        stats.native += 1;
    }
    if matches!(interp.outcome, Outcome::Failed(_)) {
        stats.runtime_errors += 1;
    }
    if interp.outcome == jit.outcome {
        stats.agreed += 1;
        None
    } else {
        finding(Kind::Disagree { interp, jit })
    }
}

/// Does this (program, kind) still reproduce the finding? Only disagreements
/// are shrunk; a refused program is a generator bug best read whole.
fn still_fails(program: &Program) -> Option<Finding> {
    let mut stats = Stats::default();
    match run_case(program, 0, &mut stats) {
        Some(f) if matches!(f.kind, Kind::Disagree { .. }) => Some(f),
        _ => None,
    }
}

fn shrink(mut program: Program, finding: Finding) -> Finding {
    if !matches!(finding.kind, Kind::Disagree { .. }) {
        return finding;
    }
    let seed = finding.seed;
    let mut best = finding;
    loop {
        let mut progressed = false;
        // Try removing each statement, deepest paths first so inner bodies
        // empty out before their enclosing statement goes.
        let mut paths = writer::stmt_paths(&program);
        paths.sort_by(|a, b| b.len().cmp(&a.len()).then(b.cmp(a)));
        for path in paths {
            let mut candidate = program.clone();
            if !writer::remove_at(&mut candidate, &path) {
                continue;
            }
            if let Some(mut f) = still_fails(&candidate) {
                f.seed = seed;
                best = f;
                program = candidate;
                progressed = true;
            }
        }
        let mut candidate = program.clone();
        if writer::shrink_loops(&mut candidate) {
            if let Some(mut f) = still_fails(&candidate) {
                f.seed = seed;
                best = f;
                program = candidate;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    best
}
