//! The differential runner: write a program, run it through the interpreter
//! and through the JIT, compare what they said. One worker per core, one
//! shared seed counter, everything reproducible from a seed.
//!
//!   lust-fuzz run    --cases N [--seed S] [--size K] [--jobs J] [--keep-going] [--no-shrink] [--fg]
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
    /// Run at normal priority. By default workers run at macOS background
    /// QoS (efficiency cores, yielding to foreground apps), which keeps the
    /// machine usable but is several times slower per case.
    foreground: bool,
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
    functions: u64,
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
        self.functions += o.functions;
        self.native_entries += o.native_entries;
        self.guard_exits += o.guard_exits;
        self.runtime_errors += o.runtime_errors;
    }
}

fn usage() -> ! {
    eprintln!("usage: lust-fuzz run --cases N [--seed S] [--size K] [--jobs J] [--keep-going] [--no-shrink] [--fg]");
    eprintln!("       lust-fuzz one --seed S [--size K]");
    eprintln!("       lust-fuzz replay --seed S [--size K]");
    std::process::exit(64)
}

thread_local! {
    static CURRENT_SEED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Run the calling thread at macOS "background" quality of service, so the
/// scheduler gives foreground apps the performance cores whenever they want
/// them: a fuzz batch then uses every idle core without making the machine
/// stutter. No-op elsewhere.
#[cfg(target_os = "macos")]
fn background_priority() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    const QOS_CLASS_BACKGROUND: u32 = 0x09;
    // SAFETY: plain libc call on the current thread with valid constants.
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn background_priority() {}

/// Resident set size of this process in bytes (macOS), or None.
#[cfg(target_os = "macos")]
fn resident_bytes() -> Option<u64> {
    #[repr(C)]
    struct TaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [i32; 2],
        system_time: [i32; 2],
        policy: i32,
        suspend_count: i32,
    }
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn task_info(target: u32, flavor: u32, info: *mut TaskBasicInfo, count: *mut u32) -> i32;
    }
    const MACH_TASK_BASIC_INFO: u32 = 20;
    let mut info = TaskBasicInfo {
        virtual_size: 0,
        resident_size: 0,
        resident_size_max: 0,
        user_time: [0; 2],
        system_time: [0; 2],
        policy: 0,
        suspend_count: 0,
    };
    let mut count = (std::mem::size_of::<TaskBasicInfo>() / 4) as u32;
    // SAFETY: task_info fills a struct of the declared flavor and size.
    let rc = unsafe { task_info(mach_task_self_, MACH_TASK_BASIC_INFO, &mut info, &mut count) };
    (rc == 0).then_some(info.resident_size)
}

#[cfg(not(target_os = "macos"))]
fn resident_bytes() -> Option<u64> {
    None
}

/// A case that runs for longer than this, or a process that grows past
/// `MAX_RESIDENT_BYTES`, is a generator bug (a program that does not
/// terminate, usually while allocating). Report the seed and abort before
/// the machine starts swapping. Memory is the real guard; the time limit is
/// generous because background-QoS workers on efficiency cores are several
/// times slower than a foreground run.
const MAX_CASE_SECONDS: f64 = 120.0;
const MAX_RESIDENT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

fn main() {
    // The workspace builds with panic=abort, so a panic inside an engine
    // takes the whole run down; at least say which seed did it.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        eprintln!("\n=== panic while running seed {} ===", CURRENT_SEED.with(|c| c.get()));
        default_hook(info);
    }));
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
        foreground: false,
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
            "--fg" => s.foreground = true,
            _ => usage(),
        }
        i += 1;
    }
    match cmd.as_str() {
        "one" => {
            print!("{}", writer::render(&writer::program(s.first_seed, s.size)));
        }
        "replay" => {
            if !s.foreground {
                background_priority();
            }
            CURRENT_SEED.with(|c| c.set(s.first_seed));
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
                    let started = Instant::now();
                    let interp = engine::run(&source, false).unwrap();
                    let interp_secs = started.elapsed().as_secs_f64();
                    let started = Instant::now();
                    let jit = engine::run(&source, true).unwrap();
                    let jit_secs = started.elapsed().as_secs_f64();
                    println!("--- seed {}: interpreter and JIT agree", s.first_seed);
                    println!("--- interpreter: {interp_secs:.3}s  {}", interp.summary());
                    println!("--- jit:         {jit_secs:.3}s  {}", jit.summary());
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
    // Per worker: the seed being run and when it started, for the watchdog.
    let in_flight: Vec<Mutex<Option<(u64, Instant)>>> =
        (0..s.jobs).map(|_| Mutex::new(None)).collect();
    let in_flight = &in_flight;
    let findings: Mutex<Vec<Finding>> = Mutex::new(Vec::new());
    let totals: Mutex<Stats> = Mutex::new(Stats::default());
    let start = Instant::now();
    std::thread::scope(|scope| {
        for w in 0..s.jobs {
            let (next, stop, done, findings, totals) = (&next, &stop, &done, &findings, &totals);
            scope.spawn(move || {
                if !s.foreground {
                    background_priority();
                }
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
                    CURRENT_SEED.with(|c| c.set(seed));
                    let program = writer::program(seed, s.size);
                    let started = Instant::now();
                    *in_flight[w].lock().unwrap() = Some((seed, started));
                    let result = run_case(&program, seed, &mut stats);
                    *in_flight[w].lock().unwrap() = None;
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
                // Watchdog: a hung or ballooning case is a generator bug;
                // name it and stop before the machine starts swapping.
                let resident = resident_bytes().unwrap_or(0);
                for slot in in_flight {
                    if let Some((seed, started)) = *slot.lock().unwrap() {
                        let secs = started.elapsed().as_secs_f64();
                        if secs > MAX_CASE_SECONDS || resident > MAX_RESIDENT_BYTES {
                            eprintln!(
                                "\n=== watchdog: seed {seed} has run {secs:.0}s, process resident {} MB — aborting (generator bug: non-terminating program?) ===",
                                resident / (1024 * 1024)
                            );
                            std::process::exit(2);
                        }
                    }
                }
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
        "jit: {} root traces, {} functions, {} native entries, {} guard exits",
        t.root_traces, t.functions, t.native_entries, t.guard_exits
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
    stats.functions += jit.functions;
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
