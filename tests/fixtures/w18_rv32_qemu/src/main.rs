#![no_std]
#![no_main]

extern crate alloc;

use alloc::format;
use alloc::vec;
use core::alloc::{Allocator, GlobalAlloc, Layout};
use core::arch::{asm, naked_asm};
use core::panic::PanicInfo;
use core::ptr;
use lust_aot_rt::{AotCtx, AotError, LustBudgetAlloc, StaticSlabAllocator};

mod lust_kernel {
    include!(concat!(env!("OUT_DIR"), "/lust_kernel.rs"));
}

#[path = "../../../../src/no_std/trace.rs"]
mod trace;

use trace::{
    CanonicalTraceCodec, DeterministicScheduler, ExternalInput, FinalOutcome, SimTaskSpec,
    TargetInfo,
};

// -----------------------------------------------------------------------------
// Global allocator backed by lust_aot_rt::StaticSlabAllocator<65536>
// -----------------------------------------------------------------------------

struct SlabGlobalAlloc(StaticSlabAllocator<65536>);

// SAFETY: Bare-metal QEMU virt single-hart (hart 0) runs without preemption/threads.
unsafe impl Sync for SlabGlobalAlloc {}

#[global_allocator]
static GLOBAL_ALLOC: SlabGlobalAlloc = SlabGlobalAlloc(StaticSlabAllocator::new());

unsafe impl GlobalAlloc for SlabGlobalAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match self.0.allocate(layout) {
            Ok(nn) => nn.as_ptr() as *mut u8,
            Err(_) => ptr::null_mut(),
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // StaticSlabAllocator is a monotonic bump slab.
    }
}

// -----------------------------------------------------------------------------
// QEMU virt 16550A UART (0x1000_0000) & SiFive Test Finisher (0x0010_0000)
// -----------------------------------------------------------------------------

const UART0_THR: *mut u8 = 0x1000_0000 as *mut u8;
const UART0_LSR: *const u8 = 0x1000_0005 as *const u8;
const SIFIVE_TEST: *mut u32 = 0x0010_0000 as *mut u32;

fn uart_putc(b: u8) {
    // SAFETY: QEMU `virt` maps 16550A UART at 0x1000_0000 with LSR at +5.
    unsafe {
        while (ptr::read_volatile(UART0_LSR) & 0x20) == 0 {}
        ptr::write_volatile(UART0_THR, b);
    }
}

fn uart_print(s: &str) {
    for &b in s.as_bytes() {
        uart_putc(b);
    }
}

fn qemu_exit(code: u32) -> ! {
    // SiFive test device: 0x5555 = FINISHER_PASS (exit 0),
    // (code << 16) | 0x3333 = FINISHER_FAIL (non-zero exit).
    let val = if code == 0 {
        0x5555
    } else {
        (code << 16) | 0x3333
    };
    // SAFETY: QEMU `virt` maps sifive,test0 at 0x0010_0000.
    unsafe {
        ptr::write_volatile(SIFIVE_TEST, val);
        loop {
            asm!("wfi", options(nomem, nostack, preserves_flags));
        }
    }
}

// -----------------------------------------------------------------------------
// RV32 Bare-Metal Entry (_start) & Panic Handler
// -----------------------------------------------------------------------------

#[unsafe(link_section = ".text.start")]
#[unsafe(no_mangle)]
#[unsafe(naked)]
pub unsafe extern "C" fn _start() -> ! {
    naked_asm!(
        // Park secondary harts if any
        "csrr t0, mhartid",
        "bnez t0, 3f",
        // Initialize stack pointer
        "la sp, _stack_start",
        // Enable RV32F FPU (mstatus.FS = 0b11, bits 14:13 = 0x6000)
        "li t0, 0x6000",
        "csrs mstatus, t0",
        // Zero .bss section
        "la t0, __sbss",
        "la t1, __ebss",
        "1:",
        "bgeu t0, t1, 2f",
        "sw zero, 0(t0)",
        "addi t0, t0, 4",
        "j 1b",
        "2:",
        "j {main}",
        "3:",
        "wfi",
        "j 3b",
        main = sym rv32_main,
    );
}

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    uart_print("PANIC:");
    if let Some(loc) = info.location() {
        let msg = format!("{}:{}\n", loc.file(), loc.line());
        uart_print(&msg);
    } else {
        uart_print("unknown\n");
    }
    qemu_exit(1)
}

// -----------------------------------------------------------------------------
// RV32 Bare-Metal Verification Suite
// -----------------------------------------------------------------------------

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01B3);
    }
    h
}

extern "C" fn rv32_main() -> ! {
    // (a) Deterministic Lust #![no_std] computation
    let mut ctx_ok = AotCtx::with_gas_limit(10_000);
    let dot_sum_res = match lust_kernel::dot_sum_with_ctx(&mut ctx_ok, 10) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("ERR:DOT_SUM:{e}\n");
            uart_print(&msg);
            qemu_exit(10);
        }
    };

    // (b1) LustBudgetAlloc OOM budget enforcement
    let budget = LustBudgetAlloc::with_limit(128);
    let l64 = Layout::from_size_align(64, 8).unwrap();
    let l128 = Layout::from_size_align(128, 8).unwrap();
    let p1 = match budget.allocate(l64) {
        Ok(p) => p,
        Err(_) => {
            uart_print("ERR:BUDGET_FIRST_ALLOC\n");
            qemu_exit(11);
        }
    };
    let budget_oom = match budget.allocate(l128) {
        Err(_) => {
            let err = AotError::from_alloc_failure(&budget, 128);
            match err {
                AotError::OutOfMemory {
                    limit_bytes: 128,
                    requested_bytes: 128,
                } => 1u32,
                _ => {
                    uart_print("ERR:BUDGET_OOM_SNAPSHOT\n");
                    qemu_exit(12);
                }
            }
        }
        Ok(_) => {
            uart_print("ERR:BUDGET_OOM_EXPECTED\n");
            qemu_exit(12);
        }
    };
    // SAFETY: p1 was allocated from `budget` with `l64`.
    unsafe { budget.deallocate(p1.cast::<u8>(), l64) };
    if budget.allocate(l128).is_err() {
        uart_print("ERR:BUDGET_REALLOC_AFTER_FREE\n");
        qemu_exit(13);
    }

    // (b2) StaticSlabAllocator OOM enforcement
    let slab = StaticSlabAllocator::<256>::new();
    if slab.allocate(l128).is_err() {
        uart_print("ERR:SLAB_FIRST_ALLOC\n");
        qemu_exit(14);
    }
    let l256 = Layout::from_size_align(256, 8).unwrap();
    let slab_oom = match slab.allocate(l256) {
        Err(_) => 1u32,
        Ok(_) => {
            uart_print("ERR:SLAB_OOM_EXPECTED\n");
            qemu_exit(15);
        }
    };

    // (c) Gas exhaustion (OutOfGas) enforcement on riscv32imafc-unknown-none-elf
    let mut ctx_oom = AotCtx::with_gas_limit(3);
    let gas_oom = match lust_kernel::dot_sum_with_ctx(&mut ctx_oom, 100) {
        Err(AotError::OutOfGas { .. }) => 1u32,
        other => {
            let msg = format!("ERR:GAS_OOM_EXPECTED:{other:?}\n");
            uart_print(&msg);
            qemu_exit(16);
        }
    };

    // (d) Deterministic DST trace + canonical codec + replay verification
    let tasks = vec![
        SimTaskSpec {
            task_id: 1,
            steps: 3,
            delta_per_step: 10,
        },
        SimTaskSpec {
            task_id: 2,
            steps: 3,
            delta_per_step: 5,
        },
        SimTaskSpec {
            task_id: 3,
            steps: 2,
            delta_per_step: 7,
        },
    ];
    let sched = DeterministicScheduler {
        seed: 42,
        run_id: alloc::string::String::from("rv32-dst-t3"),
        target: TargetInfo {
            triple: alloc::string::String::from("riscv32imafc-unknown-none-elf"),
            endianness: alloc::string::String::from("little"),
            toolchain: alloc::string::String::from("rustc-1.100.0"),
            candidate_digest: alloc::string::String::from("w18-rv32"),
            fixture_hash: alloc::string::String::from("rv32-sample-v1"),
        },
        external_inputs: vec![ExternalInput {
            name: alloc::string::String::from("sim_tasks"),
            content_hash: alloc::string::String::from("sha256:tasks-v1"),
            payload_len: 3,
        }],
        fault_at_event: None,
    };
    let trace_v1 = sched.run(&tasks);
    if let Err(e) = sched.replay_verify(&trace_v1, &tasks) {
        let msg = format!("ERR:REPLAY_VERIFY:{e}\n");
        uart_print(&msg);
        qemu_exit(17);
    }
    let trace_bytes = CanonicalTraceCodec::encode_canonical(&trace_v1);
    let decoded = match CanonicalTraceCodec::decode_canonical(&trace_bytes) {
        Ok(t) => t,
        Err(e) => {
            let msg = format!("ERR:DECODE:{e}\n");
            uart_print(&msg);
            qemu_exit(18);
        }
    };
    if CanonicalTraceCodec::compare_semantic(&trace_v1, &decoded).is_err() {
        uart_print("ERR:SEMANTIC_MISMATCH\n");
        qemu_exit(19);
    }
    let trace_fnv64 = fnv1a_64(&trace_bytes);
    let final_val = match trace_v1.final_result {
        FinalOutcome::OkValue(v) => v,
        FinalOutcome::StableError(_) => -1,
    };

    let summary = format!(
        "RV32-DST:OK|dot_sum={}|budget_oom={}|slab_oom={}|gas_oom={}|trace_len={}|trace_fnv64={:016x}|final_val={}|final_state={}\n",
        dot_sum_res,
        budget_oom,
        slab_oom,
        gas_oom,
        trace_bytes.len(),
        trace_fnv64,
        final_val,
        trace_v1.final_state
    );
    uart_print(&summary);
    qemu_exit(0)
}
