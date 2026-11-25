#![no_std]
#![no_main]

extern crate alloc;

use alloc::format;
use alloc::vec;
use core::alloc::{Allocator, GlobalAlloc, Layout};
use core::arch::asm;
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
// Global allocator backed by lust_aot_rt::StaticSlabAllocator (24 KiB for 64 KiB LM3S6965 RAM)
// -----------------------------------------------------------------------------

struct SlabGlobalAlloc(StaticSlabAllocator<24576>);

// SAFETY: Cortex-M4 bare-metal target runs single-core without preemption/threads.
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
// ARM Cortex-M4 Semihosting (SYS_WRITE0 = 0x04, SYS_EXIT = 0x18 via bkpt #0xAB)
// -----------------------------------------------------------------------------

#[inline(never)]
fn semihost_call(op: u32, arg: u32) -> u32 {
    let mut ret: u32;
    // SAFETY: ARMv7-M semihosting breakpoint handled by QEMU `-semihosting-config`.
    unsafe {
        asm!(
            "bkpt #0xAB",
            inout("r0") op => ret,
            in("r1") arg,
            options(nostack, preserves_flags)
        );
    }
    ret
}

fn semihost_write0_bytes(nul_terminated: &[u8]) {
    let _ = semihost_call(0x04, nul_terminated.as_ptr() as u32);
}

fn semihost_print(s: &str) {
    // Print in fixed stack chunks so we don't depend on heap allocation in panic/fault paths.
    let bytes = s.as_bytes();
    let mut buf = [0u8; 129];
    let mut i = 0;
    while i < bytes.len() {
        let chunk = (bytes.len() - i).min(128);
        buf[..chunk].copy_from_slice(&bytes[i..i + chunk]);
        buf[chunk] = 0;
        semihost_write0_bytes(&buf[..chunk + 1]);
        i += chunk;
    }
}

fn semihost_exit(code: u32) -> ! {
    // ADP_Stopped_ApplicationExit = 0x20026 (exit 0),
    // ADP_Stopped_RunTimeErrorUnknown = 0x20023 (non-zero exit).
    let reason: u32 = if code == 0 { 0x20026 } else { 0x20023 };
    let _ = semihost_call(0x18, reason);
    loop {
        // SAFETY: WFI halt if semihosting exit did not terminate immediately.
        unsafe { asm!("wfi", options(nomem, nostack, preserves_flags)) };
    }
}

// -----------------------------------------------------------------------------
// Cortex-M4 Vector Table, Reset Handler, Fault Handlers, Panic Handler
// -----------------------------------------------------------------------------

unsafe extern "C" {
    static mut __sbss: u32;
    static mut __ebss: u32;
    static mut __sdata: u32;
    static mut __edata: u32;
    static __sidata: u32;
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Reset() -> ! {
    // Enable FPU (CP10 & CP11 full access in CPACR = 0xE000ED88) before any FP instructions.
    // SAFETY: Standard Cortex-M4 SCB CPACR MMIO register initialization before Rust code runs.
    unsafe {
        let cpacr = 0xE000_ED88 as *mut u32;
        let val = ptr::read_volatile(cpacr);
        ptr::write_volatile(cpacr, val | (0xF << 20));
        asm!("dsb", "isb", options(nostack, preserves_flags));

        // Zero .bss
        let mut bss = &raw mut __sbss;
        let ebss = &raw mut __ebss;
        while bss < ebss {
            ptr::write_volatile(bss, 0);
            bss = bss.add(1);
        }

        // Copy .data from FLASH to RAM
        let mut data = &raw mut __sdata;
        let edata = &raw mut __edata;
        let mut idata = &raw const __sidata;
        while data < edata {
            ptr::write_volatile(data, ptr::read_volatile(idata));
            data = data.add(1);
            idata = idata.add(1);
        }
    }

    arm_main()
}

#[unsafe(no_mangle)]
pub extern "C" fn DefaultExceptionHandler() -> ! {
    semihost_print("ERR:CORTEX_M_EXCEPTION\n");
    semihost_exit(2)
}

#[unsafe(link_section = ".vector_table.reset_vector")]
#[unsafe(no_mangle)]
pub static __RESET_VECTOR: unsafe extern "C" fn() -> ! = Reset;

#[unsafe(link_section = ".vector_table.exceptions")]
#[unsafe(no_mangle)]
pub static __EXCEPTIONS: [extern "C" fn() -> !; 14] = [
    DefaultExceptionHandler, // NMI
    DefaultExceptionHandler, // HardFault
    DefaultExceptionHandler, // MemManage
    DefaultExceptionHandler, // BusFault
    DefaultExceptionHandler, // UsageFault
    DefaultExceptionHandler, // Reserved
    DefaultExceptionHandler, // Reserved
    DefaultExceptionHandler, // Reserved
    DefaultExceptionHandler, // Reserved
    DefaultExceptionHandler, // SVCall
    DefaultExceptionHandler, // DebugMon
    DefaultExceptionHandler, // Reserved
    DefaultExceptionHandler, // PendSV
    DefaultExceptionHandler, // SysTick
];

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    semihost_print("PANIC:");
    if let Some(loc) = info.location() {
        let msg = format!("{}:{}\n", loc.file(), loc.line());
        semihost_print(&msg);
    } else {
        semihost_print("unknown\n");
    }
    semihost_exit(1)
}

// -----------------------------------------------------------------------------
// DST-T3 Bare-Metal Verification Suite
// -----------------------------------------------------------------------------

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01B3);
    }
    h
}

fn arm_main() -> ! {
    // (a) Deterministic Lust #![no_std] computation
    let mut ctx_ok = AotCtx::with_gas_limit(10_000);
    let dot_sum_res = match lust_kernel::dot_sum_with_ctx(&mut ctx_ok, 10) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("ERR:DOT_SUM:{e}\n");
            semihost_print(&msg);
            semihost_exit(10);
        }
    };

    // (b1) LustBudgetAlloc OOM budget enforcement
    let budget = LustBudgetAlloc::with_limit(128);
    let l64 = Layout::from_size_align(64, 8).unwrap();
    let l128 = Layout::from_size_align(128, 8).unwrap();
    let p1 = match budget.allocate(l64) {
        Ok(p) => p,
        Err(_) => {
            semihost_print("ERR:BUDGET_FIRST_ALLOC\n");
            semihost_exit(11);
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
                    semihost_print("ERR:BUDGET_OOM_SNAPSHOT\n");
                    semihost_exit(12);
                }
            }
        }
        Ok(_) => {
            semihost_print("ERR:BUDGET_OOM_EXPECTED\n");
            semihost_exit(12);
        }
    };
    // SAFETY: p1 was allocated from `budget` with `l64`.
    unsafe { budget.deallocate(p1.cast::<u8>(), l64) };
    if budget.allocate(l128).is_err() {
        semihost_print("ERR:BUDGET_REALLOC_AFTER_FREE\n");
        semihost_exit(13);
    }

    // (b2) StaticSlabAllocator OOM enforcement
    let slab = StaticSlabAllocator::<256>::new();
    if slab.allocate(l128).is_err() {
        semihost_print("ERR:SLAB_FIRST_ALLOC\n");
        semihost_exit(14);
    }
    let l256 = Layout::from_size_align(256, 8).unwrap();
    let slab_oom = match slab.allocate(l256) {
        Err(_) => 1u32,
        Ok(_) => {
            semihost_print("ERR:SLAB_OOM_EXPECTED\n");
            semihost_exit(15);
        }
    };

    // (c) Gas exhaustion (OutOfGas) enforcement on thumbv7em-none-eabihf
    let mut ctx_oom = AotCtx::with_gas_limit(3);
    let gas_oom = match lust_kernel::dot_sum_with_ctx(&mut ctx_oom, 100) {
        Err(AotError::OutOfGas { .. }) => 1u32,
        other => {
            let msg = format!("ERR:GAS_OOM_EXPECTED:{other:?}\n");
            semihost_print(&msg);
            semihost_exit(16);
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
        run_id: alloc::string::String::from("arm-dst-t3"),
        target: TargetInfo {
            triple: alloc::string::String::from("thumbv7em-none-eabihf"),
            endianness: alloc::string::String::from("little"),
            toolchain: alloc::string::String::from("rustc-1.100.0"),
            candidate_digest: alloc::string::String::from("w18-dst-t3"),
            fixture_hash: alloc::string::String::from("dst-t3-sample-v1"),
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
        semihost_print(&msg);
        semihost_exit(17);
    }
    let trace_bytes = CanonicalTraceCodec::encode_canonical(&trace_v1);
    let decoded = match CanonicalTraceCodec::decode_canonical(&trace_bytes) {
        Ok(t) => t,
        Err(e) => {
            let msg = format!("ERR:DECODE:{e}\n");
            semihost_print(&msg);
            semihost_exit(18);
        }
    };
    if CanonicalTraceCodec::compare_semantic(&trace_v1, &decoded).is_err() {
        semihost_print("ERR:SEMANTIC_MISMATCH\n");
        semihost_exit(19);
    }
    let trace_fnv64 = fnv1a_64(&trace_bytes);
    let final_val = match trace_v1.final_result {
        FinalOutcome::OkValue(v) => v,
        FinalOutcome::StableError(_) => -1,
    };

    let summary = format!(
        "DST-T3:OK|dot_sum={}|budget_oom={}|slab_oom={}|gas_oom={}|trace_len={}|trace_fnv64={:016x}|final_val={}|final_state={}\n",
        dot_sum_res,
        budget_oom,
        slab_oom,
        gas_oom,
        trace_bytes.len(),
        trace_fnv64,
        final_val,
        trace_v1.final_state
    );
    semihost_print(&summary);
    semihost_exit(0)
}
