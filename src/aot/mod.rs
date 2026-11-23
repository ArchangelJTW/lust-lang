pub mod c_emitter;
pub mod cfg_structurer;
pub mod compiler;
pub mod cranelift_object_backend;
pub mod rust_emitter;

pub use c_emitter::{CEmitterOptions, emit_c_from_mir};
pub use cfg_structurer::{
    CfgAnalysis, LoopInfo, StructuredNode, analyze_cfg, structure_mir_function,
};
pub use compiler::{
    AotCompileOptions, AotCompiler, AotLinkOptions, AotTargetKind, RustAotBuildOutcome,
    RustCompileOptions, check_target_core_installed, encode_embedded_lust_source,
    ensure_cached_aot_rt_rlib, extract_embedded_lust_source, is_embedded_mcu_target,
    is_nostd_target, prepare_items_for_mir,
};
pub use cranelift_object_backend::{
    CRANELIFT_OBJECT_BACKEND_STATUS, CRANELIFT_OBJECT_BLOCKED_REASON,
    CraneliftClifValidationReport, compile_mir_to_object_file, validate_mir_module_clif,
};
pub use rust_emitter::{
    GasMetering, RustEmitterOptions, emit_rust_from_mir, prepare_mir_module_for_rust_aot,
};
