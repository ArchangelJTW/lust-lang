pub mod c_emitter;
pub mod cfg_structurer;
pub mod compiler;
pub mod rust_emitter;

pub use c_emitter::{CEmitterOptions, emit_c_from_mir};
pub use cfg_structurer::{
    CfgAnalysis, LoopInfo, StructuredNode, analyze_cfg, structure_mir_function,
};
pub use compiler::{
    AotCompileOptions, AotCompiler, AotLinkOptions, AotTargetKind, RustCompileOptions,
    encode_embedded_lust_source, ensure_cached_aot_rt_rlib, extract_embedded_lust_source,
    prepare_items_for_mir,
};
pub use rust_emitter::{
    GasMetering, RustEmitterOptions, emit_rust_from_mir, prepare_mir_module_for_rust_aot,
};
