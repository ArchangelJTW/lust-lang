//! Runs a generated program in-process through the embedding API, once with
//! the JIT disabled (the interpreter, our ground truth) and once with it
//! enabled, and reports what each said.

use lust::{EmbeddedProgram, LustConfig, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The entry function returned this string.
    Returned(String),
    /// The program raised a runtime error.
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Answer {
    pub outcome: Outcome,
    pub root_traces: u64,
    pub functions: u64,
    pub native_entries: u64,
    pub guard_exits: u64,
    pub execution_failures: u64,
}

impl Answer {
    pub fn summary(&self) -> String {
        let what = match &self.outcome {
            Outcome::Returned(s) => format!("returned {:?}", s),
            Outcome::Failed(e) => format!("error: {e}"),
        };
        format!(
            "{what}  [traces {}, functions {}, native entries {}, guard exits {}, failures {}]",
            self.root_traces,
            self.functions,
            self.native_entries,
            self.guard_exits,
            self.execution_failures
        )
    }
}

/// Compile and run `source`. `Err` is a compile (lexer/parser/type) error,
/// which for a generated program means the generator wrote something invalid.
pub fn run(source: &str, jit: bool) -> Result<Answer, String> {
    let mut config = LustConfig::default();
    config.set_jit_enabled(jit);
    let mut program = EmbeddedProgram::builder()
        .with_config(config)
        .module("main", source)
        .entry_module("main")
        .compile()
        .map_err(|e| e.to_string())?;
    let outcome = match program.call_raw("main.run", Vec::new()) {
        Ok(Value::String(s)) => Outcome::Returned(s.to_string()),
        Ok(other) => Outcome::Returned(format!("<non-string {other}>")),
        Err(e) => Outcome::Failed(e.to_string()),
    };
    let stats = program.jit_stats();
    Ok(Answer {
        outcome,
        root_traces: stats.root_traces_compiled,
        functions: stats.functions_compiled,
        native_entries: stats.native_trace_entries,
        guard_exits: stats.guard_exits,
        execution_failures: stats.execution_failures,
    })
}
