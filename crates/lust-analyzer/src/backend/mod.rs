mod completions;
mod highlights;
mod hover;
mod inlay_hints;
mod references;
mod rename;
mod server;
mod signature_help;
mod symbols;

pub use server::run;

#[cfg(test)]
pub(crate) use completions::{
    analyze_identifier_context, analyze_member_method_context, analyze_module_path_context,
    identifier_completions, infer_constructor_call_base_name, infer_struct_literal_base_name,
    instance_method_completions, module_path_completions, prewarm_builtins,
    resolve_base_type_name_for_context, struct_field_completions, type_for_identifier,
    CompletionKind,
};
#[cfg(test)]
pub(crate) use highlights::*;
#[cfg(test)]
pub(crate) use hover::*;
#[cfg(test)]
pub(crate) use references::*;
#[cfg(test)]
pub(crate) use rename::*;
#[cfg(test)]
pub(crate) use signature_help::*;
#[cfg(test)]
pub(crate) use symbols::*;

#[cfg(test)]
mod tests {
    include!("tests.rs");
}
