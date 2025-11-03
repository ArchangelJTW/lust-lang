mod completions;
mod hover;
mod inlay_hints;
mod server;

pub use server::run;

#[cfg(test)]
pub(crate) use completions::{
    analyze_identifier_context, analyze_member_method_context, identifier_completions,
    infer_constructor_call_base_name, infer_struct_literal_base_name, instance_method_completions,
    module_path_completions, resolve_base_type_name_for_context, struct_field_completions,
    type_for_identifier, CompletionKind,
};
#[cfg(test)]
pub(crate) use hover::*;

#[cfg(test)]
mod tests {
    include!("tests.rs");
}
