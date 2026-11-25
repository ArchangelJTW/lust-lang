// Host CLI / tooling boundary: C header generator writes output to host filesystem.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use crate::mir::ir::*;
use crate::mir::zig_backend::MirZigBackendError;
use alloc::{format, string::String};
use std::fs;
use std::path::{Path, PathBuf};

pub struct CHeaderGenerator;

impl CHeaderGenerator {
    pub fn generate_header_for_function(func: &MirFunction) -> String {
        let guard_name = format!(
            "LUST_GENERATED_{}_H",
            func.signature
                .name
                .to_ascii_uppercase()
                .replace(|c: char| !c.is_alphanumeric(), "_")
        );

        let mut s = String::new();
        s.push_str(&format!("#ifndef {guard_name}\n"));
        s.push_str(&format!("#define {guard_name}\n\n"));
        s.push_str("#include <stdint.h>\n");
        s.push_str("#include <stdbool.h>\n");
        s.push_str("#include <stddef.h>\n\n");
        s.push_str("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");

        s.push_str(
            "typedef enum {\n\
             \x20   LUST_STATUS_OK = 0,\n\
             \x20   LUST_STATUS_ERR_OUT_OF_GAS = 1,\n\
             \x20   LUST_STATUS_ERR_OUT_OF_MEMORY = 2,\n\
             \x20   LUST_STATUS_ERR_RUNTIME = 3\n\
             } lust_status_t;\n\n\
             void lust_set_gas_limit(int64_t gas);\n\
             int64_t lust_get_gas_remaining(void);\n\
             void lust_set_memory_limit(size_t bytes);\n\
             size_t lust_get_memory_used(void);\n\
             int32_t lust_get_last_status(void);\n\
             void lust_reset_arena(void);\n\n",
        );

        let proto = Self::function_prototype(func);
        s.push_str(&proto);
        s.push_str("\n\n");

        s.push_str("#ifdef __cplusplus\n}\n#endif\n\n");
        s.push_str(&format!("#endif /* {guard_name} */\n"));

        s
    }

    pub fn generate_header_for_module(module: &MirModule) -> String {
        let guard_name = "LUST_GENERATED_H";

        let mut s = String::new();
        s.push_str(&format!("#ifndef {guard_name}\n"));
        s.push_str(&format!("#define {guard_name}\n\n"));
        s.push_str("#include <stdint.h>\n");
        s.push_str("#include <stdbool.h>\n");
        s.push_str("#include <stddef.h>\n\n");
        s.push_str("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");

        s.push_str(
            "typedef enum {\n\
             \x20   LUST_STATUS_OK = 0,\n\
             \x20   LUST_STATUS_ERR_OUT_OF_GAS = 1,\n\
             \x20   LUST_STATUS_ERR_OUT_OF_MEMORY = 2,\n\
             \x20   LUST_STATUS_ERR_RUNTIME = 3\n\
             } lust_status_t;\n\n\
             void lust_set_gas_limit(int64_t gas);\n\
             int64_t lust_get_gas_remaining(void);\n\
             void lust_set_memory_limit(size_t bytes);\n\
             size_t lust_get_memory_used(void);\n\
             int32_t lust_get_last_status(void);\n\
             void lust_reset_arena(void);\n\n",
        );

        for func in &module.functions {
            let proto = Self::function_prototype(func);
            s.push_str(&proto);
            s.push('\n');
        }
        s.push('\n');

        s.push_str("#ifdef __cplusplus\n}\n#endif\n\n");
        s.push_str(&format!("#endif /* {guard_name} */\n"));

        s
    }

    pub fn write_header_for_function(
        func: &MirFunction,
        out_dir: &Path,
    ) -> Result<PathBuf, MirZigBackendError> {
        let content = Self::generate_header_for_function(func);
        let header_path = out_dir.join(format!("{}.h", func.signature.name));
        fs::write(&header_path, content)
            .map_err(|e| MirZigBackendError::WriteFile(format!("{e}")))?;
        Ok(header_path)
    }

    pub fn write_header_for_module(
        module: &MirModule,
        out_dir: &Path,
    ) -> Result<PathBuf, MirZigBackendError> {
        let content = Self::generate_header_for_module(module);
        let header_path = out_dir.join(format!("{}.h", module.name));
        fs::write(&header_path, content)
            .map_err(|e| MirZigBackendError::WriteFile(format!("{e}")))?;
        Ok(header_path)
    }

    fn function_prototype(func: &MirFunction) -> String {
        let ret_ty = mir_type_to_c(&func.signature.return_type);
        let mut params_str = String::new();
        if func.params.is_empty() {
            params_str.push_str("void");
        } else {
            for (i, p_id) in func.params.iter().enumerate() {
                if i > 0 {
                    params_str.push_str(", ");
                }
                let param_local = &func.locals[p_id.0];
                let c_ty = mir_type_to_c(&param_local.ty);
                if let Some(name) = &param_local.name {
                    params_str.push_str(&format!("{c_ty} {name}"));
                } else {
                    params_str.push_str(&format!("{c_ty} arg{i}"));
                }
            }
        }
        format!("{ret_ty} {}({params_str});", func.signature.name)
    }
}

pub fn mir_type_to_c(ty: &MirType) -> String {
    match ty {
        MirType::Int => "int64_t".to_string(),
        MirType::Float => "double".to_string(),
        MirType::Bool => "bool".to_string(),
        MirType::Unit => "void".to_string(),
        MirType::String => "const char*".to_string(),
        MirType::Array(_) => "void*".to_string(),
        MirType::Struct(name) => format!("const {}*", name),
        _ => "int64_t".to_string(),
    }
}
