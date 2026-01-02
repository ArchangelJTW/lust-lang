#![cfg(feature = "std")]

use crate::embed::native_types::ModuleStub;
use crate::{NativeExport, VM};
use std::{
    collections::BTreeMap,
    fs,
    io,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Default)]
pub struct DumpExternsOptions {
    /// When an export name is not module-qualified (has no '.' or '::'),
    /// place it into this module.
    pub default_module: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ExternFile {
    pub relative_path: PathBuf,
    pub contents: String,
}

pub fn extern_files_from_vm(vm: &VM, options: &DumpExternsOptions) -> Vec<ExternFile> {
    extern_files_from_exports(vm.exported_natives(), vm.exported_type_stubs(), options)
}

pub fn write_extern_files(
    output_root: impl AsRef<Path>,
    files: &[ExternFile],
) -> io::Result<Vec<PathBuf>> {
    let output_root = output_root.as_ref();
    if files.is_empty() {
        return Ok(Vec::new());
    }
    fs::create_dir_all(output_root)?;

    let mut written = Vec::with_capacity(files.len());
    for file in files {
        let mut relative = file.relative_path.clone();
        if relative.extension().is_none() {
            relative.set_extension("lust");
        }
        let destination = output_root.join(&relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&destination, &file.contents)?;
        written.push(relative);
    }

    Ok(written)
}

fn extern_files_from_exports(
    exports: &[NativeExport],
    type_stubs: &[ModuleStub],
    options: &DumpExternsOptions,
) -> Vec<ExternFile> {
    if exports.is_empty() && type_stubs.iter().all(ModuleStub::is_empty) {
        return Vec::new();
    }

    #[derive(Default)]
    struct CombinedModule<'a> {
        type_stub: ModuleStub,
        functions: Vec<&'a NativeExport>,
    }

    let mut combined: BTreeMap<String, CombinedModule<'_>> = BTreeMap::new();
    for stub in type_stubs {
        if stub.is_empty() {
            continue;
        }
        let entry = combined
            .entry(stub.module.clone())
            .or_insert_with(|| CombinedModule {
                type_stub: ModuleStub {
                    module: stub.module.clone(),
                    ..ModuleStub::default()
                },
                ..CombinedModule::default()
            });
        entry.type_stub.struct_defs.extend(stub.struct_defs.clone());
        entry.type_stub.enum_defs.extend(stub.enum_defs.clone());
        entry.type_stub.trait_defs.extend(stub.trait_defs.clone());
        entry.type_stub.const_defs.extend(stub.const_defs.clone());
    }

    for export in exports {
        let normalized_name = export.name().replace("::", ".");
        let (module, function) = match normalized_name.rsplit_once('.') {
            Some(parts) => (parts.0.to_string(), parts.1.to_string()),
            None => match &options.default_module {
                Some(default) if !default.trim().is_empty() => (default.clone(), normalized_name),
                _ => continue,
            },
        };
        let entry = combined
            .entry(module.clone())
            .or_insert_with(|| CombinedModule {
                type_stub: ModuleStub {
                    module,
                    ..ModuleStub::default()
                },
                ..CombinedModule::default()
            });
        if function.is_empty() {
            continue;
        }
        entry.functions.push(export);
    }

    let mut result = Vec::new();
    for (module, mut combined_entry) in combined {
        combined_entry
            .functions
            .sort_by(|a, b| a.name().cmp(b.name()));
        let mut contents = String::new();

        let mut wrote_type = false;
        let append_defs = |defs: &Vec<String>, contents: &mut String, wrote_flag: &mut bool| {
            if defs.is_empty() {
                return;
            }
            if *wrote_flag && !contents.ends_with("\n\n") && !contents.is_empty() {
                contents.push('\n');
            }
            for def in defs {
                contents.push_str(def);
                if !def.ends_with('\n') {
                    contents.push('\n');
                }
            }
            *wrote_flag = true;
        };

        append_defs(
            &combined_entry.type_stub.struct_defs,
            &mut contents,
            &mut wrote_type,
        );
        append_defs(
            &combined_entry.type_stub.enum_defs,
            &mut contents,
            &mut wrote_type,
        );
        append_defs(
            &combined_entry.type_stub.trait_defs,
            &mut contents,
            &mut wrote_type,
        );
        append_defs(
            &combined_entry.type_stub.const_defs,
            &mut contents,
            &mut wrote_type,
        );

        if !combined_entry.functions.is_empty() {
            if wrote_type && !contents.ends_with("\n\n") {
                contents.push('\n');
            }
            contents.push_str("pub extern\n");
            for export in combined_entry.functions {
                let normalized_name = export.name().replace("::", ".");
                if let Some((_, function)) = normalized_name.rsplit_once('.') {
                    let params = format_params(export);
                    let return_type = export.return_type();
                    if let Some(doc) = export.doc() {
                        contents.push_str("    -- ");
                        contents.push_str(doc);
                        if !doc.ends_with('\n') {
                            contents.push('\n');
                        }
                    }
                    contents.push_str("    function ");
                    contents.push_str(function);
                    contents.push('(');
                    contents.push_str(&params);
                    contents.push(')');
                    if !return_type.trim().is_empty() && return_type.trim() != "()" {
                        contents.push_str(": ");
                        contents.push_str(return_type);
                    }
                    contents.push('\n');
                } else if let Some(default) = &options.default_module {
                    if default == &module {
                        let params = format_params(export);
                        let return_type = export.return_type();
                        contents.push_str("    function ");
                        contents.push_str(&normalized_name);
                        contents.push('(');
                        contents.push_str(&params);
                        contents.push(')');
                        if !return_type.trim().is_empty() && return_type.trim() != "()" {
                            contents.push_str(": ");
                            contents.push_str(return_type);
                        }
                        contents.push('\n');
                    }
                }
            }
            contents.push_str("end\n");
        }

        if contents.is_empty() {
            continue;
        }
        let mut relative = relative_stub_path(&module);
        if relative.extension().is_none() {
            relative.set_extension("lust");
        }
        result.push(ExternFile {
            relative_path: relative,
            contents,
        });
    }

    result
}

fn format_params(export: &NativeExport) -> String {
    export
        .params()
        .iter()
        .map(|param| {
            let ty = param.ty().trim();
            if ty.is_empty() {
                "any"
            } else {
                ty
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn relative_stub_path(module: &str) -> PathBuf {
    let mut path = PathBuf::new();
    let mut segments: Vec<String> = module.split('.').map(|seg| seg.replace('-', "_")).collect();
    if let Some(first) = segments.first() {
        if first == "externs" {
            segments.remove(0);
        }
    }
    if let Some(first) = segments.first() {
        path.push(first);
    }
    if segments.len() > 1 {
        for seg in &segments[1..segments.len() - 1] {
            path.push(seg);
        }
        path.push(segments.last().unwrap());
    } else if let Some(first) = segments.first() {
        path.push(first);
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LustConfig, NativeExportParam};
    use tempfile::tempdir;

    #[test]
    fn writes_vm_exports_and_type_stubs() {
        let mut vm = VM::with_config(&LustConfig::default());
        vm.register_type_stubs(vec![ModuleStub {
            module: "host".to_string(),
            struct_defs: vec!["pub struct Widget\nend\n".to_string()],
            ..ModuleStub::default()
        }]);
        vm.record_exported_native(NativeExport::new(
            "host.scale",
            vec![NativeExportParam::new("value", "int")],
            "int",
        ));

        let dir = tempdir().expect("temp dir");
        let files = extern_files_from_vm(&vm, &DumpExternsOptions::default());
        assert_eq!(files.len(), 1);
        let written = write_extern_files(dir.path(), &files).expect("write externs");
        assert_eq!(written.len(), 1);
        let destination = dir.path().join(&written[0]);
        let contents = fs::read_to_string(destination).expect("read output");
        assert!(contents.contains("pub struct Widget"));
        assert!(contents.contains("pub extern"));
        assert!(contents.contains("function scale(int): int"));
    }
}

