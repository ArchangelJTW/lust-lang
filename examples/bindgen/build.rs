use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let entry = manifest_dir.join("scripts/main.lust");
    let generated = lust::bindgen::RustBindingsBuilder::new(&entry)
        .bindings_name("GameBindings")
        .generate()?;

    for path in generated.input_files() {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let output = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("game_bindings.rs");
    generated.write_to(output)?;
    Ok(())
}
