use std::env;

fn main() {
    let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // Export symbols from the CLI binary so dlopen'ed Lua 5.1 modules can
    // resolve lua_* shims (e.g., luasocket core) at runtime.
    if target_family == "unix" && target_env != "msvc" {
        println!("cargo:rustc-link-arg-bin=lust=-Wl,-export-dynamic");
    }
}
