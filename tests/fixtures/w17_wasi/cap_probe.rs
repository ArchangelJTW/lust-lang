fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("env-fs");
    match mode {
        "spin" => {
            let mut x: u64 = 0;
            loop {
                x = x.wrapping_add(1);
                std::hint::black_box(x);
            }
        }
        "stdio" => {
            println!("w17-stdio-visible");
        }
        _ => {
            let host_path = std::env::var("PATH").is_ok();
            let injected = std::env::var("LUST_WASI_TEST_ENV").unwrap_or_else(|_| "MISSING".to_string());
            let fs_ok = std::fs::read_to_string("/sandbox/hello.txt").is_ok();
            let host_leak = std::fs::read_to_string("/etc/hosts").is_ok();
            println!("host_path={host_path};injected={injected};fs_ok={fs_ok};host_leak={host_leak}");
        }
    }
}
