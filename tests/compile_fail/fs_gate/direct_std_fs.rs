//! Canary test for verifying that Clippy compile-time blacklist gate
//! rejects disallowed direct host filesystem and Path methods.

#[test]
fn canary_allowed_baseline() {
    assert_eq!(2 * 2, 4);
}

#[cfg(feature = "canary_read")]
pub fn canary_fs_read() {
    let _ = std::fs::read("canary.txt");
}

#[cfg(feature = "canary_read_to_string")]
pub fn canary_fs_read_to_string() {
    let _ = std::fs::read_to_string("canary.txt");
}

#[cfg(feature = "canary_write")]
pub fn canary_fs_write() {
    let _ = std::fs::write("canary.txt", b"data");
}

#[cfg(feature = "canary_file_open")]
pub fn canary_file_open() {
    let _ = std::fs::File::open("canary.txt");
}

#[cfg(feature = "canary_open_options")]
pub fn canary_open_options() {
    let _ = std::fs::OpenOptions::new().open("canary.txt");
}

#[cfg(feature = "canary_dir_builder")]
pub fn canary_dir_builder() {
    let _ = std::fs::DirBuilder::new().create("canary_dir");
}

#[cfg(feature = "canary_path_exists")]
pub fn canary_path_exists(p: &std::path::Path) -> bool {
    p.exists()
}

#[cfg(feature = "canary_path_canonicalize")]
pub fn canary_path_canonicalize(p: &std::path::Path) {
    let _ = p.canonicalize();
}

#[cfg(feature = "canary_default_violation")]
pub fn direct_fs_violation_canary() {
    let _ = std::fs::read("canary.txt");
}
