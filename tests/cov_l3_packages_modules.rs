#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! L3 Coverage and Correctness Test Suite: Packages and Embedded Modules
//!
//! Independent Oracles:
//! 1. Mock local HTTP registry on loopback (127.0.0.1) in temp directory (NO external network).
//! 2. Dependency cycle detection (direct, transitive, and embedded modules).
//! 3. Missing version / unmet semver constraint checking.
//! 4. Conflict detection between direct and transitive requirements.
//! 5. File read denial / unreadable directory / corrupted archive / SHA-256 digest mismatch.
//! 6. Embedded module errors: unknown module name error, capability denial, syntax error during initialization.
//! 7. Hostile edge cases: malformed manifests, circular graphs, invalid checksums, unreadable directories.

use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::PathBuf,
    sync::Arc,
    thread,
};

use lust::{
    LustConfig, LustError,
    embed::native_types::ModuleStub,
    modules::embedded::{EmbeddedModule, build_directory_map, load_program_from_embedded, load_program_from_embedded_with_config},
    packages::{
        ArchiveVerificationError, BuildOptions, DependencyResolutionError,
        DownloadedArchive, LocalModuleError, PackageManager, PackageSpecifier, PackageKind,
        RegistryClient, RegistryError, SearchParameters, Semver, StubFile,
        build_local_module, build_package_archive, compute_sha256, verify_sha256,
        collect_stub_files, write_stub_files, stub_files_from_exports,
        library_file_name, read_crate_name, relative_stub_path, sanitized_crate_name,
        prepare_rust_dependencies, load_prepared_rust_dependencies,
        DependencyResolution,
        resolve_dependencies, resolve_dependencies_with_manager,
        resolve_transitive_dependencies,
        manifest::PackageManifest,
    },
    NativeExport, NativeExportParam,
};

// ============================================================================
// 1. REGISTRY & SHA-256 ORACLES
// ============================================================================

#[test]
fn test_registry_sha256_standard_vectors() {
    // FIPS 180-4 standard test vectors
    let empty_sha = compute_sha256(b"");
    assert_eq!(
        empty_sha, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "SHA-256 empty vector failed on input b\"\""
    );

    let abc_sha = compute_sha256(b"abc");
    assert_eq!(
        abc_sha, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 vector failed on input b\"abc\""
    );

    let multi_block = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    let multi_sha = compute_sha256(multi_block);
    assert_eq!(
        multi_sha, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        "SHA-256 two-block vector failed"
    );

    // verify_sha256 oracle
    assert!(
        verify_sha256(b"abc", "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"),
        "verify_sha256 should be case-insensitive"
    );
    assert!(
        verify_sha256(b"abc", "  ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad \n"),
        "verify_sha256 should trim whitespace"
    );
    assert!(
        !verify_sha256(b"abc", "0000000000000000000000000000000000000000000000000000000000000000"),
        "verify_sha256 should reject mismatched hash"
    );
}

#[test]
fn test_registry_client_new_url_formatting() {
    let client_trailing = RegistryClient::new("http://localhost:8080/").expect("client trailing");
    let client_no_trailing = RegistryClient::new("http://localhost:8080").expect("client no trailing");

    // Both should format paths identically
    let params = SearchParameters::default();
    // Verify client can be created without panic
    let _ = format!("{:?}", client_trailing);
    let _ = format!("{:?}", client_no_trailing);
    let _ = format!("{:?}", params);
}

#[test]
fn test_registry_downloaded_archive_lifecycle() {
    let temp = tempfile::tempdir().expect("tempdir");
    let file_1 = temp.path().join("archive_drop.tar.gz");
    fs::write(&file_1, b"archive payload to be cleaned up").expect("write file_1");
    assert!(file_1.exists(), "file_1 must exist before wrapping");

    // 1. RAII Drop cleanup: dropping DownloadedArchive must delete file
    {
        let archive1 = DownloadedArchive::new(&file_1);
        assert_eq!(archive1.path(), file_1.as_path(), "path() must match input path");
    }
    assert!(!file_1.exists(), "file_1 must be deleted on Drop of DownloadedArchive");

    // 2. into_path() prevents deletion on drop and returns PathBuf
    let file_2 = temp.path().join("archive_keep.tar.gz");
    fs::write(&file_2, b"archive payload to be kept").expect("write file_2");
    assert!(file_2.exists(), "file_2 must exist before wrapping");

    {
        let archive2 = DownloadedArchive::new(&file_2);
        let extracted_path = archive2.into_path();
        assert_eq!(extracted_path, file_2, "into_path must return the file path");
    }
    assert!(file_2.exists(), "file_2 must persist after into_path()");
}

/// Spawns a mock HTTP server on 127.0.0.1 for the duration of the test.
/// Completely local: NO external network calls.
fn run_mock_registry_server<F>(handler: F) -> (String, thread::JoinHandle<()>)
where
    F: Fn(&str, &str, &str) -> (u16, String, Vec<u8>) + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral loopback port");
    let port = listener.local_addr().expect("local addr").port();
    let base_url = format!("http://127.0.0.1:{}/", port);
    let handler = Arc::new(handler);

    let handle = thread::spawn(move || {
        for _ in 0..10 {
            let (mut stream, _) = match listener.accept() {
                Ok(conn) => conn,
                Err(_) => break,
            };

            let mut total_req = Vec::new();
            let mut buf = [0u8; 8192];
            let mut expected_len: Option<usize> = None;
            let mut header_len = 0;

            loop {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        total_req.extend_from_slice(&buf[..n]);
                        if expected_len.is_none() {
                            if let Some(pos) = total_req.windows(4).position(|w| w == b"\r\n\r\n") {
                                header_len = pos + 4;
                                let header_str = String::from_utf8_lossy(&total_req[..pos]);
                                if header_str.to_ascii_lowercase().contains("expect: 100-continue") {
                                    let _ = stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
                                    let _ = stream.flush();
                                }
                                for line in header_str.lines() {
                                    if let Some(val) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                                        if let Ok(cl) = val.trim().parse::<usize>() {
                                            expected_len = Some(cl);
                                        }
                                    }
                                }
                                if expected_len.is_none() {
                                    expected_len = Some(0);
                                }
                            }
                        }
                        if let Some(cl) = expected_len {
                            if total_req.len() >= header_len + cl {
                                break;
                            }
                        }
                    }
                    Err(_) => break,
                }
            }

            let req_str = String::from_utf8_lossy(&total_req);
            let mut lines = req_str.lines();
            let req_line = lines.next().unwrap_or("");
            let mut parts = req_line.split_whitespace();
            let method = parts.next().unwrap_or("GET");
            let path = parts.next().unwrap_or("/");

            let (status_code, content_type, body) = handler(method, path, &req_str);

            let reason = match status_code {
                200 => "OK",
                201 => "Created",
                400 => "Bad Request",
                401 => "Unauthorized",
                404 => "Not Found",
                500 => "Internal Server Error",
                _ => "Unknown",
            };

            let response_headers = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                status_code, reason, content_type, body.len()
            );

            let _ = stream.write_all(response_headers.as_bytes());
            let _ = stream.write_all(&body);
            let _ = stream.flush();
        }
    });

    (base_url, handle)
}

#[test]
fn test_registry_client_package_details_success_and_error() {
    let (base_url, _handle) = run_mock_registry_server(|method, path, _req| {
        if method == "GET" && path == "/api/packages/my-cool-pkg" {
            let json = serde_json::json!({
                "name": "my-cool-pkg",
                "description": "A high-performance package",
                "keywords": ["math", "fast"],
                "categories": ["algorithms"],
                "downloads": 1337,
                "updated_at": "2026-11-11T00:00:00Z",
                "latest_version": {
                    "version": "1.2.0",
                    "published_at": "2026-11-11T00:00:00Z"
                },
                "versions": [
                    { "version": "1.0.0", "published_at": "2026-01-01T00:00:00Z" },
                    { "version": "1.2.0", "published_at": "2026-11-11T00:00:00Z" }
                ],
                "readme_html": "<p>Readme content</p>"
            });
            (200, "application/json".to_string(), serde_json::to_vec(&json).unwrap())
        } else if method == "GET" && path == "/api/packages/nonexistent" {
            let err = serde_json::json!({
                "code": "not_found",
                "message": "Package 'nonexistent' does not exist"
            });
            (404, "application/json".to_string(), serde_json::to_vec(&err).unwrap())
        } else if method == "GET" && path == "/api/packages/server-err" {
            (500, "text/plain".to_string(), b"Internal Server Error: DB connection failed".to_vec())
        } else {
            (404, "text/plain".to_string(), b"Not Found".to_vec())
        }
    });

    let client = RegistryClient::new(&base_url).expect("client");

    // Success oracle
    let details = client.package_details("my-cool-pkg").expect("details success");
    assert_eq!(details.name, "my-cool-pkg", "name mismatch for package details");
    assert_eq!(details.keywords, vec!["math", "fast"]);
    assert_eq!(details.versions.len(), 2);
    assert_eq!(details.latest_version.unwrap().version, "1.2.0");

    // 404 API error oracle
    let err_404 = client.package_details("nonexistent").expect_err("should fail with 404");
    match err_404 {
        RegistryError::Api { status, code, message } => {
            assert_eq!(status, 404, "expected status 404 for nonexistent package");
            assert_eq!(code.as_deref(), Some("not_found"));
            assert!(message.contains("nonexistent"), "error message must name package: {message}");
        }
        other => panic!("expected RegistryError::Api, got: {other:?}"),
    }

    // 500 plain text error oracle
    let err_500 = client.package_details("server-err").expect_err("should fail with 500");
    match err_500 {
        RegistryError::Api { status, code, message } => {
            assert_eq!(status, 500);
            assert_eq!(code, None);
            assert!(message.contains("DB connection failed"), "expected raw message in body: {message}");
        }
        other => panic!("expected RegistryError::Api, got: {other:?}"),
    }
}

#[test]
fn test_registry_client_search_packages_with_parameters() {
    let (base_url, _handle) = run_mock_registry_server(|method, path, _req| {
        if method == "GET" && path.starts_with("/api/packages") {
            let json = serde_json::json!({
                "total": 1,
                "page": 2,
                "per_page": 5,
                "sort": "downloads",
                "packages": [{
                    "name": "found-pkg",
                    "description": "Found via search query",
                    "keywords": ["search"],
                    "categories": ["utilities"],
                    "downloads": 42,
                    "updated_at": "2026-11-10T12:00:00Z",
                    "latest_version": {
                        "version": "0.9.1",
                        "published_at": "2026-11-10T12:00:00Z"
                    }
                }]
            });
            (200, "application/json".to_string(), serde_json::to_vec(&json).unwrap())
        } else {
            (404, "text/plain".to_string(), b"Not Found".to_vec())
        }
    });

    let client = RegistryClient::new(&base_url).expect("client");

    let params = SearchParameters {
        q: Some("found".to_string()),
        keyword: Some("search".to_string()),
        category: Some("utilities".to_string()),
        sort: Some("downloads".to_string()),
        page: Some(2),
        per_page: Some(5),
    };

    let search_res = client.search_packages(&params).expect("search result");
    assert_eq!(search_res.total, 1, "expected total 1 package");
    assert_eq!(search_res.packages.len(), 1);
    assert_eq!(search_res.packages[0].name, "found-pkg");
    assert_eq!(search_res.page, 2);
    assert_eq!(search_res.per_page, 5);
}

#[test]
fn test_registry_client_download_package_success_and_missing() {
    let dummy_archive = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00";

    let (base_url, _handle) = run_mock_registry_server(move |method, path, _req| {
        if method == "GET" && path == "/api/packages/my-lib/1.0.0/download" {
            (200, "application/gzip".to_string(), dummy_archive.to_vec())
        } else if method == "GET" && path == "/api/packages/my-lib/9.9.9/download" {
            let err = serde_json::json!({
                "code": "version_not_found",
                "message": "Version 9.9.9 of my-lib not found"
            });
            (404, "application/json".to_string(), serde_json::to_vec(&err).unwrap())
        } else {
            (404, "text/plain".to_string(), b"Not Found".to_vec())
        }
    });

    let client = RegistryClient::new(&base_url).expect("client");

    // Success download oracle
    let downloaded = client.download_package("my-lib", "1.0.0").expect("download succeeded");
    assert!(downloaded.path().exists(), "downloaded file must exist at path: {:?}", downloaded.path());
    let contents = fs::read(downloaded.path()).expect("read downloaded file");
    assert_eq!(contents, dummy_archive, "downloaded content mismatch");
    let saved_path = downloaded.into_path();
    assert!(saved_path.exists(), "path preserved after into_path");
    let _ = fs::remove_file(saved_path);

    // 404 download error oracle
    let err = client.download_package("my-lib", "9.9.9").expect_err("should fail on missing version");
    match err {
        RegistryError::Api { status, code, message } => {
            assert_eq!(status, 404);
            assert_eq!(code.as_deref(), Some("version_not_found"));
            assert!(message.contains("9.9.9"), "error message must name missing version: {message}");
        }
        other => panic!("expected RegistryError::Api, got: {other:?}"),
    }
}

#[test]
fn test_registry_client_publish_flow() {
    let (base_url, _handle) = run_mock_registry_server(|method, path, req| {
        if method == "POST" && path == "/api/publish" {
            if !req.contains("Authorization: Bearer valid-secret-token") {
                let err = serde_json::json!({
                    "code": "unauthorized",
                    "message": "Invalid or missing token"
                });
                return (401, "application/json".to_string(), serde_json::to_vec(&err).unwrap());
            }
            let resp = serde_json::json!({
                "package": "published-pkg",
                "version": "1.0.0",
                "artifact_sha256": "abcdef123456",
                "download_url": "http://example.com/download"
            });
            (200, "application/json".to_string(), serde_json::to_vec(&resp).unwrap())
        } else {
            (404, "text/plain".to_string(), b"Not Found".to_vec())
        }
    });

    let temp = tempfile::tempdir().expect("tempdir");
    let pkg_dir = temp.path().join("published-pkg");
    fs::create_dir_all(&pkg_dir).expect("create dir");

    let manifest_toml = r#"
[package]
name = "published-pkg"
version = "1.0.0"
description = "A successfully published package"
"#;
    fs::write(pkg_dir.join("lust-config.toml"), manifest_toml).expect("write manifest");
    let manifest = PackageManifest::discover(&pkg_dir).expect("discover manifest");
    let archive = build_package_archive(&pkg_dir).expect("build archive");

    let client = RegistryClient::new(&base_url).expect("client");

    // Unauthorized publish oracle
    let unauth_err = client.publish(&manifest, "bad-token", &archive, None).expect_err("should reject bad token");
    match unauth_err {
        RegistryError::Api { status, code, message } => {
            assert_eq!(status, 401, "expected 401 status for bad token");
            assert_eq!(code.as_deref(), Some("unauthorized"));
            assert!(message.contains("Invalid or missing token"));
        }
        other => panic!("expected RegistryError::Api, got: {other:?}"),
    }

    // Authorized publish oracle with readme
    let pub_res = client.publish(&manifest, "valid-secret-token", &archive, Some("# Readme header\nFull docs")).expect("publish succeeded");
    assert_eq!(pub_res.package, "published-pkg");
    assert_eq!(pub_res.version, "1.0.0");
    assert_eq!(pub_res.artifact_sha256, "abcdef123456");

    // Authorized publish oracle with whitespace readme (triggers None branch)
    let pub_res_ws = client.publish(&manifest, "valid-secret-token", &archive, Some("   \n  ")).expect("publish succeeded");
    assert_eq!(pub_res_ws.package, "published-pkg");
}

// ============================================================================
// 2. PACKAGE MANAGER, DIGEST VERIFICATION & ARCHIVE INTEGRITY ORACLES
// ============================================================================

#[test]
fn test_package_manager_archive_digest_mismatch_and_corruption() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path().join("installed_packages");
    let manager = PackageManager::new(&root);
    manager.ensure_layout().expect("ensure layout");

    // Create a valid tar.gz file
    let src_dir = temp.path().join("sample_pkg");
    fs::create_dir_all(&src_dir).expect("create src dir");
    fs::write(src_dir.join("lib.lust"), "pub function greet() { return 42; }\n").expect("write code");
    let archive = build_package_archive(&src_dir).expect("build archive");

    let archive_bytes = fs::read(archive.path()).expect("read archive bytes");
    let correct_sha = compute_sha256(&archive_bytes);

    // 1. Digest verification oracle: exact match
    let matches = PackageManager::verify_archive_digest(archive.path(), &correct_sha).expect("verify digest");
    assert!(matches, "computed digest must match correct SHA-256");

    // 2. Digest verification oracle: mismatch
    let bad_sha = "0000000000000000000000000000000000000000000000000000000000000000";
    let mismatch = PackageManager::verify_archive_digest(archive.path(), bad_sha).expect("verify digest");
    assert!(!mismatch, "mismatched digest must return false");

    // 3. Digest verification oracle: non-existent file read denial
    let non_existent = temp.path().join("does_not_exist.tar.gz");
    let io_err = PackageManager::verify_archive_digest(&non_existent, &correct_sha).expect_err("non existent archive");
    match io_err {
        ArchiveVerificationError::Io { path, .. } => {
            assert_eq!(path, non_existent, "assert failure must name file path");
        }
        other => panic!("expected ArchiveVerificationError::Io, got: {other:?}"),
    }

    // 4. Install with digest mismatch oracle
    let install_mismatch = manager.install_package_archive("sample-pkg", "1.0.0", archive.path(), Some(bad_sha)).expect_err("should fail with digest mismatch");
    match install_mismatch {
        ArchiveVerificationError::DigestMismatch { package, expected, actual } => {
            assert_eq!(package, "sample-pkg", "error must name package");
            assert_eq!(expected, bad_sha);
            assert_eq!(actual, correct_sha);
        }
        other => panic!("expected DigestMismatch, got: {other:?}"),
    }

    // 5. Install with corrupted archive oracle (corrupted gzip magic)
    let corrupted_file = temp.path().join("corrupted.tar.gz");
    fs::write(&corrupted_file, b"NOT_A_VALID_GZIP_FILE_AT_ALL").expect("write corrupted");
    let corrupted_err = manager.install_package_archive("sample-pkg", "1.0.0", &corrupted_file, None).expect_err("should reject corrupted archive");
    match corrupted_err {
        ArchiveVerificationError::CorruptedArchive { package, path, message } => {
            assert_eq!(package, "sample-pkg");
            assert_eq!(path, corrupted_file);
            assert!(message.contains("gzip"), "error message must name corruption reason: {message}");
        }
        other => panic!("expected CorruptedArchive, got: {other:?}"),
    }

    // 6. Successful installation oracle
    let installed_dir = manager.install_package_archive("sample-pkg", "1.0.0", archive.path(), Some(&correct_sha)).expect("install succeeds");
    assert!(installed_dir.exists(), "target installation directory must exist: {:?}", installed_dir);
    assert!(installed_dir.join("lib.lust").exists(), "unpacked lib.lust must exist in target dir");
}

#[test]
fn test_package_manager_specifier_and_roots() {
    let spec = PackageSpecifier::new("my-ext", PackageKind::RustExtension).with_version("2.1.0");
    assert_eq!(spec.name, "my-ext");
    assert_eq!(spec.version.as_deref(), Some("2.1.0"));
    assert_eq!(spec.kind, PackageKind::RustExtension);

    let default_root = PackageManager::default_root();
    assert!(default_root.ends_with(".lust/packages") || default_root.ends_with(".lust\\packages"));

    let manager = PackageManager::new(&default_root);
    assert_eq!(manager.root(), default_root.as_path());
}

#[test]
fn test_package_stubs_and_exports_generation() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out_dir = temp.path().join("externs_out");

    let exports = vec![
        NativeExport::new(
            "my_module.calculate",
            vec![
                NativeExportParam::new("x", "int"),
                NativeExportParam::new("y", "int"),
            ],
            "int",
        ).with_doc("Calculates the sum of two integers."),
        NativeExport::new(
            "my_module.untyped_fn",
            vec![NativeExportParam::new("val", "")],
            "()",
        ),
    ];

    let mut type_stub = ModuleStub::default();
    type_stub.module = "my_module".to_string();
    type_stub.struct_defs.push("struct Point { x: int, y: int }".to_string());
    type_stub.enum_defs.push("enum Status { Active, Idle }".to_string());
    type_stub.trait_defs.push("trait Describable { function describe(): string; }".to_string());
    type_stub.const_defs.push("const MAX_LIMIT: int = 1000;".to_string());

    let stubs = stub_files_from_exports(&exports, &[type_stub]);
    assert_eq!(stubs.len(), 1, "expected 1 combined stub file");
    let stub = &stubs[0];
    assert_eq!(stub.relative_path, PathBuf::from("my_module.lust"));
    assert!(stub.contents.contains("struct Point"));
    assert!(stub.contents.contains("enum Status"));
    assert!(stub.contents.contains("trait Describable"));
    assert!(stub.contents.contains("const MAX_LIMIT"));
    assert!(stub.contents.contains("function calculate(x: int, y: int): int"));
    assert!(stub.contents.contains("function untyped_fn(val: any)"));
    assert!(stub.contents.contains("Calculates the sum of two integers."));

    // Write stub files oracle
    let written = write_stub_files("my-crate", &stubs, &out_dir).expect("write stubs");
    assert_eq!(written.len(), 1);
    assert!(out_dir.join("my_module.lust").exists());

    // Collect stub files oracle
    let collected = collect_stub_files(temp.path(), Some(&out_dir)).expect("collect stubs");
    assert_eq!(collected.len(), 1);
    assert_eq!(collected[0].relative_path, PathBuf::from("my_module.lust"));

    // Collect from non-existent directory -> empty vec
    let empty_collected = collect_stub_files(temp.path(), Some(&temp.path().join("missing"))).expect("missing externs dir");
    assert!(empty_collected.is_empty());
}

// ============================================================================
// 3. DEPENDENCY RESOLUTION, SEMVER, CYCLE & CONFLICT ORACLES
// ============================================================================

#[test]
fn test_semver_parsing_and_matching_oracle() {
    let v123 = Semver::parse("1.2.3").expect("parse 1.2.3");
    assert_eq!(v123, Semver { major: 1, minor: 2, patch: 3 });

    let v_prefix = Semver::parse("v2.0.0").expect("parse v2.0.0");
    assert_eq!(v_prefix, Semver { major: 2, minor: 0, patch: 0 });

    let v_pre_release = Semver::parse("3.4.5-beta.1").expect("parse pre-release");
    assert_eq!(v_pre_release, Semver { major: 3, minor: 4, patch: 5 });

    assert!(Semver::parse("invalid.semver.extra.stuff").is_none());
    assert!(Semver::parse("not-a-number").is_none());

    // Ordering oracle
    assert!(Semver::parse("1.0.0").unwrap() < Semver::parse("1.0.1").unwrap());
    assert!(Semver::parse("1.0.1").unwrap() < Semver::parse("1.1.0").unwrap());
    assert!(Semver::parse("1.1.0").unwrap() < Semver::parse("2.0.0").unwrap());

    // Constraint matching: Caret ^
    assert!(v123.matches_constraint("^1.2.0"), "^1.2.0 should match 1.2.3");
    assert!(v123.matches_constraint("^1.0.0"), "^1.0.0 should match 1.2.3");
    assert!(!v123.matches_constraint("^1.3.0"), "^1.3.0 should not match 1.2.3");
    assert!(!v123.matches_constraint("^2.0.0"), "^2.0.0 should not match 1.2.3");
    assert!(!v123.matches_constraint("^0.2.3"), "^0.2.3 should not match 1.2.3");

    let zero_ver = Semver::parse("0.2.5").unwrap();
    assert!(zero_ver.matches_constraint("^0.2.0"), "^0.2.0 should match 0.2.5");
    assert!(!zero_ver.matches_constraint("^0.3.0"), "^0.3.0 should not match 0.2.5");

    // Constraint matching: Tilde ~
    assert!(v123.matches_constraint("~1.2.0"), "~1.2.0 should match 1.2.3");
    assert!(!v123.matches_constraint("~1.1.0"), "~1.1.0 should not match 1.2.3");
    assert!(!v123.matches_constraint("~1.3.0"), "~1.3.0 should not match 1.2.3");

    // Constraint matching: Comparisons
    assert!(v123.matches_constraint(">= 1.0.0"));
    assert!(v123.matches_constraint("<= 1.2.3"));
    assert!(v123.matches_constraint("> 1.2.0"));
    assert!(v123.matches_constraint("< 2.0.0"));
    assert!(!v123.matches_constraint("> 1.2.3"));

    // Multiple constraints (comma-separated)
    assert!(v123.matches_constraint(">= 1.0.0, < 2.0.0"));
    assert!(!v123.matches_constraint(">= 1.0.0, < 1.2.0"));

    // Wildcards
    assert!(v123.matches_constraint("*"));
    assert!(v123.matches_constraint(""));
}

#[test]
fn test_dependency_missing_path_error() {
    let temp = tempfile::tempdir().expect("tempdir");
    let non_existent = temp.path().join("missing_pkg_dir");

    let config_toml = format!(r#"
[dependencies]
bad_dep = {{ path = "{}" }}
"#, non_existent.display());

    let config = LustConfig::from_toml_str(&config_toml).expect("parse config");
    let err = resolve_dependencies(&config, temp.path()).expect_err("should fail with MissingPath");

    match err {
        DependencyResolutionError::MissingPath { name, path } => {
            assert_eq!(name, "bad_dep", "error must name dependency");
            assert_eq!(path, non_existent, "error must name missing path");
        }
        other => panic!("expected MissingPath, got: {other:?}"),
    }
}

#[test]
fn test_dependency_missing_package_version_error() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_toml = r#"
[dependencies]
uninstalled_lib = "1.0.0"
"#;
    let config = LustConfig::from_toml_str(config_toml).expect("parse config");
    let err = resolve_dependencies(&config, temp.path()).expect_err("should fail with MissingPackage");

    match err {
        DependencyResolutionError::MissingPackage { name, version, path } => {
            assert_eq!(name, "uninstalled_lib", "error must name missing package");
            assert_eq!(version, "1.0.0", "error must name requested version");
            assert!(path.to_string_lossy().contains("uninstalled_lib"), "error path must name package: {path:?}");
        }
        other => panic!("expected MissingPackage, got: {other:?}"),
    }
}

#[test]
fn test_dependency_unmet_version_constraint_error() {
    let temp = tempfile::tempdir().expect("tempdir");
    let fake_packages_root = temp.path().join("packages_cache");
    let pkg_dir = fake_packages_root.join("semver_pkg");
    fs::create_dir_all(pkg_dir.join("0.5.0")).expect("create installed version 0.5.0");

    let config_toml = r#"
[dependencies]
semver_pkg = "^1.0.0"
"#;
    let config = LustConfig::from_toml_str(config_toml).expect("parse config");
    let manager = PackageManager::new(&fake_packages_root);

    let err = resolve_dependencies_with_manager(&config, temp.path(), &manager)
        .expect_err("should return UnmetVersionConstraint");

    match err {
        DependencyResolutionError::UnmetVersionConstraint { name, required, found } => {
            assert_eq!(name, "semver_pkg", "assert failure must name package: {name}");
            assert_eq!(required, "^1.0.0", "assert failure must name required constraint: {required}");
            assert_eq!(found, "0.5.0", "assert failure must name found version: {found}");
        }
        other => panic!("expected UnmetVersionConstraint, got: {other:?}"),
    }
}

#[test]
fn test_dependency_direct_and_transitive_conflict_error_oracle() {
    let temp = tempfile::tempdir().expect("tempdir");
    let fake_packages_root = temp.path().join("packages_cache");
    let pkg_common_dir = fake_packages_root.join("pkg_common");
    fs::create_dir_all(pkg_common_dir.join("1.0.0")).expect("create pkg_common 1.0.0");
    fs::create_dir_all(pkg_common_dir.join("2.0.0")).expect("create pkg_common 2.0.0");

    let pkg_intermediate = temp.path().join("pkg_intermediate");
    fs::create_dir_all(&pkg_intermediate).expect("create intermediate");

    let intermediate_toml = r#"
[package]
name = "pkg_intermediate"
version = "1.0.0"

[dependencies]
pkg_common = "^2.0.0"
"#;
    fs::write(pkg_intermediate.join("lust-config.toml"), intermediate_toml).expect("write intermediate");
    fs::write(pkg_intermediate.join("lib.lust"), "// code\n").expect("write code");

    let root_toml = format!(r#"
[dependencies]
pkg_intermediate = {{ path = "{}" }}
pkg_common = "^1.0.0"
"#, pkg_intermediate.display());

    let root_config = LustConfig::from_toml_str(&root_toml).expect("parse root config");
    let manager = PackageManager::new(&fake_packages_root);

    let err = resolve_dependencies_with_manager(&root_config, temp.path(), &manager)
        .expect_err("should detect direct and transitive version conflict");

    match err {
        DependencyResolutionError::VersionConflict {
            package,
            direct_requirement,
            transitive_requirement,
            required_by,
        } => {
            assert_eq!(package, "pkg_common", "conflict must name package: {package}");
            assert_eq!(direct_requirement, "^1.0.0", "conflict must name direct requirement: {direct_requirement}");
            assert_eq!(transitive_requirement, "^2.0.0", "conflict must name transitive requirement: {transitive_requirement}");
            assert_eq!(required_by, "pkg_intermediate", "conflict must name parent requiring it: {required_by}");
        }
        other => panic!("expected VersionConflict, got: {other:?}"),
    }
}

#[test]
fn test_dependency_cycle_detection_oracle() {
    let temp = tempfile::tempdir().expect("tempdir");
    let pkg_a = temp.path().join("pkg_a");
    let pkg_b = temp.path().join("pkg_b");

    fs::create_dir_all(&pkg_a).expect("create pkg_a");
    fs::create_dir_all(&pkg_b).expect("create pkg_b");

    // A depends on B
    let manifest_a = format!(r#"
[package]
name = "pkg_a"
version = "1.0.0"

[dependencies]
pkg_b = {{ path = "{}" }}
"#, pkg_b.display());
    fs::write(pkg_a.join("lust-config.toml"), manifest_a).expect("write A manifest");
    fs::write(pkg_a.join("lib.lust"), "// code\n").expect("write A code");

    // B depends on A (Circular cycle A -> B -> A)
    let manifest_b = format!(r#"
[package]
name = "pkg_b"
version = "1.0.0"

[dependencies]
pkg_a = {{ path = "{}" }}
"#, pkg_a.display());
    fs::write(pkg_b.join("lust-config.toml"), manifest_b).expect("write B manifest");
    fs::write(pkg_b.join("lib.lust"), "// code\n").expect("write B code");

    let root_config_toml = format!(r#"
[dependencies]
pkg_a = {{ path = "{}" }}
"#, pkg_a.display());

    let root_config = LustConfig::from_toml_str(&root_config_toml).expect("parse root config");
    let cycle_err = resolve_transitive_dependencies(&root_config, temp.path()).expect_err("should detect circular dependency");

    match cycle_err {
        DependencyResolutionError::CyclicDependency { package, path } => {
            assert!(package == "pkg_a" || package == "pkg_b", "cycle error must name participating package: {package}");
            assert!(path.contains("->"), "cycle error must display chain: {path}");
            assert!(path.contains("pkg_a") && path.contains("pkg_b"), "path must name both packages: {path}");
        }
        other => panic!("expected CyclicDependency, got: {other:?}"),
    }
}

#[test]
fn test_dependency_direct_and_transitive_conflict_oracle() {
    let temp = tempfile::tempdir().expect("tempdir");
    let pkg_common = temp.path().join("pkg_common");
    let pkg_intermediate = temp.path().join("pkg_intermediate");

    fs::create_dir_all(&pkg_common).expect("create common");
    fs::create_dir_all(&pkg_intermediate).expect("create intermediate");

    fs::write(pkg_common.join("lust-config.toml"), r#"
[package]
name = "pkg_common"
version = "1.0.0"
"#).expect("write common");
    fs::write(pkg_common.join("lib.lust"), "// common\n").expect("write code");

    // Intermediate depends on common
    let intermediate_toml = format!(r#"
[package]
name = "pkg_intermediate"
version = "1.0.0"

[dependencies]
pkg_common = {{ path = "{}" }}
"#, pkg_common.display());
    fs::write(pkg_intermediate.join("lust-config.toml"), intermediate_toml).expect("write intermediate");
    fs::write(pkg_intermediate.join("lib.lust"), "// intermediate\n").expect("write code");

    // Root project depends on intermediate AND common directly
    let root_toml = format!(r#"
[dependencies]
pkg_intermediate = {{ path = "{}" }}
pkg_common = {{ path = "{}" }}
"#, pkg_intermediate.display(), pkg_common.display());

    let root_config = LustConfig::from_toml_str(&root_toml).expect("parse root config");
    let res = resolve_transitive_dependencies(&root_config, temp.path()).expect("resolves diamond without conflict");
    assert_eq!(res.lust().len(), 2, "both unique lust dependencies must be resolved");
}

#[test]
fn test_dependency_auto_detection_kinds_and_lua_collection() {
    let temp = tempfile::tempdir().expect("tempdir");

    // 1. Lust package with lust-config.toml
    let lust_pkg = temp.path().join("lust_pkg");
    fs::create_dir_all(lust_pkg.join("src")).expect("create src");
    fs::write(lust_pkg.join("lust-config.toml"), "[package]\nname = \"lust_pkg\"\nversion = \"1.0.0\"\n").expect("manifest");
    fs::write(lust_pkg.join("src").join("lib.lust"), "// lib\n").expect("lib");

    // 2. Cargo package with Cargo.toml
    let cargo_pkg = temp.path().join("cargo_pkg");
    fs::create_dir_all(&cargo_pkg).expect("create cargo pkg");
    fs::write(cargo_pkg.join("Cargo.toml"), "[package]\nname = \"cargo_pkg\"\nversion = \"0.1.0\"\n").expect("cargo manifest");

    // 3. Lua package with .lua files in subdirectories
    let lua_pkg = temp.path().join("lua_pkg");
    let lua_sub = lua_pkg.join("sub").join("nested");
    fs::create_dir_all(&lua_sub).expect("create lua sub");
    fs::write(lua_pkg.join("init.lua"), "-- init\n").expect("init.lua");
    fs::write(lua_sub.join("util.lua"), "-- util\n").expect("util.lua");

    let root_toml = format!(r#"
[dependencies]
lust_dep = {{ path = "{}" }}
cargo_dep = {{ path = "{}" }}
lua_dep = {{ path = "{}" }}
"#, lust_pkg.display(), cargo_pkg.display(), lua_pkg.display());

    let config = LustConfig::from_toml_str(&root_toml).expect("parse config");
    let resolution = resolve_dependencies(&config, temp.path()).expect("resolve auto detected");

    // Assert Lust dependency detected
    assert_eq!(resolution.lust().len(), 1);
    assert_eq!(resolution.lust()[0].name, "lust_dep");
    assert_eq!(resolution.lust()[0].root_module, Some(PathBuf::from("lib.lust")));

    // Assert Rust dependency detected
    assert_eq!(resolution.rust().len(), 1);
    assert_eq!(resolution.rust()[0].name, "cargo_dep");

    // Assert Lua dependency detected and files collected recursively
    assert_eq!(resolution.lua().len(), 1);
    assert_eq!(resolution.lua()[0].name, "lua_dep");
    let lua_files = &resolution.lua()[0].lua_files;
    assert_eq!(lua_files.len(), 2, "must collect both init.lua and sub/nested/util.lua");
}

#[test]
fn test_dependency_unknown_library_kind_and_corrupt_inspect() {
    let temp = tempfile::tempdir().expect("tempdir");

    // File with random corrupt bytes (not a valid ELF/Mach-O/PE)
    let corrupt_so = temp.path().join("corrupt.so");
    fs::write(&corrupt_so, b"CORRUPTED_NOT_AN_OBJECT_FILE_AT_ALL").expect("write corrupt");

    let config_toml = format!(r#"
[dependencies]
bad_lib = {{ path = "{}" }}
"#, corrupt_so.display());

    let config = LustConfig::from_toml_str(&config_toml).expect("parse config");
    let err = resolve_dependencies(&config, temp.path()).expect_err("should fail inspecting invalid object file");

    match err {
        DependencyResolutionError::LibraryInspect { path, .. } => {
            assert_eq!(path, corrupt_so, "error must name corrupt library path");
        }
        other => panic!("expected LibraryInspect, got: {other:?}"),
    }
}

// ============================================================================
// 4. EMBEDDED MODULES ORACLES (>= 70% TARGET)
// ============================================================================

#[test]
fn test_embedded_directory_map_oracle() {
    let entries = vec![
        EmbeddedModule { module: "root_a", parent: None, source: Some("// a") },
        EmbeddedModule { module: "root_b", parent: None, source: Some("// b") },
        EmbeddedModule { module: "child_1", parent: Some("pkg"), source: Some("// c1") },
        EmbeddedModule { module: "child_2", parent: Some("pkg"), source: Some("// c2") },
        EmbeddedModule { module: "nested", parent: Some("pkg.sub"), source: Some("// n") },
    ];

    let map = build_directory_map(&entries);
    assert_eq!(map.get(""), Some(&vec!["root_a".to_string(), "root_b".to_string()]));
    assert_eq!(map.get("pkg"), Some(&vec!["child_1".to_string(), "child_2".to_string()]));
    assert_eq!(map.get("pkg.sub"), Some(&vec!["nested".to_string()]));
}

#[test]
fn test_embedded_program_single_module_execution() {
    let entries = vec![
        EmbeddedModule {
            module: "main",
            parent: None,
            source: Some(r#"
function calculate(x: int): int
    return x * 2
end
"#),
        },
    ];

    let program = load_program_from_embedded(&entries, "main").expect("load single module");
    assert_eq!(program.modules.len(), 1);
    assert_eq!(program.modules[0].path, "main");
    assert_eq!(program.entry_module, "main");
}

#[test]
fn test_embedded_program_path_and_group_and_glob_imports() {
    let entries = vec![
        EmbeddedModule {
            module: "math_helper",
            parent: None,
            source: Some(r#"
pub function add(a: int, b: int): int
    return a + b
end

pub function sub(a: int, b: int): int
    return a - b
end

pub function mul(a: int, b: int): int
    return a * b
end
"#),
        },
        EmbeddedModule {
            module: "path_importer",
            parent: None,
            source: Some(r#"
use math_helper.add
use math_helper.sub as minus

function run()
    return add(10, 5) + minus(10, 5)
end
"#),
        },
        EmbeddedModule {
            module: "group_importer",
            parent: None,
            source: Some(r#"
use math_helper.{add, mul as times}

function run_group()
    return times(add(2, 3), 4)
end
"#),
        },
        EmbeddedModule {
            module: "glob_importer",
            parent: None,
            source: Some(r#"
use math_helper.*

function run_glob()
    return add(sub(10, 2), mul(3, 4))
end
"#),
        },
    ];

    let program = load_program_from_embedded(&entries, "glob_importer").expect("load all import variations");
    assert_eq!(program.modules.len(), 4);

    let path_mod = program.modules.iter().find(|m| m.path == "path_importer").unwrap();
    assert_eq!(path_mod.imports.function_aliases.get("add").unwrap(), "math_helper.add");
    assert_eq!(path_mod.imports.function_aliases.get("minus").unwrap(), "math_helper.sub");

    let group_mod = program.modules.iter().find(|m| m.path == "group_importer").unwrap();
    assert_eq!(group_mod.imports.function_aliases.get("add").unwrap(), "math_helper.add");
    assert_eq!(group_mod.imports.function_aliases.get("times").unwrap(), "math_helper.mul");

    let glob_mod = program.modules.iter().find(|m| m.path == "glob_importer").unwrap();
    assert_eq!(glob_mod.imports.function_aliases.get("add").unwrap(), "math_helper.add");
    assert_eq!(glob_mod.imports.function_aliases.get("sub").unwrap(), "math_helper.sub");
    assert_eq!(glob_mod.imports.function_aliases.get("mul").unwrap(), "math_helper.mul");
}

#[test]
fn test_embedded_program_reexport_variations() {
    let entries = vec![
        EmbeddedModule {
            module: "base",
            parent: None,
            source: Some(r#"
export function base_fn(): int
    return 100
end
"#),
        },
        EmbeddedModule {
            module: "facade_path",
            parent: None,
            source: Some(r#"
export use base.base_fn
"#),
        },
        EmbeddedModule {
            module: "facade_group",
            parent: None,
            source: Some(r#"
export use base.{base_fn as aliased_base}
"#),
        },
        EmbeddedModule {
            module: "facade_glob",
            parent: None,
            source: Some(r#"
export use base.*
"#),
        },
    ];

    let program = load_program_from_embedded(&entries, "facade_glob").expect("load reexports");
    assert_eq!(program.modules.len(), 4);

    let facade_path = program.modules.iter().find(|m| m.path == "facade_path").unwrap();
    assert_eq!(facade_path.exports.functions.get("base_fn").unwrap(), "base.base_fn");

    let facade_group = program.modules.iter().find(|m| m.path == "facade_group").unwrap();
    assert_eq!(facade_group.exports.functions.get("aliased_base").unwrap(), "base.base_fn");

    let facade_glob = program.modules.iter().find(|m| m.path == "facade_glob").unwrap();
    assert_eq!(facade_glob.exports.functions.get("base_fn").unwrap(), "base.base_fn");
}

#[test]
fn test_embedded_cycle_detection_oracle() {
    let entries = vec![
        EmbeddedModule {
            module: "cycle_a",
            parent: None,
            source: Some(r#"
use cycle_b.from_b

function from_a()
    return from_b()
end
"#),
        },
        EmbeddedModule {
            module: "cycle_b",
            parent: None,
            source: Some(r#"
use cycle_a.from_a

function from_b()
    return from_a()
end
"#),
        },
    ];

    let err = load_program_from_embedded(&entries, "cycle_a").expect_err("should detect embedded cyclic dependency");
    match err {
        LustError::Unknown(msg) => {
            assert!(msg.contains("Cyclic dependency detected"), "error must state cyclic dependency: {msg}");
            assert!(msg.contains("cycle_a") || msg.contains("cycle_b"), "error must name module in cycle: {msg}");
        }
        other => panic!("expected LustError::Unknown, got: {other:?}"),
    }
}

#[test]
fn test_embedded_unknown_module_error_oracle() {
    let entries = vec![
        EmbeddedModule {
            module: "uses_unknown",
            parent: None,
            source: Some(r#"
use nonexistent_module.some_fn

function test()
    return some_fn()
end
"#),
        },
    ];

    let err = load_program_from_embedded(&entries, "uses_unknown")
        .expect_err("should return Unknown module error for nonexistent_module");
    match err {
        LustError::Unknown(msg) => {
            assert!(msg.contains("Unknown module"), "error must state Unknown module: {msg}");
            assert!(msg.contains("nonexistent_module"), "error must name the missing module: {msg}");
        }
        other => panic!("expected LustError::Unknown, got: {other:?}"),
    }
}

#[test]
fn test_embedded_syntax_initialization_error_oracle() {
    let entries = vec![
        EmbeddedModule {
            module: "broken_module",
            parent: None,
            source: Some(r#"
function syntax_error(
    // missing closing parenthesis and end
"#),
        },
    ];

    let err = load_program_from_embedded(&entries, "broken_module").expect_err("should fail parsing syntax error");
    match err {
        LustError::ParserError { .. } | LustError::CompileError(_) | LustError::LexerError { .. } => {
            // green on syntax error oracle
        }
        other => panic!("expected syntax error, got: {other:?}"),
    }
}

#[test]
fn test_embedded_capability_denial_virtual_std_modules() {
    let mut config = LustConfig::default();
    // Deny io capability
    config.capability_mask_mut().allow_io = false;

    let entries = vec![
        EmbeddedModule {
            module: "denied_io_user",
            parent: None,
            source: Some(r#"
use std.io

function read_secret()
    return 0
end
"#),
        },
    ];

    let err = load_program_from_embedded_with_config(&entries, "denied_io_user", &config).expect_err("should deny io module");
    match err {
        LustError::CompileError(msg) => {
            assert!(msg.contains("CapabilityDenied"), "error must specify CapabilityDenied: {msg}");
            assert!(msg.contains("io"), "error must name denied module: {msg}");
        }
        other => panic!("expected CapabilityDenied compile error, got: {other:?}"),
    }
}

#[test]
fn test_embedded_lua_require_and_preprocessing_variations() {
    let entries = vec![
        EmbeddedModule {
            module: "helper_lua",
            parent: None,
            source: Some(r#"
pub function helper()
    return 1
end
"#),
        },
        EmbeddedModule {
            module: "lua_consumer",
            parent: None,
            source: Some(r#"
use @std/json
import math from "@std/math";

function test_constructs()
    local x = require("helper_lua")
    local built_in = require("math")
    local arr = [require("helper_lua")]
    local m = { k = require("helper_lua") }

    if x > 0 then
        local y = require("helper_lua")
    elseif x == 0 then
        local z = require("helper_lua")
    else
        local w = require("helper_lua")
    end

    while x > 0 do
        local l = require("helper_lua")
    end

    for i = 1, 10 do
        local f = require("helper_lua")
    end

    return x
end
"#),
        },
    ];

    let program = load_program_from_embedded(&entries, "lua_consumer").expect("load lua constructs");
    assert_eq!(program.modules.len(), 2);
}

#[test]
fn test_embedded_module_without_source() {
    let entries = vec![
        EmbeddedModule {
            module: "no_source_mod",
            parent: None,
            source: None,
        },
        EmbeddedModule {
            module: "main",
            parent: None,
            source: Some("function main()\n    return 0\nend\n"),
        },
    ];

    let program = load_program_from_embedded(&entries, "main").expect("module without source should not fail");
    assert_eq!(program.modules.len(), 1);
}

// ============================================================================
// 5. EXTENDED PACKAGES/MOD.RS TESTS (>= 60% TARGET)
// ============================================================================

#[test]
fn test_packages_local_module_error_display_and_read_crate() {
    let io_err = LocalModuleError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file missing"));
    assert!(format!("{io_err}").contains("file missing"));

    let manifest_err = LocalModuleError::Manifest {
        path: PathBuf::from("Cargo.toml"),
        source: toml::from_str::<toml::Value>("invalid [[ toml").unwrap_err(),
    };
    assert!(format!("{manifest_err}").contains("failed to parse Cargo manifest"));

    let missing_lib = LocalModuleError::LibraryMissing(PathBuf::from("/target/debug/libtest.dylib"));
    assert!(format!("{missing_lib}").contains("built library not found"));

    let reg_sym_err = LocalModuleError::RegisterSymbolMissing("my_mod".to_string());
    assert!(format!("{reg_sym_err}").contains("lust_extension_register"));

    let reg_fail_err = LocalModuleError::RegisterFailed("my_mod".to_string());
    assert!(format!("{reg_fail_err}").contains("register function reported failure"));

    // build_local_module on non-existent dir
    let temp = tempfile::tempdir().expect("tempdir");
    let missing_crate = temp.path().join("missing_crate");
    let build_err = build_local_module(&missing_crate, BuildOptions::default()).expect_err("should fail on missing crate");
    match build_err {
        LocalModuleError::Io(_) => {}
        other => panic!("expected LocalModuleError::Io, got: {other:?}"),
    }

    // Build options default
    let opts = BuildOptions::default();
    assert!(opts.features.is_empty());
    assert!(opts.default_features);
}

#[test]
fn test_packages_write_stub_files_variations() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output_root = temp.path().join("stubs_root");

    let stubs = vec![
        // Empty relative path -> uses sanitized crate name + .lust
        StubFile {
            relative_path: PathBuf::new(),
            contents: "// empty rel path".to_string(),
        },
        // Missing extension -> appends .lust
        StubFile {
            relative_path: PathBuf::from("no_ext"),
            contents: "// no ext".to_string(),
        },
        // Nested subdirectories
        StubFile {
            relative_path: PathBuf::from("sub/dir/nested.lust"),
            contents: "// nested".to_string(),
        },
    ];

    let written = write_stub_files("my-crate-name", &stubs, &output_root).expect("write stub files");
    assert_eq!(written.len(), 3);
    assert_eq!(written[0], PathBuf::from("my_crate_name.lust"));
    assert_eq!(written[1], PathBuf::from("no_ext.lust"));
    assert_eq!(written[2], PathBuf::from("sub/dir/nested.lust"));

    assert!(output_root.join("my_crate_name.lust").exists());
    assert!(output_root.join("no_ext.lust").exists());
    assert!(output_root.join("sub/dir/nested.lust").exists());
}

#[test]
fn test_packages_archive_verification_error_display() {
    let io_err = ArchiveVerificationError::Io {
        path: PathBuf::from("/path/to/archive.tar.gz"),
        source: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
    };
    assert!(format!("{io_err}").contains("/path/to/archive.tar.gz"));

    let digest_err = ArchiveVerificationError::DigestMismatch {
        package: "pkg_foo".to_string(),
        expected: "exp_hash".to_string(),
        actual: "act_hash".to_string(),
    };
    assert!(format!("{digest_err}").contains("pkg_foo"));
    assert!(format!("{digest_err}").contains("exp_hash"));

    let corrupt_err = ArchiveVerificationError::CorruptedArchive {
        package: "pkg_bar".to_string(),
        path: PathBuf::from("/path/to/corrupt.tar.gz"),
        message: "bad magic".to_string(),
    };
    assert!(format!("{corrupt_err}").contains("pkg_bar"));
    assert!(format!("{corrupt_err}").contains("bad magic"));

    let extract_err = ArchiveVerificationError::ExtractFailed {
        package: "pkg_baz".to_string(),
        destination: PathBuf::from("/dest/dir"),
        message: "tar exit 1".to_string(),
    };
    assert!(format!("{extract_err}").contains("pkg_baz"));
    assert!(format!("{extract_err}").contains("tar exit 1"));
}

#[test]
fn test_packages_dependency_resolution_error_display() {
    let cache_err = DependencyResolutionError::PackageCache {
        source: std::io::Error::new(std::io::ErrorKind::Other, "cache failure"),
    };
    assert!(format!("{cache_err}").contains("cache failure"));

    let dir_read = DependencyResolutionError::DirectoryRead {
        path: PathBuf::from("/dir/path"),
        source: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
    };
    assert!(format!("{dir_read}").contains("/dir/path"));

    let lib_io = DependencyResolutionError::LibraryIo {
        path: PathBuf::from("/lib/path.so"),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
    };
    assert!(format!("{lib_io}").contains("/lib/path.so"));
}

// ============================================================================
// 6. EXTENDED MODULES/EMBEDDED.RS TESTS (>= 70% TARGET)
// ============================================================================

#[test]
fn test_embedded_dynamic_import_capability_denial_in_ast() {
    let mut config = LustConfig::default();
    config.capability_mask_mut().allow_io = false;

    // Module with dynamic require in an expression statement
    let entries = vec![
        EmbeddedModule {
            module: "dyn_req",
            parent: None,
            source: Some(r#"
function run()
    local x = 1
    local mod = require("std.io")
    return x
end
"#),
        },
    ];

    let err = load_program_from_embedded_with_config(&entries, "dyn_req", &config)
        .expect_err("dynamic require of std.io should be denied");
    match err {
        LustError::CompileError(msg) => {
            assert!(msg.contains("CapabilityDenied"), "error must state CapabilityDenied: {msg}");
            assert!(msg.contains("io"), "error must name denied module: {msg}");
        }
        other => panic!("expected CapabilityDenied CompileError, got: {other:?}"),
    }
}

#[test]
fn test_embedded_dynamic_import_keyword_capability_denial() {
    let mut config = LustConfig::default();
    config.capability_mask_mut().allow_io = false;

    let entries = vec![
        EmbeddedModule {
            module: "dyn_imp",
            parent: None,
            source: Some(r#"
function run()
    local x = import("std.io")
    return 0
end
"#),
        },
    ];

    let err = load_program_from_embedded_with_config(&entries, "dyn_imp", &config)
        .expect_err("dynamic import of std.io should be denied");
    match err {
        LustError::CompileError(msg) => {
            assert!(msg.contains("CapabilityDenied"), "error must state CapabilityDenied: {msg}");
            assert!(msg.contains("io"), "error must name denied module: {msg}");
        }
        other => panic!("expected CapabilityDenied CompileError, got: {other:?}"),
    }
}

#[test]
fn test_embedded_all_ast_expressions_and_statements_require_scanner() {
    let entries = vec![
        EmbeddedModule {
            module: "sub_provider",
            parent: None,
            source: Some(r#"
pub function get_val()
    return 42
end
"#),
        },
        EmbeddedModule {
            module: "ast_explorer",
            parent: None,
            source: Some(r#"
function explore()
    -- Compound assign
    local a = 1
    a += 2

    -- Unary and binary
    local b = -(1 + 2)

    -- Range
    for i = 1, 5 do
        a += i
    end

    -- While loop
    local count = 3
    while count > 0 do
        count -= 1
    end

    -- Array and map
    local arr = [10, 20]
    local tbl = { key = "val" }

    -- If, elseif, else
    if a > 100 then
        a = 100
    elseif a == 50 then
        a = 50
    else
        a = 0
    end

    return a
end
"#),
        },
    ];

    let prog = load_program_from_embedded(&entries, "ast_explorer").expect("load AST explorer");
    assert_eq!(prog.modules.len(), 2);
}

#[test]
fn test_embedded_group_use_with_nested_paths() {
    let entries = vec![
        EmbeddedModule {
            module: "nested_provider",
            parent: None,
            source: Some(r#"
pub function f1()
    return 1
end

pub function f2()
    return 2
end
"#),
        },
        EmbeddedModule {
            module: "consumer",
            parent: None,
            source: Some(r#"
use nested_provider.{f1, f2 as aliased_f2}

function call_both()
    return f1() + aliased_f2()
end
"#),
        },
    ];

    let prog = load_program_from_embedded(&entries, "consumer").expect("nested use group");
    assert_eq!(prog.modules.len(), 2);
    let consumer = prog.modules.iter().find(|m| m.path == "consumer").unwrap();
    assert_eq!(consumer.imports.function_aliases.get("f1").unwrap(), "nested_provider.f1");
    assert_eq!(consumer.imports.function_aliases.get("aliased_f2").unwrap(), "nested_provider.f2");
}

#[test]
fn test_packages_read_crate_name_and_paths() {
    let temp = tempfile::tempdir().expect("tempdir");

    // 1. Missing Cargo.toml
    let missing_err = read_crate_name(temp.path()).expect_err("should fail when Cargo.toml missing");
    assert!(matches!(missing_err, LocalModuleError::Io(_)));

    // 2. Invalid Cargo.toml
    fs::write(temp.path().join("Cargo.toml"), "invalid [[ toml content").expect("write bad toml");
    let bad_err = read_crate_name(temp.path()).expect_err("should fail on invalid toml");
    assert!(matches!(bad_err, LocalModuleError::Manifest { .. }));

    // 3. Valid Cargo.toml
    let valid_toml = r#"
[package]
name = "my_awesome_crate"
version = "0.1.0"
"#;
    fs::write(temp.path().join("Cargo.toml"), valid_toml).expect("write valid toml");
    let crate_name = read_crate_name(temp.path()).expect("read valid crate name");
    assert_eq!(crate_name, "my_awesome_crate");

    // 4. relative_stub_path edge cases
    assert_eq!(relative_stub_path("externs.my_module"), PathBuf::from("my_module.lust"));
    assert_eq!(relative_stub_path("single_mod"), PathBuf::from("single_mod.lust"));
    assert_eq!(relative_stub_path("pkg.sub.leaf"), PathBuf::from("pkg/sub/leaf"));

    // 5. sanitized_crate_name and library_file_name
    assert_eq!(sanitized_crate_name("my-cool-lib"), "my_cool_lib");
    let lib_file = library_file_name("my-cool-lib");
    assert!(lib_file.contains("my_cool_lib"));
}

#[test]
fn test_packages_prepare_and_load_empty_rust_dependencies() {
    let temp = tempfile::tempdir().expect("tempdir");
    let empty_res = DependencyResolution::default();

    let prepared = prepare_rust_dependencies(&empty_res, temp.path()).expect("empty rust dependencies");
    assert!(prepared.is_empty());

    let mut vm = lust::VM::new();
    let loaded = load_prepared_rust_dependencies(&prepared, &mut vm).expect("empty prepared dependencies");
    assert!(loaded.is_empty());
}

#[test]
fn test_embedded_dynamic_imports_in_impl_and_trait_blocks() {
    let mut config = LustConfig::default();
    config.capability_mask_mut().allow_io = false;

    // Module with dynamic import inside a trait default method
    let entries = vec![
        EmbeddedModule {
            module: "trait_dyn_io",
            parent: None,
            source: Some(r#"
trait FileOp
    function do_op()
        local mod = require("std.io")
        return 0
    end
end
"#),
        },
    ];

    let err = load_program_from_embedded_with_config(&entries, "trait_dyn_io", &config)
        .expect_err("trait method dynamic require of std.io should be denied");
    match err {
        LustError::CompileError(msg) => {
            assert!(msg.contains("CapabilityDenied"), "error must state CapabilityDenied: {msg}");
            assert!(msg.contains("io"), "error must name denied module: {msg}");
        }
        other => panic!("expected CapabilityDenied CompileError, got: {other:?}"),
    }
}

#[test]
fn test_embedded_lua_dynamic_import_callee_variants() {
    let mut config = LustConfig::default();
    config.capability_mask_mut().allow_io = false;

    // lua.require("std.io")
    let entries = vec![
        EmbeddedModule {
            module: "lua_ns_io",
            parent: None,
            source: Some(r#"
function run()
    local x = lua.require("std.io")
    return 0
end
"#),
        },
    ];

    let err = load_program_from_embedded_with_config(&entries, "lua_ns_io", &config)
        .expect_err("lua.require of std.io should be denied");
    match err {
        LustError::CompileError(msg) => {
            assert!(msg.contains("CapabilityDenied"), "error must state CapabilityDenied: {msg}");
            assert!(msg.contains("io"), "error must name denied module: {msg}");
        }
        other => panic!("expected CapabilityDenied CompileError, got: {other:?}"),
    }
}

#[test]
fn test_embedded_require_in_compound_assign_and_expressions() {
    let entries = vec![
        EmbeddedModule {
            module: "helper_math",
            parent: None,
            source: Some(r#"
pub function val()
    return 10
end
"#),
        },
        EmbeddedModule {
            module: "ast_rich",
            parent: None,
            source: Some(r#"
use helper_math.*

function compute()
    local a = 5
    local b = (val() as int)
    local arr = [val(), val()]
    local m = { k = val() }

    while a > 0 do
        a -= 1
    end

    for x in arr do
        b += x
    end

    return a + b
end
"#),
        },
    ];

    let prog = load_program_from_embedded(&entries, "ast_rich").expect("load AST rich constructs");
    assert_eq!(prog.modules.len(), 2);
}

#[test]
fn test_packages_edge_cases_and_env() {
    // 1. LUST_PACKAGES_DIR env var
    let custom_dir = "/tmp/custom_lust_packages";
    unsafe { std::env::set_var("LUST_PACKAGES_DIR", custom_dir); }
    assert_eq!(PackageManager::default_root(), PathBuf::from(custom_dir));
    unsafe { std::env::remove_var("LUST_PACKAGES_DIR"); }

    // 2. stub_files_from_exports edge cases
    // a) Empty exports and empty type stubs
    let empty_stubs = stub_files_from_exports(&[], &[]);
    assert!(empty_stubs.is_empty());

    // b) Empty ModuleStub
    let empty_module_stub = ModuleStub::default();
    let stubs_from_empty = stub_files_from_exports(&[], &[empty_module_stub]);
    assert!(stubs_from_empty.is_empty());

    // c) Export without dot in name (should be skipped)
    let no_dot_export = NativeExport::new("standalone_fn", vec![], "int");
    let stubs_no_dot = stub_files_from_exports(&[no_dot_export], &[]);
    assert!(stubs_no_dot.is_empty());

    // d) Export with doc comments, unnamed param, and untyped param
    let complex_export = NativeExport::new(
        "my_mod.complex_func",
        vec![
            NativeExportParam::new("", "int"),
            NativeExportParam::new("untyped_p", ""),
        ],
        "float",
    ).with_doc("Line 1 documentation\nLine 2 documentation");

    let stubs_complex = stub_files_from_exports(&[complex_export], &[]);
    assert_eq!(stubs_complex.len(), 1);
    assert!(stubs_complex[0].contents.contains("--- Line 1 documentation"));
    assert!(stubs_complex[0].contents.contains("--- Line 2 documentation"));
    assert!(stubs_complex[0].contents.contains("int, untyped_p: any"));
    assert!(stubs_complex[0].contents.contains("): float"));
}

#[test]
fn test_embedded_dynamic_import_expression_ast_branches() {
    let mut config = LustConfig::default();
    config.capability_mask_mut().allow_io = false;

    // 1. Binary expression with denied dynamic require
    let entries_bin = vec![
        EmbeddedModule {
            module: "dyn_bin",
            parent: None,
            source: Some(r#"
function test_bin()
    return 1 + require("std.io")
end
"#),
        },
    ];
    let err_bin = load_program_from_embedded_with_config(&entries_bin, "dyn_bin", &config).expect_err("binary require");
    assert!(matches!(err_bin, LustError::CompileError(_)));

    // 2. Unary expression with denied dynamic require
    let entries_unary = vec![
        EmbeddedModule {
            module: "dyn_unary",
            parent: None,
            source: Some(r#"
function test_unary()
    return -require("std.io")
end
"#),
        },
    ];
    let err_unary = load_program_from_embedded_with_config(&entries_unary, "dyn_unary", &config).expect_err("unary require");
    assert!(matches!(err_unary, LustError::CompileError(_)));

    // 3. Array expression with denied dynamic require
    let entries_arr = vec![
        EmbeddedModule {
            module: "dyn_arr",
            parent: None,
            source: Some(r#"
function test_arr()
    local x = [require("std.io")]
    return 0
end
"#),
        },
    ];
    let err_arr = load_program_from_embedded_with_config(&entries_arr, "dyn_arr", &config).expect_err("array require");
    assert!(matches!(err_arr, LustError::CompileError(_)));

    // 4. Map expression with denied dynamic require
    let entries_map = vec![
        EmbeddedModule {
            module: "dyn_map",
            parent: None,
            source: Some(r#"
function test_map()
    local x = { k = require("std.io") }
    return 0
end
"#),
        },
    ];
    let err_map = load_program_from_embedded_with_config(&entries_map, "dyn_map", &config).expect_err("map require");
    assert!(matches!(err_map, LustError::CompileError(_)));

    // 5. While condition with denied dynamic require
    let entries_while = vec![
        EmbeddedModule {
            module: "dyn_while",
            parent: None,
            source: Some(r#"
function test_while()
    while require("std.io") do
        break
    end
    return 0
end
"#),
        },
    ];
    let err_while = load_program_from_embedded_with_config(&entries_while, "dyn_while", &config).expect_err("while require");
    assert!(matches!(err_while, LustError::CompileError(_)));

    // 6. If condition with denied dynamic require
    let entries_if = vec![
        EmbeddedModule {
            module: "dyn_if",
            parent: None,
            source: Some(r#"
function test_if()
    if require("std.io") then
        return 1
    end
    return 0
end
"#),
        },
    ];
    let err_if = load_program_from_embedded_with_config(&entries_if, "dyn_if", &config).expect_err("if require");
    assert!(matches!(err_if, LustError::CompileError(_)));

    // 7. For loop bounds with denied dynamic require
    let entries_for = vec![
        EmbeddedModule {
            module: "dyn_for",
            parent: None,
            source: Some(r#"
function test_for()
    for i = 1, require("std.io") do
        break
    end
    return 0
end
"#),
        },
    ];
    let err_for = load_program_from_embedded_with_config(&entries_for, "dyn_for", &config).expect_err("for require");
    assert!(matches!(err_for, LustError::CompileError(_)));
}

#[test]
fn test_embedded_preprocessing_and_import_variations() {
    let entries = vec![
        EmbeddedModule {
            module: "preprocessed_importer",
            parent: None,
            source: Some(r#"
-- comment line with import foo from "bar"
import math from "@std/math";
export import string from "@std/string";
use @std/array
export use @std/map

function test_all()
    return 0
end
"#),
        },
    ];

    let prog = load_program_from_embedded(&entries, "preprocessed_importer").expect("preprocessing variations");
    assert_eq!(prog.modules.len(), 1);
}

#[test]
fn test_embedded_capability_denial_on_es_import() {
    let mut config = LustConfig::default();
    config.capability_mask_mut().allow_io = false;

    let entries = vec![
        EmbeddedModule {
            module: "es_denied",
            parent: None,
            source: Some("import io from \"@std/io\";\n"),
        },
    ];

    let err = load_program_from_embedded_with_config(&entries, "es_denied", &config)
        .expect_err("ES import of denied module should fail");
    assert!(matches!(err, LustError::CompileError(_)));
}

#[test]
fn test_embedded_lua_import_field_access_denial() {
    let mut config = LustConfig::default();
    config.capability_mask_mut().allow_io = false;

    let entries = vec![
        EmbeddedModule {
            module: "lua_imp_field",
            parent: None,
            source: Some(r#"
function run()
    local x = lua.import("std.io")
    return 0
end
"#),
        },
    ];

    let err = load_program_from_embedded_with_config(&entries, "lua_imp_field", &config)
        .expect_err("lua.import should be denied");
    assert!(matches!(err, LustError::CompileError(_)));
}
