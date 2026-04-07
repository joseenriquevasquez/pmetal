//! Build script for compiling Metal shaders.
//!
//! This script compiles .metal shader files into .metallib libraries
//! that can be embedded into the binary at compile time.
//!
//! Supports dual compilation:
//! - Metal 3 (macOS 14.0+): All shaders in src/kernels/metal/
//! - Metal 4 (macOS 26.0+): MPP/NAX shaders in src/kernels/metal4/
//!   Only compiled when Metal compiler version >= 400 and SDK >= 26.0

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("src").join("kernels").join("metal");
    let metal4_dir = manifest_dir.join("src").join("kernels").join("metal4");

    // Check if we're on macOS
    if env::var("CARGO_CFG_TARGET_OS").unwrap() != "macos" {
        println!("cargo:warning=Metal shaders only compile on macOS");
        return;
    }

    // Verify Metal toolchain is available before attempting compilation
    check_metal_toolchain();

    // Detect Metal compiler version and SDK version
    let metal_version = detect_metal_version();
    let sdk_version = detect_sdk_version();

    // Compile Metal 3 shaders (always)
    if shaders_dir.exists() {
        compile_metal_shaders(
            &shaders_dir,
            &out_dir,
            "air64-apple-macos14.0",
            "pmetal_kernels",
        );
    } else {
        println!(
            "cargo:warning=No metal shaders directory found at {:?}",
            shaders_dir
        );
    }

    // Compile Metal 4 / MPP shaders (conditional on toolchain support)
    let has_metal4 = metal_version >= 400 && sdk_version >= 26.0;
    if has_metal4 && metal4_dir.exists() {
        let metal4_files: Vec<_> = std::fs::read_dir(&metal4_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| {
                let p = e.ok()?.path();
                if p.extension()?.to_str()? == "metal" {
                    Some(p)
                } else {
                    None
                }
            })
            .collect();

        if !metal4_files.is_empty() {
            compile_metal_shaders(
                &metal4_dir,
                &out_dir,
                "air64-apple-macos26.0",
                "pmetal_kernels_metal4",
            );
            println!("cargo:rustc-cfg=has_metal4");
            println!(
                "cargo:warning=Metal 4 / MPP shaders compiled (Metal version {metal_version}, SDK {sdk_version})"
            );
        }
    } else if metal4_dir.exists() {
        // Metal 4 shaders present but toolchain lacks support — expected on SDK < 27.
        // Not a warning; just skip silently.
    }

    // Declare has_metal4 cfg for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_metal4)");

    // Re-run if shaders change
    println!("cargo:rerun-if-changed=src/kernels/metal");
    println!("cargo:rerun-if-changed=src/kernels/metal4");

    // Link against Accelerate.framework for vDSP vector operations
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // Conditionally link IOSurface.framework when ANE feature is active
    if env::var("CARGO_FEATURE_ANE").is_ok() {
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        // AppleNeuralEngine.framework is loaded at runtime via dlopen, not linked here
    }
}

/// Detect Metal compiler version (__METAL_VERSION__).
/// Returns e.g. 310 (Metal 3.1), 320 (Metal 3.2), 400 (Metal 4.0).
fn detect_metal_version() -> u32 {
    let output = Command::new("zsh")
        .args([
            "-c",
            "echo '__METAL_VERSION__' | xcrun -sdk macosx metal -E -x metal -P - 2>/dev/null | tail -1 | tr -d '\\n'",
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let version_str = String::from_utf8_lossy(&o.stdout).trim().to_string();
            version_str.parse::<u32>().unwrap_or(0)
        }
        _ => 0,
    }
}

/// Detect macOS SDK version (e.g. 14.5, 26.2).
fn detect_sdk_version() -> f64 {
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "--show-sdk-version"])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let version_str = String::from_utf8_lossy(&o.stdout).trim().to_string();
            version_str.parse::<f64>().unwrap_or(0.0)
        }
        _ => 0.0,
    }
}

/// Verify that the Metal compiler toolchain is available via xcrun.
///
/// Provides actionable installation instructions on failure instead of
/// a cryptic "Failed to run Metal compiler" panic.
fn check_metal_toolchain() {
    // Check xcrun itself
    let xcrun_ok = Command::new("xcrun").args(["--find", "metal"]).output();

    match xcrun_ok {
        Ok(output) if output.status.success() => {} // Metal compiler found
        Ok(_) => {
            // xcrun exists but can't find metal
            panic!(
                "\n\
                 ╔══════════════════════════════════════════════════════════════════╗\n\
                 ║  Metal compiler not found                                       ║\n\
                 ╚══════════════════════════════════════════════════════════════════╝\n\n\
                 PMetal requires the Metal shader compiler to build.\n\n\
                 To install:\n\
                 1. Install Xcode from the App Store (or Command Line Tools):\n\
                    xcode-select --install\n\n\
                 2. Accept the Xcode license:\n\
                    sudo xcodebuild -license accept\n\n\
                 3. Download the Metal toolchain:\n\
                    xcodebuild -downloadComponent MetalToolchain\n\n\
                 4. Restart your terminal (or reboot) after installation.\n\n\
                 If you have Xcode installed but metal is not found, try:\n\
                    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer\n"
            );
        }
        Err(e) => {
            // xcrun itself not found
            panic!(
                "\n\
                 ╔══════════════════════════════════════════════════════════════════╗\n\
                 ║  xcrun not found — Xcode Command Line Tools required            ║\n\
                 ╚══════════════════════════════════════════════════════════════════╝\n\n\
                 PMetal requires Xcode Command Line Tools to compile Metal shaders.\n\n\
                 To install:\n\
                    xcode-select --install\n\n\
                 After installation, restart your terminal and try again.\n\n\
                 Error: {e}\n"
            );
        }
    }
}

fn compile_metal_shaders(shaders_dir: &Path, out_dir: &Path, target: &str, lib_name: &str) {
    let cache_root = out_dir.join(".cache");
    std::fs::create_dir_all(&cache_root).expect("Failed to create shader compiler cache");

    // Determine metal language standard from target
    let std_flag = if target.contains("macos26") {
        "-std=metal4.0"
    } else {
        "-std=metal3.1"
    };

    let metal_files: Vec<_> = std::fs::read_dir(shaders_dir)
        .expect("Failed to read shaders directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "metal" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if metal_files.is_empty() {
        println!("cargo:warning=No .metal files found in {:?}", shaders_dir);
        return;
    }

    // Compile each .metal file to .air (intermediate representation)
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_file = out_dir.join(format!("{}_{}.air", lib_name, stem));

        println!("cargo:rerun-if-changed={}", metal_file.display());

        let output = Command::new("xcrun")
            .env("HOME", out_dir)
            .env("XDG_CACHE_HOME", &cache_root)
            .args([
                "-sdk",
                "macosx",
                "metal",
                // Metal language standard
                std_flag,
                // Optimization flags
                "-O3",
                // Enable fast math for ML workloads
                "-ffast-math",
                // Target Apple Silicon
                "-target",
                target,
                // Include path for shared headers (both metal/ and metal4/)
                "-I",
                shaders_dir.to_str().unwrap(),
                "-I",
                shaders_dir
                    .parent()
                    .unwrap()
                    .join("metal")
                    .to_str()
                    .unwrap(),
                // Compile to AIR
                "-c",
                metal_file.to_str().unwrap(),
                "-o",
                air_file.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to run Metal compiler (xcrun verified but execution failed)");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "Failed to compile Metal shader: {}\n\n--- compiler output ---\n{}",
                metal_file.display(),
                stderr
            );
        }

        air_files.push(air_file);
    }

    // Link all .air files into a single .metallib
    let metallib_file = out_dir.join(format!("{}.metallib", lib_name));

    let mut cmd = Command::new("xcrun");
    cmd.env("HOME", out_dir);
    cmd.env("XDG_CACHE_HOME", &cache_root);
    cmd.args(["-sdk", "macosx", "metallib"]);

    for air_file in &air_files {
        cmd.arg(air_file.to_str().unwrap());
    }

    cmd.args(["-o", metallib_file.to_str().unwrap()]);

    let output = cmd.output().expect("Failed to run metallib linker");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "Failed to link Metal library {}\n\n--- linker output ---\n{}",
            lib_name, stderr
        );
    }

    // Set env var for embedding — uppercase lib name
    let env_name = format!(
        "{}_PATH",
        lib_name.to_uppercase().replace("pmetal_", "PMETAL_")
    );
    println!("cargo:rustc-env={}={}", env_name, metallib_file.display());

    println!(
        "Successfully compiled {} Metal shaders to {:?} (target: {})",
        metal_files.len(),
        metallib_file,
        target
    );
}
