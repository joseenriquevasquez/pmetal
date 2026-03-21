//! Build script for compiling Metal shaders.
//!
//! This script compiles .metal shader files into .metallib libraries
//! that can be embedded into the binary at compile time.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("src").join("kernels").join("metal");

    // Check if we're on macOS
    if env::var("CARGO_CFG_TARGET_OS").unwrap() != "macos" {
        println!("cargo:warning=Metal shaders only compile on macOS");
        return;
    }

    // Verify Metal toolchain is available before attempting compilation
    check_metal_toolchain();

    // Find all .metal files
    if shaders_dir.exists() {
        compile_metal_shaders(&shaders_dir, &out_dir);
    } else {
        println!(
            "cargo:warning=No metal shaders directory found at {:?}",
            shaders_dir
        );
    }

    // Re-run if shaders change
    println!("cargo:rerun-if-changed=src/kernels/metal");

    // Link against Accelerate.framework for vDSP vector operations
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // Conditionally link IOSurface.framework when ANE feature is active
    if env::var("CARGO_FEATURE_ANE").is_ok() {
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        // AppleNeuralEngine.framework is loaded at runtime via dlopen, not linked here
    }
}

/// Verify that the Metal compiler toolchain is available via xcrun.
///
/// Provides actionable installation instructions on failure instead of
/// a cryptic "Failed to run Metal compiler" panic.
fn check_metal_toolchain() {
    // Check xcrun itself
    let xcrun_ok = Command::new("xcrun")
        .args(["--find", "metal"])
        .output();

    match xcrun_ok {
        Ok(output) if output.status.success() => return, // Metal compiler found
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

fn compile_metal_shaders(shaders_dir: &Path, out_dir: &Path) {
    let cache_root = out_dir.join(".cache");
    std::fs::create_dir_all(&cache_root).expect("Failed to create shader compiler cache");

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
        let air_file = out_dir.join(format!("{}.air", stem));

        println!("cargo:rerun-if-changed={}", metal_file.display());

        let output = Command::new("xcrun")
            .env("HOME", out_dir)
            .env("XDG_CACHE_HOME", &cache_root)
            .args([
                "-sdk",
                "macosx",
                "metal",
                // Optimization flags
                "-O3",
                // Enable fast math for ML workloads
                "-ffast-math",
                // Target Apple Silicon
                "-target",
                "air64-apple-macos14.0",
                // Include path for shared headers
                "-I",
                shaders_dir.to_str().unwrap(),
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
    let metallib_file = out_dir.join("pmetal_kernels.metallib");

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
            "Failed to link Metal library\n\n--- linker output ---\n{}",
            stderr
        );
    }

    println!(
        "cargo:rustc-env=PMETAL_METALLIB_PATH={}",
        metallib_file.display()
    );

    println!("Successfully compiled Metal shaders to {:?}", metallib_file);
}
