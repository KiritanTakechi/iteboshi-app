fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let metal_enabled = std::env::var("CARGO_FEATURE_METAL").is_ok();

    match (target_os.as_str(), metal_enabled) {
        ("macos", false) => {
            println!("cargo:rustc-cfg=feature=\"metal\"");
            println!(
                "cargo:warning=Build script: Automatically enabling 'metal' feature for macOS."
            );
        }

        ("macos", true) => {
            println!(
                "cargo:warning=Build script: User explicitly enabled 'metal' feature for macOS."
            );
        }

        (_, true) => {
            println!(
                "cargo:warning=Build script: User explicitly enabled 'metal' feature, but target is not macOS ('{}'). Metal may not be available.",
                target_os
            );
        }

        (_, false) => {
            println!(
                "cargo:warning=Build script: Target is not macOS ('{}'), 'metal' feature not enabled.",
                target_os
            );
        }
    }

    tauri_build::build()
}
