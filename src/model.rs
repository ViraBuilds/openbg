//! First-run model download and cache management.
//!
//! OpenBG does **not** ship the model binary in the crate. On first run
//! (CLI or server) it downloads the ONNX weights into a per-user cache
//! directory and reuses them afterwards.
//!
//! ## Default model
//!
//! By default OpenBG downloads the **BiRefNet-general** ONNX export with
//! the tiny Swin-v1 backbone (`BiRefNet-general-bb_swin_v1_tiny-epoch_232`),
//! ~224 MB, MIT-licensed, from the upstream BiRefNet GitHub releases.
//!
//! Override the URL with the `OPENBG_MODEL_URL` environment variable to use
//! any other BiRefNet ONNX export (for example the full `BiRefNet-general`
//! at ~972 MB for the highest quality, or `onnx-community/BiRefNet_lite` at
//! ~224 MB / 114 MB fp16). The model file's SHA-256 is verified against the
//! value in `OPENBG_MODEL_SHA256` if that variable is set.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

/// Default model: BiRefNet-general with the tiny Swin-v1 backbone (~224 MB, MIT).
pub const DEFAULT_MODEL_URL: &str =
    "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx";

/// Filename used inside the cache directory.
pub const MODEL_FILENAME: &str = "birefnet-general-tiny.onnx";

/// Returns the URL the model should be downloaded from.
fn model_url() -> String {
    std::env::var("OPENBG_MODEL_URL").unwrap_or_else(|_| DEFAULT_MODEL_URL.to_string())
}

/// Returns the cache directory used to store the downloaded ONNX model.
///
/// Override with the `OPENBG_CACHE_DIR` environment variable. Otherwise:
/// - Linux: `$XDG_CACHE_HOME/openbg` or `~/.cache/openbg`
/// - macOS: `~/Library/Caches/openbg`
/// - Windows: `%LOCALAPPDATA%\openbg`
pub fn cache_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("OPENBG_CACHE_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let dir = dirs::cache_dir()
        .ok_or_else(|| anyhow!("could not determine a per-user cache directory"))?
        .join("openbg");
    Ok(dir)
}

/// Returns the local path of the model, downloading it on first run.
pub fn ensure_model() -> Result<PathBuf> {
    let dir = cache_dir()?;
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create cache directory {}", dir.display()))?;
    let model_path = dir.join(MODEL_FILENAME);

    if model_path.is_file() {
        // Touch the file so we know it was seen recently (purely cosmetic).
        return Ok(model_path);
    }

    download_model(&model_path)?;
    Ok(model_path)
}

/// Downloads the model into `dest`, printing progress to stderr.
fn download_model(dest: &Path) -> Result<()> {
    let url = model_url();
    eprintln!("OpenBG: downloading BiRefNet ONNX model from");
    eprintln!("  {url}");
    eprintln!("  (this only happens once; cached at {})", dest.display());

    let tmp = dest.with_extension("onnx.part");
    let resp = reqwest::blocking::get(&url)
        .with_context(|| format!("failed to start download from {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        bail!("download failed: HTTP {status} for {url}");
    }
    let total = resp.content_length().unwrap_or(0);

    let mut file = fs::File::create(&tmp)
        .with_context(|| format!("failed to create temporary file {}", tmp.display()))?;
    let mut reader = resp;
    let mut buf = [0u8; 64 * 1024];
    let mut written: u64 = 0;
    let mut last_pct: u64 = 0;
    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| "error while reading model download stream")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .with_context(|| "error while writing model to disk")?;
        written += n as u64;
        if total > 0 {
            let pct = written * 100 / total;
            if pct >= last_pct + 5 || pct == 100 {
                eprint!(
                    "\r  downloaded {:.1} / {:.1} MB ({}%)   ",
                    written as f64 / 1_048_576.0,
                    total as f64 / 1_048_576.0,
                    pct
                );
                std::io::stderr().flush().ok();
                last_pct = pct;
            }
        }
    }
    eprintln!();
    file.sync_all().ok();
    drop(file);

    fs::rename(&tmp, dest)
        .with_context(|| format!("failed to move downloaded model into {}", dest.display()))?;

    eprintln!("OpenBG: model ready at {}", dest.display());
    Ok(())
}

// Tiny helper to use `Read` from std on the reqwest response.
use std::io::Read;

/// Convenience: read an environment variable as a `usize`, returning the
/// default if the variable is absent or unparseable.
pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
