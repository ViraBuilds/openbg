//! Command-line interface for OpenBG: single-file, batch, and serve modes.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use tracing::info;

use crate::engine::{Engine, OutputFormat, OutputSize, RemoveBgOptions};
use crate::server;

/// Free, open-source remove.bg replacement powered by BiRefNet (MIT).
#[derive(Parser, Debug)]
#[command(name = "openbg", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Remove the background from a single image.
    #[command(name = "remove")]
    Remove(RemoveArgs),
    /// Remove the background from every image in a folder.
    #[command(name = "batch")]
    Batch(BatchArgs),
    /// Start the HTTP API server and web demo (remove.bg-compatible).
    #[command(name = "serve")]
    Serve(ServeArgs),
}

/// Shared image-processing options, mirroring remove.bg's API fields.
#[derive(Args, Debug, Clone)]
pub struct ImageOpts {
    /// Output size. One of: auto, preview, full, regular (remove.bg compatibility).
    #[arg(long, default_value = "auto")]
    pub size: String,

    /// Output format: auto, png, jpg, zip. Only `auto`, `png` and `jpg` are honoured.
    #[arg(long, default_value = "auto")]
    pub format: String,

    /// Background colour as `#rrggbb` (e.g. `#ffffff`). Omit for transparent.
    #[arg(long)]
    pub bg_color: Option<String>,
}

#[derive(Args, Debug)]
pub struct RemoveArgs {
    /// Input image path (png, jpg, webp, bmp, tiff).
    pub input: PathBuf,

    /// Output image path. If omitted, `<input>-openbg.png` is written next to the input.
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    #[command(flatten)]
    pub image: ImageOpts,
}

#[derive(Args, Debug)]
pub struct BatchArgs {
    /// Folder containing input images.
    pub input_dir: PathBuf,

    /// Folder to write processed images into. Created if it does not exist.
    #[arg(short = 'o', long = "output")]
    pub output_dir: PathBuf,

    /// Process images recursively (descending into subdirectories).
    #[arg(long)]
    pub recursive: bool,

    /// Skip an output file if it already exists and is newer than the input.
    #[arg(long)]
    pub skip_existing: bool,

    #[command(flatten)]
    pub image: ImageOpts,
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// TCP port to listen on.
    #[arg(long, default_value = "8080")]
    pub port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Disable the built-in web demo served at `/`.
    #[arg(long)]
    pub no_web_demo: bool,
}

/// Entry point for `openbg remove`.
pub async fn run_remove(args: RemoveArgs) -> Result<()> {
    let engine = Engine::load().await?;
    let opts = parse_image_opts(&args.image)?;
    let output = args
        .output
        .unwrap_or_else(|| default_output_path(&args.input, &opts.format));
    process_one(&engine, &args.input, &output, opts).await
}

/// Entry point for `openbg batch`.
pub async fn run_batch(args: BatchArgs) -> Result<()> {
    let engine = Engine::load().await?;
    let opts = parse_image_opts(&args.image)?;

    std::fs::create_dir_all(&args.output_dir).with_context(|| {
        format!("failed to create output directory {}", args.output_dir.display())
    })?;

    let images = collect_images(&args.input_dir, args.recursive)?;
    if images.is_empty() {
        eprintln!("openbg: no images found in {}", args.input_dir.display());
        return Ok(());
    }

    let total = images.len();
    eprintln!("openbg: processing {total} image(s)…");
    let mut ok = 0usize;
    let mut failed: Vec<(PathBuf, String)> = Vec::new();
    for (i, input) in images.iter().enumerate() {
        let rel = input
            .strip_prefix(&args.input_dir)
            .unwrap_or(input)
            .to_path_buf();
        let output = args.output_dir.join(&rel);
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        if args.skip_existing && output.is_file() && newer_than(&output, input) {
            eprintln!("[{}/{}] skip {}", i + 1, total, rel.display());
            continue;
        }

        eprintln!("[{}/{}] {}", i + 1, total, rel.display());
        match process_one(&engine, input, &output, opts.clone()).await {
            Ok(_) => ok += 1,
            Err(e) => {
                eprintln!("  failed: {e:#}");
                failed.push((input.clone(), format!("{e:#}")));
            }
        }
    }

    eprintln!("openbg: done — {ok}/{total} succeeded");
    if !failed.is_empty() {
        eprintln!("openbg: {} file(s) failed:", failed.len());
        for (p, err) in &failed {
            eprintln!("  {}: {err}", p.display());
        }
        bail!("batch completed with {} failure(s)", failed.len());
    }
    Ok(())
}

/// Entry point for `openbg serve`.
pub async fn run_serve(args: ServeArgs) -> Result<()> {
    let engine = Engine::load().await?;
    let addr = format!("{}:{}", args.host, args.port);
    info!(%addr, "OpenBG API server starting");
    server::serve(engine, &addr, args.no_web_demo).await
}

// ---- helpers ----------------------------------------------------------------

fn parse_image_opts(image: &ImageOpts) -> Result<RemoveBgOptions> {
    let format = match image.format.to_ascii_lowercase().as_str() {
        "auto" => OutputFormat::Auto,
        "png" => OutputFormat::Png,
        "jpg" | "jpeg" => OutputFormat::Jpeg,
        other => bail!("unsupported --format `{other}` (use auto, png, jpg)"),
    };
    let size = match image.size.to_ascii_lowercase().as_str() {
        "auto" => OutputSize::Auto,
        "preview" => OutputSize::Preview,
        "full" => OutputSize::Full,
        "regular" => OutputSize::Regular,
        other => bail!("unsupported --size `{other}` (use auto, preview, full, regular)"),
    };
    let bg_color = match &image.bg_color {
        Some(c) => Some(parse_hex_color(c).with_context(|| "invalid --bg-color")?),
        None => None,
    };
    Ok(RemoveBgOptions {
        format,
        size,
        bg_color,
    })
}

fn parse_hex_color(s: &str) -> Result<[u8; 3]> {
    let s = s.trim().trim_start_matches('#');
    if s.len() != 6 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("expected `#rrggbb`, got `{s}`");
    }
    let mut bytes = [0u8; 3];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .with_context(|| format!("invalid hex byte in `{s}`"))?;
    }
    Ok(bytes)
}

fn default_output_path(input: &std::path::Path, format: &OutputFormat) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let ext = match format {
        OutputFormat::Png | OutputFormat::Auto => "png",
        OutputFormat::Jpeg => "jpg",
    };
    let name = format!("{stem}-openbg.{ext}");
    let parent = input.parent().unwrap_or_else(|| std::path::Path::new("."));
    parent.join(name)
}

fn is_image_ext(p: &std::path::Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp" | "tif" | "tiff")
    )
}

fn collect_images(dir: &std::path::Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_images_inner(dir, recursive, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_images_inner(
    dir: &std::path::Path,
    recursive: bool,
    out: &mut Vec<PathBuf>,
) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read directory {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            if recursive {
                collect_images_inner(&path, recursive, out)?;
            }
        } else if ft.is_file() && is_image_ext(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn newer_than(a: &std::path::Path, b: &std::path::Path) -> bool {
    let (Ok(am), Ok(bm)) = (
        std::fs::metadata(a).and_then(|m| m.modified()),
        std::fs::metadata(b).and_then(|m| m.modified()),
    ) else {
        return false;
    };
    am >= bm
}

async fn process_one(
    engine: &Engine,
    input: &std::path::Path,
    output: &std::path::Path,
    opts: RemoveBgOptions,
) -> Result<()> {
    let bytes = std::fs::read(input)
        .with_context(|| format!("failed to read input image {}", input.display()))?;
    let result = engine.remove_bg_from_bytes(&bytes, opts).await?;
    std::fs::write(output, &result.image_bytes)
        .with_context(|| format!("failed to write output image {}", output.display()))?;
    eprintln!(
        "  wrote {} ({}, mask ratio {:.2}%)",
        output.display(),
        result.format,
        result.alpha_ratio * 100.0
    );
    Ok(())
}
