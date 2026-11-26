//! OpenBG binary entry point.
//!
//! The same binary acts as a CLI, a batch processor, and an HTTP API server,
//! matching the three usage modes remove.bg users need to migrate to.
//!
//! - `openbg input.png -o out.png` — single image
//! - `openbg batch ./in_dir -o ./out_dir` — process a whole folder
//! - `openbg serve --port 8080` — wire-compatible remove.bg HTTP API + web demo
//!
//! Build with `cargo build --release`. Run `cargo run --release -- --help`
//! for the full option list.

use std::process::ExitCode;

use clap::Parser;

use openbg::cli::{Cli, Command};

#[tokio::main]
async fn main() -> ExitCode {
    // Initialise structured logging. `RUST_LOG=openbg=info` is a good default.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("openbg=info,warn")),
        )
        .init();

    let cli = Cli::parse();

    let result = match cli.command {
        Command::Remove(args) => openbg::cli::run_remove(args).await,
        Command::Batch(args) => openbg::cli::run_batch(args).await,
        Command::Serve(args) => openbg::cli::run_serve(args).await,
    };

    if let Err(e) = result {
        eprintln!("openbg: {e:#}");
        return ExitCode::from(1);
    }
    ExitCode::from(0)
}
