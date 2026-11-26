//! OpenBG — a free, open-source, self-hostable remove.bg replacement.
//!
//! OpenBG runs the MIT-licensed BiRefNet ONNX model fully on your own
//! machine through the ONNX Runtime. It ships as a single binary that can
//! act as a command-line tool, a batch processor, and an HTTP API server
//! that is wire-compatible with the (now-shutdown) remove.bg standalone API.
//!
//! See the [`cli`], [`server`] and [`engine`] module docs for details.

pub mod cli;
pub mod engine;
pub mod model;
pub mod server;

pub use engine::{Engine, OutputFormat, RemoveBgOptions};
