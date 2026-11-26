//! Inference engine: loads the ONNX model, preprocesses images, runs BiRefNet,
//! and post-processes the resulting mask into a transparent (or solid-coloured)
//! output image.
//!
//! ## Preprocessing (matches the BiRefNet `ViTFeatureExtractor`)
//!
//! - Resize to 1024×1024 with bilinear interpolation.
//! - Rescale: pixel / 255.
//! - Normalize: (x - mean) / std, with ImageNet mean `[0.485, 0.456, 0.406]`
//!   and std `[0.229, 0.224, 0.225]`.
//! - Layout: NCHW float32.
//!
//! ## Postprocessing
//!
//! - Sigmoid the model's logits to get a [0, 1] mask.
//! - Scale to 0–255 u8.
//! - Resize the mask back to the original image resolution with Lanczos3.
//! - Compose as the alpha channel of the original RGB image.
//! - Optionally composite onto a solid `bg_color`, then encode as JPEG.
//! - Apply the `size` cap (longest side: preview=1024, regular=2048, full=∞).

use std::io::Cursor;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use image::imageops::FilterType;
use image::{ImageBuffer, ImageEncoder, ImageReader, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::value::{DynValue, Tensor};
use ort::inputs;

use crate::model;

const MODEL_SIZE: u32 = 1024;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A cheaply-clonable handle to a loaded ONNX Runtime session.
///
/// All inference runs are serialised through a `std::sync::Mutex`, which is
/// the recommended pattern for `ort` 2.x (`Session::run` takes `&mut self`).
#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Auto,
    Png,
    Jpeg,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Auto => f.write_str("auto"),
            OutputFormat::Png => f.write_str("png"),
            OutputFormat::Jpeg => f.write_str("jpeg"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputSize {
    Auto,
    Preview,
    Regular,
    Full,
}

#[derive(Debug, Clone)]
pub struct RemoveBgOptions {
    pub format: OutputFormat,
    pub size: OutputSize,
    pub bg_color: Option<[u8; 3]>,
}

pub struct RemoveResult {
    pub image_bytes: Vec<u8>,
    pub format: OutputFormat,
    /// Mean foreground probability across the mask (0.0–1.0).
    pub alpha_ratio: f32,
}

impl Engine {
    /// Loads the ONNX model from disk, downloading it on first run.
    pub async fn load() -> Result<Engine> {
        let path = model::ensure_model()?;
        let env = ort::init()
            .with_name("openbg")
            .build()?;
        let session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("failed to set graph optimization level: {e}"))?
            .with_intra_threads(num_threads())
            .map_err(|e| anyhow!("failed to set intra-op thread count: {e}"))?
            .commit_from_file(&path)?;
        let input_name = session
            .inputs()
            .first()
            .context("model has no input tensors")?
            .name()
            .to_string();
        let output_name = session
            .outputs()
            .first()
            .context("model has no output tensors")?
            .name()
            .to_string();
        tracing::info!(%input_name, %output_name, "OpenBG engine ready");
        Ok(Engine {
            inner: Arc::new(EngineInner {
                session: Mutex::new(session),
                input_name,
                output_name,
            }),
        })
    }

    /// Removes the background from an image encoded as raw bytes.
    ///
    /// The blocking ONNX Runtime call is moved onto a `spawn_blocking` thread
    /// so the tokio worker stays free to handle other requests.
    pub async fn remove_bg_from_bytes(
        &self,
        bytes: &[u8],
        opts: RemoveBgOptions,
    ) -> Result<RemoveResult> {
        let inner = self.inner.clone();
        let bytes = bytes.to_vec();
        tokio::task::spawn_blocking(move || inner.remove_bg_blocking(&bytes, opts))
            .await
            .context("inference task panicked")?
    }
}

impl EngineInner {
    fn remove_bg_blocking(&self, bytes: &[u8], opts: RemoveBgOptions) -> Result<RemoveResult> {
        // 1. Decode the input into an RGB8 image.
        let img = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .context("failed to detect image format")?
            .decode()
            .context("failed to decode input image")?
            .to_rgb8();
        let (orig_w, orig_h) = (img.width(), img.height());

        // 2. Resize to 1024×1024 with bilinear filtering (matches the ViT preprocessor).
        let resized =
            image::imageops::resize(&img, MODEL_SIZE, MODEL_SIZE, FilterType::Triangle);
        // `RgbImage` is indexed by `(x, y)`, not by raw byte offset; pull out the
        // contiguous byte buffer so we can iterate channels efficiently.
        let resized_raw: Vec<u8> = resized.into_raw();

        // 3. Build the NCHW float32 tensor with ImageNet normalisation.
        let mut input =
            Array4::<f32>::zeros((1, 3, MODEL_SIZE as usize, MODEL_SIZE as usize));
        for y in 0..MODEL_SIZE as usize {
            for x in 0..MODEL_SIZE as usize {
                let idx = (y * MODEL_SIZE as usize + x) * 3;
                for c in 0..3 {
                    let v = resized_raw[idx + c] as f32 / 255.0;
                    input[[0, c, y, x]] = (v - MEAN[c]) / STD[c];
                }
            }
        }
        let tensor = Tensor::from_array(input)?;

        // 4. Run inference and copy out the mask while holding the session lock.
        let mask_data: Vec<f32> = {
            let mut session = self
                .session
                .lock()
                .map_err(|e| anyhow::anyhow!("session mutex poisoned: {e}"))?;
            let outputs: SessionOutputs<'_> =
                session.run(inputs![self.input_name.as_str() => tensor])?;
            let value: &DynValue = outputs
                .get(self.output_name.as_str())
                .unwrap_or_else(|| &outputs[0usize]);
            let (_shape, slice) = value
                .try_extract_tensor::<f32>()
                .context("failed to extract mask tensor as f32")?;
            slice.to_vec()
        };

        // 5. Sigmoid the logits and scale to a u8 mask.
        let mask_len = (MODEL_SIZE as usize) * (MODEL_SIZE as usize);
        if mask_data.len() < mask_len {
            bail!(
                "model output too small: got {} elements, need {}",
                mask_data.len(),
                mask_len
            );
        }
        let mut mask_u8 = vec![0u8; mask_len];
        let mut alpha_sum: f32 = 0.0;
        for i in 0..mask_len {
            let s = 1.0 / (1.0 + (-mask_data[i]).exp());
            mask_u8[i] = (s * 255.0).round() as u8;
            alpha_sum += s;
        }
        let alpha_ratio = alpha_sum / mask_len as f32;

        // 6. Resize the mask back to the original resolution (Lanczos3 for sharp edges).
        let mask_img: ImageBuffer<Luma<u8>, Vec<u8>> =
            ImageBuffer::from_raw(MODEL_SIZE, MODEL_SIZE, mask_u8)
                .context("failed to build mask image")?;
        let mask_full =
            image::imageops::resize(&mask_img, orig_w, orig_h, FilterType::Lanczos3);

        // 7. Compose the RGBA output.
        let mut rgba = RgbaImage::new(orig_w, orig_h);
        for (x, y, pixel) in img.enumerate_pixels() {
            let alpha = mask_full.get_pixel(x, y)[0];
            rgba.put_pixel(x, y, Rgba([pixel[0], pixel[1], pixel[2], alpha]));
        }

        // 8. Apply the optional longest-side cap.
        let final_image = apply_size_cap(rgba, opts.size);

        // 9. Encode the result.
        let (image_bytes, format) = encode_output(final_image, &opts)?;

        Ok(RemoveResult {
            image_bytes,
            format,
            alpha_ratio,
        })
    }
}

fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn apply_size_cap(img: RgbaImage, size: OutputSize) -> RgbaImage {
    let cap = match size {
        OutputSize::Auto | OutputSize::Full => return img,
        OutputSize::Preview => 1024u32,
        OutputSize::Regular => 2048u32,
    };
    let (w, h) = (img.width(), img.height());
    let longest = w.max(h);
    if longest <= cap {
        return img;
    }
    let scale = cap as f32 / longest as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);
    image::imageops::resize(&img, new_w, new_h, FilterType::Lanczos3)
}

fn encode_output(img: RgbaImage, opts: &RemoveBgOptions) -> Result<(Vec<u8>, OutputFormat)> {
    let format = match opts.format {
        OutputFormat::Auto => {
            if opts.bg_color.is_some() {
                OutputFormat::Jpeg
            } else {
                OutputFormat::Png
            }
        }
        f => f,
    };

    let mut buf: Vec<u8> = Vec::with_capacity(256 * 1024);
    match format {
        OutputFormat::Png => {
            if let Some(color) = opts.bg_color {
                let rgb = composite_onto_color(&img, color);
                rgb.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)?;
            } else {
                img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)?;
            }
        }
        OutputFormat::Jpeg => {
            let color = opts.bg_color.unwrap_or([255, 255, 255]);
            let rgb = composite_onto_color(&img, color);
            let (w, h) = (rgb.width(), rgb.height());
            let raw = rgb.into_raw();
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 92);
            encoder.write_image(&raw, w, h, image::ExtendedColorType::Rgb8)?;
        }
        OutputFormat::Auto => unreachable!(),
    }
    Ok((buf, format))
}

fn composite_onto_color(rgba: &RgbaImage, color: [u8; 3]) -> RgbImage {
    let (w, h) = rgba.dimensions();
    let mut rgb = RgbImage::new(w, h);
    for (x, y, pixel) in rgba.enumerate_pixels() {
        let a = pixel[3] as f32 / 255.0;
        let inv = 1.0 - a;
        let r = (pixel[0] as f32 * a + color[0] as f32 * inv) as u8;
        let g = (pixel[1] as f32 * a + color[1] as f32 * inv) as u8;
        let b = (pixel[2] as f32 * a + color[2] as f32 * inv) as u8;
        rgb.put_pixel(x, y, Rgb([r, g, b]));
    }
    rgb
}
