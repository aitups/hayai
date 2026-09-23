//! Gemma4 unified multimodal embedders (`mmproj` GGUFs, projector types
//! `gemma4uv` vision / `gemma4ua` audio).
//!
//! Vision (`gemma4uv`) is **not** a ViT: im2col(patch) → LayerNorm → `patch_embd`
//! (+bias) → LayerNorm → 2D position tables → LayerNorm → RMSNorm →
//! `mm.input_projection`. Audio (`gemma4ua`) is encoder-free: frame the 16 kHz PCM
//! into `n_mel_bins`(=640)-sample frames, RMSNorm over the frame, then
//! `mm.a.input_projection`. Both mirror llama.cpp `clip.cpp`.

use crate::quant::QuantMatrix;
use crate::gguf_stream::GgufCatalog;
use crate::gguf_types::GgufError;

/// A decoded media item fed to the LM (embedding rows + how many LM tokens).
pub struct MediaEmbeddings {
    /// `n_tokens × text_hidden`, one row per LM token the media expands to.
    pub rows: Vec<Vec<f32>>,
    /// Whether the LM block should use non-causal attention (vision does).
    pub non_causal: bool,
}

/// Loaded multimodal projector (vision and/or audio).
pub struct ClipEmbedder {
    // ── vision ──
    has_vision: bool,
    n_embd: usize,
    patch_size: usize,
    patch_dim: usize,
    patch_embd: Option<QuantMatrix>,
    patch_bias: Vec<f32>,
    pn1_w: Vec<f32>,
    pn1_b: Vec<f32>,
    pn2_w: Vec<f32>,
    pn2_b: Vec<f32>,
    pn3_w: Vec<f32>,
    pn3_b: Vec<f32>,
    pos_embd: Vec<f32>,
    pos_size: usize,
    mm_in_proj: Option<QuantMatrix>,
    vision_eps: f32,
    min_pixels: usize,
    max_pixels: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],

    // ── audio ──
    has_audio: bool,
    a_in_proj: Option<QuantMatrix>,
    audio_frame: usize,
    audio_eps: f32,

    text_hidden: usize,
}

fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        x[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

fn rms_norm(x: &mut [f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn gemv(m: &QuantMatrix, x: &[f32]) -> Result<Vec<f32>, GgufError> {
    let mut y = vec![0.0f32; m.nrows];
    m.gemv(x, &mut y)?;
    Ok(y)
}

impl ClipEmbedder {
    /// Open an `mmproj` GGUF and load the projector weights.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, GgufError> {
        let mut cat = GgufCatalog::open(path.as_ref())?;
        let has_vision = matches!(
            cat.metadata.get("clip.has_vision_encoder"),
            Some(crate::MetadataValue::Bool(true))
        );
        let has_audio = matches!(
            cat.metadata.get("clip.has_audio_encoder"),
            Some(crate::MetadataValue::Bool(true))
        );
        if !has_vision && !has_audio {
            return Err(GgufError::Msg(
                "clip GGUF has neither a vision nor an audio encoder".into(),
            ));
        }

        let mut me = Self {
            has_vision: false,
            n_embd: 0,
            patch_size: 0,
            patch_dim: 0,
            patch_embd: None,
            patch_bias: Vec::new(),
            pn1_w: Vec::new(),
            pn1_b: Vec::new(),
            pn2_w: Vec::new(),
            pn2_b: Vec::new(),
            pn3_w: Vec::new(),
            pn3_b: Vec::new(),
            pos_embd: Vec::new(),
            pos_size: 0,
            mm_in_proj: None,
            vision_eps: 1e-6,
            min_pixels: 0,
            max_pixels: 0,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            has_audio: false,
            a_in_proj: None,
            audio_frame: 640,
            audio_eps: 1e-6,
            text_hidden: 0,
        };

        if has_vision {
            let pe = cat.tensor("v.patch_embd.weight")?.clone();
            let patch_dim = pe.ncols();
            let n_embd = pe.nrows();
            // Effective patch: metadata patch_size × merge (gemma4uv bakes a 3× merge).
            let base_ps = cat.meta_u32("clip.vision.patch_size").unwrap_or(16).max(1) as usize;
            let merge = cat
                .meta_u32("clip.vision.projector.scale_factor")
                .unwrap_or(3)
                .max(1) as usize;
            let patch_size = base_ps * merge;
            if patch_dim != patch_size * patch_size * 3 {
                return Err(GgufError::Msg(format!(
                    "gemma4uv: patch_dim {patch_dim} != patch_size²·3 ({}²·3)",
                    patch_size
                )));
            }
            let pos = cat.tensor("v.position_embd.weight")?.clone();
            let pos_size = pos.dims.get(1).copied().unwrap_or(0) as usize;
            let proj = cat.load_quant_matrix("mm.input_projection.weight")?;
            let text_hidden = proj.nrows;
            me.patch_embd = Some(cat.load_quant_matrix("v.patch_embd.weight")?);
            me.patch_bias = cat.dequant_f32("v.patch_embd.bias")?;
            me.pn1_w = cat.dequant_f32("v.patch_norm.1.weight")?;
            me.pn1_b = cat.dequant_f32("v.patch_norm.1.bias")?;
            me.pn2_w = cat.dequant_f32("v.patch_norm.2.weight")?;
            me.pn2_b = cat.dequant_f32("v.patch_norm.2.bias")?;
            me.pn3_w = cat.dequant_f32("v.patch_norm.3.weight")?;
            me.pn3_b = cat.dequant_f32("v.patch_norm.3.bias")?;
            me.pos_embd = cat.dequant_f32("v.position_embd.weight")?;
            me.pos_size = pos_size;
            me.mm_in_proj = Some(proj);
            me.vision_eps = cat
                .meta_f32("clip.vision.attention.layer_norm_epsilon")
                .unwrap_or(1e-6);
            // 70..1120 merged tokens (llama.cpp `set_limit_image_tokens(70, 1120)`).
            let patch_area = patch_size * patch_size;
            me.min_pixels = 70 * patch_area;
            me.max_pixels = 1120 * patch_area;
            if let Some(v) = cat.metadata.get("clip.vision.image_mean").and_then(|v| v.as_f32_array()) {
                if v.len() == 3 {
                    me.image_mean = [v[0], v[1], v[2]];
                }
            }
            if let Some(v) = cat.metadata.get("clip.vision.image_std").and_then(|v| v.as_f32_array()) {
                if v.len() == 3 {
                    me.image_std = [v[0], v[1], v[2]];
                }
            }
            me.n_embd = n_embd;
            me.patch_size = patch_size;
            me.patch_dim = patch_dim;
            me.text_hidden = text_hidden;
            me.has_vision = true;
        }

        if has_audio {
            let proj = cat.load_quant_matrix("mm.a.input_projection.weight")?;
            // Frame length is the projection's *input* dim (gemma4ua: 640 samples @16k);
            // `clip.audio.num_mel_bins` (128) is a stale metadata value here.
            me.audio_frame = proj.ncols;
            me.audio_eps = cat
                .meta_f32("clip.audio.attention.layer_norm_epsilon")
                .unwrap_or(1e-6);
            if me.text_hidden == 0 {
                me.text_hidden = proj.nrows;
            }
            me.a_in_proj = Some(proj);
            me.has_audio = true;
        }

        Ok(me)
    }

    pub fn has_vision(&self) -> bool {
        self.has_vision
    }

    pub fn has_audio(&self) -> bool {
        self.has_audio
    }

    pub fn text_hidden(&self) -> usize {
        self.text_hidden
    }

    /// Number of LM tokens an image of `(w, h)` expands to (`(w/ps)*(h/ps)`).
    pub fn image_token_count(&self, w: usize, h: usize) -> usize {
        (w / self.patch_size) * (h / self.patch_size)
    }

    /// Resize an RGB image to a 48-aligned size within the pixel budget
    /// (aspect-preserving; llama.cpp `calc_size_preserved_ratio` + PAD_CEIL).
    fn target_size(&self, w: usize, h: usize) -> (usize, usize) {
        let align = self.patch_size;
        let area = (w * h).max(1);
        let target_area = area.clamp(self.min_pixels, self.max_pixels) as f64;
        let scale = (target_area / area as f64).sqrt();
        // Round each side **up** to a multiple of `align` so the area never drops
        // below `min_pixels` (llama.cpp aligns and pads to a 48-multiple).
        let round_up = |v: f64| -> usize {
            let v = v.ceil().max(align as f64) as usize;
            v.div_ceil(align) * align
        };
        (round_up(w as f64 * scale), round_up(h as f64 * scale))
    }

    /// Encode an image (RGB8) into `n_tokens × text_hidden` LM embeddings.
    pub fn encode_image_rgb8(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<MediaEmbeddings, GgufError> {
        if !self.has_vision {
            return Err(GgufError::Msg("mmproj has no vision encoder".into()));
        }
        let (tw, th) = self.target_size(width, height);
        // Bilinear resize (RGB8) to the target size, then normalize.
        let resized = resize_rgb_bilinear(pixels, width, height, tw, th);
        let ps = self.patch_size;
        let n_cols = tw / ps;
        let n_rows = th / ps;
        let n_patches = n_cols * n_rows;
        let patch_embd = self.patch_embd.as_ref().unwrap();
        let mm_in_proj = self.mm_in_proj.as_ref().unwrap();

        let mut out = Vec::with_capacity(n_patches);
        let mut pv = vec![0.0f32; self.patch_dim];
        for py in 0..n_rows {
            for px in 0..n_cols {
                // im2col: patch vector index = c*ps² + dy*ps + dx (PyTorch unfold).
                for dy in 0..ps {
                    for dx in 0..ps {
                        let sx = px * ps + dx;
                        let sy = py * ps + dy;
                        let o = (sy * tw + sx) * 3;
                        for c in 0..3 {
                            let v = resized[o + c] as f32 / 255.0;
                            let v = (v - self.image_mean[c]) / self.image_std[c];
                            pv[c * ps * ps + dy * ps + dx] = v;
                        }
                    }
                }
                layer_norm(&mut pv, &self.pn1_w, &self.pn1_b, 1e-5);
                let mut e = gemv(patch_embd, &pv)?;
                for i in 0..e.len() {
                    e[i] += self.patch_bias[i];
                }
                layer_norm(&mut e, &self.pn2_w, &self.pn2_b, 1e-5);
                // 2D positions: table 0 = x, table 1 = y (row-major, x fastest).
                let (ix, iy) = (px, py);
                let t = self.pos_size * self.n_embd;
                for i in 0..self.n_embd {
                    let ex = self.pos_embd[ix * self.n_embd + i];
                    let ey = self.pos_embd[t + iy * self.n_embd + i];
                    e[i] += ex + ey;
                }
                layer_norm(&mut e, &self.pn3_w, &self.pn3_b, 1e-5);
                rms_norm(&mut e, self.vision_eps);
                let proj = gemv(mm_in_proj, &e)?;
                out.push(proj);
            }
        }
        Ok(MediaEmbeddings {
            rows: out,
            // Vision blocks use non-causal attention (llama.cpp `mtmd_decode_use_non_causal`).
            non_causal: true,
        })
    }

    /// Encode 16 kHz mono PCM into frame embeddings (`ceil(n/640)` rows).
    pub fn encode_audio_16k(&self, samples: &[f32]) -> Result<MediaEmbeddings, GgufError> {
        let proj = self
            .a_in_proj
            .as_ref()
            .ok_or_else(|| GgufError::Msg("mmproj has no audio encoder".into()))?;
        let frame = self.audio_frame;
        let n_tokens = samples.len().div_ceil(frame).max(1);
        let mut out = Vec::with_capacity(n_tokens);
        let mut buf = vec![0.0f32; frame];
        for t in 0..n_tokens {
            for (f, b) in buf.iter_mut().enumerate() {
                let src = t * frame + f;
                *b = if src < samples.len() { samples[src] } else { 0.0 };
            }
            rms_norm(&mut buf, self.audio_eps);
            out.push(gemv(proj, &buf)?);
        }
        Ok(MediaEmbeddings {
            rows: out,
            // Audio blocks are causal in llama.cpp.
            non_causal: false,
        })
    }
}

/// Decode an image file (PNG/JPEG/BMP/GIF/WebP/TIFF) to interleaved RGB8.
pub fn load_image_rgb8(
    path: impl AsRef<std::path::Path>,
) -> Result<(Vec<u8>, usize, usize), GgufError> {
    let img = image::open(path.as_ref())
        .map_err(|e| GgufError::Msg(format!("image decode: {e}")))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    Ok((img.into_raw(), w as usize, h as usize))
}

/// Decode an audio file (mp3/aac/flac/ogg-vorbis/wav/alac) to 16 kHz mono f32,
/// linearly resampled (llama.cpp uses miniaudio; quality here is sufficient for
/// the gemma4ua frame embedder).
pub fn decode_audio_16k(path: impl AsRef<std::path::Path>) -> Result<Vec<f32>, GgufError> {
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .map_err(|e| GgufError::Msg(format!("audio open {}: {e}", path.display())))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| GgufError::Msg(format!("audio probe: {e}")))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| GgufError::Msg("audio: no decodable track".into()))?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(16_000) as usize;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| GgufError::Msg(format!("audio decoder: {e}")))?;

    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(e) => return Err(GgufError::Msg(format!("audio packet: {e}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let spec = *buf.spec();
                let ch = spec.channels.count().max(1);
                let frames = buf.frames();
                let mut sb = symphonia::core::audio::SampleBuffer::<f32>::new(
                    frames as u64,
                    spec,
                );
                sb.copy_interleaved_ref(buf);
                let samples = sb.samples();
                for f in 0..frames {
                    let mut s = 0.0f32;
                    for c in 0..ch {
                        s += samples[f * ch + c];
                    }
                    mono.push(s / ch as f32);
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(GgufError::Msg(format!("audio decode: {e}"))),
        }
    }

    if sample_rate == 16_000 || mono.is_empty() {
        return Ok(mono);
    }
    let ratio = 16_000.0 / sample_rate as f64;
    let out_len = (mono.len() as f64 * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let i0 = (src.floor() as usize).min(mono.len() - 1);
        let i1 = (i0 + 1).min(mono.len() - 1);
        let w = (src - i0 as f64) as f32;
        out.push(mono[i0] * (1.0 - w) + mono[i1] * w);
    }
    Ok(out)
}

/// Bilinear RGB8 resize (aspect ratio already handled by `target_size`).
fn resize_rgb_bilinear(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dw * dh * 3];
    if sw == 0 || sh == 0 {
        return dst;
    }
    for y in 0..dh {
        let sy = (y as f32 + 0.5) * sh as f32 / dh as f32 - 0.5;
        let y0 = sy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..dw {
            let sx = (x as f32 + 0.5) * sw as f32 / dw as f32 - 0.5;
            let x0 = sx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = (sx - x0 as f32).clamp(0.0, 1.0);
            for c in 0..3 {
                let p00 = src[(y0 * sw + x0) * 3 + c] as f32;
                let p01 = src[(y0 * sw + x1) * 3 + c] as f32;
                let p10 = src[(y1 * sw + x0) * 3 + c] as f32;
                let p11 = src[(y1 * sw + x1) * 3 + c] as f32;
                let top = p00 + (p01 - p00) * wx;
                let bot = p10 + (p11 - p10) * wx;
                let v = top + (bot - top) * wy;
                dst[(y * dw + x) * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}
