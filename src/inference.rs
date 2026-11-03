//! Optional cloud inference on the result: image-to-image (the sketch redrawn by an image model on
//! fal.ai) and image-to-text (a description by a vision model on DeepSeek). Each run is a
//! background job that the UI polls; dropping the job cancels it.

use std::{
    io::{Cursor, ErrorKind},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, TryRecvError, channel},
    },
    time::{Duration, Instant},
};

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use chrono::{DateTime, Local};
use eframe::egui::{self, ColorImage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::img::RgbImage;

pub const DEFAULT_IMAGE_PROMPT: &str = "\
Redraw this pencil sketch in the style of a Leonardo da Vinci notebook drawing.

STYLE (change only this):
- Brown/sepia iron-gall ink and red chalk on aged, slightly yellowed paper
- Fine parallel hatching and cross-hatching for shading, delicate contour lines
- Subtle paper texture, faint stains, Renaissance study-sheet look

KEEP EXACTLY THE SAME (do not change):
- The composition, layout, proportions and position of every element
- All shapes, objects and details I drew – add nothing, remove nothing
- ALL handwritten text, labels, numbers and annotations: same words,
  same spelling, same position, same language, written left-to-right
  (do NOT mirror the text, do NOT translate or rephrase it)

Output only the restyled sketch, same aspect ratio as the input.";

pub const DEFAULT_TEXT_PROMPT: &str = "\
This is a scan of a hand-drawn sketch on paper. Describe what it shows and transcribe all \
handwritten text, labels and numbers exactly as written. Answer in plain text without Markdown.";

const FAL_QUEUE: &str = "https://queue.fal.run";
const DEEPSEEK_CHAT: &str = "https://api.deepseek.com/chat/completions";
/// Long side of the image sent to the image model; its output is 1K to 4K anyway.
const IMAGE_UPLOAD_SIDE: u32 = 2048;
/// DeepSeek scales images to about 1300×1300 pixels in total before looking at them.
const TEXT_UPLOAD_SIDE: u32 = 1568;
const UPLOAD_JPEG_QUALITY: u8 = 90;
/// Long side of the overlay texture; larger textures don't fit every GPU.
const OVERLAY_MAX_SIDE: u32 = 4096;
const POLL_INTERVAL: Duration = Duration::from_millis(1000);
/// Gives up on an image job that hasn't finished after this long, queue time included.
const IMAGE_TIMEOUT: Duration = Duration::from_secs(10 * 60);
/// Per HTTP request; a thinking text model can take minutes to answer.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const MAX_RESPONSE_BYTES: u64 = 128 * 1024 * 1024;
const CANCELLED: &str = "cancelled";

/// Output size of the image model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Resolution {
    R1K,
    R2K,
    R4K,
}

impl Resolution {
    pub const ALL: [Resolution; 3] = [Resolution::R1K, Resolution::R2K, Resolution::R4K];

    pub fn label(self) -> &'static str {
        match self {
            Resolution::R1K => "1K",
            Resolution::R2K => "2K",
            Resolution::R4K => "4K",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageSettings {
    /// fal.ai key; empty uses `$FAL_KEY`.
    pub api_key: String,
    /// fal.ai endpoint that takes `prompt` and `image_urls`.
    pub model: String,
    pub prompt: String,
    pub resolution: Resolution,
}

impl Default for ImageSettings {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "google/nano-banana-2.1/edit".into(),
            prompt: DEFAULT_IMAGE_PROMPT.into(),
            resolution: Resolution::R1K,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextSettings {
    /// DeepSeek key; empty uses `$DEEPSEEK_API_KEY`.
    pub api_key: String,
    pub model: String,
    pub prompt: String,
    /// Let the model reason before answering: more careful, but slower and more expensive.
    pub thinking: bool,
}

impl Default for TextSettings {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "deepseek-flash".into(),
            prompt: DEFAULT_TEXT_PROMPT.into(),
            thinking: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceSettings {
    pub image: ImageSettings,
    pub text: TextSettings,
    /// Opacity of the image-to-image overlay over the result.
    pub opacity: f32,
}

impl Default for InferenceSettings {
    fn default() -> Self {
        Self {
            image: ImageSettings::default(),
            text: TextSettings::default(),
            opacity: 1.0,
        }
    }
}

/// What the job thread shares with the UI.
#[derive(Clone)]
struct Progress {
    cancel: Arc<AtomicBool>,
    stage: Arc<Mutex<String>>,
    ctx: egui::Context,
}

impl Progress {
    fn set(&self, stage: impl Into<String>) {
        *self.stage.lock().unwrap() = stage.into();
        self.ctx.request_repaint();
    }

    fn check(&self) -> Result<(), String> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(CANCELLED.into())
        } else {
            Ok(())
        }
    }

    /// Sleeps for `d`, waking up early when cancelled.
    fn sleep(&self, d: Duration) -> Result<(), String> {
        let end = Instant::now() + d;
        while Instant::now() < end {
            self.check()?;
            std::thread::sleep(Duration::from_millis(50));
        }
        self.check()
    }
}

/// An inference request running on its own thread. Dropping it cancels the request.
pub struct Job<T> {
    progress: Progress,
    rx: Receiver<Result<T, String>>,
    started: Instant,
}

impl<T: Send + 'static> Job<T> {
    fn spawn(
        ctx: egui::Context,
        run: impl FnOnce(&Progress) -> Result<T, String> + Send + 'static,
    ) -> Self {
        let progress = Progress {
            cancel: Arc::new(AtomicBool::new(false)),
            stage: Arc::new(Mutex::new("Preparing…".into())),
            ctx,
        };
        let (tx, rx) = channel();
        let p = progress.clone();
        std::thread::spawn(move || {
            let result = run(&p);
            // The receiver is gone if the job was cancelled; nobody wants the result then.
            if tx.send(result).is_ok() {
                p.ctx.request_repaint();
            }
        });
        Self {
            progress,
            rx,
            started: Instant::now(),
        }
    }
}

impl<T> Job<T> {
    /// What the job is doing right now, for showing while it runs.
    pub fn stage(&self) -> String {
        self.progress.stage.lock().unwrap().clone()
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    fn try_take(&self) -> Option<Result<T, String>> {
        match self.rx.try_recv() {
            Ok(result) => Some(result),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err("the request stopped unexpectedly".into())),
        }
    }
}

impl<T> Drop for Job<T> {
    fn drop(&mut self) {
        self.progress.cancel.store(true, Ordering::Relaxed);
    }
}

/// One inference mode as the UI sees it. `D` is what a finished job's output `T` turns into,
/// e.g. a texture for an image.
pub enum Task<T, D = T> {
    Off,
    Running(Job<T>),
    Done(D),
    Failed(String),
}

impl<T, D> Task<T, D> {
    pub fn is_on(&self) -> bool {
        !matches!(self, Task::Off)
    }

    /// Moves a finished job on to `Done` or `Failed`.
    pub fn poll(&mut self, done: impl FnOnce(T) -> D) {
        if let Task::Running(job) = self
            && let Some(result) = job.try_take()
        {
            *self = match result {
                Ok(output) => Task::Done(done(output)),
                Err(e) => Task::Failed(e),
            };
        }
    }
}

/// A successful run: the model's output, and the directory the run was saved in or why that
/// failed.
pub struct Finished<T> {
    pub output: T,
    pub saved: Result<PathBuf, String>,
}

/// Redraws `image` with the image model. The result is meant to be shown stretched over the
/// input, which it matches in composition and roughly in aspect ratio. A successful run is saved
/// in a new directory in `save_in`, see [`save_run`].
pub fn image_to_image(
    ctx: egui::Context,
    image: Arc<RgbImage>,
    grayscale: bool,
    settings: ImageSettings,
    save_in: PathBuf,
) -> Job<Finished<ColorImage>> {
    let started = Local::now();
    Job::spawn(ctx, move |p| {
        let (input, output) = run_fal(p, &image, grayscale, &settings)?;
        let ext = image::guess_format(&output).map_or("png", |f| f.extensions_str()[0]);
        // Saved before decoding, so that the paid-for image is kept even if it can't be shown.
        let saved = save_run(
            &save_in,
            started,
            "image-to-image",
            &input,
            &settings.prompt,
            (&format!("output.{ext}"), &output),
        );
        let output = decode_overlay(&output)?;
        Ok(Finished { output, saved })
    })
}

/// Asks the vision model about `image`. A successful run is saved like [`image_to_image`]'s.
pub fn image_to_text(
    ctx: egui::Context,
    image: Arc<RgbImage>,
    grayscale: bool,
    settings: TextSettings,
    save_in: PathBuf,
) -> Job<Finished<String>> {
    let started = Local::now();
    Job::spawn(ctx, move |p| {
        let (input, output) = run_deepseek(p, &image, grayscale, &settings)?;
        let saved = save_run(
            &save_in,
            started,
            "image-to-text",
            &input,
            &settings.prompt,
            ("output.txt", output.as_bytes()),
        );
        Ok(Finished { output, saved })
    })
}

/// Keeps a run in a new directory in `parent`, named after its start time and mode, e.g.
/// `2026-11-03_18_30_05_image-to-image`: the image exactly as sent (`input.jpg`), the prompt
/// and the output. Runs started within the same second get numbered directories.
fn save_run(
    parent: &Path,
    started: DateTime<Local>,
    mode: &str,
    input_jpeg: &[u8],
    prompt: &str,
    (output_name, output): (&str, &[u8]),
) -> Result<PathBuf, String> {
    let failed = |path: &Path, e: std::io::Error| format!("{}: {e}", path.display());
    std::fs::create_dir_all(parent).map_err(|e| failed(parent, e))?;
    let name = format!("{}_{mode}", started.format("%Y-%m-%d_%H_%M_%S"));
    let mut dir = parent.join(&name);
    let mut n = 1;
    // `create_dir`, not a check for existence first, so that two runs never share a directory.
    loop {
        match std::fs::create_dir(&dir) {
            Ok(()) => break,
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                dir = parent.join(format!("{name}_{n}"));
                n += 1;
            }
            Err(e) => return Err(failed(&dir, e)),
        }
    }
    let files = [
        ("input.jpg", input_jpeg),
        ("prompt.txt", prompt.as_bytes()),
        (output_name, output),
    ];
    for (file, bytes) in files {
        let path = dir.join(file);
        std::fs::write(&path, bytes).map_err(|e| failed(&path, e))?;
    }
    Ok(dir)
}

/// fal.ai's queue: submit, poll the status until done, then fetch the result and download the
/// image. Unlike a single blocking call this can be cancelled, which stops the billing if the
/// request is still waiting in the queue. Returns the JPEG sent and the image file received.
fn run_fal(
    p: &Progress,
    image: &RgbImage,
    grayscale: bool,
    s: &ImageSettings,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let key = api_key(&s.api_key, "FAL_KEY", "fal.ai")?;
    let auth = format!("Key {key}");
    let model = s.model.trim().trim_matches('/');
    if model.is_empty() {
        return Err("No image-to-image model set".into());
    }
    let jpeg = encode_jpeg(image, grayscale, IMAGE_UPLOAD_SIDE)?;
    let upload = data_uri(&jpeg);
    p.check()?;
    p.set("Uploading…");
    let agent = agent();
    let body = json!({
        "prompt": s.prompt,
        "image_urls": [upload],
        "num_images": 1,
        "aspect_ratio": "auto",
        "output_format": "png",
        "resolution": s.resolution.label(),
    });
    let submitted = json_response(
        agent
            .post(format!("{FAL_QUEUE}/{model}"))
            .header("Authorization", &auth)
            .send_json(&body),
    )?;
    // The submit response names the URLs to use, which for an endpoint like `owner/app/edit`
    // leave out the `/edit`.
    let url = |name: &str| {
        submitted[name]
            .as_str()
            .map(String::from)
            .ok_or_else(|| format!("fal.ai sent no {name}"))
    };
    let (status_url, response_url, cancel_url) =
        (url("status_url")?, url("response_url")?, url("cancel_url")?);

    let deadline = Instant::now() + IMAGE_TIMEOUT;
    let waited = (|| loop {
        p.sleep(POLL_INTERVAL)?;
        let status = json_response(agent.get(&status_url).header("Authorization", &auth).call())?;
        match status["status"].as_str() {
            Some("IN_QUEUE") => p.set(match status["queue_position"].as_u64() {
                Some(n) if n > 0 => format!("Waiting in queue (position {n})…"),
                _ => "Waiting in queue…".into(),
            }),
            Some("IN_PROGRESS") => p.set("Generating…"),
            Some("COMPLETED") => {
                return match status["error"].as_str() {
                    Some(e) => Err(e.to_string()),
                    None => Ok(()),
                };
            }
            other => return Err(format!("fal.ai reported status {other:?}")),
        }
        if Instant::now() > deadline {
            return Err(format!(
                "gave up after {} minutes",
                IMAGE_TIMEOUT.as_secs() / 60
            ));
        }
    })();
    if let Err(e) = waited {
        if e == CANCELLED || e.starts_with("gave up") {
            // Best effort; once the model is running it may still finish (and be billed).
            let _ = agent
                .put(&cancel_url)
                .header("Authorization", &auth)
                .send_empty();
        }
        return Err(e);
    }

    p.set("Downloading…");
    let result = json_response(
        agent
            .get(&response_url)
            .header("Authorization", &auth)
            .call(),
    )?;
    let image_url = result["images"][0]["url"]
        .as_str()
        .ok_or("fal.ai returned no image")?;
    let bytes = match image_url.strip_prefix("data:") {
        Some(rest) => decode_data_uri(rest)?,
        // A public CDN link: no key for it.
        None => bytes_response(agent.get(image_url).call())?,
    };
    p.check()?;
    Ok((jpeg, bytes))
}

/// One OpenAI-style chat completion with the image attached. It can't be aborted halfway; a
/// cancelled request just has its answer dropped. Returns the JPEG sent and the answer.
fn run_deepseek(
    p: &Progress,
    image: &RgbImage,
    grayscale: bool,
    s: &TextSettings,
) -> Result<(Vec<u8>, String), String> {
    let key = api_key(&s.api_key, "DEEPSEEK_API_KEY", "DeepSeek")?;
    let jpeg = encode_jpeg(image, grayscale, TEXT_UPLOAD_SIDE)?;
    let upload = data_uri(&jpeg);
    p.check()?;
    p.set(if s.thinking {
        "Thinking…"
    } else {
        "Waiting for the answer…"
    });
    let body = json!({
        "model": s.model.trim(),
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": s.prompt },
                { "type": "image_url", "image_url": { "url": upload } },
            ],
        }],
        "thinking": { "type": if s.thinking { "enabled" } else { "disabled" } },
        "stream": false,
    });
    let response = json_response(
        agent()
            .post(DEEPSEEK_CHAT)
            .header("Authorization", format!("Bearer {key}"))
            .send_json(&body),
    )?;
    Ok((jpeg, chat_answer(&response)?))
}

fn chat_answer(response: &Value) -> Result<String, String> {
    let choice = &response["choices"][0];
    match choice["message"]["content"].as_str().map(str::trim) {
        Some(text) if !text.is_empty() => Ok(text.to_string()),
        _ => Err(match choice["finish_reason"].as_str() {
            Some(reason) => format!("the model returned no text (finish reason: {reason})"),
            None => "the model returned no text".into(),
        }),
    }
}

/// The key typed into the settings, else the environment variable.
fn api_key(typed: &str, env: &str, service: &str) -> Result<String, String> {
    let typed = typed.trim();
    if !typed.is_empty() {
        return Ok(typed.to_string());
    }
    std::env::var(env)
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .ok_or_else(|| {
            format!("No {service} API key. Enter it under Inference on the left, or set ${env}.")
        })
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(REQUEST_TIMEOUT))
        // Error responses carry the message worth showing in their body.
        .http_status_as_error(false)
        .user_agent(concat!("overheadcamera/", env!("CARGO_PKG_VERSION")))
        .build()
        .into()
}

fn bytes_response(
    response: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
) -> Result<Vec<u8>, String> {
    let mut response = response.map_err(|e| format!("network error: {e}"))?;
    let status = response.status().as_u16();
    let bytes = response
        .body_mut()
        .with_config()
        .limit(MAX_RESPONSE_BYTES)
        .read_to_vec()
        .map_err(|e| format!("network error: {e}"))?;
    if !(200..300).contains(&status) {
        return Err(api_error(status, &String::from_utf8_lossy(&bytes)));
    }
    Ok(bytes)
}

fn json_response(
    response: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
) -> Result<Value, String> {
    let bytes = bytes_response(response)?;
    serde_json::from_slice(&bytes).map_err(|e| format!("unexpected response: {e}"))
}

/// A short message for an HTTP error, taken from the JSON body if it has one.
fn api_error(status: u16, body: &str) -> String {
    let json: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let message = [
        &json["error"]["message"],
        &json["error"],
        &json["detail"],
        &json["message"],
    ]
    .into_iter()
    .find_map(|v| v.as_str())
    .map(String::from)
    .or_else(|| {
        // FastAPI-style validation errors: `{"detail": [{"msg": …}, …]}`.
        let msgs: Vec<&str> = json["detail"]
            .as_array()?
            .iter()
            .filter_map(|d| d["msg"].as_str())
            .collect();
        (!msgs.is_empty()).then(|| msgs.join("; "))
    })
    .unwrap_or_else(|| body.trim().chars().take(300).collect());
    let hint = match status {
        401 | 403 => " (check the API key)",
        402 => " (check the account balance)",
        429 => " (too many requests; try again later)",
        _ => "",
    };
    format!("HTTP {status}: {message}{hint}")
}

/// `image` as JPEG with the long side at most `max_side`.
fn encode_jpeg(image: &RgbImage, grayscale: bool, max_side: u32) -> Result<Vec<u8>, String> {
    let (w, h) = (image.width as u32, image.height as u32);
    let mut img = if grayscale {
        let luma = image.data.as_chunks::<3>().0.iter().map(|p| p[0]).collect();
        image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(w, h, luma).unwrap())
    } else {
        image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, image.data.clone()).unwrap())
    };
    if w.max(h) > max_side {
        img = img.resize(max_side, max_side, image::imageops::FilterType::CatmullRom);
    }
    let mut out = Vec::new();
    // Not `encode_image`, which would see a `DynamicImage` as RGBA and turn gray into color.
    img.write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(
        &mut out,
        UPLOAD_JPEG_QUALITY,
    ))
    .map_err(|e| format!("could not encode the image: {e}"))?;
    Ok(out)
}

fn data_uri(jpeg: &[u8]) -> String {
    format!("data:image/jpeg;base64,{}", BASE64.encode(jpeg))
}

/// The bytes of a data URI, given without its `data:` prefix.
fn decode_data_uri(rest: &str) -> Result<Vec<u8>, String> {
    let (_, payload) = rest
        .split_once(";base64,")
        .ok_or("unsupported data URI in the response")?;
    BASE64
        .decode(payload)
        .map_err(|e| format!("invalid data URI in the response: {e}"))
}

fn decode_overlay(bytes: &[u8]) -> Result<ColorImage, String> {
    let img = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| format!("could not read the generated image: {e}"))?;
    let img = if img.width().max(img.height()) > OVERLAY_MAX_SIDE {
        img.resize(
            OVERLAY_MAX_SIDE,
            OVERLAY_MAX_SIDE,
            image::imageops::FilterType::Triangle,
        )
    } else {
        img
    };
    let rgba = img.to_rgba8();
    Ok(ColorImage::from_rgba_unmultiplied(
        [rgba.width() as usize, rgba.height() as usize],
        rgba.as_raw(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_errors_show_the_message() {
        assert_eq!(
            api_error(
                401,
                r#"{"error":{"message":"Authentication Fails","type":"auth"}}"#
            ),
            "HTTP 401: Authentication Fails (check the API key)"
        );
        assert_eq!(
            api_error(
                422,
                r#"{"detail":[{"msg":"field required"},{"msg":"bad url"}]}"#
            ),
            "HTTP 422: field required; bad url"
        );
        assert_eq!(
            api_error(403, r#"{"detail":"No access"}"#),
            "HTTP 403: No access (check the API key)"
        );
        assert_eq!(api_error(502, "Bad Gateway\n"), "HTTP 502: Bad Gateway");
    }

    #[test]
    fn upload_is_a_small_jpeg_data_uri() {
        let mut img = RgbImage::new(4000, 1000);
        img.data.fill(180);
        let jpeg = encode_jpeg(&img, true, 2048).unwrap();
        let uri = data_uri(&jpeg);
        let back = decode_data_uri(uri.strip_prefix("data:").unwrap()).unwrap();
        assert_eq!(back, jpeg);
        let decoded = image::load_from_memory(&back).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (2048, 512));
        // A grayscale result goes up with one channel: the frame header (SOF0) says so.
        let sof = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        assert_eq!(jpeg[sof + 9], 1);
    }

    #[test]
    fn overlay_is_capped_and_rgba() {
        let mut png = Vec::new();
        image::DynamicImage::new_rgb8(5000, 100)
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let overlay = decode_overlay(&png).unwrap();
        assert_eq!(overlay.size, [OVERLAY_MAX_SIDE as usize, 82]);
    }

    #[test]
    fn runs_are_saved_in_their_own_directories() {
        use chrono::TimeZone;
        let parent =
            std::env::temp_dir().join(format!("overheadcamera-runs-{}", std::process::id()));
        let started = Local.with_ymd_and_hms(2026, 11, 3, 18, 30, 5).unwrap();
        let save = |output: &[u8]| {
            save_run(
                &parent,
                started,
                "image-to-text",
                b"jpeg",
                "Describe it",
                ("output.txt", output),
            )
            .unwrap()
        };
        let first = save(b"A cat.");
        assert_eq!(first, parent.join("2026-11-03_18_30_05_image-to-text"));
        assert_eq!(std::fs::read(first.join("input.jpg")).unwrap(), b"jpeg");
        assert_eq!(
            std::fs::read_to_string(first.join("prompt.txt")).unwrap(),
            "Describe it"
        );
        assert_eq!(
            std::fs::read_to_string(first.join("output.txt")).unwrap(),
            "A cat."
        );
        // Within the same second: numbered, the first run stays as it was.
        let second = save(b"A dog.");
        assert_eq!(second, parent.join("2026-11-03_18_30_05_image-to-text_1"));
        assert_eq!(
            std::fs::read_to_string(first.join("output.txt")).unwrap(),
            "A cat."
        );
        std::fs::remove_dir_all(&parent).unwrap();
    }

    #[test]
    fn chat_answer_reads_the_content() {
        let ok =
            json!({"choices": [{"message": {"content": "  A cat.\n"}, "finish_reason": "stop"}]});
        assert_eq!(chat_answer(&ok).unwrap(), "A cat.");
        let empty = json!({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]});
        assert!(chat_answer(&empty).unwrap_err().contains("length"));
    }

    #[test]
    fn dropping_a_job_cancels_it() {
        let ctx = egui::Context::default();
        let (seen_tx, seen_rx) = channel();
        let job = Job::spawn(ctx, move |p| {
            let r = p.sleep(Duration::from_secs(30));
            seen_tx.send(r.clone()).unwrap();
            r
        });
        drop(job);
        let seen = seen_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(seen, Err(CANCELLED.to_string()));
    }

    /// Waits for `job`, printing its progress.
    fn finish<T>(job: Job<T>) -> Result<T, String> {
        loop {
            if let Some(result) = job.try_take() {
                return result;
            }
            println!("{} ({} s)", job.stage(), job.elapsed().as_secs());
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    /// Runs both modes against the real services, each only if its key is set. Costs a little
    /// money. `FAL_KEY=… DEEPSEEK_API_KEY=… cargo test -- --ignored --nocapture inference_smoke`
    #[test]
    #[ignore]
    fn inference_smoke() {
        // A dark cross on light paper.
        let (w, h) = (800, 600);
        let mut img = RgbImage::new(w, h);
        img.data.fill(245);
        for y in 0..h {
            for x in 0..w {
                if (100..700).contains(&x) && (295..305).contains(&y)
                    || (395..405).contains(&x) && (100..500).contains(&y)
                {
                    img.data[(y * w + x) * 3..][..3].fill(30);
                }
            }
        }
        let img = Arc::new(img);
        let ctx = egui::Context::default();
        let save_in = std::env::temp_dir().join("overheadcamera-inference-smoke");
        if std::env::var("DEEPSEEK_API_KEY").is_ok() {
            let settings = TextSettings::default();
            let job = image_to_text(ctx.clone(), img.clone(), true, settings, save_in.clone());
            let run = finish(job).unwrap();
            println!("image-to-text: {}", run.output);
            println!("saved in {}", run.saved.unwrap().display());
        }
        if std::env::var("FAL_KEY").is_ok() {
            let settings = ImageSettings::default();
            let run = finish(image_to_image(ctx, img, true, settings, save_in)).unwrap();
            println!(
                "image-to-image: {}×{}",
                run.output.size[0], run.output.size[1]
            );
            println!("saved in {}", run.saved.unwrap().display());
        }
    }
}
