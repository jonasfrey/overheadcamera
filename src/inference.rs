//! Optional inference on the result: image-to-image (the sketch redrawn by an image model, on
//! fal.ai or by a local Python model) and image-to-text (a description by a vision model on
//! DeepSeek). Each run is a background job that the UI polls; dropping the job cancels it.
//!
//! The local model runs in a Python process ([`local_img2img.py`](LOCAL_WORKER_SOURCE)) that
//! the app starts on the first run and keeps, with the model loaded, until it is stopped; see
//! [`LocalWorker`].

use std::{
    collections::VecDeque,
    io::{BufRead, BufReader, Cursor, ErrorKind, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc::{Receiver, RecvTimeoutError, TryRecvError, channel},
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

/// Stable Diffusion's text encoder keeps only the first 77 tokens, so this is short.
pub const DEFAULT_LOCAL_PROMPT: &str = "\
A Leonardo da Vinci notebook drawing: sepia iron-gall ink and red chalk on aged, yellowed \
paper, fine hatching and cross-hatching, delicate contour lines, Renaissance study sheet";

pub const DEFAULT_TEXT_PROMPT: &str = "\
This is a scan of a hand-drawn sketch on paper. Describe what it shows and transcribe all \
handwritten text, labels and numbers exactly as written. Answer in plain text without Markdown.";

/// The Python worker that runs the local model; written next to the app's cache and started
/// with the user's Python.
const LOCAL_WORKER_SOURCE: &str = include_str!("local_img2img.py");
const LOCAL_WORKER_FILE: &str = "local_img2img.py";
pub const DEFAULT_LOCAL_MODEL: &str = "stabilityai/sdxl-turbo";
/// What the worker needs, installed into the app's own environment.
pub const LOCAL_PACKAGES: [&str; 5] =
    ["torch", "diffusers", "transformers", "accelerate", "pillow"];
/// PyTorch's wheels without CUDA, for machines without an NVIDIA driver.
const TORCH_CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";
/// How long Python may take to start and report that it's ready (no model loading yet).
const WORKER_START_TIMEOUT: Duration = Duration::from_secs(60);
/// How often a waiting run looks at its cancel flag.
const WORKER_POLL: Duration = Duration::from_millis(50);
/// Last lines of the worker's stderr, shown when it dies.
const STDERR_TAIL_LINES: usize = 12;

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

/// Where image-to-image runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageBackend {
    /// fal.ai's queue API.
    Fal,
    /// A diffusers model in a local Python process.
    Local,
}

impl ImageBackend {
    pub const ALL: [ImageBackend; 2] = [ImageBackend::Fal, ImageBackend::Local];

    pub fn label(self) -> &'static str {
        match self {
            ImageBackend::Fal => "Online (fal.ai)",
            ImageBackend::Local => "Local (Python, SDXL-Turbo)",
        }
    }
}

/// Settings of the local model; what `local_img2img.py` is sent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalSettings {
    /// Empty: the app's own environment, installed on first use. A bare command name such as
    /// `python3.11`: the same, created with that interpreter. A path (`~` allowed): that
    /// interpreter as it is, which must have torch and diffusers installed.
    pub python: String,
    /// Hugging Face model id or local directory of a diffusers image-to-image pipeline.
    pub model: String,
    /// `auto`, `cuda`, `cuda:1`, `mps` or `cpu`.
    pub device: String,
    pub prompt: String,
    /// How far the result may depart from the input: 0.1 keeps it almost as it is, 0.9 draws an
    /// almost new image.
    pub strength: f32,
    /// Inference steps; the worker raises it so that `steps * strength >= 1`.
    pub steps: u32,
    /// Random seed; negative picks a new one each run.
    pub seed: i64,
    /// Long side of the image the model works on, in pixels. SDXL-Turbo is made for 512.
    pub size: u32,
}

impl Default for LocalSettings {
    fn default() -> Self {
        Self {
            python: String::new(),
            model: DEFAULT_LOCAL_MODEL.into(),
            device: "auto".into(),
            prompt: DEFAULT_LOCAL_PROMPT.into(),
            strength: 0.5,
            steps: 4,
            seed: -1,
            size: 512,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageSettings {
    pub backend: ImageBackend,
    /// fal.ai key; empty uses `$FAL_KEY`.
    pub api_key: String,
    /// fal.ai endpoint that takes `prompt` and `image_urls`.
    pub model: String,
    pub prompt: String,
    pub resolution: Resolution,
    pub local: LocalSettings,
}

impl Default for ImageSettings {
    fn default() -> Self {
        Self {
            backend: ImageBackend::Fal,
            api_key: String::new(),
            model: "google/nano-banana-2.1/edit".into(),
            prompt: DEFAULT_IMAGE_PROMPT.into(),
            resolution: Resolution::R1K,
            local: LocalSettings::default(),
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

/// Redraws `image` with the image model, online or with the local model in `worker` as the
/// settings say. The result is meant to be shown stretched over the input, which it matches in
/// composition and roughly in aspect ratio. A successful run is saved in a new directory in
/// `save_in`, see [`save_run`].
pub fn image_to_image(
    ctx: egui::Context,
    image: Arc<RgbImage>,
    grayscale: bool,
    settings: ImageSettings,
    save_in: PathBuf,
    worker: LocalWorker,
) -> Job<Finished<ColorImage>> {
    let started = Local::now();
    Job::spawn(ctx, move |p| {
        let ((input, output), prompt) = match settings.backend {
            ImageBackend::Fal => (run_fal(p, &image, grayscale, &settings)?, &settings.prompt),
            ImageBackend::Local => (
                worker.run(p, &image, grayscale, &settings.local)?,
                &settings.local.prompt,
            ),
        };
        let ext = image::guess_format(&output).map_or("png", |f| f.extensions_str()[0]);
        // Saved before decoding, so that the paid-for image is kept even if it can't be shown.
        let saved = save_run(
            &save_in,
            started,
            "image-to-image",
            &input,
            prompt,
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

/// The local image model's Python process, shared by all runs. Started on the first run and
/// kept, with the model loaded, until [`LocalWorker::stop`] or the app ends. Clones share the
/// process.
///
/// Unless the settings name a Python interpreter, the worker runs in an environment of its own
/// (a venv under the app's data directory) that it creates, and installs the packages into, the
/// first time it is needed.
#[derive(Clone)]
pub struct LocalWorker {
    process: Arc<Mutex<Option<WorkerProcess>>>,
    /// PID of the process a run is waiting on (the worker, or pip while installing), 0 if none;
    /// for stopping it while the run holds the lock.
    pid: Arc<AtomicU32>,
    /// Where the worker script is written; Python also looks here first for modules.
    script_dir: PathBuf,
    /// The app's own Python environment.
    env_dir: PathBuf,
    /// The system interpreter that creates the environment.
    base_python: String,
    /// What gets installed into the environment.
    packages: Vec<String>,
}

impl Default for LocalWorker {
    fn default() -> Self {
        Self::in_dirs(default_script_dir(), default_env_dir())
    }
}

impl LocalWorker {
    pub fn in_dirs(script_dir: PathBuf, env_dir: PathBuf) -> Self {
        Self {
            process: Arc::new(Mutex::new(None)),
            pid: Arc::new(AtomicU32::new(0)),
            script_dir,
            env_dir,
            base_python: "python3".into(),
            packages: LOCAL_PACKAGES.iter().map(|&s| s.into()).collect(),
        }
    }

    /// Whether the Python process exists (which is where the model's memory goes).
    pub fn is_running(&self) -> bool {
        self.pid.load(Ordering::Relaxed) != 0
    }

    /// The app's own Python environment, where the packages go unless an interpreter is set.
    pub fn env_dir(&self) -> &Path {
        &self.env_dir
    }

    pub fn env_exists(&self) -> bool {
        self.env_python().exists()
    }

    fn env_python(&self) -> PathBuf {
        self.env_dir.join("bin/python")
    }

    /// Ends the process, freeing the memory the model takes. A run in progress fails; the next
    /// run starts a new process and loads the model again.
    pub fn stop(&self) {
        if let Ok(mut guard) = self.process.try_lock() {
            // Dropping the process kills it.
            *guard = None;
            self.pid.store(0, Ordering::Relaxed);
        } else {
            // A run holds the lock: it notices the death and clears the process itself.
            let pid = self.pid.load(Ordering::Relaxed);
            if pid != 0 {
                // SAFETY: a plain system call with a PID the app spawned itself.
                unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
            }
        }
    }

    /// Stops the model and deletes the app's own environment, so the next run installs it afresh
    /// (e.g. after a driver change). Deleting the gigabytes happens in the background.
    pub fn remove_env(&self) -> Result<(), String> {
        self.stop();
        if !self.env_dir.exists() {
            return Ok(());
        }
        let doomed = self.env_dir.with_file_name(format!(
            "{}.deleting-{}",
            self.env_dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            std::process::id()
        ));
        std::fs::rename(&self.env_dir, &doomed)
            .map_err(|e| format!("{}: {e}", self.env_dir.display()))?;
        std::thread::spawn(move || {
            if let Err(e) = std::fs::remove_dir_all(&doomed) {
                eprintln!("could not delete {}: {e}", doomed.display());
            }
        });
        Ok(())
    }

    /// Runs the model on `image`: starts Python if it isn't running (or the interpreter was
    /// changed), then sends the request and relays the worker's progress to `p`. Returns the
    /// JPEG sent and the PNG received.
    fn run(
        &self,
        p: &Progress,
        image: &RgbImage,
        grayscale: bool,
        s: &LocalSettings,
    ) -> Result<(Vec<u8>, Vec<u8>), String> {
        if s.model.trim().is_empty() {
            return Err("No local model set".into());
        }
        // Twice the model's size: the worker resamples to the exact size it needs anyway, and a
        // single downscale from here keeps the lines cleaner than two.
        let jpeg = encode_jpeg(image, grayscale, s.size.clamp(64, 2048) * 2)?;
        p.check()?;
        p.set("Waiting for the previous run to finish…");
        let mut guard = self.process.lock().unwrap_or_else(|e| e.into_inner());
        p.check()?;
        // A path names the interpreter to use as it is; a bare command name (or nothing) is
        // the system Python that the app's own environment gets created with.
        let typed = s.python.trim();
        let managed = !typed.contains('/');
        let python = if managed {
            let base = if typed.is_empty() {
                &self.base_python
            } else {
                typed
            };
            self.ensure_env(p, base)?
        } else {
            expand_home(typed)
        };
        let result = match self.generate(&mut guard, p, &python, &jpeg, s) {
            Err(e) if e.missing_packages && managed => {
                // The packages aren't there (first use, or a half-done install): install them
                // and try once more with a fresh process.
                *guard = None;
                self.install_packages(p)?;
                self.generate(&mut guard, p, &python, &jpeg, s)
            }
            result => result,
        };
        let output = result.map_err(|e| {
            if e.missing_packages && !managed {
                format!(
                    "{} Or clear the Python field under Inference, and the app installs them \
                     into an environment of its own (a bare command name such as python3 \
                     creates that environment with it).",
                    e.message
                )
            } else {
                e.message
            }
        })?;
        Ok((jpeg, output))
    }

    /// Starts the worker in `python` unless it is running already, and runs the request.
    fn generate(
        &self,
        guard: &mut Option<WorkerProcess>,
        p: &Progress,
        python: &str,
        jpeg: &[u8],
        s: &LocalSettings,
    ) -> Result<Vec<u8>, GenerateError> {
        if guard
            .as_mut()
            .is_none_or(|w| !w.alive() || w.python != python)
        {
            *guard = None;
            self.pid.store(0, Ordering::Relaxed);
            p.set("Starting Python…");
            let worker = WorkerProcess::spawn(python, &self.script_dir, p)?;
            self.pid.store(worker.child.id(), Ordering::Relaxed);
            *guard = Some(worker);
        }
        let worker = guard.as_mut().unwrap();
        let result = worker.generate(p, jpeg, s);
        if result.is_err() && !worker.alive() {
            *guard = None;
            self.pid.store(0, Ordering::Relaxed);
        }
        result
    }

    /// The interpreter of the app's own environment, creating the environment with `base` if
    /// needed.
    fn ensure_env(&self, p: &Progress, base: &str) -> Result<String, String> {
        let python = self.env_python();
        if !python.exists() {
            p.set("Creating a Python environment…");
            if let Some(parent) = self.env_dir.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            let mut cmd = Command::new(base);
            cmd.args(["-m", "venv"]).arg(&self.env_dir);
            let (status, tail) =
                run_relayed(&mut cmd, p, &self.pid, "Creating a Python environment").map_err(
                    |e| {
                        relay_error(
                            e,
                            format!(
                                "could not run `{base} -m venv`: {{e}}. Install Python 3, or \
                                 enter the path of an interpreter under Inference on the left."
                            ),
                        )
                    },
                )?;
            if !status.success() {
                let hint = if tail.contains("ensurepip") {
                    " On Debian or Ubuntu: sudo apt install python3-venv"
                } else {
                    ""
                };
                return Err(format!(
                    "could not create the Python environment in {} ({status}):\n{tail}{hint}",
                    self.env_dir.display()
                ));
            }
        }
        Ok(python.to_string_lossy().into_owned())
    }

    /// `pip install`s the packages into the app's environment, showing pip's output as progress.
    /// Without an NVIDIA driver, torch comes from PyTorch's CPU index first: without the CUDA
    /// libraries it is a tenth of the size.
    fn install_packages(&self, p: &Progress) -> Result<(), String> {
        let stage = "Installing PyTorch and diffusers (a few GB, once)";
        p.set(format!("{stage}…"));
        let cpu_torch = !has_nvidia_driver() && self.packages.iter().any(|p| p == "torch");
        let mut steps: Vec<Vec<&str>> = Vec::new();
        if cpu_torch {
            steps.push(vec!["--index-url", TORCH_CPU_INDEX, "torch"]);
        }
        steps.push(self.packages.iter().map(String::as_str).collect());
        for args in steps {
            let mut cmd = Command::new(self.env_python());
            cmd.args(["-m", "pip", "install", "--disable-pip-version-check"])
                .args(&args);
            let (status, tail) = run_relayed(&mut cmd, p, &self.pid, stage).map_err(|e| {
                relay_error(
                    e,
                    format!("could not run pip in {}: {{e}}", self.env_dir.display()),
                )
            })?;
            if !status.success() {
                return Err(format!(
                    "installing the Python packages failed ({status}):\n{tail}\nInstall them \
                     by hand with `{} -m pip install {}`, or delete the environment under \
                     Inference and try again.",
                    self.env_python().display(),
                    self.packages.join(" ")
                ));
            }
        }
        Ok(())
    }
}

/// The message for a failed [`run_relayed`]: a cancel stays one, anything else fills `{e}` in
/// `template`.
fn relay_error(e: std::io::Error, template: String) -> String {
    if e.to_string() == CANCELLED {
        CANCELLED.into()
    } else {
        template.replace("{e}", &e.to_string())
    }
}

/// Runs `cmd` to its end, showing its latest output line as `{stage}: {line}` and killing it on
/// a cancel. `pid` holds its PID meanwhile. Returns the exit status and the last output lines.
fn run_relayed(
    cmd: &mut Command,
    p: &Progress,
    pid: &AtomicU32,
    stage: &str,
) -> std::io::Result<(std::process::ExitStatus, String)> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    pid.store(child.id(), Ordering::Relaxed);
    let (tx, rx) = channel();
    for reader in [
        Box::new(child.stdout.take().unwrap()) as Box<dyn std::io::Read + Send>,
        Box::new(child.stderr.take().unwrap()),
    ] {
        let tx = tx.clone();
        std::thread::spawn(move || forward_lines(reader, &tx));
    }
    drop(tx);
    let mut tail = VecDeque::new();
    loop {
        if p.check().is_err() {
            let _ = child.kill();
            let _ = child.wait();
            pid.store(0, Ordering::Relaxed);
            return Err(std::io::Error::other(CANCELLED));
        }
        match rx.recv_timeout(WORKER_POLL) {
            Ok(line) => {
                eprintln!("[{stage}] {line}");
                if tail.len() == STDERR_TAIL_LINES {
                    tail.pop_front();
                }
                tail.push_back(line.clone());
                // Bars and such stay on one line.
                let shown: String = line.chars().take(90).collect();
                p.set(format!("{stage}: {shown}"));
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    let status = child.wait();
    pid.store(0, Ordering::Relaxed);
    let tail = tail
        .into_iter()
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    Ok((status?, tail))
}

/// Sends `reader`'s output line by line, where a carriage return (progress bars) ends a line
/// too. Stops when the reader or the receiver is gone.
fn forward_lines(mut reader: impl std::io::Read, tx: &std::sync::mpsc::Sender<String>) {
    let mut buf = [0u8; 4096];
    let mut pending = Vec::new();
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        for &b in &buf[..n] {
            if b == b'\n' || b == b'\r' {
                let line = String::from_utf8_lossy(&pending).trim().to_string();
                pending.clear();
                if !line.is_empty() && tx.send(line).is_err() {
                    return;
                }
            } else {
                pending.push(b);
            }
        }
    }
    let line = String::from_utf8_lossy(&pending).trim().to_string();
    if !line.is_empty() {
        let _ = tx.send(line);
    }
}

fn has_nvidia_driver() -> bool {
    Path::new("/proc/driver/nvidia/version").exists()
}

/// Why a request failed; `missing_packages` when the worker couldn't import torch or diffusers.
struct GenerateError {
    message: String,
    missing_packages: bool,
}

impl From<String> for GenerateError {
    fn from(message: String) -> Self {
        Self {
            message,
            missing_packages: false,
        }
    }
}

/// One Python process running `local_img2img.py`, with a thread each forwarding its stdout
/// (the protocol) and stderr (kept for error messages). Dropping it kills the process.
struct WorkerProcess {
    python: String,
    child: Child,
    stdin: ChildStdin,
    /// Lines of stdout, until the process closes it.
    events: Receiver<String>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
}

impl WorkerProcess {
    fn spawn(python: &str, dir: &Path, p: &Progress) -> Result<Self, String> {
        let failed = |path: &Path, e: std::io::Error| format!("{}: {e}", path.display());
        std::fs::create_dir_all(dir).map_err(|e| failed(dir, e))?;
        let script = dir.join(LOCAL_WORKER_FILE);
        if std::fs::read_to_string(&script).ok().as_deref() != Some(LOCAL_WORKER_SOURCE) {
            std::fs::write(&script, LOCAL_WORKER_SOURCE).map_err(|e| failed(&script, e))?;
        }
        let mut child = Command::new(python)
            .arg(&script)
            .current_dir(dir)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                format!(
                    "could not start `{python}`: {e}. Enter the Python to use under Inference on \
                     the left; it needs torch and diffusers installed."
                )
            })?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, events) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        let tail = stderr_tail.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                eprintln!("[local model] {line}");
                let mut tail = tail.lock().unwrap();
                if tail.len() == STDERR_TAIL_LINES {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
        });
        let mut worker = Self {
            python: python.to_string(),
            child,
            stdin,
            events,
            stderr_tail,
        };
        let deadline = Instant::now() + WORKER_START_TIMEOUT;
        loop {
            p.check()?;
            if Instant::now() > deadline {
                return Err("Python started but did not answer".into());
            }
            let event = match worker.events.recv_timeout(WORKER_POLL) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return Err(worker.died()),
            };
            if event_of(&event)["event"] == "ready" {
                return Ok(worker);
            }
        }
    }

    fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn send(&mut self, command: &Value) -> Result<(), String> {
        writeln!(self.stdin, "{command}")
            .and_then(|()| self.stdin.flush())
            .map_err(|e| format!("could not talk to the local model process: {e}"))
    }

    /// Sends `jpeg` and the settings and waits for the image, relaying the worker's progress.
    /// A cancel is passed on, and the worker's reply to it awaited, so that the next run starts
    /// with a quiet worker.
    fn generate(
        &mut self,
        p: &Progress,
        jpeg: &[u8],
        s: &LocalSettings,
    ) -> Result<Vec<u8>, GenerateError> {
        self.send(&json!({
            "cmd": "generate",
            "image": BASE64.encode(jpeg),
            "prompt": s.prompt,
            "model": s.model.trim(),
            "device": s.device.trim(),
            "size": s.size,
            "strength": s.strength,
            "steps": s.steps,
            "seed": s.seed,
        }))?;
        let mut cancel_sent = false;
        loop {
            if !cancel_sent && p.check().is_err() {
                self.send(&json!({ "cmd": "cancel" }))?;
                cancel_sent = true;
            }
            let line = match self.events.recv_timeout(WORKER_POLL) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return Err(self.died().into()),
            };
            let event = event_of(&line);
            match event["event"].as_str() {
                Some("stage") => p.set(event["text"].as_str().unwrap_or("Working…")),
                Some("progress") => p.set(format!(
                    "Generating step {}/{}…",
                    event["step"], event["steps"]
                )),
                Some("done") => {
                    p.check()?;
                    let image = event["image"].as_str().unwrap_or_default();
                    return BASE64.decode(image).map_err(|e| {
                        format!("the local model sent an unreadable image: {e}").into()
                    });
                }
                Some("error") => {
                    let message = event["message"].as_str().unwrap_or("unknown error");
                    return Err(GenerateError {
                        message: if cancel_sent && message == CANCELLED {
                            CANCELLED.into()
                        } else {
                            message.into()
                        },
                        missing_packages: event["missing"] == true,
                    });
                }
                _ => {}
            }
        }
    }

    /// Why the process is gone: its exit status and the last of what it printed.
    fn died(&mut self) -> String {
        let status = match self.child.wait() {
            Ok(status) => status.to_string(),
            Err(e) => format!("unknown status: {e}"),
        };
        let tail = self.stderr_tail.lock().unwrap();
        let tail = tail.iter().cloned().collect::<Vec<_>>().join("\n");
        if tail.trim().is_empty() {
            format!("the local model process stopped ({status})")
        } else {
            format!("the local model process stopped ({status}):\n{tail}")
        }
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A line of the worker's stdout as JSON; `Null` for anything else.
fn event_of(line: &str) -> Value {
    serde_json::from_str(line).unwrap_or(Value::Null)
}

/// `~/.cache/overheadcamera` (or `$XDG_CACHE_HOME`), else the temp dir.
fn default_script_dir() -> PathBuf {
    xdg_dir("XDG_CACHE_HOME", ".cache").join("overheadcamera")
}

/// `~/.local/share/overheadcamera/venv` (or under `$XDG_DATA_HOME`): next to the app's settings.
fn default_env_dir() -> PathBuf {
    xdg_dir("XDG_DATA_HOME", ".local/share").join("overheadcamera/venv")
}

fn xdg_dir(var: &str, under_home: &str) -> PathBuf {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(under_home)))
        .unwrap_or_else(std::env::temp_dir)
}

fn expand_home(path: &str) -> String {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home)
            .join(rest)
            .to_string_lossy()
            .into_owned(),
        _ => path.to_string(),
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
        let img = Arc::new(test_sketch());
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
            let worker = LocalWorker::default();
            let run = finish(image_to_image(ctx, img, true, settings, save_in, worker)).unwrap();
            println!(
                "image-to-image: {}×{}",
                run.output.size[0], run.output.size[1]
            );
            println!("saved in {}", run.saved.unwrap().display());
        }
    }

    /// The real local model. Downloads SDXL-Turbo on first use, into the app's own environment unless
    /// `$OVERHEAD_TEST_PYTHON` names an interpreter.
    /// `cargo test --release -- --ignored --nocapture local_model_smoke`
    #[test]
    #[ignore]
    fn local_model_smoke() {
        let img = Arc::new(test_sketch());
        let ctx = egui::Context::default();
        let save_in = std::env::temp_dir().join("overheadcamera-inference-smoke");
        let settings = ImageSettings {
            backend: ImageBackend::Local,
            local: LocalSettings {
                python: std::env::var("OVERHEAD_TEST_PYTHON").unwrap_or_default(),
                ..LocalSettings::default()
            },
            ..ImageSettings::default()
        };
        let worker = LocalWorker::default();
        let job = image_to_image(ctx, img, true, settings, save_in, worker.clone());
        let run = finish(job).unwrap();
        println!("local: {}×{}", run.output.size[0], run.output.size[1]);
        println!("saved in {}", run.saved.unwrap().display());
        worker.stop();
    }

    /// A dark cross on light paper.
    fn test_sketch() -> RgbImage {
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
        img
    }

    /// Fake `torch`, `diffusers` and `PIL` packages, so the worker script runs without the real
    /// ones. The pipeline returns the input image after calling the step callback `steps`
    /// times, sleeping `step_ms` per step; the fake PIL reads a JPEG's size and writes a gray
    /// PNG of the resized size.
    fn stub_modules(dir: &Path, step_ms: u32) {
        std::fs::create_dir_all(dir.join("torch/backends")).unwrap();
        std::fs::create_dir_all(dir.join("diffusers")).unwrap();
        std::fs::create_dir_all(dir.join("PIL")).unwrap();
        std::fs::write(dir.join("PIL/__init__.py"), "").unwrap();
        std::fs::write(
            dir.join("PIL/Image.py"),
            "import struct, zlib\n\
             LANCZOS = 1\n\
             def _jpeg_size(data):\n\
             \x20   i = 2\n\
             \x20   while i < len(data):\n\
             \x20       if data[i] != 0xFF: i += 1; continue\n\
             \x20       m = data[i + 1]\n\
             \x20       if m in (0xC0, 0xC1, 0xC2):\n\
             \x20           h, w = struct.unpack('>HH', data[i + 5:i + 9]); return (w, h)\n\
             \x20       if m == 0xD8 or 0xD0 <= m <= 0xD7: i += 2; continue\n\
             \x20       i += 2 + struct.unpack('>H', data[i + 2:i + 4])[0]\n\
             \x20   raise ValueError('not a JPEG')\n\
             def _chunk(t, d):\n\
             \x20   return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d))\n\
             def _png(w, h):\n\
             \x20   raw = b''.join(b'\\x00' + b'\\x80' * (w * 3) for _ in range(h))\n\
             \x20   return (b'\\x89PNG\\r\\n\\x1a\\n' + _chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))\n\
             \x20       + _chunk(b'IDAT', zlib.compress(raw)) + _chunk(b'IEND', b''))\n\
             class _Img:\n\
             \x20   def __init__(self, size): self.size = tuple(size)\n\
             \x20   width = property(lambda s: s.size[0])\n\
             \x20   height = property(lambda s: s.size[1])\n\
             \x20   def convert(self, mode): return self\n\
             \x20   def resize(self, size, resample=None): return _Img(size)\n\
             \x20   def save(self, out, format=None): out.write(_png(*self.size))\n\
             def open(f): return _Img(_jpeg_size(f.read()))\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("torch/__init__.py"),
            "float16 = 'f16'\nfloat32 = 'f32'\n\
             class cuda:\n    @staticmethod\n    def is_available(): return False\n\
             \x20   @staticmethod\n    def empty_cache(): pass\n\
             class backends:\n    class mps:\n        @staticmethod\n        def is_available(): return False\n\
             class Generator:\n    def __init__(self, device=None): pass\n\
             \x20   def manual_seed(self, seed): self.seed = seed; return self\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("diffusers/__init__.py"),
            format!(
                "import sys, time\n\
                 class _Out:\n    def __init__(self, images): self.images = images\n\
                 class AutoPipelineForImage2Image:\n\
                 \x20   @classmethod\n\
                 \x20   def from_pretrained(cls, model, **kw):\n\
                 \x20       if 'variant' in kw: raise OSError('no fp16 variant')\n\
                 \x20       print('loading', model, kw, file=sys.stderr)\n\
                 \x20       return cls()\n\
                 \x20   def to(self, device): return self\n\
                 \x20   def set_progress_bar_config(self, **kw): pass\n\
                 \x20   def __call__(self, prompt, image, strength, num_inference_steps, \
                 guidance_scale, generator, callback_on_step_end):\n\
                 \x20       steps = max(int(num_inference_steps * strength), 1)\n\
                 \x20       for i in range(steps):\n\
                 \x20           time.sleep({step_ms} / 1000)\n\
                 \x20           callback_on_step_end(self, i, 0, {{}})\n\
                 \x20       return _Out([image])\n"
            ),
        )
        .unwrap();
    }

    fn progress() -> Progress {
        Progress {
            cancel: Arc::new(AtomicBool::new(false)),
            stage: Arc::new(Mutex::new(String::new())),
            ctx: egui::Context::default(),
        }
    }

    /// A fresh directory for a test, with a worker whose script and environment go there.
    fn test_worker(name: &str) -> (PathBuf, LocalWorker) {
        let dir = std::env::temp_dir().join(format!(
            "overheadcamera-worker-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let worker = LocalWorker::in_dirs(dir.clone(), dir.join("env"));
        (dir, worker)
    }

    /// Settings that use the system Python by its path (where the stubs next to the script are
    /// found) instead of an environment of the app's own.
    fn system_python() -> LocalSettings {
        let out = Command::new("sh")
            .args(["-c", "command -v python3"])
            .output()
            .unwrap();
        LocalSettings {
            python: String::from_utf8(out.stdout).unwrap().trim().to_string(),
            ..LocalSettings::default()
        }
    }

    /// A wheel holding the stub packages, built with Python's zipfile module.
    fn stub_wheel(dir: &Path) -> PathBuf {
        let src = dir.join("wheel-src");
        stub_modules(&src, 0);
        let info = src.join("ocstub-0.1.dist-info");
        std::fs::create_dir_all(&info).unwrap();
        std::fs::write(
            info.join("METADATA"),
            "Metadata-Version: 2.1\nName: ocstub\nVersion: 0.1\n",
        )
        .unwrap();
        std::fs::write(
            info.join("WHEEL"),
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        .unwrap();
        std::fs::write(info.join("RECORD"), "").unwrap();
        let wheel = dir.join("ocstub-0.1-py3-none-any.whl");
        let status = Command::new("python3")
            .args(["-m", "zipfile", "-c"])
            .arg(&wheel)
            .args(["torch", "diffusers", "PIL", "ocstub-0.1.dist-info"])
            .current_dir(&src)
            .status()
            .unwrap();
        assert!(status.success());
        wheel
    }

    #[test]
    fn local_worker_runs_the_script_and_keeps_the_process() {
        let (dir, worker) = test_worker("run");
        stub_modules(&dir, 0);
        let settings = LocalSettings {
            seed: 7,
            ..system_python()
        };
        let p = progress();
        let img = test_sketch();

        let (sent, received) = worker.run(&p, &img, true, &settings).unwrap();
        assert!(worker.is_running());
        let pid = worker.pid.load(Ordering::Relaxed);
        let sent = image::load_from_memory(&sent).unwrap();
        assert_eq!((sent.width(), sent.height()), (800, 600));
        // The stub hands back the fitted input: 800×600 scaled to 512, rounded to 64s.
        let received = image::load_from_memory(&received).unwrap();
        assert_eq!((received.width(), received.height()), (512, 384));
        assert_eq!(p.stage.lock().unwrap().as_str(), "Generating step 2/2…");

        // Another run reuses the process and the loaded model.
        worker.run(&p, &img, false, &settings).unwrap();
        assert_eq!(worker.pid.load(Ordering::Relaxed), pid);

        // Checked before anything is sent.
        let wrong = LocalSettings {
            model: String::new(),
            ..settings.clone()
        };
        assert_eq!(
            worker.run(&p, &img, true, &wrong).unwrap_err(),
            "No local model set"
        );

        worker.stop();
        assert!(!worker.is_running());
        // A changed interpreter path or a stopped process starts a new one.
        worker.run(&p, &img, true, &settings).unwrap();
        assert_ne!(worker.pid.load(Ordering::Relaxed), pid);
        worker.stop();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn local_worker_cancels_a_run_and_stays_usable() {
        let (dir, worker) = test_worker("cancel");
        stub_modules(&dir, 300);
        let settings = LocalSettings {
            steps: 20,
            strength: 0.5,
            ..system_python()
        };
        let img = test_sketch();

        let p = progress();
        let cancel = p.cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(500));
            cancel.store(true, Ordering::Relaxed);
        });
        let started = Instant::now();
        assert_eq!(
            worker.run(&p, &img, true, &settings).unwrap_err(),
            CANCELLED
        );
        // Stopped after a step or two, not after all ten.
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(worker.is_running());

        // The next run on the same process succeeds.
        let quick = LocalSettings {
            steps: 2,
            ..settings
        };
        worker.run(&progress(), &img, true, &quick).unwrap();
        worker.stop();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn local_worker_explains_missing_packages_and_a_dead_process() {
        let (dir, worker) = test_worker("missing");
        // A `torch` that can't be imported stands in for one that isn't installed.
        std::fs::write(
            dir.join("torch.py"),
            "raise ImportError('No module named torch')",
        )
        .unwrap();
        let settings = system_python();
        let img = test_sketch();
        let error = worker.run(&progress(), &img, true, &settings).unwrap_err();
        assert!(error.contains("pip install torch diffusers"), "{error}");
        // With an interpreter of the user's, nothing is installed; the way out is named.
        assert!(error.contains("clear the Python field"), "{error}");
        assert!(worker.is_running());
        assert!(!worker.env_exists());

        // Killed from outside (as `stop` does during a run): the run fails with the reason and
        // the next run starts a fresh process.
        let pid = worker.pid.load(Ordering::Relaxed);
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
        let error = worker.run(&progress(), &img, true, &settings).unwrap_err();
        assert!(error.contains("pip install torch diffusers"), "{error}");
        assert_ne!(worker.pid.load(Ordering::Relaxed), pid);

        let no_python = LocalSettings {
            python: "/nonexistent/python".into(),
            ..settings
        };
        let error = worker.run(&progress(), &img, true, &no_python).unwrap_err();
        assert!(
            error.starts_with("could not start `/nonexistent/python`"),
            "{error}"
        );
        assert!(!worker.is_running());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// With no interpreter set, the first run creates the app's environment, finds the packages
    /// missing, installs them (here: the stub wheel) and runs in the new environment.
    #[test]
    fn local_worker_sets_up_its_own_environment() {
        let (dir, mut worker) = test_worker("env");
        let wheel = stub_wheel(&dir);
        worker.packages = vec![wheel.to_string_lossy().into_owned()];
        // A bare command name (as earlier versions saved by default) means the app's own
        // environment, created with that interpreter.
        let settings = LocalSettings {
            python: "python3".into(),
            ..LocalSettings::default()
        };
        let img = test_sketch();
        assert!(!worker.env_exists());

        let p = progress();
        let (_, received) = worker.run(&p, &img, true, &settings).unwrap();
        assert!(worker.env_exists());
        let received = image::load_from_memory(&received).unwrap();
        assert_eq!((received.width(), received.height()), (512, 384));
        let process = worker.process.lock().unwrap();
        let python = process.as_ref().unwrap().python.clone();
        drop(process);
        assert_eq!(python, worker.env_python().to_string_lossy());
        assert!(dir.join("env/lib").exists());

        // The second run just uses the environment.
        worker.run(&p, &img, true, &settings).unwrap();

        // Deleting the environment stops the model; the directory is gone at once.
        worker.remove_env().unwrap();
        assert!(!worker.is_running());
        assert!(!worker.env_exists());
        assert!(!dir.join("env").exists());
        // Wait for the background deletion, so that the test directory can go.
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline
            && std::fs::read_dir(&dir).unwrap().any(|e| {
                e.unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains("deleting")
            })
        {
            std::thread::sleep(Duration::from_millis(50));
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A cancel while pip runs kills it and reports a cancel, not a failure.
    #[test]
    fn local_worker_cancels_an_install() {
        let (dir, worker) = test_worker("env-cancel");
        // Missing packages, and an install of a wheel that doesn't exist: pip would fail, but
        // it is cancelled before it can.
        let settings = LocalSettings::default();
        let img = test_sketch();
        let p = progress();
        let cancel = p.cancel.clone();
        let stage = p.stage.clone();
        std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(60);
            while Instant::now() < deadline
                && !stage.lock().unwrap().starts_with("Installing PyTorch")
            {
                std::thread::sleep(Duration::from_millis(10));
            }
            cancel.store(true, Ordering::Relaxed);
        });
        assert_eq!(
            worker.run(&p, &img, true, &settings).unwrap_err(),
            CANCELLED
        );
        assert!(worker.env_exists());
        assert!(!worker.is_running());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn relayed_output_splits_on_carriage_returns() {
        let (tx, rx) = channel();
        forward_lines(
            Cursor::new(b"Collecting torch\n  10%\r  20%\r done\n\n tail"),
            &tx,
        );
        drop(tx);
        let lines: Vec<String> = rx.iter().collect();
        assert_eq!(lines, ["Collecting torch", "10%", "20%", "done", "tail"]);
    }
}
