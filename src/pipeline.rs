//! Background processing: decoding + stacking of camera frames, and rectification + enhancement.
//!
//! ```text
//! capture ──RawFrame──▶ decode ×2 ──RgbImage──▶ stack ──stacked frame──▶ process ──result──▶ UI
//!                                               (average)                (warp, enhance)
//! ```
//!
//! Decoding and stacking keep up with the camera (a 4K MJPEG frame takes ~20 ms to decode, hence
//! two decoders) so stacking gets every frame; the process thread always works on the newest
//! stacked frame and skips the ones it is too slow for.

use std::{
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use eframe::egui::{self, ColorImage};
use rayon::prelude::*;
use zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};
use zune_jpeg::JpegDecoder;

use crate::{
    camera::{Encoding, RawFrame},
    enhance::{ColorMode, EnhanceSettings, enhance},
    img::{FrameOrientation, RgbImage},
    sheet::SheetSettings,
    stack::Stacker,
    warp::warp,
};

/// Display textures don't need more than this; full resolution is kept for saving.
const PREVIEW_MAX_SIDE: usize = 1920;
const RESULT_MAX_SIDE: usize = 2048;
/// Upper bound on how often the (expensive) process step runs.
const MIN_PROCESS_INTERVAL: Duration = Duration::from_millis(50);
const DECODE_THREADS: usize = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct ProcessSettings {
    /// TL, TR, BR, BL in normalized camera image coordinates.
    pub corners: [[f32; 2]; 4],
    pub sheet: SheetSettings,
    pub enhance: EnhanceSettings,
}

pub struct Processed {
    pub image: Arc<RgbImage>,
    pub display: ColorImage,
    pub grayscale: bool,
}

#[derive(Clone, Default)]
pub struct Stats {
    pub camera_fps: f32,
    pub decode_ms: f32,
    pub stack_ms: f32,
    pub process_ms: f32,
    pub stacked_frames: u32,
    pub decode_errors: u64,
    pub output_size: Option<(usize, usize)>,
}

struct State {
    settings: ProcessSettings,
    settings_gen: u64,
    stack_depth: u32,
    frozen: bool,
    reset_stack: bool,
    orientation: FrameOrientation,
    /// Counts orientation changes; frames are tagged with it so that none turned the old way
    /// get stacked after a change.
    orientation_gen: u64,
    stacked: Option<Arc<RgbImage>>,
    stacked_gen: u64,
    preview: Option<ColorImage>,
    result: Option<Processed>,
    error: Option<String>,
    stats: Stats,
    last_frame_at: Option<Instant>,
    stop: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap()
    }
}

pub struct Pipeline {
    shared: Arc<Shared>,
    frame_tx: SyncSender<RawFrame>,
    threads: Vec<JoinHandle<()>>,
}

impl Pipeline {
    pub fn new(ctx: egui::Context, settings: ProcessSettings, stack_depth: u32) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                settings,
                settings_gen: 0,
                stack_depth,
                frozen: false,
                reset_stack: false,
                orientation: FrameOrientation::default(),
                orientation_gen: 0,
                stacked: None,
                stacked_gen: 0,
                preview: None,
                result: None,
                error: None,
                stats: Stats::default(),
                last_frame_at: None,
                stop: false,
            }),
            wake: Condvar::new(),
        });
        // Small queues: if decoding falls behind, the capture thread drops frames instead of
        // building up latency.
        let (frame_tx, frame_rx) = sync_channel(2);
        let (decoded_tx, decoded_rx) = sync_channel(2);
        let frame_rx = Arc::new(Mutex::new(frame_rx));
        let mut threads: Vec<_> = (0..DECODE_THREADS)
            .map(|i| {
                let shared = shared.clone();
                let frame_rx = frame_rx.clone();
                let decoded_tx = decoded_tx.clone();
                spawn(&format!("decode{i}"), move || {
                    decode_thread(&shared, &frame_rx, &decoded_tx)
                })
            })
            .collect();
        threads.push(spawn("stack", {
            let shared = shared.clone();
            let ctx = ctx.clone();
            move || stack_thread(&shared, &decoded_rx, &ctx)
        }));
        threads.push(spawn("process", {
            let shared = shared.clone();
            move || process_thread(&shared, &ctx)
        }));
        Self {
            shared,
            frame_tx,
            threads,
        }
    }

    /// Where capture threads deliver their frames.
    pub fn frame_sink(&self) -> SyncSender<RawFrame> {
        self.frame_tx.clone()
    }

    pub fn set_settings(&self, settings: &ProcessSettings) {
        let mut st = self.shared.lock();
        if st.settings != *settings {
            st.settings = settings.clone();
            st.settings_gen += 1;
            self.shared.wake.notify_all();
        }
    }

    pub fn set_stack_depth(&self, depth: u32) {
        self.shared.lock().stack_depth = depth;
    }

    pub fn set_frozen(&self, frozen: bool) {
        self.shared.lock().frozen = frozen;
    }

    pub fn reset_stack(&self) {
        self.shared.lock().reset_stack = true;
    }

    /// Turns new frames by `orientation`. `change`, the turn from the previous orientation, is
    /// also applied to the current image right away, so that a frozen image turns as well.
    pub fn set_orientation(&self, orientation: FrameOrientation, change: Option<FrameOrientation>) {
        let (stacked, generation) = {
            let mut st = self.shared.lock();
            if st.orientation == orientation {
                return;
            }
            st.orientation = orientation;
            st.orientation_gen += 1;
            (st.stacked.clone(), st.stacked_gen)
        };
        let (Some(change), Some(stacked)) = (change, stacked) else {
            return;
        };
        let turned = Arc::new(stacked.oriented(change));
        let preview = turned.to_color_image(PREVIEW_MAX_SIDE);
        let mut st = self.shared.lock();
        // Unless a frame with the new orientation has arrived meanwhile.
        if st.stacked_gen == generation {
            st.stacked = Some(turned);
            st.stacked_gen += 1;
            st.preview = Some(preview);
            self.shared.wake.notify_all();
        }
    }

    pub fn take_preview(&self) -> Option<ColorImage> {
        self.shared.lock().preview.take()
    }

    pub fn take_result(&self) -> Option<Processed> {
        self.shared.lock().result.take()
    }

    /// The newest stacked camera frame at full resolution.
    pub fn latest_frame(&self) -> Option<Arc<RgbImage>> {
        self.shared.lock().stacked.clone()
    }

    pub fn stats(&self) -> Stats {
        self.shared.lock().stats.clone()
    }

    pub fn error(&self) -> Option<String> {
        self.shared.lock().error.clone()
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.shared.lock().stop = true;
        self.shared.wake.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

fn spawn(name: &str, f: impl FnOnce() + Send + 'static) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name(name.into())
        .spawn(f)
        .expect("failed to spawn pipeline thread")
}

fn decode_thread(
    shared: &Shared,
    frames: &Mutex<Receiver<RawFrame>>,
    decoded: &SyncSender<(RgbImage, u64)>,
) {
    loop {
        let received = frames
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_millis(200));
        let raw = match received {
            Ok(raw) => raw,
            Err(RecvTimeoutError::Timeout) => {
                let mut st = shared.lock();
                if st.stop {
                    return;
                }
                if st
                    .last_frame_at
                    .is_none_or(|t| t.elapsed() > Duration::from_secs(1))
                {
                    st.stats.camera_fps = 0.0;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => return,
        };

        let (frozen, orientation, orientation_gen) = {
            let mut st = shared.lock();
            if st.stop {
                return;
            }
            let now = Instant::now();
            if let Some(prev) = st.last_frame_at.replace(now) {
                let fps = 1.0 / (now - prev).as_secs_f32().max(1e-3);
                st.stats.camera_fps = if st.stats.camera_fps == 0.0 {
                    fps
                } else {
                    st.stats.camera_fps * 0.9 + fps * 0.1
                };
            }
            (st.frozen, st.orientation, st.orientation_gen)
        };
        if frozen {
            continue;
        }

        let start = Instant::now();
        match decode(&raw) {
            Ok(frame) => {
                let frame = if orientation.is_identity() {
                    frame
                } else {
                    frame.oriented(orientation)
                };
                let ms = start.elapsed().as_secs_f32() * 1000.0;
                let mut st = shared.lock();
                st.stats.decode_ms = st.stats.decode_ms * 0.9 + ms * 0.1;
                drop(st);
                match decoded.try_send((frame, orientation_gen)) {
                    Ok(()) | Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => return,
                }
            }
            // MJPEG streams occasionally contain truncated frames; just skip them.
            Err(_) => shared.lock().stats.decode_errors += 1,
        }
    }
}

fn stack_thread(shared: &Shared, frames: &Receiver<(RgbImage, u64)>, ctx: &egui::Context) {
    let mut stacker = Stacker::default();
    let mut stacker_gen = 0;
    loop {
        let (frame, frame_gen) = match frames.recv_timeout(Duration::from_millis(200)) {
            Ok(frame) => frame,
            Err(RecvTimeoutError::Timeout) => {
                if shared.lock().stop {
                    return;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => return,
        };
        let (depth, reset, preview_pending, orientation_gen) = {
            let mut st = shared.lock();
            (
                st.stack_depth,
                std::mem::take(&mut st.reset_stack),
                st.preview.is_some(),
                st.orientation_gen,
            )
        };
        if frame_gen != orientation_gen {
            continue; // Turned the old way.
        }

        let start = Instant::now();
        if reset || frame_gen != stacker_gen {
            stacker.reset();
            stacker_gen = frame_gen;
        }
        let stacked = Arc::new(stacker.push(&frame, depth));
        // Skip building a preview while the UI hasn't even shown the previous one.
        let preview = (!preview_pending).then(|| stacked.to_color_image(PREVIEW_MAX_SIDE));
        let ms = start.elapsed().as_secs_f32() * 1000.0;

        let mut st = shared.lock();
        st.stats.stack_ms = st.stats.stack_ms * 0.9 + ms * 0.1;
        st.stats.stacked_frames = stacker.count();
        st.stacked = Some(stacked);
        st.stacked_gen += 1;
        if preview.is_some() {
            st.preview = preview;
        }
        shared.wake.notify_all();
        drop(st);
        ctx.request_repaint();
    }
}

fn process_thread(shared: &Shared, ctx: &egui::Context) {
    let (mut seen_frame, mut seen_settings) = (0, u64::MAX);
    loop {
        let (frame, settings) = {
            let mut st = shared.lock();
            loop {
                if st.stop {
                    return;
                }
                if st.stacked.is_some()
                    && (st.stacked_gen != seen_frame || st.settings_gen != seen_settings)
                {
                    break;
                }
                st = shared
                    .wake
                    .wait_timeout(st, Duration::from_millis(500))
                    .unwrap()
                    .0;
            }
            seen_frame = st.stacked_gen;
            seen_settings = st.settings_gen;
            (st.stacked.clone().unwrap(), st.settings.clone())
        };

        let start = Instant::now();
        let result = process(&frame, &settings);
        let elapsed = start.elapsed();
        let mut st = shared.lock();
        match result {
            Ok(image) => {
                st.stats.output_size = Some((image.width, image.height));
                st.result = Some(Processed {
                    display: image.to_color_image(RESULT_MAX_SIDE),
                    image: Arc::new(image),
                    grayscale: settings.enhance.mode != ColorMode::Color,
                });
                st.error = None;
            }
            Err(e) => st.error = Some(e),
        }
        let ms = elapsed.as_secs_f32() * 1000.0;
        st.stats.process_ms = if st.stats.process_ms == 0.0 {
            ms
        } else {
            st.stats.process_ms * 0.8 + ms * 0.2
        };
        drop(st);
        ctx.request_repaint();

        if let Some(rest) = MIN_PROCESS_INTERVAL.checked_sub(start.elapsed()) {
            std::thread::sleep(rest);
        }
    }
}

/// Rectifies the selected sheet and applies the enhancement settings.
pub fn process(frame: &RgbImage, settings: &ProcessSettings) -> Result<RgbImage, String> {
    let quad = settings.corners.map(|[x, y]| {
        [
            x as f64 * frame.width as f64,
            y as f64 * frame.height as f64,
        ]
    });
    let (w, h) = settings.sheet.output_size(quad);
    let warped = warp(frame, quad, w, h, settings.sheet.interpolation)
        .ok_or_else(|| "The four corners don't form a valid area.".to_string())?;
    Ok(enhance(&warped, &settings.enhance))
}

fn decode(raw: &RawFrame) -> Result<RgbImage, String> {
    match raw.encoding {
        Encoding::Mjpeg => {
            let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
            let mut decoder = JpegDecoder::new_with_options(ZCursor::new(&raw.data[..]), options);
            let pixels = decoder.decode().map_err(|e| format!("{e:?}"))?;
            let info = decoder.info().ok_or("missing JPEG header")?;
            let (w, h) = (info.width as usize, info.height as usize);
            if pixels.len() != w * h * 3 {
                return Err("unexpected JPEG pixel layout".into());
            }
            Ok(RgbImage::from_raw(w, h, pixels))
        }
        Encoding::Yuyv => {
            let (w, h) = (raw.width, raw.height);
            let stride = raw.bytes_per_line.max(w * 2);
            if raw.data.len() < stride * (h - 1) + w * 2 {
                return Err("short YUYV frame".into());
            }
            let mut out = RgbImage::new(w, h);
            out.data
                .par_chunks_mut(w * 3)
                .enumerate()
                .for_each(|(y, row)| {
                    let src = &raw.data[y * stride..y * stride + w * 2];
                    for (px, yuyv) in row
                        .as_chunks_mut::<6>()
                        .0
                        .iter_mut()
                        .zip(src.as_chunks::<4>().0)
                    {
                        let (u, v) = (yuyv[1] as i32 - 128, yuyv[3] as i32 - 128);
                        px[..3].copy_from_slice(&yuv_to_rgb(yuyv[0], u, v));
                        px[3..].copy_from_slice(&yuv_to_rgb(yuyv[2], u, v));
                    }
                });
            Ok(out)
        }
        Encoding::I420 => {
            let (w, h) = (raw.width, raw.height);
            let stride = raw.bytes_per_line.max(w);
            let (cw, chroma_stride) = (w.div_ceil(2), stride.div_ceil(2));
            let luma_len = stride * h;
            let chroma_len = chroma_stride * h.div_ceil(2);
            if raw.data.len() < luma_len + 2 * chroma_len {
                return Err("short I420 frame".into());
            }
            let (luma, chroma) = raw.data.split_at(luma_len);
            let (u_plane, v_plane) = chroma.split_at(chroma_len);
            let mut out = RgbImage::new(w, h);
            out.data
                .par_chunks_mut(w * 3)
                .enumerate()
                .for_each(|(y, row)| {
                    let luma = &luma[y * stride..][..w];
                    let u = &u_plane[y / 2 * chroma_stride..][..cw];
                    let v = &v_plane[y / 2 * chroma_stride..][..cw];
                    for (x, px) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                        *px = yuv_to_rgb(luma[x], u[x / 2] as i32 - 128, v[x / 2] as i32 - 128);
                    }
                });
            Ok(out)
        }
    }
}

/// BT.601 limited range; `u` and `v` are already centered on zero.
#[inline]
fn yuv_to_rgb(luma: u8, u: i32, v: i32) -> [u8; 3] {
    let c = 298 * (luma as i32 - 16);
    [
        ((c + 409 * v + 128) >> 8).clamp(0, 255) as u8,
        ((c - 100 * u - 208 * v + 128) >> 8).clamp(0, 255) as u8,
        ((c + 516 * u + 128) >> 8).clamp(0, 255) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Capture, list_devices};

    #[test]
    fn decodes_i420() {
        // 2×2 frames with one shared chroma sample: white, black and two mid grays, then all red
        // (BT.601: Y 81, U 90, V 240).
        let gray = RawFrame {
            encoding: Encoding::I420,
            width: 2,
            height: 2,
            bytes_per_line: 2,
            data: vec![235, 16, 126, 126, 128, 128],
        };
        let rgb = decode(&gray).unwrap();
        assert_eq!(&rgb.data[..6], &[255, 255, 255, 0, 0, 0]);
        assert_eq!(&rgb.data[6..9], &[128, 128, 128]);
        let red = RawFrame {
            data: vec![81, 81, 81, 81, 90, 240],
            ..gray
        };
        let rgb = decode(&red).unwrap();
        assert!(
            rgb.data
                .as_chunks::<3>()
                .0
                .iter()
                .all(|p| p[0] > 250 && p[1] < 5 && p[2] < 5)
        );
        let short = RawFrame {
            data: vec![0; 5],
            ..red
        };
        assert!(decode(&short).is_err());
    }

    /// Runs real camera frames through the whole pipeline and writes the stacked frame and the
    /// result as PNGs: `cargo test --release -- --ignored --nocapture camera_smoke`.
    /// `OVERHEAD_TEST_DEVICE` picks the device (default: first), `OVERHEAD_TEST_OUT` the folder.
    #[test]
    #[ignore = "needs a camera"]
    fn camera_smoke() {
        let devices = list_devices();
        for d in &devices {
            let modes: Vec<_> = d.modes.iter().map(|m| m.label()).collect();
            println!("{}: {modes:?}", d.label());
        }
        let wanted = std::env::var_os("OVERHEAD_TEST_DEVICE").map(std::path::PathBuf::from);
        let device = devices
            .iter()
            .find(|d| wanted.as_ref().is_none_or(|w| &d.path == w))
            .expect("no camera found");
        let mode = device.default_mode().unwrap();
        println!("using {} in {}", device.label(), mode.label());

        let (tx, rx) = sync_channel(2);
        let capture = Capture::start(&device.path, mode, tx);
        let mut stacker = Stacker::default();
        let mut stacked = RgbImage::new(0, 0);
        let (mut decode_time, mut stack_time) = (Duration::ZERO, Duration::ZERO);
        let frames = 30;
        let mut start = None;
        for _ in 0..frames {
            let raw = rx.recv_timeout(Duration::from_secs(5)).unwrap_or_else(|e| {
                panic!("no frame ({e}): {:?}", capture.status.lock().unwrap().error)
            });
            start.get_or_insert_with(Instant::now);
            let t = Instant::now();
            let frame = decode(&raw).expect("decode failed");
            decode_time += t.elapsed();
            let t = Instant::now();
            stacked = stacker.push(&frame, 16);
            stack_time += t.elapsed();
        }
        let elapsed = start.unwrap().elapsed();
        drop(capture);
        println!(
            "{:.1} fps received, decode {:.1} ms, stack {:.1} ms per frame",
            (frames - 1) as f32 / elapsed.as_secs_f32(),
            decode_time.as_secs_f32() * 1000.0 / frames as f32,
            stack_time.as_secs_f32() * 1000.0 / frames as f32,
        );

        println!(
            "detected sheet: {:?}",
            crate::detect::detect_sheet(&stacked)
        );
        let settings = ProcessSettings {
            corners: [[0.2, 0.15], [0.8, 0.15], [0.8, 0.85], [0.2, 0.85]],
            sheet: SheetSettings::default(),
            enhance: EnhanceSettings::scan(),
        };
        let t = Instant::now();
        let out = process(&stacked, &settings).unwrap();
        println!("process {:?} -> {}x{}", t.elapsed(), out.width, out.height);

        let dir = std::env::var_os("OVERHEAD_TEST_OUT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        for (name, img) in [("smoke_stacked.png", &stacked), ("smoke_result.png", &out)] {
            let path = dir.join(name);
            image::save_buffer(
                &path,
                &img.data,
                img.width as u32,
                img.height as u32,
                image::ColorType::Rgb8,
            )
            .unwrap();
            println!("wrote {}", path.display());
        }
    }
}
