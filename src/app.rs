use std::{
    borrow::Cow,
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, Sender, channel},
    },
    time::{Duration, Instant},
};

use eframe::egui::{
    self, Align, Align2, Color32, ComboBox, CornerRadius, CursorIcon, FontId, Key,
    KeyboardShortcut, Mesh, Modifiers, PointerButton, Pos2, ProgressBar, Rect, Response, RichText,
    Sense, Shape, Slider, Stroke, StrokeKind, TextureHandle, TextureOptions, Ui, Vec2, pos2, vec2,
};
use serde::{Deserialize, Serialize};

use crate::{
    camera::{Capture, ControlKind, Controls, DeviceInfo, Mode, list_devices},
    detect::{detect_sheet, order_corners},
    enhance::{ColorMode, EnhanceSettings},
    img::{FrameOrientation, RgbImage},
    pipeline::{Pipeline, ProcessSettings},
    rtsp::{self, StreamSettings},
    sheet::{MAX_OUTPUT_SIDE, Orientation, Paper, SheetSettings},
    warp::Interpolation,
};

const DEFAULT_CORNERS: [[f32; 2]; 4] = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]];
const CORNER_LABELS: [&str; 4] = ["TL", "TR", "BR", "BL"];
const HANDLE_RADIUS: f32 = 7.0;
const GRAB_RADIUS: f32 = 20.0;
const ACCENT: Color32 = Color32::from_rgb(255, 196, 0);
const STREAM_LABEL: &str = "Phone / network stream (RTSP)";
const STREAM_KEY: &str = "stream";
/// Every save also writes the result under this fixed name, for handing it to other tools such as
/// an LLM agent.
const SHARE_FILE: &str = "latest.png";
/// Long side of the shared copy. Claude and GPT models downscale larger images anyway; a smaller
/// file is quicker to upload and costs fewer tokens.
const SHARE_MAX_SIDE: u32 = 1568;
const TOP_EDGE: Color32 = Color32::from_rgb(80, 200, 255);
/// Space between a view's border and its image while the whole image is shown.
const VIEW_MARGIN: f32 = 12.0;
/// Zoom factor per point of mouse wheel scrolling; a wheel notch (40 to 80 points, depending on
/// the system) zooms by roughly 10 to 20 %.
const WHEEL_ZOOM_SPEED: f32 = 0.0025;
/// Zooming in stops when one image pixel covers this many points.
const MAX_PIXEL_SIZE: f32 = 16.0;
const MINIMAP_SIZE: Vec2 = vec2(200.0, 150.0);
/// Height of the button row above each image.
const VIEW_BAR_HEIGHT: f32 = 30.0;
/// The side-by-side divider keeps both views at least this wide.
const MIN_VIEW_WIDTH: f32 = 160.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum View {
    Camera,
    Both,
    Result,
}

/// Everything that is remembered between sessions.
#[derive(Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    device_name: Option<String>,
    device_path: Option<PathBuf>,
    mode: Option<Mode>,
    /// Camera control values the user changed, per device name, re-applied when it is opened.
    camera_controls: BTreeMap<String, BTreeMap<String, i32>>,
    /// Use the RTSP stream (e.g. the phone) instead of a USB camera.
    use_stream: bool,
    stream: StreamSettings,
    corners: [[f32; 2]; 4],
    /// Corners per source, so switching between e.g. the overhead camera and the phone keeps both.
    corners_by_source: BTreeMap<String, [[f32; 2]; 4]>,
    /// How the camera image is turned, also per source.
    orientation: FrameOrientation,
    orientation_by_source: BTreeMap<String, FrameOrientation>,
    sheet: SheetSettings,
    enhance: EnhanceSettings,
    stack_depth: u32,
    output_dir: String,
    view: View,
    /// Where the divider between camera and result sits in the side-by-side view (0..1).
    split: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            device_name: None,
            device_path: None,
            mode: None,
            camera_controls: BTreeMap::new(),
            use_stream: false,
            stream: StreamSettings::default(),
            corners: DEFAULT_CORNERS,
            corners_by_source: BTreeMap::new(),
            orientation: FrameOrientation::default(),
            orientation_by_source: BTreeMap::new(),
            sheet: SheetSettings::default(),
            enhance: EnhanceSettings::default(),
            stack_depth: 8,
            output_dir: default_output_dir(),
            view: View::Both,
            split: 0.5,
        }
    }
}

/// The two images the app shows.
#[derive(Clone, Copy)]
enum Picture {
    Camera,
    Result,
}

enum JobResult {
    Saved(Result<PathBuf, String>),
    Copied(Result<(), String>),
}

struct Status {
    text: String,
    error: bool,
    since: Instant,
}

/// Zoom and pan of an image view. At `zoom` 1 the whole image fits into the view; `center` is the
/// image point (0..1) in the middle of the view.
struct ZoomPan {
    zoom: f32,
    center: Vec2,
    detail: Option<Detail>,
}

/// The display textures are downscaled, so a zoomed-in view shows the visible part of the
/// full-resolution image from this texture instead.
struct Detail {
    source: Arc<RgbImage>,
    /// Source pixels `[x0, y0, x1, y1)` in the texture, downscaled by `factor`.
    region: [usize; 4],
    factor: usize,
    tex: TextureHandle,
}

impl Default for ZoomPan {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            center: vec2(0.5, 0.5),
            detail: None,
        }
    }
}

impl ZoomPan {
    fn zoomed(&self) -> bool {
        self.zoom > 1.0
    }

    /// Screen rect of the whole image (`size` in pixels) in `view`; the view clips what lies outside.
    fn image_rect(&self, view: Rect, size: Vec2) -> Rect {
        let area = view.shrink(VIEW_MARGIN);
        let shown = fit(area, size).size() * self.zoom;
        Rect::from_min_size(area.center() - self.center * shown, shown)
    }

    /// The mouse wheel (or a pinch) zooms around the pointer, dragging pans when `pan` is set,
    /// pressing in the minimap moves the view there and a double-click shows the whole image again.
    fn update(&mut self, ui: &Ui, response: &Response, view: Rect, size: Vec2, pan: bool) {
        let area = view.shrink(VIEW_MARGIN);
        if !area.is_positive() {
            return;
        }
        let fitted = fit(area, size).size();
        if self.zoomed() {
            let map_rect = minimap_rect(view, size);
            let map = ui
                .interact(
                    map_rect,
                    response.id.with("minimap"),
                    Sense::click_and_drag(),
                )
                .on_hover_cursor(CursorIcon::Crosshair);
            // A quick click can start and end within one frame, so check `clicked` too.
            if (map.is_pointer_button_down_on() || map.clicked())
                && let Some(pos) = map.interact_pointer_pos()
            {
                self.center = (pos - map_rect.min) / map_rect.size();
            }
        }
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        let factor = pinch * (scroll * WHEEL_ZOOM_SPEED).exp();
        if factor != 1.0
            && response.contains_pointer()
            && let Some(pointer) = ui.ctx().pointer_hover_pos()
        {
            self.zoom_at(pointer, factor, view, size);
        }
        if pan && response.dragged() {
            self.center -= response.drag_delta() / (fitted * self.zoom);
        }
        if response.double_clicked() {
            self.zoom = 1.0;
        }
        self.clamp(view, size);
    }

    /// Zooms by `factor`, keeping the image point under `pointer` where it is.
    fn zoom_at(&mut self, pointer: Pos2, factor: f32, view: Rect, size: Vec2) {
        let area = view.shrink(VIEW_MARGIN);
        let fitted = fit(area, size).size();
        let max_zoom = (MAX_PIXEL_SIZE * size.x / fitted.x).max(1.0);
        let anchor = (pointer - self.image_rect(view, size).min) / (fitted * self.zoom);
        self.zoom = (self.zoom * factor).clamp(1.0, max_zoom);
        self.center = anchor + (area.center() - pointer) / (fitted * self.zoom);
    }

    /// Keeps the view covered by the image; along an axis where the image is smaller than the
    /// view, centers it.
    fn clamp(&mut self, view: Rect, size: Vec2) {
        let area = view.shrink(VIEW_MARGIN);
        self.zoom = self.zoom.max(1.0);
        let shown = fit(area, size).size() * self.zoom;
        for axis in 0..2 {
            let half = area.size()[axis] / shown[axis] / 2.0;
            self.center[axis] = if half >= 0.5 {
                0.5
            } else {
                self.center[axis].clamp(half, 1.0 - half)
            };
        }
    }

    /// While zoomed in further than the display texture (`tex_size`) resolves, draws the visible
    /// part of the full-resolution `source` over it.
    fn draw_detail(
        &mut self,
        painter: &egui::Painter,
        name: &str,
        source: Option<&Arc<RgbImage>>,
        tex_size: Vec2,
        view: Rect,
        img_rect: Rect,
    ) {
        let ppp = painter.pixels_per_point();
        let visible = img_rect.intersect(view);
        let Some(source) = source.filter(|s| {
            self.zoomed()
                && img_rect.width() * ppp > tex_size.x
                && s.width as f32 > tex_size.x
                && visible.is_positive()
        }) else {
            self.detail = None;
            return;
        };
        let size = vec2(source.width as f32, source.height as f32);
        let lo = (visible.min - img_rect.min) / img_rect.size() * size;
        let hi = (visible.max - img_rect.min) / img_rect.size() * size;
        let (x0, y0) = (lo.x.max(0.0) as usize, lo.y.max(0.0) as usize);
        let x1 = (hi.x.ceil() as usize).min(source.width);
        let y1 = (hi.y.ceil() as usize).min(source.height);
        // At least one source pixel per screen pixel, whole blocks only.
        let factor = (((x1 - x0) as f32 / (visible.width() * ppp)) as usize).max(1);
        let region = [
            x0,
            y0,
            x0 + (x1 - x0) / factor * factor,
            y0 + (y1 - y0) / factor * factor,
        ];
        if region[2] == x0 || region[3] == y0 {
            self.detail = None;
            return;
        }
        let current = self.detail.as_ref().is_some_and(|d| {
            Arc::ptr_eq(&d.source, source) && d.region == region && d.factor == factor
        });
        if !current {
            let image = source.region_to_color_image(region, factor);
            match &mut self.detail {
                Some(d) => {
                    d.tex.set(image, TextureOptions::LINEAR);
                    d.source = source.clone();
                    d.region = region;
                    d.factor = factor;
                }
                None => {
                    self.detail = Some(Detail {
                        source: source.clone(),
                        region,
                        factor,
                        tex: painter
                            .ctx()
                            .load_texture(name, image, TextureOptions::LINEAR),
                    })
                }
            }
        }
        if let Some(d) = &self.detail {
            let at = |x: usize, y: usize| {
                img_rect.min + vec2(x as f32, y as f32) / size * img_rect.size()
            };
            let rect = Rect::from_min_max(at(region[0], region[1]), at(region[2], region[3]));
            painter.image(d.tex.id(), rect, unit_uv(), Color32::WHITE);
        }
    }

    /// The minimap with the visible part outlined while zoomed in, otherwise a hint on hover.
    /// `source_width` is the full-resolution width, for showing the zoom level in percent.
    fn draw_overlay(
        &self,
        painter: &egui::Painter,
        tex: &TextureHandle,
        view: Rect,
        img_rect: Rect,
        source_width: f32,
        hovered: bool,
    ) {
        let corner = view.right_bottom() - Vec2::splat(VIEW_MARGIN);
        if !self.zoomed() {
            if hovered {
                draw_tag(
                    painter,
                    corner,
                    Align2::RIGHT_BOTTOM,
                    "Scroll to zoom".into(),
                );
            }
            return;
        }
        let map = minimap_rect(view, tex.size_vec2());
        painter.rect_filled(
            map.expand(3.0),
            CornerRadius::same(3),
            Color32::from_black_alpha(200),
        );
        painter.image(tex.id(), map, unit_uv(), Color32::WHITE);
        let visible = img_rect.intersect(view);
        let to_map = |p: Pos2| map.min + (p - img_rect.min) / img_rect.size() * map.size();
        painter.rect_stroke(
            Rect::from_min_max(to_map(visible.min), to_map(visible.max)),
            CornerRadius::ZERO,
            Stroke::new(1.5, ACCENT),
            StrokeKind::Middle,
        );
        let percent = img_rect.width() * painter.pixels_per_point() / source_width * 100.0;
        draw_tag(
            painter,
            map.left_top() - vec2(3.0, 6.0),
            Align2::LEFT_BOTTOM,
            format!("{percent:.0} % · double-click: fit"),
        );
    }
}

pub struct OverheadApp {
    settings: Settings,
    pipeline: Pipeline,
    devices: Vec<DeviceInfo>,
    device: Option<usize>,
    capture: Option<Capture>,
    controls: Option<Controls>,
    /// Key into `Settings::corners_by_source` of the running source.
    source_key: Option<String>,
    /// Stream settings being edited; applied on "Connect".
    stream_draft: StreamSettings,
    preview: Option<TextureHandle>,
    result_tex: Option<TextureHandle>,
    result: Option<(Arc<RgbImage>, bool)>,
    frozen: bool,
    /// Corner being dragged and the offset from the pointer to the corner at grab time.
    drag: Option<(usize, Vec2)>,
    selected_corner: Option<usize>,
    /// Corners right-clicked so far for a new sheet outline; the old outline stays in use until
    /// the fourth one.
    placing: Vec<[f32; 2]>,
    camera_zoom: ZoomPan,
    result_zoom: ZoomPan,
    status: Option<Status>,
    jobs_tx: Sender<JobResult>,
    jobs_rx: Receiver<JobResult>,
    clipboard: Arc<Mutex<Option<arboard::Clipboard>>>,
}

impl OverheadApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let settings: Settings = cc
            .storage
            .and_then(|s| eframe::get_value(s, eframe::APP_KEY))
            .unwrap_or_default();
        let pipeline = Pipeline::new(
            cc.egui_ctx.clone(),
            process_settings(&settings),
            settings.stack_depth,
        );
        let (jobs_tx, jobs_rx) = channel();
        let stream_draft = settings.stream.clone();
        let mut app = Self {
            settings,
            pipeline,
            devices: Vec::new(),
            device: None,
            capture: None,
            controls: None,
            source_key: None,
            stream_draft,
            preview: None,
            result_tex: None,
            result: None,
            frozen: false,
            drag: None,
            selected_corner: None,
            placing: Vec::new(),
            camera_zoom: ZoomPan::default(),
            result_zoom: ZoomPan::default(),
            status: None,
            jobs_tx,
            jobs_rx,
            clipboard: Arc::new(Mutex::new(None)),
        };
        app.pipeline.set_orientation(app.settings.orientation, None);
        app.rescan_devices();
        app
    }

    fn set_status(&mut self, text: impl Into<String>, error: bool) {
        self.status = Some(Status {
            text: text.into(),
            error,
            since: Instant::now(),
        });
    }

    fn rescan_devices(&mut self) {
        let current = self.device.map(|i| self.devices[i].path.clone());
        if !self.settings.use_stream {
            // Release the current device first so it enumerates normally.
            self.capture = None;
            self.controls = None;
        }
        self.devices = list_devices();
        self.device = None;
        if self.settings.use_stream {
            if self.capture.is_none() {
                self.start_stream();
            }
            return;
        }

        let wanted_path = current.or_else(|| self.settings.device_path.clone());
        let index = self
            .devices
            .iter()
            .position(|d| {
                Some(&d.path) == wanted_path.as_ref()
                    && Some(&d.name) == self.settings.device_name.as_ref()
            })
            .or_else(|| {
                self.devices
                    .iter()
                    .position(|d| Some(&d.name) == self.settings.device_name.as_ref())
            })
            .or((!self.devices.is_empty()).then_some(0));
        match index {
            Some(i) => self.start_camera(i, self.settings.mode),
            None => self.set_status(
                "No camera found. Plug one in and press \"Rescan\", or pick the phone stream.",
                true,
            ),
        }
    }

    fn start_camera(&mut self, index: usize, mode: Option<Mode>) {
        self.capture = None;
        self.controls = None;
        let device = &self.devices[index];
        let Some(mode) = mode
            .filter(|m| device.modes.contains(m))
            .or(device.default_mode())
        else {
            return;
        };
        let name = device.name.clone();
        self.device = Some(index);
        self.settings.use_stream = false;
        self.settings.device_name = Some(name.clone());
        self.settings.device_path = Some(device.path.clone());
        self.settings.mode = Some(mode);
        self.switch_source(format!("camera:{name}"));
        let device = &self.devices[index];
        self.pipeline.reset_stack();
        self.capture = Some(Capture::start(
            &device.path,
            mode,
            self.pipeline.frame_sink(),
        ));

        match Controls::open(&device.path) {
            Ok(mut controls) => {
                if let Some(saved) = self.settings.camera_controls.get(&device.name) {
                    // Apply in device order so e.g. "auto exposure off" precedes "exposure time".
                    for i in 0..controls.list.len() {
                        let c = &controls.list[i];
                        if let Some(&v) = saved.get(&c.name)
                            && v != c.value
                        {
                            let _ = controls.set(i, v);
                        }
                    }
                }
                self.controls = Some(controls);
            }
            Err(e) => self.set_status(format!("Camera controls unavailable: {e}"), true),
        }
    }

    fn start_stream(&mut self) {
        self.capture = None;
        self.controls = None;
        self.device = None;
        self.settings.use_stream = true;
        self.switch_source(STREAM_KEY.into());
        self.pipeline.reset_stack();
        self.capture = Some(rtsp::start(
            self.settings.stream.clone(),
            self.pipeline.frame_sink(),
        ));
    }

    /// Remembers the corners of the source being left and restores those of the new one.
    fn switch_source(&mut self, key: String) {
        self.remember_corners();
        if self.source_key.as_ref() != Some(&key) {
            // Don't show the previous source's image while the new one starts.
            self.preview = None;
            self.camera_zoom = ZoomPan::default();
            self.placing.clear();
        }
        if let Some(&corners) = self.settings.corners_by_source.get(&key) {
            self.settings.corners = corners;
        }
        if let Some(&orientation) = self.settings.orientation_by_source.get(&key) {
            self.settings.orientation = orientation;
        }
        self.pipeline
            .set_orientation(self.settings.orientation, None);
        self.source_key = Some(key);
    }

    fn remember_corners(&mut self) {
        if let Some(key) = &self.source_key {
            self.settings
                .corners_by_source
                .insert(key.clone(), self.settings.corners);
            self.settings
                .orientation_by_source
                .insert(key.clone(), self.settings.orientation);
        }
    }

    /// Why the source delivers no frames right now, if it doesn't.
    fn capture_problem(&self) -> Option<String> {
        let status = self.capture.as_ref()?.status.lock().unwrap().clone();
        if status.running {
            return None;
        }
        Some(match status.error {
            Some(e) if self.settings.use_stream => format!("{e}\nRetrying every few seconds…"),
            Some(e) => e,
            None if self.settings.use_stream => {
                format!("Connecting to {}…", self.settings.stream.host())
            }
            None => "Starting camera…".into(),
        })
    }

    /// Size of the camera image as shown, i.e. after turning it.
    fn frame_size(&self) -> Option<(f32, f32)> {
        let frame = self.pipeline.latest_frame()?;
        Some((frame.width as f32, frame.height as f32))
    }

    /// Turns the camera image a quarter turn clockwise. The corners stay on the sheet, so the
    /// result keeps its orientation.
    fn rotate_camera(&mut self, ctx: &egui::Context) {
        let turn = |[x, y]: [f32; 2]| [1.0 - y, x];
        self.settings.orientation = self.settings.orientation.rotated();
        self.settings.corners = self.settings.corners.map(turn);
        self.turned_camera(ctx, turn, FrameOrientation::QUARTER_TURN);
    }

    /// Mirrors the camera image left-right. The corners stay on the sheet but swap sides, so the
    /// result is mirrored too, which fixes a camera that delivers mirrored pictures.
    fn mirror_camera(&mut self, ctx: &egui::Context) {
        let flip = |[x, y]: [f32; 2]| [1.0 - x, y];
        self.settings.orientation = self.settings.orientation.mirrored();
        let c = self.settings.corners.map(flip);
        self.settings.corners = [c[1], c[0], c[3], c[2]];
        self.turned_camera(ctx, flip, FrameOrientation::MIRROR);
    }

    /// Moves everything else that lives in camera image coordinates along with a turn.
    fn turned_camera(
        &mut self,
        ctx: &egui::Context,
        map: impl Fn([f32; 2]) -> [f32; 2],
        change: FrameOrientation,
    ) {
        for p in &mut self.placing {
            *p = map(*p);
        }
        let [x, y] = map([self.camera_zoom.center.x, self.camera_zoom.center.y]);
        self.camera_zoom.center = vec2(x, y);
        self.pipeline
            .set_orientation(self.settings.orientation, Some(change));
        ctx.request_repaint();
    }

    /// Picks up new images from the pipeline and finished background jobs.
    fn poll(&mut self, ctx: &egui::Context) {
        if let Some(image) = self.pipeline.take_preview() {
            match &mut self.preview {
                Some(tex) => tex.set(image, TextureOptions::LINEAR),
                None => {
                    self.preview = Some(ctx.load_texture("camera", image, TextureOptions::LINEAR))
                }
            }
        }
        if let Some(processed) = self.pipeline.take_result() {
            match &mut self.result_tex {
                Some(tex) => tex.set(processed.display, TextureOptions::LINEAR),
                None => {
                    self.result_tex =
                        Some(ctx.load_texture("result", processed.display, TextureOptions::LINEAR))
                }
            }
            self.result = Some((processed.image, processed.grayscale));
        }
        while let Ok(job) = self.jobs_rx.try_recv() {
            match job {
                JobResult::Saved(Ok(path)) => self.set_status(
                    format!("Saved {} and updated {SHARE_FILE}", path.display()),
                    false,
                ),
                JobResult::Saved(Err(e)) => self.set_status(format!("Saving failed: {e}"), true),
                JobResult::Copied(Ok(())) => self.set_status("Copied image to clipboard", false),
                JobResult::Copied(Err(e)) => self.set_status(format!("Copy failed: {e}"), true),
            }
        }
    }

    /// The full-resolution image behind a view, and whether it is grayscale.
    fn picture(&self, which: Picture) -> Option<(Arc<RgbImage>, bool)> {
        match which {
            Picture::Camera => self.pipeline.latest_frame().map(|f| (f, false)),
            Picture::Result => self.result.clone(),
        }
    }

    fn save(&mut self, ctx: &egui::Context, which: Picture) {
        let Some((image, gray)) = self.picture(which) else {
            self.set_status("Nothing to save yet", true);
            return;
        };
        let name = match which {
            Picture::Camera => "camera",
            Picture::Result => "sketch",
        };
        let dir = expand_home(&self.settings.output_dir);
        let tx = self.jobs_tx.clone();
        let ctx = ctx.clone();
        self.set_status("Saving…", false);
        std::thread::spawn(move || {
            let _ = tx.send(JobResult::Saved(save_png(&image, gray, &dir, name)));
            ctx.request_repaint();
        });
    }

    fn copy(&mut self, ctx: &egui::Context, which: Picture) {
        let Some((image, _)) = self.picture(which) else {
            self.set_status("Nothing to copy yet", true);
            return;
        };
        let clipboard = self.clipboard.clone();
        let tx = self.jobs_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let rgba: Vec<u8> = image
                .data
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|&[r, g, b]| [r, g, b, 255])
                .collect();
            // The clipboard object has to stay alive: on X11 it serves the data to other apps.
            let mut guard = clipboard.lock().unwrap();
            if guard.is_none() {
                *guard = arboard::Clipboard::new().ok();
            }
            let result = match guard.as_mut() {
                Some(cb) => cb
                    .set_image(arboard::ImageData {
                        width: image.width,
                        height: image.height,
                        bytes: Cow::Owned(rgba),
                    })
                    .map_err(|e| e.to_string()),
                None => Err("no clipboard available".into()),
            };
            let _ = tx.send(JobResult::Copied(result));
            ctx.request_repaint();
        });
    }

    fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
        self.pipeline.set_frozen(frozen);
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let save = ctx
            .input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::S)));
        // egui turns Ctrl+C into a Copy event rather than a key press.
        let copy = ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Copy)));
        let freeze = ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::F));
        if save {
            self.save(ctx, Picture::Result);
        }
        if copy {
            self.copy(ctx, Picture::Result);
        }
        if freeze {
            self.set_frozen(!self.frozen);
        }

        // A focused widget (e.g. a slider) uses the arrow keys itself.
        let widget_focused = ctx.memory(|m| m.focused().is_some());
        if !self.placing.is_empty() && !widget_focused && ctx.input(|i| i.key_pressed(Key::Escape))
        {
            self.placing.clear();
        }
        if let Some(i) = self.selected_corner.filter(|_| !widget_focused) {
            let (dx, dy, shift, escape) = ctx.input(|inp| {
                let axis = |neg: Key, pos: Key| {
                    inp.key_pressed(pos) as i32 as f32 - inp.key_pressed(neg) as i32 as f32
                };
                (
                    axis(Key::ArrowLeft, Key::ArrowRight),
                    axis(Key::ArrowUp, Key::ArrowDown),
                    inp.modifiers.shift,
                    inp.key_pressed(Key::Escape),
                )
            });
            if escape {
                self.selected_corner = None;
            } else if (dx != 0.0 || dy != 0.0)
                && let Some((w, h)) = self.frame_size()
            {
                let step = if shift { 10.0 } else { 1.0 };
                let c = &mut self.settings.corners[i];
                c[0] = (c[0] + dx * step / w).clamp(0.0, 1.0);
                c[1] = (c[1] + dy * step / h).clamp(0.0, 1.0);
            }
        }
    }

    fn toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading("Overhead Camera");
            ui.separator();
            ui.selectable_value(&mut self.settings.view, View::Camera, "Camera");
            ui.selectable_value(&mut self.settings.view, View::Both, "Side by side");
            ui.selectable_value(&mut self.settings.view, View::Result, "Result");
            ui.separator();
            let mut frozen = self.frozen;
            if ui
                .toggle_value(&mut frozen, "Freeze")
                .on_hover_text("Stop taking new camera frames, e.g. to tune the enhancement (F)")
                .changed()
            {
                self.set_frozen(frozen);
            }
        });
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            if let Some(status) = &self.status
                && (status.error || status.since.elapsed() < Duration::from_secs(8))
            {
                let color = if status.error {
                    ui.visuals().error_fg_color
                } else {
                    ui.visuals().text_color()
                };
                ui.label(RichText::new(&status.text).color(color));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let stats = self.pipeline.stats();
                let mut parts = vec![format!("camera {:.1} fps", stats.camera_fps)];
                parts.push(format!("decode {:.0} ms", stats.decode_ms));
                parts.push(format!("stack {:.0} ms", stats.stack_ms));
                parts.push(format!("process {:.0} ms", stats.process_ms));
                if let Some((w, h)) = stats.output_size {
                    parts.push(format!("output {w}×{h}"));
                }
                if stats.decode_errors > 0 {
                    parts.push(format!("{} corrupt frames", stats.decode_errors));
                }
                ui.label(RichText::new(parts.join("  ·  ")).weak());
            });
        });
    }

    fn settings_panel(&mut self, ui: &mut Ui) {
        ui.spacing_mut().slider_width = 150.0;
        egui::CollapsingHeader::new(RichText::new("Camera").strong())
            .default_open(true)
            .show(ui, |ui| self.camera_section(ui));
        egui::CollapsingHeader::new(RichText::new("Noise reduction").strong())
            .default_open(true)
            .show(ui, |ui| self.stack_section(ui));
        egui::CollapsingHeader::new(RichText::new("Sheet & crop").strong())
            .default_open(true)
            .show(ui, |ui| self.sheet_section(ui));
        egui::CollapsingHeader::new(RichText::new("Enhancement").strong())
            .default_open(true)
            .show(ui, |ui| self.enhance_section(ui));
        egui::CollapsingHeader::new(RichText::new("Export").strong())
            .default_open(true)
            .show(ui, |ui| self.export_section(ui));
    }

    fn camera_section(&mut self, ui: &mut Ui) {
        let mut new_device = None;
        let mut new_mode = None;
        let mut pick_stream = false;
        let use_stream = self.settings.use_stream;
        let device_label = if use_stream {
            STREAM_LABEL.to_string()
        } else {
            self.device
                .map(|i| self.devices[i].label())
                .unwrap_or_else(|| "No camera".into())
        };
        ComboBox::from_id_salt("device")
            .width(ui.available_width() - 70.0)
            .selected_text(device_label)
            .show_ui(ui, |ui| {
                for (i, d) in self.devices.iter().enumerate() {
                    if ui
                        .selectable_label(!use_stream && self.device == Some(i), d.label())
                        .clicked()
                    {
                        new_device = Some(i);
                    }
                }
                if ui.selectable_label(use_stream, STREAM_LABEL).clicked() && !use_stream {
                    pick_stream = true;
                }
            });
        if use_stream {
            self.stream_ui(ui);
        } else if let Some(index) = self.device {
            let current = self.settings.mode;
            ComboBox::from_id_salt("mode")
                .width(ui.available_width() - 70.0)
                .selected_text(current.map(|m| m.label()).unwrap_or_default())
                .show_ui(ui, |ui| {
                    for m in &self.devices[index].modes {
                        if ui
                            .selectable_label(current == Some(*m), m.label())
                            .clicked()
                        {
                            new_mode = Some(*m);
                        }
                    }
                });
        }
        let status = self
            .capture
            .as_ref()
            .map(|c| c.status.lock().unwrap().clone());
        let error = status
            .as_ref()
            .and_then(|s| s.error.clone().filter(|_| !s.running));
        ui.horizontal(|ui| {
            if ui.button("Rescan").clicked() {
                self.rescan_devices();
            }
            let Some(status) = status.filter(|_| error.is_none()) else {
                return;
            };
            let text = match status.negotiated.filter(|_| status.running) {
                Some(m) if use_stream => {
                    let fps = m.fps();
                    let fps = if fps > 0.0 {
                        format!(" @ {fps:.0} fps")
                    } else {
                        String::new()
                    };
                    format!("receiving {}×{}{fps}", m.width, m.height)
                }
                Some(m) => format!("streaming {}", m.label()),
                None if use_stream => "connecting…".into(),
                None => "starting…".into(),
            };
            ui.label(RichText::new(text).weak());
        });
        // On its own line so that it wraps instead of widening the panel.
        if let Some(e) = error {
            ui.label(RichText::new(e).color(ui.visuals().error_fg_color));
        }
        if let Some(i) = new_device {
            self.start_camera(i, None);
        } else if pick_stream {
            self.start_stream();
        } else if let (Some(i), Some(m)) = (self.device, new_mode) {
            self.start_camera(i, Some(m));
        }

        egui::CollapsingHeader::new("Camera settings (exposure, white balance, …)")
            .default_open(false)
            .show(ui, |ui| self.controls_ui(ui));
    }

    fn stream_ui(&mut self, ui: &mut Ui) {
        let mut submit = false;
        let field_width = ui.available_width() - 140.0;
        egui::Grid::new("stream").num_columns(2).show(ui, |ui| {
            let draft = &mut self.stream_draft;
            let mut field = |ui: &mut Ui, label: &str, value: &mut String, password: bool| {
                ui.label(label);
                let response = ui.add(
                    egui::TextEdit::singleline(value)
                        .password(password)
                        .desired_width(field_width),
                );
                submit |= response.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                ui.end_row();
            };
            field(ui, "URL", &mut draft.url, false);
            field(ui, "Username", &mut draft.username, false);
            field(ui, "Password", &mut draft.password, true);
        });
        let changed = self.stream_draft != self.settings.stream;
        ui.horizontal(|ui| {
            submit |= ui
                .button(if changed { "Connect" } else { "Reconnect" })
                .clicked();
            if changed && ui.button("Undo").clicked() {
                self.stream_draft = self.settings.stream.clone();
            }
        });
        if submit {
            self.settings.stream = self.stream_draft.clone();
            self.start_stream();
        }
        ui.label(
            RichText::new(
                "In the phone's IP Camera app: RTSP server on, H.264, same Wi-Fi as this PC. The \
                 stream reconnects by itself when the phone is switched on later.",
            )
            .weak(),
        );
    }

    fn controls_ui(&mut self, ui: &mut Ui) {
        let Some(controls) = &mut self.controls else {
            ui.label(if self.settings.use_stream {
                "Set exposure, focus and white balance in the phone app."
            } else {
                "Not available for this camera."
            });
            return;
        };
        ui.label(
            RichText::new("Manual exposure and white balance give the most stable image.").weak(),
        );
        let mut write = None;
        for (i, c) in controls.list.iter().enumerate() {
            let enabled = !c.inactive && !c.read_only;
            let mut value = c.value;
            ui.add_enabled_ui(enabled, |ui| match c.kind {
                ControlKind::Integer => {
                    let slider = Slider::new(&mut value, c.min..=c.max)
                        .step_by(c.step as f64)
                        .text(&c.name);
                    if ui.add(slider).changed() {
                        write = Some((i, value));
                    }
                }
                ControlKind::Boolean => {
                    let mut on = value != 0;
                    if ui.checkbox(&mut on, &c.name).changed() {
                        write = Some((i, on as i32));
                    }
                }
                ControlKind::Menu => {
                    let current = c
                        .menu
                        .iter()
                        .find(|(idx, _)| *idx == value)
                        .map(|(_, name)| name.clone())
                        .unwrap_or_else(|| value.to_string());
                    ComboBox::from_label(&c.name)
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for (idx, name) in &c.menu {
                                if ui.selectable_label(*idx == value, name).clicked() {
                                    write = Some((i, *idx));
                                }
                            }
                        });
                }
            });
        }
        let reset = ui.button("Reset camera settings to defaults").clicked();

        let device_name = self.settings.device_name.clone().unwrap_or_default();
        let mut error = None;
        if let Some((i, value)) = write {
            let name = controls.list[i].name.clone();
            match controls.set(i, value) {
                Ok(()) => {
                    self.settings
                        .camera_controls
                        .entry(device_name.clone())
                        .or_default()
                        .insert(name, value);
                }
                Err(e) => error = Some(format!("Could not set {name}: {e}")),
            }
        }
        if reset {
            for i in 0..controls.list.len() {
                let default = controls.list[i].default;
                // Fails for controls that became inactive due to an earlier reset; that's fine.
                let _ = controls.set(i, default);
            }
            self.settings.camera_controls.remove(&device_name);
        }
        if let Some(e) = error {
            self.set_status(e, true);
        }
    }

    fn stack_section(&mut self, ui: &mut Ui) {
        ui.add(
            Slider::new(&mut self.settings.stack_depth, 1..=64)
                .logarithmic(true)
                .text("frames averaged"),
        );
        self.pipeline.set_stack_depth(self.settings.stack_depth);
        let stacked = self.pipeline.stats().stacked_frames;
        let depth = self.settings.stack_depth;
        ui.add(
            ProgressBar::new(stacked as f32 / depth as f32)
                .text(format!("{stacked} / {depth} frames")),
        );
        ui.horizontal(|ui| {
            if ui.button("Restart averaging").clicked() {
                self.pipeline.reset_stack();
            }
            let mut frozen = self.frozen;
            if ui.checkbox(&mut frozen, "Freeze input").changed() {
                self.set_frozen(frozen);
            }
        });
        ui.label(
            RichText::new("More frames = less noise, but new strokes take longer to fully appear.")
                .weak(),
        );
    }

    fn sheet_section(&mut self, ui: &mut Ui) {
        let sheet = &mut self.settings.sheet;
        ComboBox::from_label("Paper")
            .selected_text(sheet.paper.label())
            .show_ui(ui, |ui| {
                for p in Paper::ALL {
                    ui.selectable_value(&mut sheet.paper, p, p.label());
                }
            });
        if sheet.paper == Paper::Custom {
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut sheet.custom_mm[0])
                        .range(1.0..=2000.0)
                        .suffix(" mm"),
                );
                ui.label("×");
                ui.add(
                    egui::DragValue::new(&mut sheet.custom_mm[1])
                        .range(1.0..=2000.0)
                        .suffix(" mm"),
                );
            });
        }
        ui.add_enabled_ui(
            !matches!(sheet.paper, Paper::FromCorners | Paper::Square),
            |ui| {
                ui.horizontal(|ui| {
                    ui.radio_value(&mut sheet.orientation, Orientation::Auto, "Auto");
                    ui.radio_value(&mut sheet.orientation, Orientation::Portrait, "Portrait");
                    ui.radio_value(&mut sheet.orientation, Orientation::Landscape, "Landscape");
                });
            },
        );

        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Detect sheet")
                .on_hover_text("Find a bright sheet on a darker desk")
                .clicked()
            {
                match self
                    .pipeline
                    .latest_frame()
                    .as_deref()
                    .and_then(detect_sheet)
                {
                    Some(corners) => {
                        self.settings.corners = corners;
                        self.placing.clear();
                        self.set_status("Sheet detected — fine-tune the corners if needed", false);
                    }
                    None => self.set_status(
                        "No sheet found. Right-click the four sheet corners in the camera view.",
                        true,
                    ),
                }
            }
            let c = self.settings.corners;
            if ui
                .button("Rotate result 90°")
                .on_hover_text("Rotate the result clockwise")
                .clicked()
            {
                self.settings.corners = [c[3], c[0], c[1], c[2]];
            }
            if ui.button("Mirror result").clicked() {
                self.settings.corners = [c[1], c[0], c[3], c[2]];
            }
            if ui.button("Reset corners").clicked() {
                self.settings.corners = DEFAULT_CORNERS;
                self.placing.clear();
            }
        });

        let sheet = &mut self.settings.sheet;
        ui.checkbox(&mut sheet.native_resolution, "Native resolution")
            .on_hover_text("Output roughly as many pixels as the camera sees on the sheet");
        ui.add_enabled(
            !sheet.native_resolution,
            Slider::new(&mut sheet.long_side, 256..=MAX_OUTPUT_SIDE as u32)
                .logarithmic(true)
                .suffix(" px")
                .text("long side"),
        );
        ComboBox::from_label("Interpolation")
            .selected_text(format!("{:?}", sheet.interpolation))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut sheet.interpolation, Interpolation::Bicubic, "Bicubic");
                ui.selectable_value(
                    &mut sheet.interpolation,
                    Interpolation::Bilinear,
                    "Bilinear",
                );
            });
        if let Some(size) = self.pipeline.stats().output_size {
            let dpi = sheet
                .dpi(size)
                .map(|d| format!(" (≈ {d:.0} dpi)"))
                .unwrap_or_default();
            ui.label(format!("Output: {} × {} px{dpi}", size.0, size.1));
        }
        ui.label(
            RichText::new(
                "Right-click the four sheet corners in the camera view, in any order, or drag the \
                 corner handles. Click a handle, then use the arrow keys for pixel-precise \
                 placement (Shift = 10 px). The blue edge is the top. Scroll to zoom, drag to \
                 pan, double-click to see the whole image.",
            )
            .weak(),
        );
    }

    fn enhance_section(&mut self, ui: &mut Ui) {
        let e = &mut self.settings.enhance;
        ui.horizontal_wrapped(|ui| {
            ui.label("Presets:");
            if ui.button("Natural").clicked() {
                *e = EnhanceSettings::natural();
            }
            if ui.button("Scan").clicked() {
                *e = EnhanceSettings::scan();
            }
            if ui.button("Pencil").clicked() {
                *e = EnhanceSettings::pencil();
            }
            if ui.button("Line art").clicked() {
                *e = EnhanceSettings::line_art();
            }
        });
        ui.horizontal(|ui| {
            ui.radio_value(&mut e.mode, ColorMode::Color, "Color");
            ui.radio_value(&mut e.mode, ColorMode::Gray, "Grayscale");
            ui.radio_value(&mut e.mode, ColorMode::BlackWhite, "Black & white");
        });
        ui.checkbox(&mut e.flatten, "Even out lighting")
            .on_hover_text("Removes shadows, vignetting and color casts so the paper turns white");
        ui.add_enabled(
            e.flatten,
            Slider::new(&mut e.flatten_block, 16..=256)
                .logarithmic(true)
                .suffix(" px")
                .text("lighting detail"),
        );
        ui.add(Slider::new(&mut e.black_point, 0.0..=0.9).text("black point"));
        ui.add(Slider::new(&mut e.white_point, 0.1..=1.0).text("white point"));
        if e.white_point <= e.black_point + 0.02 {
            e.white_point = (e.black_point + 0.02).min(1.0);
        }
        ui.add(
            Slider::new(&mut e.gamma, 0.3..=3.0)
                .logarithmic(true)
                .text("gamma"),
        );
        ui.add_enabled(
            e.mode == ColorMode::Color,
            Slider::new(&mut e.saturation, 0.0..=3.0).text("saturation"),
        );
        ui.add(Slider::new(&mut e.sharpen_amount, 0.0..=3.0).text("sharpen"));
        ui.add_enabled(
            e.sharpen_amount > 0.0,
            Slider::new(&mut e.sharpen_radius, 0.5..=6.0)
                .suffix(" px")
                .text("sharpen radius"),
        );
        ui.add_enabled(
            e.mode == ColorMode::BlackWhite,
            Slider::new(&mut e.threshold, 0.05..=0.98).text("threshold"),
        );
    }

    fn export_section(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Folder");
            ui.add(
                egui::TextEdit::singleline(&mut self.settings.output_dir)
                    .desired_width(ui.available_width() - 60.0),
            );
            if ui.button("Open").clicked() {
                let dir = expand_home(&self.settings.output_dir);
                let _ = std::fs::create_dir_all(&dir);
                if let Err(e) = std::process::Command::new("xdg-open").arg(&dir).spawn() {
                    self.set_status(format!("Could not open folder: {e}"), true);
                }
            }
        });
        let share = Path::new(&self.settings.output_dir).join(SHARE_FILE);
        ui.label(
            RichText::new(format!(
                "Save and Copy are above each image; Ctrl+S and Ctrl+C save and copy the result. \
                 Every save also updates {} (at most {SHARE_MAX_SIDE} px), e.g. for Claude Code or \
                 Codex: \"look at {0}\".",
                share.display()
            ))
            .weak(),
        );
    }

    fn central(&mut self, ui: &mut Ui) {
        let rect = ui.available_rect_before_wrap();
        match self.settings.view {
            View::Camera => self.camera_pane(ui, rect),
            View::Result => self.result_pane(ui, rect),
            View::Both => {
                // A draggable divider; double-click puts it back in the middle.
                let min = (MIN_VIEW_WIDTH / rect.width()).min(0.5);
                let split_x = |split: f32| rect.left() + rect.width() * split.clamp(min, 1.0 - min);
                let x = split_x(self.settings.split);
                let handle = Rect::from_x_y_ranges(x - 4.0..=x + 4.0, rect.y_range());
                let divider = ui.interact(
                    handle,
                    ui.scope_id().with("divider"),
                    Sense::click_and_drag(),
                );
                if divider.dragged()
                    && let Some(p) = divider.interact_pointer_pos()
                {
                    self.settings.split = (p.x - rect.left()) / rect.width();
                }
                if divider.double_clicked() {
                    self.settings.split = 0.5;
                }
                self.settings.split = self.settings.split.clamp(min, 1.0 - min);
                let active = divider.hovered() || divider.dragged();
                if active {
                    ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
                }
                let x = split_x(self.settings.split);
                let (width, color) = if active {
                    (2.0, ACCENT)
                } else {
                    (1.0, Color32::from_gray(70))
                };
                ui.painter()
                    .vline(x, rect.y_range(), Stroke::new(width, color));
                self.camera_pane(
                    ui,
                    Rect::from_x_y_ranges(rect.left()..=x - 4.0, rect.y_range()),
                );
                self.result_pane(
                    ui,
                    Rect::from_x_y_ranges(x + 4.0..=rect.right(), rect.y_range()),
                );
            }
        }
    }

    /// The camera image with its buttons above it.
    fn camera_pane(&mut self, ui: &mut Ui, rect: Rect) {
        let (bar, view) = rect.split_top_bottom_at_y(rect.top() + VIEW_BAR_HEIGHT);
        let [mut copy, mut rotate, mut mirror, mut save] = [false; 4];
        view_bar(ui, bar, "Camera", |ui| {
            copy = ui
                .button("Copy")
                .on_hover_text("Copy the camera image to the clipboard")
                .clicked();
            rotate = ui
                .button("Rotate 90°")
                .on_hover_text(
                    "Turn the camera image clockwise. The corners stay on the sheet, so the \
                     result keeps its orientation.",
                )
                .clicked();
            mirror = ui
                .button("Mirror")
                .on_hover_text(
                    "Mirror the camera image left-right, e.g. for a camera that delivers \
                     mirrored pictures. The result is mirrored too.",
                )
                .clicked();
            save = ui
                .button("Save")
                .on_hover_text("Save the camera image as PNG in the export folder")
                .clicked();
        });
        let ctx = ui.ctx().clone();
        if copy {
            self.copy(&ctx, Picture::Camera);
        }
        if rotate {
            self.rotate_camera(&ctx);
        }
        if mirror {
            self.mirror_camera(&ctx);
        }
        if save {
            self.save(&ctx, Picture::Camera);
        }
        self.camera_view(ui, view);
    }

    /// The result with its buttons above it.
    fn result_pane(&mut self, ui: &mut Ui, rect: Rect) {
        let (bar, view) = rect.split_top_bottom_at_y(rect.top() + VIEW_BAR_HEIGHT);
        let [mut copy, mut save] = [false; 2];
        view_bar(ui, bar, "Result", |ui| {
            copy = ui
                .button("Copy")
                .on_hover_text("Copy the result to the clipboard (Ctrl+C)")
                .clicked();
            save = ui
                .button("Save")
                .on_hover_text("Save the result as PNG in the export folder (Ctrl+S)")
                .clicked();
        });
        let ctx = ui.ctx().clone();
        if copy {
            self.copy(&ctx, Picture::Result);
        }
        if save {
            self.save(&ctx, Picture::Result);
        }
        self.result_view(ui, view);
    }

    fn camera_view(&mut self, ui: &mut Ui, rect: Rect) {
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, CornerRadius::same(4), Color32::from_gray(24));
        let Some(tex) = self.preview.clone() else {
            let error = self
                .capture
                .as_ref()
                .and_then(|c| c.status.lock().unwrap().error.clone());
            let text = if self.settings.use_stream {
                self.capture_problem()
                    .unwrap_or_else(|| "Waiting for the stream…".into())
            } else if self.devices.is_empty() {
                "No camera found".to_string()
            } else if let Some(e) = error {
                format!("{e}\nPick another camera on the left or close the other application.")
            } else {
                "Waiting for camera…".to_string()
            };
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                text,
                FontId::proportional(18.0),
                Color32::GRAY,
            );
            return;
        };
        let size = tex.size_vec2();

        // A left drag that starts on a corner handle moves the corner; any other drag pans.
        if response.drag_started() {
            let img_rect = self.camera_zoom.image_rect(rect, size);
            let pts = self.settings.corners.map(|c| to_screen(img_rect, c));
            let origin = ui.input(|i| i.pointer.press_origin());
            self.drag = origin
                .filter(|_| {
                    self.placing.is_empty() && response.drag_started_by(PointerButton::Primary)
                })
                .and_then(|o| nearest_corner(&pts, o).map(|i| (i, pts[i] - o)));
            if let Some((i, _)) = self.drag {
                self.selected_corner = Some(i);
            }
        }
        let panning = response.dragged()
            && self.drag.is_none()
            && !response.dragged_by(PointerButton::Secondary);
        self.camera_zoom.update(ui, &response, rect, size, panning);
        let img_rect = self.camera_zoom.image_rect(rect, size);
        painter.image(tex.id(), img_rect, unit_uv(), Color32::WHITE);
        let source = self
            .camera_zoom
            .zoomed()
            .then(|| self.pipeline.latest_frame())
            .flatten();
        self.camera_zoom.draw_detail(
            &painter,
            "camera detail",
            source.as_ref(),
            size,
            rect,
            img_rect,
        );

        if response.dragged()
            && let (Some((i, offset)), Some(pos)) = (self.drag, response.interact_pointer_pos())
        {
            self.settings.corners[i] = to_norm(img_rect, pos + offset);
        }
        if response.drag_stopped() {
            self.drag = None;
        }
        // Right-clicks place a new outline corner by corner; when four corners exist, the first
        // right-click starts over.
        if response.secondary_clicked()
            && let Some(pos) = response.interact_pointer_pos()
            && img_rect.contains(pos)
        {
            self.placing.push(to_norm(img_rect, pos));
            self.selected_corner = None;
            if let Ok(placed) = <[[f32; 2]; 4]>::try_from(self.placing.as_slice()) {
                self.settings.corners = order_normalized(placed, size);
                self.placing.clear();
            }
        }
        let pts = self.settings.corners.map(|c| to_screen(img_rect, c));
        if response.clicked() {
            self.selected_corner = response
                .interact_pointer_pos()
                .filter(|_| self.placing.is_empty())
                .and_then(|p| nearest_corner(&pts, p));
        }
        if response.clicked() || response.drag_started() {
            // Hand the arrow keys over to corner nudging.
            ui.memory_mut(|m| {
                if let Some(id) = m.focused() {
                    m.surrender_focus(id);
                }
            });
        }
        let pointer = response.hover_pos();
        let hovered = pointer
            .filter(|_| self.placing.is_empty())
            .and_then(|p| nearest_corner(&pts, p));
        if self.drag.is_some() || (panning && self.camera_zoom.zoomed()) {
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
        } else if hovered.is_some() {
            ui.ctx().set_cursor_icon(CursorIcon::Grab);
        }

        if self.placing.is_empty() {
            dim_outside(&painter, img_rect, pts);
            painter.add(Shape::closed_line(pts.to_vec(), Stroke::new(1.5, ACCENT)));
            painter.line_segment([pts[0], pts[1]], Stroke::new(3.0, TOP_EDGE));
            let center = pts.iter().fold(Vec2::ZERO, |acc, p| acc + p.to_vec2()) / 4.0;
            for (i, &p) in pts.iter().enumerate() {
                let active = self.selected_corner == Some(i) || hovered == Some(i);
                let fill = if active {
                    ACCENT
                } else {
                    Color32::from_black_alpha(160)
                };
                painter.circle_filled(p, HANDLE_RADIUS, fill);
                painter.circle_stroke(p, HANDLE_RADIUS, Stroke::new(2.0, ACCENT));
                // Label outside the quad so it doesn't cover the content.
                let away = (p.to_vec2() - center).normalized() * 18.0;
                painter.text(
                    p + away,
                    Align2::CENTER_CENTER,
                    CORNER_LABELS[i],
                    FontId::monospace(13.0),
                    ACCENT,
                );
            }
        } else {
            // The new outline so far, with a rubber band to the pointer.
            let placed: Vec<Pos2> = self
                .placing
                .iter()
                .map(|&c| to_screen(img_rect, c))
                .collect();
            let line = placed.iter().copied().chain(pointer).collect();
            painter.add(Shape::line(line, Stroke::new(1.5, ACCENT)));
            for (i, &p) in placed.iter().enumerate() {
                painter.circle_filled(p, HANDLE_RADIUS, ACCENT);
                painter.text(
                    p + vec2(0.0, -18.0),
                    Align2::CENTER_CENTER,
                    (i + 1).to_string(),
                    FontId::monospace(13.0),
                    ACCENT,
                );
            }
        }

        let loupe_corner = self.drag.map(|(i, _)| i).or(self.selected_corner);
        if let Some(i) = loupe_corner {
            let minimap = self.camera_zoom.zoomed().then(|| minimap_rect(rect, size));
            draw_loupe(
                &painter,
                &tex,
                rect,
                self.settings.corners[i],
                pts[i],
                minimap,
            );
        }

        // The image shown is the last one received; say so when the source is gone.
        let visible = img_rect.intersect(rect);
        if let Some(problem) = self.capture_problem() {
            let galley = painter.layout(
                problem,
                FontId::proportional(14.0),
                Color32::WHITE,
                visible.width() - 40.0,
            );
            let banner = Rect::from_min_size(
                visible.left_top() + vec2(8.0, 8.0),
                galley.size() + vec2(16.0, 10.0),
            );
            painter.rect_filled(
                banner,
                CornerRadius::same(4),
                Color32::from_black_alpha(200),
            );
            painter.galley(banner.min + vec2(8.0, 5.0), galley, Color32::WHITE);
        }

        let source_width = source.map_or(size.x, |s| s.width as f32);
        let idle = !response.dragged() && response.hovered();
        self.camera_zoom
            .draw_overlay(&painter, &tex, rect, img_rect, source_width, idle);
        if let Some(p) = pointer.filter(|p| idle && hovered.is_none() && img_rect.contains(*p)) {
            let text = match self.placing.len() {
                0 => "Right-click: set corner".to_string(),
                n => format!("Right-click: corner {} of 4 · Esc: cancel", n + 1),
            };
            // Beside the pointer, on the side with more room.
            let h = if p.x > rect.center().x {
                Align::Max
            } else {
                Align::Min
            };
            let v = if p.y > rect.bottom() - 60.0 {
                Align::Max
            } else {
                Align::Min
            };
            let offset = vec2(
                if h == Align::Max { -14.0 } else { 14.0 },
                if v == Align::Max { -10.0 } else { 20.0 },
            );
            draw_tag(&painter, p + offset, Align2([h, v]), text);
        }
    }

    fn result_view(&mut self, ui: &mut Ui, rect: Rect) {
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, CornerRadius::same(4), Color32::from_gray(24));
        if let Some(e) = self.pipeline.error() {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                e,
                FontId::proportional(16.0),
                ui.visuals().error_fg_color,
            );
            return;
        }
        let Some(tex) = self.result_tex.clone() else {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "No result yet",
                FontId::proportional(18.0),
                Color32::GRAY,
            );
            return;
        };
        let size = tex.size_vec2();
        self.result_zoom
            .update(ui, &response, rect, size, response.dragged());
        let img_rect = self.result_zoom.image_rect(rect, size);
        painter.image(tex.id(), img_rect, unit_uv(), Color32::WHITE);
        let source = self.result.as_ref().map(|(image, _)| image.clone());
        self.result_zoom.draw_detail(
            &painter,
            "result detail",
            source.as_ref(),
            size,
            rect,
            img_rect,
        );
        if response.dragged() && self.result_zoom.zoomed() {
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
        }
        painter.rect_stroke(
            img_rect,
            CornerRadius::ZERO,
            Stroke::new(1.0, Color32::from_gray(80)),
            StrokeKind::Outside,
        );
        if self.frozen {
            painter.text(
                rect.left_top() + vec2(16.0, 16.0),
                Align2::LEFT_TOP,
                "FROZEN",
                FontId::proportional(14.0),
                ACCENT,
            );
        }
        let source_width = source.map_or(size.x, |s| s.width as f32);
        let idle = !response.dragged() && response.hovered();
        self.result_zoom
            .draw_overlay(&painter, &tex, rect, img_rect, source_width, idle);
    }
}

impl eframe::App for OverheadApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        self.handle_shortcuts(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
        egui::Panel::left("settings")
            .resizable(true)
            .default_size(360.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink(false)
                    .show(ui, |ui| self.settings_panel(ui));
            });
        egui::CentralPanel::default().show(ui, |ui| self.central(ui));

        self.pipeline
            .set_settings(&process_settings(&self.settings));
        // Keep status texts (capture errors, timeouts) fresh even without new frames.
        ctx.request_repaint_after(Duration::from_millis(500));
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.remember_corners();
        eframe::set_value(storage, eframe::APP_KEY, &self.settings);
    }
}

/// A row of buttons above a view, with the view's name on the right.
fn view_bar(ui: &mut Ui, bar: Rect, name: &str, add_buttons: impl FnOnce(&mut Ui)) {
    let layout = egui::Layout::left_to_right(Align::Center);
    ui.scope_builder(egui::UiBuilder::new().max_rect(bar).layout(layout), |ui| {
        add_buttons(ui);
        ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
            ui.weak(name);
        });
    });
}

fn process_settings(s: &Settings) -> ProcessSettings {
    ProcessSettings {
        corners: s.corners,
        sheet: s.sheet.clone(),
        enhance: s.enhance.clone(),
    }
}

fn unit_uv() -> Rect {
    Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0))
}

/// Largest rect with the aspect ratio of `size` that fits centered into `outer`.
fn fit(outer: Rect, size: Vec2) -> Rect {
    let scale = (outer.width() / size.x).min(outer.height() / size.y);
    Rect::from_center_size(outer.center(), size * scale)
}

/// Screen position of a normalized image point.
fn to_screen(img_rect: Rect, p: [f32; 2]) -> Pos2 {
    img_rect.min + vec2(p[0], p[1]) * img_rect.size()
}

/// Normalized image point at a screen position, clamped to the image.
fn to_norm(img_rect: Rect, p: Pos2) -> [f32; 2] {
    let n = (p - img_rect.min) / img_rect.size();
    [n.x.clamp(0.0, 1.0), n.y.clamp(0.0, 1.0)]
}

/// The corner handle within grabbing distance of `pos`, the nearest one if several are.
fn nearest_corner(pts: &[Pos2; 4], pos: Pos2) -> Option<usize> {
    (0..4)
        .map(|i| (i, pts[i].distance(pos)))
        .filter(|(_, d)| *d <= GRAB_RADIUS)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// Puts corners clicked in any order into TL, TR, BR, BL order, judged in the pixels of an image
/// of `size`.
fn order_normalized(corners: [[f32; 2]; 4], size: Vec2) -> [[f32; 2]; 4] {
    let px = corners.map(|[x, y]| [(x * size.x) as f64, (y * size.y) as f64]);
    order_corners(px).map(|[x, y]| [(x / size.x as f64) as f32, (y / size.y as f64) as f32])
}

/// Where the minimap of an image of `size` goes: bottom-right in `view`, at most a third of it.
fn minimap_rect(view: Rect, size: Vec2) -> Rect {
    let bounds = Rect::from_min_size(Pos2::ZERO, MINIMAP_SIZE.min(view.size() / 3.0));
    let map = fit(bounds, size).size();
    Rect::from_min_size(view.right_bottom() - map - Vec2::splat(VIEW_MARGIN), map)
}

/// Small text on a dark background, placed at `pos` according to `align`.
fn draw_tag(painter: &egui::Painter, pos: Pos2, align: Align2, text: String) {
    let galley = painter.layout_no_wrap(text, FontId::proportional(12.0), Color32::WHITE);
    let rect = align.anchor_size(pos, galley.size() + vec2(8.0, 4.0));
    painter.rect_filled(rect, CornerRadius::same(3), Color32::from_black_alpha(200));
    painter.galley(rect.min + vec2(4.0, 2.0), galley, Color32::WHITE);
}

/// Darkens the camera image outside the selected quad.
fn dim_outside(painter: &egui::Painter, img_rect: Rect, pts: [Pos2; 4]) {
    // Sort into screen-clockwise order so each quad corner pairs with the nearest image corner,
    // independent of how the user rotated the corner assignment.
    let q =
        order_corners(pts.map(|p| [p.x as f64, p.y as f64])).map(|[x, y]| pos2(x as f32, y as f32));
    let r = [
        img_rect.left_top(),
        img_rect.right_top(),
        img_rect.right_bottom(),
        img_rect.left_bottom(),
    ];
    let color = Color32::from_black_alpha(140);
    let mut mesh = Mesh::default();
    for i in 0..4 {
        let j = (i + 1) % 4;
        let base = mesh.vertices.len() as u32;
        for p in [r[i], r[j], q[j], q[i]] {
            mesh.colored_vertex(p, color);
        }
        mesh.add_triangle(base, base + 1, base + 2);
        mesh.add_triangle(base, base + 2, base + 3);
    }
    painter.add(Shape::mesh(mesh));
}

/// Magnified view around a corner for precise placement, in the view corner farthest from the
/// handle that `avoid` (the minimap) leaves free.
fn draw_loupe(
    painter: &egui::Painter,
    tex: &TextureHandle,
    view: Rect,
    corner: [f32; 2],
    screen: Pos2,
    avoid: Option<Rect>,
) {
    const SIZE: f32 = 180.0;
    const HALF_TEXELS: f32 = 24.0;
    let inner = view.shrink(VIEW_MARGIN);
    let spots = [
        inner.left_top(),
        inner.right_top() - vec2(SIZE, 0.0),
        inner.left_bottom() - vec2(0.0, SIZE),
        inner.right_bottom() - Vec2::splat(SIZE),
    ]
    .map(|min| Rect::from_min_size(min, Vec2::splat(SIZE)));
    let Some(rect) = spots
        .into_iter()
        .filter(|r| avoid.is_none_or(|a| !r.intersects(a)))
        .max_by(|a, b| {
            a.center()
                .distance(screen)
                .total_cmp(&b.center().distance(screen))
        })
    else {
        return;
    };
    let [tw, th] = tex.size().map(|v| v as f32);
    let half = vec2(HALF_TEXELS / tw, HALF_TEXELS / th);
    let uv = Rect::from_min_max(
        pos2(corner[0], corner[1]) - half,
        pos2(corner[0], corner[1]) + half,
    );
    painter.rect_filled(rect, CornerRadius::ZERO, Color32::BLACK);
    painter.image(tex.id(), rect, uv, Color32::WHITE);
    let c = rect.center();
    let stroke = Stroke::new(1.0, ACCENT);
    painter.line_segment([pos2(rect.left(), c.y), pos2(c.x - 6.0, c.y)], stroke);
    painter.line_segment([pos2(c.x + 6.0, c.y), pos2(rect.right(), c.y)], stroke);
    painter.line_segment([pos2(c.x, rect.top()), pos2(c.x, c.y - 6.0)], stroke);
    painter.line_segment([pos2(c.x, c.y + 6.0), pos2(c.x, rect.bottom())], stroke);
    painter.rect_stroke(
        rect,
        CornerRadius::ZERO,
        Stroke::new(2.0, ACCENT),
        StrokeKind::Outside,
    );
}

fn default_output_dir() -> String {
    let pictures = std::process::Command::new("xdg-user-dir")
        .arg("PICTURES")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| PathBuf::from(s.trim()))
        .filter(|p| p.is_dir())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Pictures")))
        .unwrap_or_else(|| PathBuf::from("."));
    pictures
        .join("overheadcamera")
        .to_string_lossy()
        .into_owned()
}

fn expand_home(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(path),
    }
}

/// Saves `image` as `<name>_<timestamp>.png` in `dir` and a smaller copy as [`SHARE_FILE`].
fn save_png(image: &RgbImage, grayscale: bool, dir: &Path, name: &str) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let stamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    let mut path = dir.join(format!("{name}_{stamp}.png"));
    let mut n = 1;
    while path.exists() {
        path = dir.join(format!("{name}_{stamp}_{n}.png"));
        n += 1;
    }
    let (w, h) = (image.width as u32, image.height as u32);
    let img = if grayscale {
        let luma = image.data.as_chunks::<3>().0.iter().map(|p| p[0]).collect();
        image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(w, h, luma).unwrap())
    } else {
        image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, image.data.clone()).unwrap())
    };
    img.save(&path).map_err(|e| e.to_string())?;

    let share = if w.max(h) > SHARE_MAX_SIDE {
        img.resize(
            SHARE_MAX_SIDE,
            SHARE_MAX_SIDE,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        img
    };
    // Written under another name first so that readers never see a half-written file.
    let partial = dir.join(".latest-partial.png");
    share.save(&partial).map_err(|e| e.to_string())?;
    std::fs::rename(&partial, dir.join(SHARE_FILE)).map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_keeps_the_point_under_the_pointer() {
        let view = Rect::from_min_size(Pos2::ZERO, vec2(824.0, 624.0));
        let size = vec2(4000.0, 3000.0);
        let mut zp = ZoomPan::default();
        // The whole image is fitted into the 800×600 area inside the margin.
        assert_eq!(
            zp.image_rect(view, size),
            Rect::from_min_size(pos2(12.0, 12.0), vec2(800.0, 600.0))
        );

        let pointer = pos2(212.0, 162.0);
        let before = to_norm(zp.image_rect(view, size), pointer);
        zp.zoom_at(pointer, 3.0, view, size);
        zp.clamp(view, size);
        assert_eq!(zp.zoom, 3.0);
        let after = to_norm(zp.image_rect(view, size), pointer);
        assert!((after[0] - before[0]).abs() < 1e-5 && (after[1] - before[1]).abs() < 1e-5);

        // Panning past the edge stops with the image edge at the view edge.
        zp.center = vec2(-1.0, 2.0);
        zp.clamp(view, size);
        let shown = zp.image_rect(view, size);
        assert!((shown.left() - 12.0).abs() < 1e-3 && (shown.bottom() - 612.0).abs() < 1e-3);

        // Zoom is limited to MAX_PIXEL_SIZE points per image pixel, and never below fitting.
        zp.zoom_at(pointer, 1e6, view, size);
        assert_eq!(zp.image_rect(view, size).width(), size.x * MAX_PIXEL_SIZE);
        zp.zoom_at(pointer, 1e-6, view, size);
        zp.clamp(view, size);
        assert_eq!(zp.zoom, 1.0);
        assert_eq!(zp.center, vec2(0.5, 0.5));
    }

    #[test]
    fn clicked_corners_are_ordered() {
        // Clicked as BR, TL, BL, TR on a wide image.
        let clicked = [[0.8, 0.9], [0.2, 0.1], [0.1, 0.8], [0.9, 0.2]];
        let ordered = order_normalized(clicked, vec2(1600.0, 900.0));
        assert_eq!(ordered, [[0.2, 0.1], [0.9, 0.2], [0.8, 0.9], [0.1, 0.8]]);
    }

    #[test]
    fn save_png_never_overwrites_and_keeps_gray_small() {
        let dir = std::env::temp_dir().join(format!("overheadcamera-test-{}", std::process::id()));
        let mut img = RgbImage::new(4, 3);
        img.data.fill(200);
        let a = save_png(&img, true, &dir, "sketch").unwrap();
        let b = save_png(&img, false, &dir, "sketch").unwrap();
        assert_ne!(a, b);
        let gray = image::open(&a).unwrap();
        assert_eq!(gray.color(), image::ColorType::L8);
        assert_eq!((gray.width(), gray.height()), (4, 3));
        assert_eq!(image::open(&b).unwrap().color(), image::ColorType::Rgb8);
        assert_eq!(image::open(dir.join(SHARE_FILE)).unwrap().width(), 4);

        let wide = RgbImage::new(3200, 100);
        let c = save_png(&wide, true, &dir, "sketch").unwrap();
        assert_eq!(image::open(&c).unwrap().width(), 3200);
        let shared = image::open(dir.join(SHARE_FILE)).unwrap();
        assert_eq!((shared.width(), shared.height()), (SHARE_MAX_SIDE, 49));
        assert_eq!(shared.color(), image::ColorType::L8);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
