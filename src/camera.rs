//! V4L2 camera access: device discovery, the capture thread and camera controls.

use std::{
    io,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{SyncSender, TrySendError},
    },
    thread::JoinHandle,
};

use linuxvideo::{
    BufType, CapabilityFlags, Device, Fract,
    controls::{Cid, ControlFlags, CtrlType},
    format::{FrameSizes, PixFormat, PixelFormat},
};
use serde::{Deserialize, Serialize};

/// Pixel encodings we know how to decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Encoding {
    Mjpeg,
    Yuyv,
    /// Planar 4:2:0 (Y, then U, then V), as delivered by the RTSP source.
    I420,
}

impl Encoding {
    fn from_pixel_format(pf: PixelFormat) -> Option<Self> {
        if pf == PixelFormat::MJPG || pf == PixelFormat::JPEG {
            Some(Self::Mjpeg)
        } else if pf == PixelFormat::YUYV {
            Some(Self::Yuyv)
        } else {
            None
        }
    }

    fn pixel_format(self) -> PixelFormat {
        match self {
            Self::Mjpeg => PixelFormat::MJPG,
            Self::Yuyv => PixelFormat::YUYV,
            Self::I420 => PixelFormat::from_fourcc(*b"YU12"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Mjpeg => "MJPEG",
            Self::Yuyv => "YUYV",
            Self::I420 => "I420",
        }
    }
}

/// A capture mode: encoding, resolution and the fastest frame interval for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mode {
    pub encoding: Encoding,
    pub width: u32,
    pub height: u32,
    /// Frame interval in seconds = num / den.
    pub interval_num: u32,
    pub interval_den: u32,
}

impl Mode {
    pub fn fps(&self) -> f32 {
        self.interval_den as f32 / self.interval_num.max(1) as f32
    }

    pub fn label(&self) -> String {
        format!(
            "{}×{} {} @ {:.0} fps",
            self.width,
            self.height,
            self.encoding.label(),
            self.fps()
        )
    }
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub path: PathBuf,
    pub name: String,
    /// Sorted largest resolution first, MJPEG before YUYV.
    pub modes: Vec<Mode>,
}

impl DeviceInfo {
    pub fn label(&self) -> String {
        format!("{} ({})", self.name, self.path.display())
    }

    /// The highest resolution mode, preferring MJPEG (YUYV is usually limited to low fps).
    pub fn default_mode(&self) -> Option<Mode> {
        self.modes.first().copied()
    }
}

/// Lists all video capture devices that offer at least one decodable mode.
pub fn list_devices() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();
    let Ok(iter) = linuxvideo::list() else {
        return devices;
    };
    for device in iter.flatten() {
        let Ok(path) = device.path() else { continue };
        if !path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("video"))
        {
            continue;
        }
        let Ok(caps) = device.capabilities() else {
            continue;
        };
        if !caps
            .device_capabilities()
            .contains(CapabilityFlags::VIDEO_CAPTURE)
        {
            continue;
        }
        let modes = enumerate_modes(&device);
        if modes.is_empty() {
            continue;
        }
        devices.push(DeviceInfo {
            path,
            name: caps.card().to_string(),
            modes,
        });
    }
    devices.sort_by(|a, b| a.path.cmp(&b.path));
    devices
}

fn enumerate_modes(device: &Device) -> Vec<Mode> {
    let mut modes = Vec::new();
    for fmt in device.formats(BufType::VIDEO_CAPTURE).flatten() {
        let Some(encoding) = Encoding::from_pixel_format(fmt.pixel_format()) else {
            continue;
        };
        let Ok(sizes) = device.frame_sizes(fmt.pixel_format()) else {
            continue;
        };
        let sizes: Vec<(u32, u32)> = match sizes {
            FrameSizes::Discrete(list) => list.iter().map(|s| (s.width(), s.height())).collect(),
            FrameSizes::Stepwise(s) | FrameSizes::Continuous(s) => {
                vec![(s.max_width(), s.max_height())]
            }
        };
        for (width, height) in sizes {
            let interval = device
                .frame_intervals(fmt.pixel_format(), width, height)
                .map(|ivals| ivals.min())
                .unwrap_or(Fract::new(1, 30));
            let mode = Mode {
                encoding,
                width,
                height,
                interval_num: interval.numerator(),
                interval_den: interval.denominator(),
            };
            if !modes.contains(&mode) {
                modes.push(mode);
            }
        }
    }
    modes.sort_by_key(|m| {
        (
            std::cmp::Reverse(m.width as u64 * m.height as u64),
            m.encoding != Encoding::Mjpeg,
        )
    });
    modes
}

/// A frame exactly as it came from the driver, before decoding.
pub struct RawFrame {
    pub encoding: Encoding,
    pub width: usize,
    pub height: usize,
    pub bytes_per_line: usize,
    pub data: Vec<u8>,
}

#[derive(Default, Clone)]
pub struct CaptureStatus {
    /// The format the driver actually negotiated.
    pub negotiated: Option<Mode>,
    pub error: Option<String>,
    pub running: bool,
}

/// Owns the capture thread; dropping it stops the thread and releases the device.
pub struct Capture {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    pub status: Arc<Mutex<CaptureStatus>>,
}

impl Capture {
    pub fn start(path: &Path, mode: Mode, sink: SyncSender<RawFrame>) -> Self {
        let path = path.to_owned();
        Self::spawn(move |stop, status| {
            if let Err(e) = capture_loop(&path, mode, &sink, stop, status) {
                let message = if e.raw_os_error() == Some(libc::EBUSY) {
                    "camera is in use by another application".to_string()
                } else {
                    e.to_string()
                };
                status.lock().unwrap().error = Some(format!("{}: {message}", path.display()));
            }
        })
    }

    /// Runs `body` on the capture thread. It has to return soon after `stop` is set.
    pub fn spawn(body: impl FnOnce(&AtomicBool, &Mutex<CaptureStatus>) + Send + 'static) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(CaptureStatus::default()));
        let thread = {
            let stop = stop.clone();
            let status = status.clone();
            std::thread::Builder::new()
                .name("capture".into())
                .spawn(move || {
                    body(&stop, &status);
                    status.lock().unwrap().running = false;
                })
                .expect("failed to spawn capture thread")
        };
        Self {
            stop,
            thread: Some(thread),
            status,
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            // Bounded by the poll timeouts in the capture loops.
            let _ = thread.join();
        }
    }
}

fn capture_loop(
    path: &Path,
    mode: Mode,
    sink: &SyncSender<RawFrame>,
    stop: &AtomicBool,
    status: &Mutex<CaptureStatus>,
) -> io::Result<()> {
    let device = Device::open(path)?;
    let capture = device.video_capture(PixFormat::new(
        mode.width,
        mode.height,
        mode.encoding.pixel_format(),
    ))?;
    let format = capture.format();
    let (width, height) = (format.width(), format.height());
    let bytes_per_line = format.bytes_per_line() as usize;
    let Some(encoding) = Encoding::from_pixel_format(format.pixel_format()) else {
        return Err(io::Error::other(format!(
            "driver negotiated unsupported pixel format {}",
            format.pixel_format()
        )));
    };
    let interval = capture
        .set_frame_interval(Fract::new(mode.interval_num, mode.interval_den))
        .unwrap_or(Fract::new(mode.interval_num, mode.interval_den));
    {
        let mut status = status.lock().unwrap();
        status.negotiated = Some(Mode {
            encoding,
            width,
            height,
            interval_num: interval.numerator(),
            interval_den: interval.denominator(),
        });
        status.running = true;
        status.error = None;
    }

    let mut stream = capture.into_stream()?;
    let fd = stream.as_raw_fd();
    while !stop.load(Ordering::Relaxed) {
        // Wait with a timeout so a stalled camera can't keep us from noticing `stop`.
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut pfd, 1, 200) };
        if ready < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if ready == 0 {
            continue;
        }

        let frame = stream.dequeue(|buf| {
            Ok((!buf.is_error() && !buf.is_empty()).then(|| RawFrame {
                encoding,
                width: width as usize,
                height: height as usize,
                bytes_per_line,
                data: buf.to_vec(),
            }))
        })?;
        if let Some(frame) = frame {
            match sink.try_send(frame) {
                // A full queue means processing is behind; dropping keeps latency low.
                Ok(()) | Err(TrySendError::Full(_)) => {}
                Err(TrySendError::Disconnected(_)) => break,
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlKind {
    Integer,
    Boolean,
    Menu,
}

#[derive(Clone, Debug)]
pub struct ControlInfo {
    pub cid: Cid,
    pub name: String,
    pub kind: ControlKind,
    pub min: i32,
    pub max: i32,
    pub step: i32,
    pub default: i32,
    pub value: i32,
    /// e.g. manual exposure time while auto exposure is on.
    pub inactive: bool,
    pub read_only: bool,
    pub menu: Vec<(i32, String)>,
}

/// A second handle on the device used only for controls, so they can be changed while streaming.
pub struct Controls {
    device: Device,
    pub list: Vec<ControlInfo>,
}

impl Controls {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut this = Self {
            device: Device::open(path)?,
            list: Vec::new(),
        };
        this.refresh();
        Ok(this)
    }

    /// Re-reads all controls; flags and values of other controls can change after a write.
    pub fn refresh(&mut self) {
        let mut list = Vec::new();
        for desc in self.device.controls().flatten() {
            let kind = match desc.control_type() {
                CtrlType::INTEGER => ControlKind::Integer,
                CtrlType::BOOLEAN => ControlKind::Boolean,
                CtrlType::MENU => ControlKind::Menu,
                _ => continue,
            };
            let flags = desc.flags();
            if flags.contains(ControlFlags::WRITE_ONLY) {
                continue;
            }
            let Ok(value) = self.device.read_control_raw(desc.id()) else {
                continue;
            };
            let menu = if kind == ControlKind::Menu {
                self.device
                    .enumerate_menu(&desc)
                    .flatten()
                    .map(|item| (item.index() as i32, item.name().to_string()))
                    .collect()
            } else {
                Vec::new()
            };
            list.push(ControlInfo {
                cid: desc.id(),
                name: desc.name().to_string(),
                kind,
                min: desc.minimum(),
                max: desc.maximum(),
                step: desc.step().max(1),
                default: desc.default_value(),
                value,
                inactive: flags.contains(ControlFlags::INACTIVE),
                read_only: flags.contains(ControlFlags::READ_ONLY)
                    || flags.contains(ControlFlags::GRABBED),
                menu,
            });
        }
        self.list = list;
    }

    pub fn set(&mut self, index: usize, value: i32) -> io::Result<()> {
        let cid = self.list[index].cid;
        let result = self.device.write_control_raw(cid, value);
        self.refresh();
        result
    }
}
