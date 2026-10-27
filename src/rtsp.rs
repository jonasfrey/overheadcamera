//! Network cameras over RTSP, such as a phone running an IP camera app.
//!
//! `retina` runs the RTSP session (interleaved over TCP, with digest or basic authentication) and
//! reassembles the H.264 frames, OpenH264 decodes them into I420 planes. Both are built by cargo,
//! so nothing has to be installed on the system. Lost connections are retried, so the phone can be
//! switched on after the app was started.

use std::{
    pin::pin,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{SyncSender, TrySendError},
    },
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use openh264::{decoder::Decoder, formats::YUVSource};
use percent_encoding::percent_decode_str;
use retina::{
    client::{Credentials, PlayOptions, Session, SessionOptions, SetupOptions, Transport},
    codec::{CodecItem, FrameFormat, ParametersRef},
};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::camera::{Capture, CaptureStatus, Encoding, Mode, RawFrame};

const RETRY_DELAY: Duration = Duration::from_secs(2);
/// Fail fast while the phone is off; we retry anyway.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Connecting plus the RTSP handshake and the first picture.
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(15);
/// The phone app paused or the Wi-Fi dropped without closing the connection.
const STALL_TIMEOUT: Duration = Duration::from_secs(5);
/// How often waiting on the network checks whether the capture was stopped.
const STOP_POLL: Duration = Duration::from_millis(200);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamSettings {
    pub url: String,
    pub username: String,
    pub password: String,
}

impl Default for StreamSettings {
    fn default() -> Self {
        Self {
            url: "rtsp://192.168.1.52:8554/live".into(),
            username: String::new(),
            password: String::new(),
        }
    }
}

impl StreamSettings {
    /// `host:port` of the URL, for messages.
    pub fn host(&self) -> &str {
        let url = self.url.trim();
        let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
        let authority = rest.split('/').next().unwrap_or(rest);
        authority
            .rsplit_once('@')
            .map_or(authority, |(_, host)| host)
    }

    /// The URL without credentials (retina refuses those) and the login: from the username and
    /// password fields, or else from the URL.
    fn target(&self) -> Result<(Url, Option<Credentials>), String> {
        let mut url = Url::parse(self.url.trim()).map_err(|e| format!("Invalid URL: {e}"))?;
        if url.scheme() != "rtsp" {
            return Err("The URL has to start with rtsp://".into());
        }
        let decode = |s: &str| percent_decode_str(s).decode_utf8_lossy().into_owned();
        let in_url = Credentials {
            username: decode(url.username()),
            password: decode(url.password().unwrap_or_default()),
        };
        let _ = url.set_username("");
        let _ = url.set_password(None);
        let creds = if !self.username.is_empty() {
            Some(Credentials {
                username: self.username.clone(),
                password: self.password.clone(),
            })
        } else if !in_url.username.is_empty() {
            Some(in_url)
        } else {
            None
        };
        Ok((url, creds))
    }
}

/// Why a connection attempt ended.
enum Ended {
    /// The capture was dropped.
    Stopped,
    /// A message for the user.
    Failed(String),
}

/// Receives the stream, reconnecting whenever it fails, until the capture is dropped.
pub fn start(settings: StreamSettings, sink: SyncSender<RawFrame>) -> Capture {
    Capture::spawn(move |stop, status| {
        // The worker thread lets retina finish a TEARDOWN in the background after a session ends.
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("rtsp")
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(e) => {
                status.lock().unwrap().error = Some(format!("Could not start networking: {e}"));
                return;
            }
        };
        while !stop.load(Ordering::Relaxed) {
            match runtime.block_on(receive(&settings, &sink, stop, status)) {
                Ok(()) | Err(Ended::Stopped) => return,
                Err(Ended::Failed(e)) => {
                    let mut st = status.lock().unwrap();
                    st.running = false;
                    st.error = Some(e);
                }
            }
            let retry_at = Instant::now() + RETRY_DELAY;
            while Instant::now() < retry_at && !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    })
}

/// One connection attempt. `Ok` means the app is shutting down.
async fn receive(
    settings: &StreamSettings,
    sink: &SyncSender<RawFrame>,
    stop: &AtomicBool,
    status: &Mutex<CaptureStatus>,
) -> Result<(), Ended> {
    let host = settings.host();
    let (url, creds) = settings.target().map_err(Ended::Failed)?;
    let started = Instant::now();
    let options = SessionOptions::default()
        .creds(creds)
        .user_agent(format!("overheadcamera/{}", env!("CARGO_PKG_VERSION")));
    let mut session = bounded(
        Session::describe(url, options),
        started + CONNECT_TIMEOUT,
        stop,
        || unreachable(host),
    )
    .await?
    .map_err(|e| failed(&e, settings))?;

    let streams = session.streams();
    let Some(video) = streams
        .iter()
        .position(|s| s.media() == "video" && s.encoding_name() == "h264")
    else {
        let sent: Vec<&str> = streams.iter().map(|s| s.encoding_name()).collect();
        return Err(Ended::Failed(format!(
            "{host} sends {}, not H.264. Set the camera app to H.264.",
            if sent.is_empty() { "nothing".into() } else { sent.join(", ") }
        )));
    };
    let mut deadline = started + FIRST_FRAME_TIMEOUT;
    let no_video = || format!("No video from {host} (timed out).");
    let setup = SetupOptions::default()
        .transport(Transport::Tcp(Default::default()))
        // Annex B with SPS/PPS before every key frame, the way the decoder takes it.
        .frame_format(FrameFormat::SIMPLE);
    bounded(session.setup(video, setup), deadline, stop, no_video)
        .await?
        .map_err(|e| failed(&e, settings))?;
    let session = bounded(session.play(PlayOptions::default()), deadline, stop, no_video)
        .await?
        .map_err(|e| failed(&e, settings))?;
    let mut frames = session.demuxed().map_err(|e| failed(&e, settings))?;

    let mut decoder = Decoder::new()
        .map_err(|e| Ended::Failed(format!("Could not start the H.264 decoder: {e}")))?;
    // Decoding has to start at a key frame.
    let mut synced = false;
    let mut decode_error = None;
    let mut size = None;
    loop {
        let item = bounded(frames.next(), deadline, stop, || match &decode_error {
            Some(e) => format!("The video from {host} could not be decoded ({e})."),
            None => no_video(),
        })
        .await?;
        let frame = match item {
            Some(Ok(CodecItem::VideoFrame(frame))) => frame,
            Some(Ok(_)) => continue,
            Some(Err(e)) => return Err(failed(&e, settings)),
            None => return Err(Ended::Failed(format!("{host} stopped streaming."))),
        };
        synced |= frame.is_random_access_point();
        if !synced {
            continue;
        }
        let raw = match decoder.decode(frame.data()) {
            Ok(Some(yuv)) => pack_i420(&yuv),
            Ok(None) => continue,
            Err(e) => {
                // Usually a lost packet; the next key frame recovers.
                decode_error = Some(e.to_string());
                continue;
            }
        };
        deadline = Instant::now() + STALL_TIMEOUT;
        if size != Some((raw.width, raw.height)) {
            size = Some((raw.width, raw.height));
            // A frame interval, e.g. (1, 30); unknown leaves the rate at 0.
            let (interval_num, interval_den) = match frames.streams()[video].parameters() {
                Some(ParametersRef::Video(p)) => p.frame_rate().unwrap_or_default(),
                _ => (0, 0),
            };
            let mut st = status.lock().unwrap();
            st.negotiated = Some(Mode {
                encoding: Encoding::I420,
                width: raw.width as u32,
                height: raw.height as u32,
                interval_num,
                interval_den,
            });
            st.running = true;
            st.error = None;
        }
        match sink.try_send(raw) {
            // A full queue means processing is behind; dropping keeps latency low.
            Ok(()) | Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
    }
}

/// Awaits `future`, giving up when the capture is stopped or at `deadline`.
async fn bounded<F: Future>(
    future: F,
    deadline: Instant,
    stop: &AtomicBool,
    timeout_message: impl FnOnce() -> String,
) -> Result<F::Output, Ended> {
    let mut future = pin!(future);
    loop {
        if stop.load(Ordering::Relaxed) {
            return Err(Ended::Stopped);
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(Ended::Failed(timeout_message()));
        }
        if let Ok(output) = tokio::time::timeout(left.min(STOP_POLL), future.as_mut()).await {
            return Ok(output);
        }
    }
}

/// Copies the decoder's padded planes into one tightly packed I420 frame.
fn pack_i420(yuv: &impl YUVSource) -> RawFrame {
    let (width, height) = yuv.dimensions();
    let (y_stride, u_stride, v_stride) = yuv.strides();
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let mut data = Vec::with_capacity(width * height + 2 * chroma_width * chroma_height);
    for (plane, stride, w, h) in [
        (yuv.y(), y_stride, width, height),
        (yuv.u(), u_stride, chroma_width, chroma_height),
        (yuv.v(), v_stride, chroma_width, chroma_height),
    ] {
        // A short plane makes a short frame, which decoding rejects.
        for row in plane.chunks(stride.max(1)).take(h) {
            data.extend_from_slice(&row[..w.min(row.len())]);
        }
    }
    RawFrame {
        encoding: Encoding::I420,
        width,
        height,
        bytes_per_line: width,
        data,
    }
}

fn unreachable(host: &str) -> String {
    format!(
        "Can't reach {host}. Is the phone on, in the same Wi-Fi, and the camera app's RTSP server \
         running?"
    )
}

/// Turns a retina error into a message for the user.
fn failed(error: &retina::Error, settings: &StreamSettings) -> Ended {
    let host = settings.host();
    Ended::Failed(match error.status_code() {
        Some(401) => format!("{host} rejected the login. Check the username and password."),
        Some(404) => format!("{host} has no stream at this path. Check the URL."),
        _ => {
            // The first line is the message, the rest is connection and packet context.
            let text = error.to_string();
            let message = text.lines().next().unwrap_or_default();
            if message.starts_with("Unable to connect") {
                unreachable(host)
            } else {
                format!("{host}: {message}")
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use openh264::formats::YUVSlices;

    use super::*;

    #[test]
    fn packs_padded_planes() {
        // 4×2 picture in planes padded to a stride of 8 (luma) and 4 (chroma).
        let y = [1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0];
        let (u, v) = ([9, 10, 0, 0], [11, 12, 0, 0]);
        let raw = pack_i420(&YUVSlices::new((&y, &u, &v), (4, 2), (8, 4, 4)));
        assert_eq!((raw.width, raw.height, raw.bytes_per_line), (4, 2, 4));
        assert_eq!(raw.data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn host_strips_scheme_credentials_and_path() {
        let mut s = StreamSettings::default();
        assert_eq!(s.host(), "192.168.1.52:8554");
        s.url = " rtsp://me:secret@phone.local:554/h264?x=1".into();
        assert_eq!(s.host(), "phone.local:554");
    }

    #[test]
    fn credentials_move_from_url_to_login() {
        let mut s = StreamSettings {
            url: " rtsp://me:pa%40ss@phone.local:554/live ".into(),
            ..Default::default()
        };
        let (url, creds) = s.target().unwrap();
        assert_eq!(url.as_str(), "rtsp://phone.local:554/live");
        let creds = creds.unwrap();
        assert_eq!((&*creds.username, &*creds.password), ("me", "pa@ss"));

        // The login fields win over the URL.
        s.username = "admin".into();
        s.password = "pw".into();
        assert_eq!(s.target().unwrap().1.unwrap().username, "admin");

        s.url = "rtsp://phone.local/live".into();
        s.username.clear();
        assert!(s.target().unwrap().1.is_none());
        s.url = "http://phone.local/live".into();
        assert!(s.target().is_err());
    }

    /// Receives frames from a real stream: `cargo test --release -- --ignored --nocapture
    /// stream_smoke`. `OVERHEAD_TEST_RTSP` is the URL (default `rtsp://127.0.0.1:8554/live`).
    #[test]
    #[ignore = "needs an RTSP stream"]
    fn stream_smoke() {
        let settings = StreamSettings {
            url: std::env::var("OVERHEAD_TEST_RTSP")
                .unwrap_or_else(|_| "rtsp://127.0.0.1:8554/live".into()),
            ..Default::default()
        };
        let (tx, rx) = std::sync::mpsc::sync_channel(2);
        let capture = start(settings, tx);
        let frames = 60;
        let mut first = None;
        for _ in 0..frames {
            let raw = rx.recv_timeout(Duration::from_secs(20)).unwrap_or_else(|e| {
                panic!("no frame ({e}): {:?}", capture.status.lock().unwrap().error)
            });
            first.get_or_insert_with(Instant::now);
            let chroma = raw.width.div_ceil(2) * raw.height.div_ceil(2);
            assert_eq!(raw.data.len(), raw.width * raw.height + 2 * chroma);
        }
        let elapsed = first.unwrap().elapsed();
        let mode = capture.status.lock().unwrap().negotiated.unwrap();
        println!(
            "{}, received {:.1} fps",
            mode.label(),
            (frames - 1) as f64 / elapsed.as_secs_f64()
        );
    }
}
