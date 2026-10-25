//! Network cameras over RTSP, such as a phone running an IP camera app.
//!
//! GStreamer handles the RTSP session, authentication and H.264 decoding in a `gst-launch-1.0`
//! child process (the pipeline from `phoneascmaera/huawei_rtsp_setup.md`). It sends the decoded
//! frames back through a pipe as YUV4MPEG2: a header line with the frame size, then per frame a
//! `FRAME` line followed by the raw I420 planes. Lost connections are retried, so the phone can be
//! switched on after the app was started.

use std::{
    collections::VecDeque,
    io::{self, BufRead, BufReader, Read},
    os::fd::AsRawFd,
    process::{ChildStdout, Command, Stdio},
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{SyncSender, TrySendError},
    },
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use crate::camera::{Capture, CaptureStatus, Encoding, Mode, RawFrame};

const RETRY_DELAY: Duration = Duration::from_secs(2);
/// Connecting (bounded by `tcp-timeout` in the pipeline) plus the RTSP handshake.
const FIRST_DATA_TIMEOUT: Duration = Duration::from_secs(15);
/// The phone app paused or the Wi-Fi dropped without closing the connection.
const STALL_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_SIDE: usize = 16384;

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
}

/// Receives the stream, reconnecting whenever it fails, until the capture is dropped.
pub fn start(settings: StreamSettings, sink: SyncSender<RawFrame>) -> Capture {
    Capture::spawn(move |stop, status| {
        while !stop.load(Ordering::Relaxed) {
            match receive(&settings, &sink, stop, status) {
                Ok(()) => return,
                Err(e) => {
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

/// One connection attempt. `Ok` means stop was requested, `Err` says why the stream ended.
fn receive(
    settings: &StreamSettings,
    sink: &SyncSender<RawFrame>,
    stop: &AtomicBool,
    status: &Mutex<CaptureStatus>,
) -> Result<(), String> {
    let mut child = Command::new("gst-launch-1.0")
        .arg("-q")
        .args(pipeline(settings))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => {
                "GStreamer is not installed (gst-launch-1.0 not found).".to_string()
            }
            _ => format!("Could not start GStreamer: {e}"),
        })?;
    let stderr = child.stderr.take().unwrap();
    let log = std::thread::spawn(move || {
        let mut lines = VecDeque::new();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if !line.trim().is_empty() {
                if lines.len() == 40 {
                    lines.pop_front();
                }
                lines.push_back(line);
            }
        }
        lines
    });

    let mut reader = BufReader::with_capacity(
        1 << 20,
        PollReader {
            pipe: child.stdout.take().unwrap(),
            stop,
            last_data: Instant::now(),
            received: false,
        },
    );
    let result = read_frames(&mut reader, sink, status);
    let _ = child.kill();
    let _ = child.wait();
    let log: Vec<String> = log.join().unwrap_or_default().into();
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    // `Ok` here means the pipeline is gone, i.e. the app is shutting down.
    result.map_err(|e| describe_failure(&e, &log, settings))
}

/// The pipeline from the setup notes, ending in raw frames on stdout instead of a window.
fn pipeline(s: &StreamSettings) -> Vec<String> {
    let mut args = vec![
        "rtspsrc".to_string(),
        format!("location={}", quote(s.url.trim())),
        "latency=50".into(),
        "protocols=tcp".into(),
        // Fail fast while the phone is off; we retry anyway.
        "tcp-timeout=5000000".into(),
    ];
    if !s.username.is_empty() {
        args.push(format!("user-id={}", quote(&s.username)));
        args.push(format!("user-pw={}", quote(&s.password)));
    }
    // Asking for BT.601 makes videoconvert convert whatever colorimetry the phone signals into
    // the one `decode` expects.
    let rest = "! rtph264depay ! h264parse ! avdec_h264 ! videoconvert n-threads=4 \
                ! video/x-raw,format=I420,colorimetry=bt601 ! y4menc ! fdsink fd=1 sync=false";
    args.extend(rest.split_whitespace().map(String::from));
    args
}

/// Quotes a property value for gst-launch's pipeline syntax.
fn quote(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn read_frames(
    reader: &mut impl BufRead,
    sink: &SyncSender<RawFrame>,
    status: &Mutex<CaptureStatus>,
) -> io::Result<()> {
    let header = read_header(reader)?;
    let mut first = true;
    while let Some(data) = read_frame(reader, &header)? {
        if first {
            first = false;
            let mut st = status.lock().unwrap();
            st.negotiated = Some(Mode {
                encoding: Encoding::I420,
                width: header.width as u32,
                height: header.height as u32,
                interval_num: header.fps.1,
                interval_den: header.fps.0,
            });
            st.running = true;
            st.error = None;
        }
        let frame = RawFrame {
            encoding: Encoding::I420,
            width: header.width,
            height: header.height,
            bytes_per_line: header.width,
            data,
        };
        match sink.try_send(frame) {
            // A full queue means processing is behind; dropping keeps latency low.
            Ok(()) | Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
    }
    Err(io::ErrorKind::UnexpectedEof.into())
}

#[derive(Debug, PartialEq)]
struct Y4mHeader {
    width: usize,
    height: usize,
    /// Frame rate as numerator and denominator, 0:0 if unknown.
    fps: (u32, u32),
}

impl Y4mHeader {
    fn frame_len(&self) -> usize {
        self.width * self.height + 2 * self.width.div_ceil(2) * self.height.div_ceil(2)
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Reads one header or frame line without the newline; `None` at the end of the stream.
fn read_line(reader: &mut impl BufRead) -> io::Result<Option<String>> {
    let mut line = Vec::new();
    reader.by_ref().take(1024).read_until(b'\n', &mut line)?;
    if line.is_empty() {
        return Ok(None);
    }
    if line.pop() != Some(b'\n') {
        return Err(invalid("YUV4MPEG2 line too long"));
    }
    Ok(Some(String::from_utf8_lossy(&line).into_owned()))
}

fn read_header(reader: &mut impl BufRead) -> io::Result<Y4mHeader> {
    let line = read_line(reader)?.ok_or(io::ErrorKind::UnexpectedEof)?;
    let mut tokens = line.split_ascii_whitespace();
    if tokens.next() != Some("YUV4MPEG2") {
        return Err(invalid("not a YUV4MPEG2 stream"));
    }
    let mut header = Y4mHeader {
        width: 0,
        height: 0,
        fps: (0, 0),
    };
    for token in tokens {
        let mut chars = token.chars();
        let tag = chars.next();
        let value = chars.as_str();
        match tag {
            Some('W') => header.width = value.parse().map_err(|_| invalid("bad width"))?,
            Some('H') => header.height = value.parse().map_err(|_| invalid("bad height"))?,
            Some('F') => {
                if let Some((num, den)) = value.split_once(':') {
                    header.fps = (num.parse().unwrap_or(0), den.parse().unwrap_or(0));
                }
            }
            // 420jpeg, 420mpeg2, 420paldv only differ in chroma siting.
            Some('C') if !value.starts_with("420") => {
                return Err(invalid(format!("unsupported chroma format {value}")));
            }
            _ => {}
        }
    }
    if !(1..=MAX_SIDE).contains(&header.width) || !(1..=MAX_SIDE).contains(&header.height) {
        return Err(invalid("bad frame size"));
    }
    Ok(header)
}

/// Reads the next frame's planes; `None` at the end of the stream.
fn read_frame(reader: &mut impl BufRead, header: &Y4mHeader) -> io::Result<Option<Vec<u8>>> {
    let Some(line) = read_line(reader)? else {
        return Ok(None);
    };
    if !line.starts_with("FRAME") {
        return Err(invalid("lost frame sync in YUV4MPEG2 stream"));
    }
    let mut data = vec![0; header.frame_len()];
    reader.read_exact(&mut data)?;
    Ok(Some(data))
}

/// Reads from the GStreamer pipe, giving up when asked to stop or when no data arrives.
struct PollReader<'a> {
    pipe: ChildStdout,
    stop: &'a AtomicBool,
    last_data: Instant,
    received: bool,
}

impl Read for PollReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.stop.load(Ordering::Relaxed) {
                return Err(io::Error::other("stopped"));
            }
            let timeout = if self.received {
                STALL_TIMEOUT
            } else {
                FIRST_DATA_TIMEOUT
            };
            if self.last_data.elapsed() > timeout {
                return Err(io::ErrorKind::TimedOut.into());
            }
            let mut pfd = libc::pollfd {
                fd: self.pipe.as_raw_fd(),
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
            if ready > 0 {
                let n = self.pipe.read(buf)?;
                if n > 0 {
                    self.last_data = Instant::now();
                    self.received = true;
                }
                return Ok(n);
            }
        }
    }
}

/// Turns GStreamer's error output into a message for the user.
fn describe_failure(error: &io::Error, log: &[String], settings: &StreamSettings) -> String {
    let host = settings.host();
    let has = |needle: &str| log.iter().any(|l| l.contains(needle));
    if has("(401)") || has("Unauthorized") {
        return format!("{host} rejected the login. Check the username and password.");
    }
    if has("(404)") {
        return format!("{host} has no stream at this path. Check the URL.");
    }
    if has("Failed to connect") || has("Could not open resource") {
        return format!(
            "Can't reach {host}. Is the phone on, in the same Wi-Fi, and the camera app's RTSP \
             server running?"
        );
    }
    if let Some(line) = log.iter().find(|l| l.contains("no element")) {
        let element = line.rsplit("no element").next().unwrap_or_default().trim();
        return format!(
            "GStreamer is missing the {element} element. Install gstreamer1.0-libav and \
             gstreamer1.0-plugins-good/-bad."
        );
    }
    if has("not-negotiated") || has("Internal data stream error") {
        return "The stream could not be decoded. Set the camera app to H.264.".into();
    }
    if let Some(line) = log.iter().find(|l| l.starts_with("ERROR")) {
        let message = line.rsplit(": ").next().unwrap_or(line);
        return format!("{host}: {message}");
    }
    match error.kind() {
        io::ErrorKind::TimedOut => format!("No video from {host} (timed out)."),
        io::ErrorKind::UnexpectedEof => format!("{host} stopped streaming."),
        _ => format!("{host}: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_y4m_frames() {
        let mut stream = b"YUV4MPEG2 W4 H2 F30000:1001 Ip A1:1 C420jpeg XYSCSS=420JPEG\n".to_vec();
        for v in [10u8, 20] {
            stream.extend_from_slice(b"FRAME\n");
            stream.extend(std::iter::repeat_n(v, 4 * 2 + 2 * 2));
        }
        let mut reader = &stream[..];
        let header = read_header(&mut reader).unwrap();
        assert_eq!(
            header,
            Y4mHeader {
                width: 4,
                height: 2,
                fps: (30000, 1001)
            }
        );
        assert_eq!(
            read_frame(&mut reader, &header).unwrap(),
            Some(vec![10; 12])
        );
        assert_eq!(
            read_frame(&mut reader, &header).unwrap(),
            Some(vec![20; 12])
        );
        assert_eq!(read_frame(&mut reader, &header).unwrap(), None);
    }

    #[test]
    fn odd_sizes_round_chroma_up() {
        let header = read_header(&mut &b"YUV4MPEG2 W3 H3 C420\n"[..]).unwrap();
        assert_eq!(header.frame_len(), 9 + 2 * 4);
    }

    #[test]
    fn rejects_other_chroma_and_garbage() {
        assert!(read_header(&mut &b"YUV4MPEG2 W4 H2 C444\n"[..]).is_err());
        assert!(read_header(&mut &b"RIFF....\n"[..]).is_err());
        let header = read_header(&mut &b"YUV4MPEG2 W2 H2\n"[..]).unwrap();
        assert!(read_frame(&mut &b"garbage\n"[..], &header).is_err());
        // Truncated frame.
        assert!(read_frame(&mut &b"FRAME\n\x01\x02"[..], &header).is_err());
    }

    #[test]
    fn host_strips_scheme_credentials_and_path() {
        let mut s = StreamSettings::default();
        assert_eq!(s.host(), "192.168.1.52:8554");
        s.url = " rtsp://me:secret@phone.local:554/h264?x=1".into();
        assert_eq!(s.host(), "phone.local:554");
    }

    #[test]
    fn quotes_gst_values() {
        assert_eq!(quote(r#"pa"ss\word"#), r#""pa\"ss\\word""#);
    }
}
