# Huawei → PC · RTSP camera

**Huawei app:** IP Camera — ShenYao (`com.shenyaocn.android.WebCam`); install via AppGallery / trusted APK source (if available). No Google Play Services required for RTSP.

**Phone:** Same Wi-Fi/LAN as PC · Rear camera · H.264/AVC · 1920×1080 @ 30 fps (test 4K later) · bitrate 8,000,000 bps (4K: 12–20 Mbps) · Audio OFF · Hardware encoder ON (if available) · RTSP server ON · Username/password set · note phone IP, port, **actual RTSP path** (example: `192.168.1.52:8554/live`). Focus: AF or manual/lock if supported.

**PC:** GStreamer + H.264 decoder/plugins installed · same LAN · connect **to phone IP** (no PC IP entered on phone for RTSP server).

```bash
gst-launch-1.0 rtspsrc location="rtsp://192.168.1.52:8554/live" user-id="USERNAME" user-pw="PASSWORD" latency=50 protocols=tcp ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

**Troubleshooting:** `401 Unauthorized` → credentials · no stream → check actual RTSP URL/server/network · lag → 1080p, 30 fps, lower latency; test 4K only after smooth playback · audio disabled in phone app.
