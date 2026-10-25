gst-launch-1.0 \
  rtspsrc location="rtsp://192.168.1.52:8554/live" \
  user-id="YOUR_USERNAME" user-pw="YOUR_PASSWORD" \
  latency=50 protocols=tcp \
  ! rtph264depay \
  ! h264parse \
  ! avdec_h264 \
  ! videoconvert \
  ! autovideosink sync=false