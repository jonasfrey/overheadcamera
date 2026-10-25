//! Live digitizer for sketches drawn under an overhead USB camera.

mod app;
mod camera;
mod detect;
mod enhance;
mod img;
mod pipeline;
mod rtsp;
mod sheet;
mod stack;
mod warp;

use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Overhead Camera")
            .with_app_id("overheadcamera")
            .with_inner_size([1500.0, 950.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "overheadcamera",
        options,
        Box::new(|cc| Ok(Box::new(app::OverheadApp::new(cc)))),
    )
}
