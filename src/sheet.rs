//! Geometry of the output sheet: paper format, orientation and output resolution.

use serde::{Deserialize, Serialize};

use crate::warp::Interpolation;

pub const MAX_OUTPUT_SIDE: usize = 8192;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Paper {
    A3,
    A4,
    A5,
    A6,
    Letter,
    Square,
    Custom,
    /// Keep whatever aspect ratio the selected corners have.
    FromCorners,
}

impl Paper {
    pub const ALL: [Paper; 8] = [
        Paper::A4,
        Paper::A5,
        Paper::A3,
        Paper::A6,
        Paper::Letter,
        Paper::Square,
        Paper::Custom,
        Paper::FromCorners,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Paper::A3 => "A3",
            Paper::A4 => "A4",
            Paper::A5 => "A5",
            Paper::A6 => "A6",
            Paper::Letter => "US Letter",
            Paper::Square => "Square",
            Paper::Custom => "Custom",
            Paper::FromCorners => "Free (from corners)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Orientation {
    /// Portrait or landscape, whichever is closer to the selected corners.
    Auto,
    Portrait,
    Landscape,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SheetSettings {
    pub paper: Paper,
    pub orientation: Orientation,
    /// Custom paper size in mm.
    pub custom_mm: [f32; 2],
    /// Match the camera's sampling density instead of using `long_side`.
    pub native_resolution: bool,
    pub long_side: u32,
    pub interpolation: Interpolation,
}

impl Default for SheetSettings {
    fn default() -> Self {
        Self {
            paper: Paper::A4,
            orientation: Orientation::Auto,
            custom_mm: [200.0, 200.0],
            native_resolution: true,
            long_side: 2480,
            interpolation: Interpolation::Bicubic,
        }
    }
}

impl SheetSettings {
    /// Physical size (short, long) in mm, if the format has one.
    pub fn paper_mm(&self) -> Option<(f32, f32)> {
        let (a, b) = match self.paper {
            Paper::A3 => (297.0, 420.0),
            Paper::A4 => (210.0, 297.0),
            Paper::A5 => (148.0, 210.0),
            Paper::A6 => (105.0, 148.0),
            Paper::Letter => (215.9, 279.4),
            Paper::Custom => (self.custom_mm[0].max(1.0), self.custom_mm[1].max(1.0)),
            Paper::Square | Paper::FromCorners => return None,
        };
        Some((a.min(b), a.max(b)))
    }

    /// Output size in pixels for the given corners (TL, TR, BR, BL in source pixels).
    pub fn output_size(&self, quad: [[f64; 2]; 4]) -> (usize, usize) {
        let dist = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
        let quad_w = ((dist(quad[0], quad[1]) + dist(quad[3], quad[2])) / 2.0).max(1.0);
        let quad_h = ((dist(quad[0], quad[3]) + dist(quad[1], quad[2])) / 2.0).max(1.0);

        let short_over_long = match self.paper {
            Paper::FromCorners => None,
            Paper::Square => Some(1.0),
            _ => self.paper_mm().map(|(s, l)| (s / l) as f64),
        };
        let aspect = match short_over_long {
            None => quad_w / quad_h,
            Some(r) => {
                let landscape = match self.orientation {
                    Orientation::Auto => quad_w > quad_h,
                    Orientation::Portrait => false,
                    Orientation::Landscape => true,
                };
                if landscape { 1.0 / r } else { r }
            }
        };
        let long = if self.native_resolution {
            quad_w.max(quad_h)
        } else {
            self.long_side as f64
        };
        let (w, h) = if aspect >= 1.0 {
            (long, long / aspect)
        } else {
            (long * aspect, long)
        };
        let clamp = |v: f64| (v.round() as usize).clamp(16, MAX_OUTPUT_SIDE);
        (clamp(w), clamp(h))
    }

    /// Effective resolution of an output of the given size, if the paper has a physical size.
    pub fn dpi(&self, size: (usize, usize)) -> Option<f32> {
        let (_, long_mm) = self.paper_mm()?;
        Some(size.0.max(size.1) as f32 / (long_mm / 25.4))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a4_auto_orientation_follows_corners() {
        let s = SheetSettings {
            native_resolution: false,
            long_side: 2970,
            ..Default::default()
        };
        let landscape = [[0.0, 0.0], [300.0, 0.0], [300.0, 200.0], [0.0, 200.0]];
        assert_eq!(s.output_size(landscape), (2970, 2100));
        let portrait = [[0.0, 0.0], [200.0, 0.0], [200.0, 300.0], [0.0, 300.0]];
        assert_eq!(s.output_size(portrait), (2100, 2970));
    }

    #[test]
    fn native_uses_longest_edge() {
        let s = SheetSettings {
            paper: Paper::FromCorners,
            ..Default::default()
        };
        let quad = [[0.0, 0.0], [400.0, 0.0], [400.0, 100.0], [0.0, 100.0]];
        assert_eq!(s.output_size(quad), (400, 100));
    }
}
