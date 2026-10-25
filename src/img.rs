//! Minimal RGB8 image type shared by all processing stages.

use eframe::egui::{Color32, ColorImage};
use rayon::prelude::*;

#[derive(Clone)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    /// Interleaved RGB, row-major, no padding.
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height * 3],
        }
    }

    pub fn from_raw(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width * height * 3);
        Self {
            width,
            height,
            data,
        }
    }

    /// Converts to an egui image, box-downscaling by an integer factor so that
    /// neither side exceeds `max_side` (display textures don't need full resolution).
    pub fn to_color_image(&self, max_side: usize) -> ColorImage {
        let factor = self.width.max(self.height).div_ceil(max_side.max(1)).max(1);
        if factor == 1 {
            let pixels = self
                .data
                .par_chunks_exact(3)
                .map(|p| Color32::from_rgb(p[0], p[1], p[2]))
                .collect();
            return ColorImage::new([self.width, self.height], pixels);
        }

        let out_w = self.width / factor;
        let out_h = self.height / factor;
        let norm = (factor * factor) as u32;
        let mut pixels = vec![Color32::BLACK; out_w * out_h];
        pixels
            .par_chunks_mut(out_w)
            .enumerate()
            .for_each(|(oy, row)| {
                for (ox, px) in row.iter_mut().enumerate() {
                    let mut sum = [0u32; 3];
                    for dy in 0..factor {
                        let start = ((oy * factor + dy) * self.width + ox * factor) * 3;
                        for p in self.data[start..start + factor * 3].as_chunks::<3>().0 {
                            sum[0] += p[0] as u32;
                            sum[1] += p[1] as u32;
                            sum[2] += p[2] as u32;
                        }
                    }
                    *px = Color32::from_rgb(
                        (sum[0] / norm) as u8,
                        (sum[1] / norm) as u8,
                        (sum[2] / norm) as u8,
                    );
                }
            });
        ColorImage::new([out_w, out_h], pixels)
    }

    /// Single-channel luminance (Rec. 601 weights).
    pub fn to_gray(&self) -> Vec<u8> {
        self.data
            .par_chunks_exact(3)
            .map(|p| luma(p[0], p[1], p[2]))
            .collect()
    }
}

#[inline]
pub fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 77 + g as u32 * 150 + b as u32 * 29 + 128) >> 8) as u8
}
