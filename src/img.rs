//! Minimal RGB8 image type shared by all processing stages.

use eframe::egui::{Color32, ColorImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// How camera frames are turned before anything else sees them: mirrored left-right if `mirror`,
/// then rotated clockwise by `quarter_turns` × 90°.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FrameOrientation {
    pub quarter_turns: u8,
    pub mirror: bool,
}

impl FrameOrientation {
    pub const QUARTER_TURN: Self = Self {
        quarter_turns: 1,
        mirror: false,
    };
    pub const MIRROR: Self = Self {
        quarter_turns: 0,
        mirror: true,
    };

    pub fn is_identity(self) -> bool {
        self.quarter_turns.is_multiple_of(4) && !self.mirror
    }

    /// This orientation followed by a further quarter turn clockwise.
    pub fn rotated(self) -> Self {
        Self {
            quarter_turns: (self.quarter_turns + 1) % 4,
            ..self
        }
    }

    /// This orientation followed by mirroring left-right.
    pub fn mirrored(self) -> Self {
        // Mirroring after a rotation equals mirroring first and rotating the other way.
        Self {
            quarter_turns: (4 - self.quarter_turns % 4) % 4,
            mirror: !self.mirror,
        }
    }
}

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
        self.region_to_color_image([0, 0, self.width, self.height], factor)
    }

    /// Converts the pixels in `[x0, y0, x1, y1)` to an egui image, box-downscaled by `factor`.
    /// Pixels that don't fill a whole block at the right and bottom edge are dropped. The region
    /// must lie inside the image and hold at least one block.
    pub fn region_to_color_image(&self, [x0, y0, x1, y1]: [usize; 4], factor: usize) -> ColorImage {
        debug_assert!(x1 <= self.width && y1 <= self.height);
        let out_w = (x1 - x0) / factor;
        let out_h = (y1 - y0) / factor;
        let mut pixels = vec![Color32::BLACK; out_w * out_h];
        if factor == 1 {
            pixels
                .par_chunks_mut(out_w)
                .enumerate()
                .for_each(|(oy, row)| {
                    let start = ((y0 + oy) * self.width + x0) * 3;
                    let src = self.data[start..start + out_w * 3].as_chunks::<3>().0;
                    for (px, p) in row.iter_mut().zip(src) {
                        *px = Color32::from_rgb(p[0], p[1], p[2]);
                    }
                });
            return ColorImage::new([out_w, out_h], pixels);
        }

        let norm = (factor * factor) as u32;
        pixels
            .par_chunks_mut(out_w)
            .enumerate()
            .for_each(|(oy, row)| {
                for (ox, px) in row.iter_mut().enumerate() {
                    let mut sum = [0u32; 3];
                    for dy in 0..factor {
                        let start = ((y0 + oy * factor + dy) * self.width + x0 + ox * factor) * 3;
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

    /// The image mirrored and rotated as `o` says.
    pub fn oriented(&self, o: FrameOrientation) -> RgbImage {
        let (w, h) = (self.width, self.height);
        let turns = o.quarter_turns % 4;
        let src = self.data.as_chunks::<3>().0;
        if turns.is_multiple_of(2) {
            // Rows stay rows: copied, possibly in reverse.
            let flip_rows = turns == 2;
            let reverse = flip_rows != o.mirror;
            let mut out = RgbImage::new(w, h);
            out.data
                .par_chunks_mut(w * 3)
                .enumerate()
                .for_each(|(y, row)| {
                    let sy = if flip_rows { h - 1 - y } else { y };
                    let src_row = &src[sy * w..(sy + 1) * w];
                    let dst = row.as_chunks_mut::<3>().0;
                    if reverse {
                        for (d, s) in dst.iter_mut().zip(src_row.iter().rev()) {
                            *d = *s;
                        }
                    } else {
                        dst.copy_from_slice(src_row);
                    }
                });
            return out;
        }

        // Quarter turns read columns; going tile by tile keeps reads and writes in the cache.
        const TILE: usize = 64;
        let (out_w, out_h) = (h, w);
        let mut out = RgbImage::new(out_w, out_h);
        out.data
            .par_chunks_mut(out_w * 3 * TILE)
            .enumerate()
            .for_each(|(band, rows)| {
                let dst = rows.as_chunks_mut::<3>().0;
                let band_h = dst.len() / out_w;
                for x0 in (0..out_w).step_by(TILE) {
                    for dy in 0..band_h {
                        let y = band * TILE + dy;
                        for x in x0..(x0 + TILE).min(out_w) {
                            // Position in the mirrored image, then in this one.
                            let (mx, my) = if turns == 1 {
                                (y, h - 1 - x)
                            } else {
                                (w - 1 - y, x)
                            };
                            let sx = if o.mirror { w - 1 - mx } else { mx };
                            dst[dy * out_w + x] = src[my * w + sx];
                        }
                    }
                }
            });
        debug_assert_eq!(out.height, out_h);
        out
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_crops_and_downscales() {
        // 4×2 gray image, pixel value = 10 * index.
        let data = (0..8u8).flat_map(|i| [i * 10; 3]).collect();
        let img = RgbImage::from_raw(4, 2, data);
        let gray = |c: &ColorImage| c.pixels.iter().map(|p| p.r()).collect::<Vec<_>>();

        let crop = img.region_to_color_image([1, 0, 3, 2], 1);
        assert_eq!(crop.size, [2, 2]);
        assert_eq!(gray(&crop), [10, 20, 50, 60]);

        let block = img.region_to_color_image([1, 0, 3, 2], 2);
        assert_eq!(block.size, [1, 1]);
        assert_eq!(gray(&block), [35]);

        // Incomplete blocks at the edge are dropped; the full image path still works.
        assert_eq!(img.region_to_color_image([0, 0, 3, 2], 2).size, [1, 1]);
        assert_eq!(gray(&img.to_color_image(2)), [25, 45]);
    }

    #[test]
    fn orientations_turn_and_compose() {
        // 3×2 image, pixel value = index:  0 1 2
        //                                  3 4 5
        let img = RgbImage::from_raw(3, 2, (0..6u8).flat_map(|i| [i; 3]).collect());
        let values = |i: &RgbImage| {
            (
                i.width,
                i.height,
                i.data.iter().step_by(3).copied().collect::<Vec<_>>(),
            )
        };
        let o = |quarter_turns, mirror| FrameOrientation {
            quarter_turns,
            mirror,
        };

        assert_eq!(
            values(&img.oriented(o(0, false))),
            (3, 2, vec![0, 1, 2, 3, 4, 5])
        );
        assert_eq!(
            values(&img.oriented(o(1, false))),
            (2, 3, vec![3, 0, 4, 1, 5, 2])
        );
        assert_eq!(
            values(&img.oriented(o(2, false))),
            (3, 2, vec![5, 4, 3, 2, 1, 0])
        );
        assert_eq!(
            values(&img.oriented(o(3, false))),
            (2, 3, vec![2, 5, 1, 4, 0, 3])
        );
        assert_eq!(
            values(&img.oriented(o(0, true))),
            (3, 2, vec![2, 1, 0, 5, 4, 3])
        );

        // Larger than a tile, with ragged edges: compare against a pixel-by-pixel reference.
        let (w, h) = (150, 70);
        let big = RgbImage::from_raw(
            w,
            h,
            (0..w * h)
                .flat_map(|i| [i as u8, (i >> 8) as u8, 7])
                .collect(),
        );
        for turns in 0..4 {
            for mirror in [false, true] {
                let out = big.oriented(o(turns, mirror));
                let (out_w, out_h) = (out.width, out.height);
                assert_eq!((out_w, out_h), if turns % 2 == 1 { (h, w) } else { (w, h) });
                for y in 0..out_h {
                    for x in 0..out_w {
                        let (mx, my) = match turns {
                            0 => (x, y),
                            1 => (y, h - 1 - x),
                            2 => (w - 1 - x, h - 1 - y),
                            _ => (w - 1 - y, x),
                        };
                        let sx = if mirror { w - 1 - mx } else { mx };
                        let (i, j) = ((y * out_w + x) * 3, (my * w + sx) * 3);
                        assert_eq!(out.data[i..i + 3], big.data[j..j + 3]);
                    }
                }
            }
        }

        // Turning an already turned image equals turning the original by the combined orientation.
        for turns in 0..4 {
            for mirror in [false, true] {
                let before = o(turns, mirror);
                let turned = img.oriented(before);
                let rotated = turned.oriented(FrameOrientation::QUARTER_TURN);
                assert_eq!(values(&img.oriented(before.rotated())), values(&rotated));
                let mirrored = turned.oriented(FrameOrientation::MIRROR);
                assert_eq!(values(&img.oriented(before.mirrored())), values(&mirrored));
            }
        }
    }
}
