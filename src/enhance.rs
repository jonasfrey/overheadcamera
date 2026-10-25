//! Digital clean-up of the rectified sheet: lighting correction, levels, sharpening, color modes.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::img::RgbImage;

const CHUNK: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorMode {
    Color,
    Gray,
    BlackWhite,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EnhanceSettings {
    /// Divide by an estimate of the paper brightness to remove shading and color casts.
    pub flatten: bool,
    /// Size in output pixels of the blocks used to estimate the paper brightness.
    pub flatten_block: u32,
    pub black_point: f32,
    pub white_point: f32,
    pub gamma: f32,
    pub saturation: f32,
    pub sharpen_amount: f32,
    /// Gaussian sigma in output pixels.
    pub sharpen_radius: f32,
    pub mode: ColorMode,
    pub threshold: f32,
}

impl Default for EnhanceSettings {
    fn default() -> Self {
        Self::scan()
    }
}

impl EnhanceSettings {
    /// Untouched camera colors.
    pub fn natural() -> Self {
        Self {
            flatten: false,
            flatten_block: 64,
            black_point: 0.0,
            white_point: 1.0,
            gamma: 1.0,
            saturation: 1.0,
            sharpen_amount: 0.0,
            sharpen_radius: 1.5,
            mode: ColorMode::Color,
            threshold: 0.6,
        }
    }

    /// White paper, crisp strokes, colors kept.
    pub fn scan() -> Self {
        Self {
            flatten: true,
            black_point: 0.25,
            white_point: 0.92,
            gamma: 1.0,
            saturation: 1.3,
            sharpen_amount: 0.8,
            ..Self::natural()
        }
    }

    /// Pencil sketch as clean grayscale.
    pub fn pencil() -> Self {
        Self {
            mode: ColorMode::Gray,
            black_point: 0.3,
            white_point: 0.9,
            gamma: 0.8,
            sharpen_amount: 1.0,
            ..Self::scan()
        }
    }

    /// Pure black and white line art.
    pub fn line_art() -> Self {
        Self {
            mode: ColorMode::BlackWhite,
            black_point: 0.0,
            white_point: 1.0,
            sharpen_amount: 0.5,
            threshold: 0.75,
            ..Self::scan()
        }
    }
}

pub fn enhance(img: &RgbImage, s: &EnhanceSettings) -> RgbImage {
    let (w, h) = (img.width, img.height);
    let mut rgb: Vec<f32> = if s.flatten {
        flatten(img, s.flatten_block.max(8) as usize)
    } else {
        img.data.par_iter().map(|&v| v as f32 / 255.0).collect()
    };

    apply_levels(&mut rgb, s.black_point, s.white_point, s.gamma);

    let mut out = RgbImage::new(w, h);
    match s.mode {
        ColorMode::Color => {
            if s.saturation != 1.0 {
                rgb.par_chunks_exact_mut(3).for_each(|p| {
                    let l = lum(p);
                    for c in p.iter_mut() {
                        *c = (l + (*c - l) * s.saturation).clamp(0.0, 1.0);
                    }
                });
            }
            if s.sharpen_amount > 0.0 {
                let l: Vec<f32> = rgb.par_chunks_exact(3).map(lum).collect();
                let blurred = gaussian_blur(&l, w, h, s.sharpen_radius);
                rgb.par_chunks_exact_mut(3)
                    .zip(l.par_iter().zip(blurred.par_iter()))
                    .for_each(|(p, (&l, &b))| {
                        let d = s.sharpen_amount * (l - b);
                        for c in p.iter_mut() {
                            *c += d;
                        }
                    });
            }
            out.data
                .par_iter_mut()
                .zip(rgb.par_iter())
                .for_each(|(d, &v)| *d = to_u8(v));
        }
        ColorMode::Gray | ColorMode::BlackWhite => {
            let mut l: Vec<f32> = rgb.par_chunks_exact(3).map(lum).collect();
            drop(rgb);
            if s.sharpen_amount > 0.0 {
                let blurred = gaussian_blur(&l, w, h, s.sharpen_radius);
                l.par_iter_mut()
                    .zip(blurred.par_iter())
                    .for_each(|(v, &b)| *v += s.sharpen_amount * (*v - b));
            }
            let threshold = (s.mode == ColorMode::BlackWhite).then_some(s.threshold);
            out.data
                .par_chunks_exact_mut(3)
                .zip(l.par_iter())
                .for_each(|(p, &v)| {
                    let v = match threshold {
                        Some(t) => {
                            if v >= t {
                                255
                            } else {
                                0
                            }
                        }
                        None => to_u8(v),
                    };
                    p.fill(v);
                });
        }
    }
    out
}

#[inline]
fn lum(p: &[f32]) -> f32 {
    0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

fn apply_levels(values: &mut [f32], black: f32, white: f32, gamma: f32) {
    if black == 0.0 && white == 1.0 && gamma == 1.0 {
        return;
    }
    const N: usize = 4096;
    let range = (white - black).max(1e-3);
    let inv_gamma = 1.0 / gamma.max(0.05);
    let lut: Vec<f32> = (0..N)
        .map(|i| {
            let v = i as f32 / (N - 1) as f32;
            ((v - black) / range).clamp(0.0, 1.0).powf(inv_gamma)
        })
        .collect();
    values.par_chunks_mut(CHUNK).for_each(|chunk| {
        for v in chunk {
            *v = lut[(v.clamp(0.0, 1.0) * (N - 1) as f32 + 0.5) as usize];
        }
    });
}

/// Returns the image divided by a smooth estimate of the bare paper color, as RGB floats.
///
/// The paper color of each block is the mean of its brightest 10% of pixels, so pencil strokes
/// (which are darker) don't pull the estimate down. A median and mean filter over the block grid
/// then removes outlier blocks, e.g. ones that are almost completely shaded.
fn flatten(img: &RgbImage, block: usize) -> Vec<f32> {
    let (w, h) = (img.width, img.height);
    let gw = w.div_ceil(block);
    let gh = h.div_ceil(block);

    let mut grid: Vec<[f32; 3]> = (0..gw * gh)
        .into_par_iter()
        .map(|i| {
            let (bx, by) = (i % gw, i / gw);
            let xs = bx * block..((bx + 1) * block).min(w);
            let ys = by * block..((by + 1) * block).min(h);
            let mut hist = [0u32; 256];
            let mut n = 0u32;
            for y in ys.clone() {
                for x in xs.clone() {
                    let p = &img.data[(y * w + x) * 3..][..3];
                    hist[crate::img::luma(p[0], p[1], p[2]) as usize] += 1;
                    n += 1;
                }
            }
            let target = (n / 10).max(1);
            let mut cum = 0;
            let mut threshold = 255;
            for t in (0..256).rev() {
                cum += hist[t];
                if cum >= target {
                    threshold = t as u8;
                    break;
                }
            }
            let mut sum = [0u64; 3];
            let mut count = 0u64;
            for y in ys.clone() {
                for x in xs.clone() {
                    let p = &img.data[(y * w + x) * 3..][..3];
                    if crate::img::luma(p[0], p[1], p[2]) >= threshold {
                        sum[0] += p[0] as u64;
                        sum[1] += p[1] as u64;
                        sum[2] += p[2] as u64;
                        count += 1;
                    }
                }
            }
            let count = count.max(1) as f32 * 255.0;
            sum.map(|s| s as f32 / count)
        })
        .collect();

    grid = filter_grid(&grid, gw, gh, |values| {
        values.sort_by(f32::total_cmp);
        values[values.len() / 2]
    });
    for _ in 0..2 {
        grid = filter_grid(&grid, gw, gh, |values| {
            values.iter().sum::<f32>() / values.len() as f32
        });
    }

    // Grid cell centers sit half a block in from the image border; beyond them the estimate is
    // extrapolated linearly, since lighting typically keeps falling off towards the edges.
    let interp = |pos: usize, cells: usize| -> (usize, usize, f32) {
        if cells == 1 {
            return (0, 0, 0.0);
        }
        let g = (pos as f32 + 0.5) / block as f32 - 0.5;
        let i0 = (g.floor().max(0.0) as usize).min(cells - 2);
        (i0, i0 + 1, g - i0 as f32)
    };
    let mut out = vec![0f32; w * h * 3];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let (y0, y1, fy) = interp(y, gh);
        for x in 0..w {
            let (x0, x1, fx) = interp(x, gw);
            let src = &img.data[(y * w + x) * 3..][..3];
            for c in 0..3 {
                let top = grid[y0 * gw + x0][c] * (1.0 - fx) + grid[y0 * gw + x1][c] * fx;
                let bottom = grid[y1 * gw + x0][c] * (1.0 - fx) + grid[y1 * gw + x1][c] * fx;
                let bg = (top * (1.0 - fy) + bottom * fy).max(1.0 / 255.0);
                row[x * 3 + c] = (src[c] as f32 / 255.0 / bg).min(1.0);
            }
        }
    });
    out
}

/// Applies `reduce` per channel over each cell's 3×3 neighborhood.
fn filter_grid(
    grid: &[[f32; 3]],
    gw: usize,
    gh: usize,
    reduce: impl Fn(&mut Vec<f32>) -> f32,
) -> Vec<[f32; 3]> {
    let mut out = vec![[0f32; 3]; grid.len()];
    let mut values = Vec::with_capacity(9);
    for (i, cell) in out.iter_mut().enumerate() {
        let (x, y) = ((i % gw) as isize, (i / gw) as isize);
        for (c, v) in cell.iter_mut().enumerate() {
            values.clear();
            for dy in -1..=1 {
                for dx in -1..=1 {
                    values.push(grid_at(grid, gw, gh, x + dx, y + dy, c));
                }
            }
            *v = reduce(&mut values);
        }
    }
    out
}

/// Grid value with linear extrapolation one cell beyond the border, so that filtering doesn't
/// flatten a lighting gradient at the edges (which replicating the edge cells would).
fn grid_at(grid: &[[f32; 3]], gw: usize, gh: usize, x: isize, y: isize, c: usize) -> f32 {
    let (w, h) = (gw as isize, gh as isize);
    let get = |x, y| grid_at(grid, gw, gh, x, y, c);
    if w > 1 && x < 0 {
        return (2.0 * get(0, y) - get(1, y)).max(0.0);
    }
    if w > 1 && x >= w {
        return (2.0 * get(w - 1, y) - get(w - 2, y)).max(0.0);
    }
    if h > 1 && y < 0 {
        return (2.0 * get(x, 0) - get(x, 1)).max(0.0);
    }
    if h > 1 && y >= h {
        return (2.0 * get(x, h - 1) - get(x, h - 2)).max(0.0);
    }
    grid[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize][c]
}

/// Separable Gaussian blur of a single-channel image, edges clamped.
fn gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let sigma = sigma.clamp(0.3, 10.0);
    let r = (sigma * 3.0).ceil() as isize;
    let mut kernel: Vec<f32> = (-r..=r)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f32 = kernel.iter().sum();
    kernel.iter_mut().for_each(|k| *k /= norm);

    let mut tmp = vec![0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src_row = &src[y * w..(y + 1) * w];
        for (x, out) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (k, &kv) in kernel.iter().enumerate() {
                let sx = (x as isize + k as isize - r).clamp(0, w as isize - 1) as usize;
                acc += src_row[sx] * kv;
            }
            *out = acc;
        }
    });

    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (k, &kv) in kernel.iter().enumerate() {
            let sy = (y as isize + k as isize - r).clamp(0, h as isize - 1) as usize;
            let src_row = &tmp[sy * w..(sy + 1) * w];
            for (o, &s) in row.iter_mut().zip(src_row) {
                *o += s * kv;
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_removes_gradient() {
        // Paper whose brightness falls off from 240 to 120 left to right.
        let (w, h) = (256, 64);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = (240 - x * 120 / w) as u8;
                img.data[(y * w + x) * 3..][..3].fill(v);
            }
        }
        let flat = flatten(&img, 32);
        for v in flat {
            assert!(v > 0.94, "{v}");
        }
    }

    #[test]
    fn blur_preserves_constant() {
        let src = vec![0.5f32; 20 * 10];
        for v in gaussian_blur(&src, 20, 10, 2.0) {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn line_art_is_binary() {
        let mut img = RgbImage::new(64, 64);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = (i % 251) as u8;
        }
        let out = enhance(&img, &EnhanceSettings::line_art());
        assert!(out.data.iter().all(|&v| v == 0 || v == 255));
    }
}
