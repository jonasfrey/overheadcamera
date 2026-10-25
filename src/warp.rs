//! Perspective correction: maps an arbitrary quadrilateral of the camera image onto a rectangle.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::img::RgbImage;

pub type Mat3 = [[f64; 3]; 3];

/// Homography mapping each `from[i]` onto `to[i]`. `None` if the points are degenerate.
pub fn homography(from: [[f64; 2]; 4], to: [[f64; 2]; 4]) -> Option<Mat3> {
    // Solve the 8x8 system A h = b for h = [h00 h01 h02 h10 h11 h12 h20 h21], h22 = 1.
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let [x, y] = from[i];
        let [u, v] = to[i];
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for col in 0..8 {
        let pivot = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        let pivot_row = a[col];
        for (r, row) in a.iter_mut().enumerate() {
            if r != col {
                let f = row[col] / pivot_row[col];
                for (v, p) in row.iter_mut().zip(pivot_row).skip(col) {
                    *v -= f * p;
                }
            }
        }
    }
    let h: Vec<f64> = (0..8).map(|i| a[i][8] / a[i][i]).collect();
    Some([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
}

pub fn apply(h: &Mat3, x: f64, y: f64) -> [f64; 2] {
    let w = h[2][0] * x + h[2][1] * y + h[2][2];
    [
        (h[0][0] * x + h[0][1] * y + h[0][2]) / w,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / w,
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolation {
    Bilinear,
    Bicubic,
}

/// Renders `quad` (TL, TR, BR, BL in source pixel coordinates) of `src` into a `width`×`height` image.
pub fn warp(
    src: &RgbImage,
    quad: [[f64; 2]; 4],
    width: usize,
    height: usize,
    interpolation: Interpolation,
) -> Option<RgbImage> {
    let (w, h) = (width as f64, height as f64);
    let out_to_src = homography([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], quad)?;
    let mut out = RgbImage::new(width, height);
    out.data
        .par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                // Sample at pixel centers; image coordinates have pixel centers at +0.5.
                let [sx, sy] = apply(&out_to_src, x as f64 + 0.5, y as f64 + 0.5);
                let (sx, sy) = (sx as f32 - 0.5, sy as f32 - 0.5);
                let px = match interpolation {
                    Interpolation::Bilinear => sample_bilinear(src, sx, sy),
                    Interpolation::Bicubic => sample_bicubic(src, sx, sy),
                };
                row[x * 3..x * 3 + 3].copy_from_slice(&px);
            }
        });
    Some(out)
}

#[inline]
fn texel(src: &RgbImage, x: isize, y: isize) -> &[u8] {
    let x = x.clamp(0, src.width as isize - 1) as usize;
    let y = y.clamp(0, src.height as isize - 1) as usize;
    let i = (y * src.width + x) * 3;
    &src.data[i..i + 3]
}

fn sample_bilinear(src: &RgbImage, x: f32, y: f32) -> [u8; 3] {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let p00 = texel(src, x0, y0);
    let p10 = texel(src, x0 + 1, y0);
    let p01 = texel(src, x0, y0 + 1);
    let p11 = texel(src, x0 + 1, y0 + 1);
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as f32 + (p10[c] as f32 - p00[c] as f32) * fx;
        let bottom = p01[c] as f32 + (p11[c] as f32 - p01[c] as f32) * fx;
        out[c] = (top + (bottom - top) * fy + 0.5).clamp(0.0, 255.0) as u8;
    }
    out
}

/// Catmull-Rom weights for the 4 taps around a sample with fractional offset `t`.
#[inline]
fn cubic_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    ]
}

fn sample_bicubic(src: &RgbImage, x: f32, y: f32) -> [u8; 3] {
    let (x0, y0) = (x.floor(), y.floor());
    let wx = cubic_weights(x - x0);
    let wy = cubic_weights(y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let mut acc = [0f32; 3];
    for (j, wyj) in wy.iter().enumerate() {
        let mut row = [0f32; 3];
        for (i, wxi) in wx.iter().enumerate() {
            let p = texel(src, x0 + i as isize - 1, y0 + j as isize - 1);
            row[0] += p[0] as f32 * wxi;
            row[1] += p[1] as f32 * wxi;
            row[2] += p[2] as f32 * wxi;
        }
        acc[0] += row[0] * wyj;
        acc[1] += row[1] * wyj;
        acc[2] += row[2] * wyj;
    }
    acc.map(|v| (v + 0.5).clamp(0.0, 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homography_maps_corners() {
        let from = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]];
        let to = [[10.0, 20.0], [300.0, 5.0], [320.0, 200.0], [0.0, 180.0]];
        let h = homography(from, to).unwrap();
        for i in 0..4 {
            let p = apply(&h, from[i][0], from[i][1]);
            assert!((p[0] - to[i][0]).abs() < 1e-6 && (p[1] - to[i][1]).abs() < 1e-6);
        }
    }

    #[test]
    fn degenerate_quad_is_rejected() {
        let from = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let to = [[5.0, 5.0]; 4];
        assert!(homography(from, to).is_none());
    }

    #[test]
    fn identity_warp_preserves_image() {
        let mut src = RgbImage::new(8, 6);
        for (i, v) in src.data.iter_mut().enumerate() {
            *v = (i * 7 % 256) as u8;
        }
        let quad = [[0.0, 0.0], [8.0, 0.0], [8.0, 6.0], [0.0, 6.0]];
        for interp in [Interpolation::Bilinear, Interpolation::Bicubic] {
            let out = warp(&src, quad, 8, 6, interp).unwrap();
            assert_eq!(out.data, src.data);
        }
    }
}
