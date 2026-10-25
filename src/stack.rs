//! Temporal frame stacking to reduce sensor noise.
//!
//! The first `depth` frames after a reset are averaged exactly (cumulative mean); after that the
//! accumulator becomes an exponential moving average with weight `1/depth`. This keeps memory at a
//! single 16-bit buffer regardless of depth (a ring buffer of 4K frames would need gigabytes) and
//! keeps reacting to new pencil strokes while the user draws.

use rayon::prelude::*;

use crate::img::RgbImage;

const CHUNK: usize = 64 * 1024;

#[derive(Default)]
pub struct Stacker {
    width: usize,
    height: usize,
    /// Pixel values in 8.8 fixed point.
    acc: Vec<u16>,
    count: u32,
}

impl Stacker {
    pub fn reset(&mut self) {
        self.count = 0;
    }

    /// Number of frames currently contributing (saturates at the depth).
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Adds a frame and returns the current average.
    pub fn push(&mut self, frame: &RgbImage, depth: u32) -> RgbImage {
        if frame.width != self.width || frame.height != self.height {
            self.width = frame.width;
            self.height = frame.height;
            self.acc = vec![0; frame.data.len()];
            self.count = 0;
        }
        let depth = depth.max(1);
        let k = (self.count + 1).min(depth);
        self.count = k;
        // acc += (x - acc) / k, with 1/k as a 16-bit fixed-point reciprocal. For k >= 2 the
        // product fits in i32: 65280 * 32768 < 2^31. The mean is written in the same pass, as
        // this is memory bound at 4K.
        let recip = (65536 / k) as i32;
        let mut out = RgbImage::new(self.width, self.height);
        self.acc
            .par_chunks_mut(CHUNK)
            .zip(frame.data.par_chunks(CHUNK))
            .zip(out.data.par_chunks_mut(CHUNK))
            .for_each(|((acc, src), dst)| {
                if k == 1 {
                    for ((a, &s), d) in acc.iter_mut().zip(src).zip(dst) {
                        *a = (s as u16) << 8;
                        *d = s;
                    }
                    return;
                }
                for ((a, &s), d) in acc.iter_mut().zip(src).zip(dst) {
                    let cur = *a as i32;
                    let next = (cur + (((((s as i32) << 8) - cur) * recip) >> 16)).clamp(0, 0xFF00);
                    *a = next as u16;
                    *d = ((next + 128) >> 8).min(255) as u8;
                }
            });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(v: u8) -> RgbImage {
        RgbImage::from_raw(2, 1, vec![v; 6])
    }

    #[test]
    fn averages_exactly_until_full() {
        let mut s = Stacker::default();
        s.push(&solid(10), 4);
        s.push(&solid(20), 4);
        s.push(&solid(30), 4);
        let mean = s.push(&solid(40), 4);
        assert_eq!(s.count(), 4);
        assert_eq!(mean.data[0], 25);
    }

    #[test]
    fn depth_one_is_passthrough() {
        let mut s = Stacker::default();
        s.push(&solid(10), 1);
        assert_eq!(s.push(&solid(200), 1).data, vec![200; 6]);
    }

    #[test]
    fn converges_to_new_value() {
        let mut s = Stacker::default();
        let mut mean = s.push(&solid(0), 8);
        for _ in 0..200 {
            mean = s.push(&solid(255), 8);
        }
        assert!(mean.data[0] >= 254);
    }
}
