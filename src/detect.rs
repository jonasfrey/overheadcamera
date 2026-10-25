//! Automatic detection of a bright paper sheet lying on a darker desk.

use crate::img::RgbImage;

/// Returns the sheet corners as TL, TR, BR, BL in normalized (0..1) image coordinates.
pub fn detect_sheet(img: &RgbImage) -> Option<[[f32; 2]; 4]> {
    if img.width < 16 || img.height < 16 {
        return None;
    }
    let factor = img.width.max(img.height).div_ceil(800).max(1);
    let (w, h, gray) = downscale_gray(img, factor);
    let threshold = otsu(&gray);
    let mask: Vec<bool> = gray.iter().map(|&v| v > threshold).collect();

    let component = largest_component(&mask, w, h)?;
    if component.len() < w * h / 30 {
        return None;
    }

    let mut in_comp = vec![false; w * h];
    for &i in &component {
        in_comp[i] = true;
    }
    let boundary: Vec<[f64; 2]> = component
        .iter()
        .filter(|&&i| {
            let (x, y) = (i % w, i / w);
            x == 0
                || y == 0
                || x == w - 1
                || y == h - 1
                || !in_comp[i - 1]
                || !in_comp[i + 1]
                || !in_comp[i - w]
                || !in_comp[i + w]
        })
        .map(|&i| [(i % w) as f64 + 0.5, (i / w) as f64 + 0.5])
        .collect();

    let hull = convex_hull(boundary);
    if hull.len() < 4 {
        return None;
    }
    let quad = largest_quad(&hull)?;
    let quad = order_corners(quad);

    let scale = factor as f64;
    Some(quad.map(|[x, y]| {
        [
            ((x * scale) / img.width as f64).clamp(0.0, 1.0) as f32,
            ((y * scale) / img.height as f64).clamp(0.0, 1.0) as f32,
        ]
    }))
}

fn downscale_gray(img: &RgbImage, factor: usize) -> (usize, usize, Vec<u8>) {
    let gray = img.to_gray();
    let (w, h) = (img.width / factor, img.height / factor);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0u32;
            for dy in 0..factor {
                let row = (y * factor + dy) * img.width + x * factor;
                sum += gray[row..row + factor]
                    .iter()
                    .map(|&v| v as u32)
                    .sum::<u32>();
            }
            out[y * w + x] = (sum / (factor * factor) as u32) as u8;
        }
    }
    (w, h, out)
}

fn otsu(gray: &[u8]) -> u8 {
    let mut hist = [0u64; 256];
    for &v in gray {
        hist[v as usize] += 1;
    }
    let total = gray.len() as f64;
    let sum_all: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();
    let (mut w_bg, mut sum_bg) = (0.0, 0.0);
    let (mut best, mut best_var) = (127u8, -1.0);
    for (t, &count) in hist.iter().enumerate() {
        w_bg += count as f64;
        if w_bg == 0.0 {
            continue;
        }
        let w_fg = total - w_bg;
        if w_fg == 0.0 {
            break;
        }
        sum_bg += t as f64 * count as f64;
        let mean_bg = sum_bg / w_bg;
        let mean_fg = (sum_all - sum_bg) / w_fg;
        let var = w_bg * w_fg * (mean_bg - mean_fg).powi(2);
        if var > best_var {
            best_var = var;
            best = t as u8;
        }
    }
    best
}

/// Indices of the largest 4-connected region of `true` pixels.
fn largest_component(mask: &[bool], w: usize, h: usize) -> Option<Vec<usize>> {
    let mut visited = vec![false; mask.len()];
    let mut best: Vec<usize> = Vec::new();
    let mut stack = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        let mut comp = Vec::new();
        visited[start] = true;
        stack.push(start);
        while let Some(i) = stack.pop() {
            comp.push(i);
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if mask[j] && !visited[j] {
                    visited[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        if comp.len() > best.len() {
            best = comp;
        }
    }
    (!best.is_empty()).then_some(best)
}

fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Andrew's monotone chain.
fn convex_hull(mut pts: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(pts.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 {
            Box::new(pts.iter())
        } else {
            Box::new(pts.iter().rev())
        };
        for &p in iter {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

fn quad_area(q: &[[f64; 2]; 4]) -> f64 {
    let mut a = 0.0;
    for i in 0..4 {
        let (p, n) = (q[i], q[(i + 1) % 4]);
        a += p[0] * n[1] - n[0] * p[1];
    }
    a.abs() / 2.0
}

/// Approximates the hull by 4 of its points: for a range of rotations, take the extreme points
/// along the two diagonal directions and keep the quadrilateral with the largest area.
fn largest_quad(hull: &[[f64; 2]]) -> Option<[[f64; 2]; 4]> {
    let mut best: Option<([[f64; 2]; 4], f64)> = None;
    for step in 0..90 {
        let theta = (step as f64).to_radians();
        let (s, c) = theta.sin_cos();
        let mut ext = [
            (f64::MAX, 0usize),
            (f64::MIN, 0),
            (f64::MIN, 0),
            (f64::MAX, 0),
        ];
        for (i, p) in hull.iter().enumerate() {
            let u = p[0] * c + p[1] * s;
            let v = -p[0] * s + p[1] * c;
            if u + v < ext[0].0 {
                ext[0] = (u + v, i);
            }
            if u - v > ext[1].0 {
                ext[1] = (u - v, i);
            }
            if u + v > ext[2].0 {
                ext[2] = (u + v, i);
            }
            if u - v < ext[3].0 {
                ext[3] = (u - v, i);
            }
        }
        let quad = ext.map(|(_, i)| hull[i]);
        let area = quad_area(&quad);
        if best.is_none_or(|(_, a)| area > a) {
            best = Some((quad, area));
        }
    }
    best.filter(|(_, a)| *a > 0.0).map(|(q, _)| q)
}

/// Orders corners clockwise (on screen) starting with the one closest to the top-left.
pub fn order_corners(mut q: [[f64; 2]; 4]) -> [[f64; 2]; 4] {
    let cx = q.iter().map(|p| p[0]).sum::<f64>() / 4.0;
    let cy = q.iter().map(|p| p[1]).sum::<f64>() / 4.0;
    // With y pointing down, increasing atan2 runs clockwise on screen.
    q.sort_by(|a, b| {
        (a[1] - cy)
            .atan2(a[0] - cx)
            .total_cmp(&(b[1] - cy).atan2(b[0] - cx))
    });
    let first = (0..4)
        .min_by(|&i, &j| (q[i][0] + q[i][1]).total_cmp(&(q[j][0] + q[j][1])))
        .unwrap();
    q.rotate_left(first);
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_rotated_sheet() {
        // A bright, slightly rotated rectangle on a dark background.
        let (w, h) = (400, 300);
        let corners = [[100.0, 60.0], [300.0, 80.0], [285.0, 240.0], [85.0, 220.0]];
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let p = [x as f64 + 0.5, y as f64 + 0.5];
                let inside = (0..4).all(|i| cross(corners[i], corners[(i + 1) % 4], p) >= 0.0);
                img.data[(y * w + x) * 3..][..3].fill(if inside { 230 } else { 40 });
            }
        }
        let found = detect_sheet(&img).unwrap();
        for (f, c) in found.iter().zip(corners) {
            let (fx, fy) = (f[0] as f64 * w as f64, f[1] as f64 * h as f64);
            assert!(
                (fx - c[0]).abs() < 3.0 && (fy - c[1]).abs() < 3.0,
                "{f:?} vs {c:?}"
            );
        }
    }

    #[test]
    fn orders_corners_clockwise_from_top_left() {
        let q = order_corners([[10.0, 10.0], [0.0, 10.0], [10.0, 0.0], [0.0, 0.0]]);
        assert_eq!(q, [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
    }
}
