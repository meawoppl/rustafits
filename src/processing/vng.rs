//! Full-resolution gradient-based (VNG) demosaic.
//!
//! Classic eight-direction "variable number of gradients" interpolation over a
//! 5×5 window: eight directional gradients are measured around every interior
//! CFA site, the directions whose gradient falls under an adaptive threshold are
//! kept, and the missing colours are estimated from the colour averages of the
//! kept directions only — so interpolation follows edges instead of crossing
//! them.
//!
//! Unlike [`super::debayer::super_pixel_debayer_f32`] this keeps the native
//! pixel grid: the output is a full-resolution planar RGB buffer.

use crate::types::BayerPattern;

/// Unit vectors of the eight gradient directions: the four axial ones first,
/// then the four diagonals. Every gradient and sample table below is generated
/// by rotating one canonical form (north for axial, north-east for diagonal)
/// onto these vectors, so all eight directions stay consistent by construction.
const DIRECTIONS: [(i32, i32); 8] = [
    (0, -1),  // N
    (1, 0),   // E
    (0, 1),   // S
    (-1, 0),  // W
    (1, -1),  // NE
    (1, 1),   // SE
    (-1, 1),  // SW
    (-1, -1), // NW
];

/// Half of the eight directions are diagonal; the first four are axial.
const FIRST_DIAGONAL: usize = 4;

/// Sample pairs per direction whose absolute differences form its gradient.
const GRADIENT_TERMS: usize = 6;

/// The first two gradient terms carry full weight, the remaining four a half
/// weight — they sit one pixel off the direction's own axis.
const FULL_WEIGHT_TERMS: usize = 2;

/// Sample offsets per direction used to average the CFA values along it.
const ARM_SAMPLES: usize = 6;

type Offset = (i32, i32);

/// For each direction, the six `(a, b)` offset pairs contributing `|p(a) − p(b)|`
/// to its gradient. Entries `0..FULL_WEIGHT_TERMS` count at weight 1, the rest
/// at weight 0.5.
fn gradient_terms() -> [[(Offset, Offset); GRADIENT_TERMS]; 8] {
    let mut terms = [[((0, 0), (0, 0)); GRADIENT_TERMS]; 8];
    for (d, &(ux, uy)) in DIRECTIONS.iter().enumerate() {
        let two = (ux * 2, uy * 2);
        terms[d] = if d < FIRST_DIAGONAL {
            // Axial: `q` is the direction turned 90°, so the four half-weight
            // terms repeat the axial difference on the two flanking columns.
            let (qx, qy) = (-uy, ux);
            [
                ((ux, uy), (-ux, -uy)),
                (two, (0, 0)),
                ((ux + qx, uy + qy), (-ux + qx, -uy + qy)),
                ((ux - qx, uy - qy), (-ux - qx, -uy - qy)),
                ((two.0 + qx, two.1 + qy), (qx, qy)),
                ((two.0 - qx, two.1 - qy), (-qx, -qy)),
            ]
        } else {
            // Diagonal: `a` and `b` are the direction's horizontal and vertical
            // components, and the half-weight terms step along the two staircases
            // flanking the diagonal.
            let (a, b) = ((ux, 0), (0, uy));
            [
                ((ux, uy), (-ux, -uy)),
                (two, (0, 0)),
                ((ux + b.0, uy + b.1), b),
                ((ux + a.0, uy + a.1), a),
                (b, (-a.0, -a.1)),
                (a, (-b.0, -b.1)),
            ]
        };
    }
    terms
}

/// For each direction, the six offsets whose CFA samples are averaged per colour
/// when that direction is selected. Each arm reaches two pixels out along the
/// direction and one pixel to either side of it.
fn arm_offsets() -> [[Offset; ARM_SAMPLES]; 8] {
    let mut arms = [[(0, 0); ARM_SAMPLES]; 8];
    for (d, &(ux, uy)) in DIRECTIONS.iter().enumerate() {
        let two = (ux * 2, uy * 2);
        arms[d] = if d < FIRST_DIAGONAL {
            let (qx, qy) = (-uy, ux);
            [
                (ux, uy),
                two,
                (ux + qx, uy + qy),
                (ux - qx, uy - qy),
                (two.0 + qx, two.1 + qy),
                (two.0 - qx, two.1 - qy),
            ]
        } else {
            let (a, b) = ((ux, 0), (0, uy));
            [
                (ux, uy),
                two,
                (ux + b.0, uy + b.1),
                (ux + a.0, uy + a.1),
                b,
                a,
            ]
        };
    }
    arms
}

/// Colour index (0 = R, 1 = G, 2 = B) of each of the four CFA sites in the
/// pattern's 2×2 tile, indexed by `(y & 1) * 2 + (x & 1)`.
fn tile_colors(pattern: BayerPattern) -> [usize; 4] {
    match pattern {
        BayerPattern::Rggb => [0, 1, 1, 2],
        BayerPattern::Bggr => [2, 1, 1, 0],
        BayerPattern::Grbg => [1, 0, 2, 1],
        BayerPattern::Gbrg => [1, 2, 0, 1],
        BayerPattern::None => [0, 0, 0, 0],
    }
}

/// Full-resolution gradient-based demosaic. Input: one CFA plane, row-major,
/// any float scale (negatives allowed). Output: PLANAR RGB, len `3 * width * height`,
/// ordered `[R plane | G plane | B plane]`. `pattern` must not be
/// [`BayerPattern::None`] — a `None` pattern replicates the input into all three
/// planes rather than guessing a mosaic.
pub fn vng_debayer_f32(
    data: &[f32],
    width: usize,
    height: usize,
    pattern: BayerPattern,
) -> Vec<f32> {
    debug_assert!(
        pattern != BayerPattern::None,
        "vng_debayer_f32 needs a real Bayer pattern"
    );
    debug_assert_eq!(data.len(), width * height, "CFA plane length");

    let plane = width * height;
    let mut out = vec![0f32; 3 * plane];
    if pattern == BayerPattern::None {
        for p in 0..3 {
            out[p * plane..][..plane].copy_from_slice(&data[..plane]);
        }
        return out;
    }

    let colors = tile_colors(pattern);
    let color_at = |x: usize, y: usize| colors[(y & 1) * 2 + (x & 1)];

    // Interior = every site with a full 5×5 window. `saturating_sub` keeps the
    // range empty (rather than panicking) on images narrower/shorter than that.
    let x_hi = width.saturating_sub(2);
    let y_hi = height.saturating_sub(2);

    bilinear_border(data, width, height, &colors, x_hi, y_hi, &mut out);

    let terms = gradient_terms();
    let arms = arm_offsets();

    for y in 2..y_hi {
        for x in 2..x_hi {
            let center = y * width + x;
            let base = data[center];
            let center_color = color_at(x, y);

            let sample = |dx: i32, dy: i32| -> f32 {
                data[(y as i32 + dy) as usize * width + (x as i32 + dx) as usize]
            };

            let mut gradients = [0f32; 8];
            for (d, dir_terms) in terms.iter().enumerate() {
                let mut g = 0f32;
                for (t, &((ax, ay), (bx, by))) in dir_terms.iter().enumerate() {
                    let diff = (sample(ax, ay) - sample(bx, by)).abs();
                    g += if t < FULL_WEIGHT_TERMS {
                        diff
                    } else {
                        0.5 * diff
                    };
                }
                gradients[d] = g;
            }

            let mut g_min = gradients[0];
            let mut g_max = gradients[0];
            for &g in &gradients[1..] {
                if g < g_min {
                    g_min = g;
                }
                if g > g_max {
                    g_max = g;
                }
            }
            // Equal min and max (a flat neighbourhood) admits all eight.
            let threshold = 1.5 * g_min + 0.5 * (g_max - g_min);

            // Per-colour running mean over the selected directions. Counts are
            // tracked per colour because a direction's arm need not contain all
            // three — an absent colour must not dilute the others' average.
            let mut color_sum = [0f32; 3];
            let mut color_dirs = [0u32; 3];
            for (d, arm) in arms.iter().enumerate() {
                if gradients[d] > threshold {
                    continue;
                }
                let mut arm_sum = [0f32; 3];
                let mut arm_count = [0u32; 3];
                for &(dx, dy) in arm {
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    let c = color_at(nx, ny);
                    arm_sum[c] += data[ny * width + nx];
                    arm_count[c] += 1;
                }
                for c in 0..3 {
                    if arm_count[c] > 0 {
                        color_sum[c] += arm_sum[c] / arm_count[c] as f32;
                        color_dirs[c] += 1;
                    }
                }
            }

            // Every arm reaches the site two steps along the direction, which
            // always carries the centre's own colour, so this mean exists.
            let center_mean = color_sum[center_color] / color_dirs[center_color] as f32;
            for c in 0..3 {
                out[c * plane + center] = if c == center_color {
                    base
                } else if color_dirs[c] > 0 {
                    base + (color_sum[c] / color_dirs[c] as f32 - center_mean)
                } else {
                    base
                };
            }
        }
    }

    out
}

/// Bilinear demosaic for the two-pixel frame the 5×5 gradient window cannot
/// reach. Missing colours come from the in-bounds neighbours that carry them:
/// the four cardinal ones at a non-green site (green) or a green site (red and
/// blue), the four diagonal ones for the colour opposite the centre's.
#[allow(clippy::too_many_arguments)]
fn bilinear_border(
    data: &[f32],
    width: usize,
    height: usize,
    colors: &[usize; 4],
    x_hi: usize,
    y_hi: usize,
    out: &mut [f32],
) {
    const CARDINAL: [Offset; 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];
    const DIAGONAL: [Offset; 4] = [(-1, -1), (1, -1), (1, 1), (-1, 1)];

    let plane = width * height;
    let color_at = |x: usize, y: usize| colors[(y & 1) * 2 + (x & 1)];

    for y in 0..height {
        for x in 0..width {
            if (2..y_hi).contains(&y) && (2..x_hi).contains(&x) {
                continue;
            }
            let center = y * width + x;
            let base = data[center];
            let center_color = color_at(x, y);

            let mut sum = [0f32; 3];
            let mut count = [0u32; 3];
            // A green centre finds red and blue on its cardinal neighbours; a
            // red or blue centre finds green there and its opposite diagonally.
            let extra: &[Offset] = if center_color == 1 { &[] } else { &DIAGONAL };
            for &(dx, dy) in CARDINAL.iter().chain(extra.iter()) {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                let c = color_at(nx, ny);
                if c == center_color {
                    continue;
                }
                sum[c] += data[ny * width + nx];
                count[c] += 1;
            }

            for c in 0..3 {
                out[c * plane + center] = if c == center_color {
                    base
                } else if count[c] > 0 {
                    sum[c] / count[c] as f32
                } else {
                    base
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BayerPattern;

    fn cfa_fill(w: usize, h: usize, pattern: BayerPattern, r: f32, g: f32, b: f32) -> Vec<f32> {
        // Fill a mosaic where every R site = r, G site = g, B site = b for `pattern`.
        let (ri, g0, g1, bi) = match pattern {
            BayerPattern::Rggb => (0, 1, 2, 3),
            BayerPattern::Bggr => (3, 1, 2, 0),
            BayerPattern::Grbg => (1, 0, 3, 2),
            BayerPattern::Gbrg => (2, 3, 0, 1),
            BayerPattern::None => unreachable!(),
        };
        let vals =
            |slot: usize| [r, g, g, b][[ri, g0, g1, bi].iter().position(|&s| s == slot).unwrap()];
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                vals((y % 2) * 2 + (x % 2))
            })
            .collect()
    }

    #[test]
    fn constant_channels_reconstruct_exactly() {
        for pattern in [
            BayerPattern::Rggb,
            BayerPattern::Bggr,
            BayerPattern::Grbg,
            BayerPattern::Gbrg,
        ] {
            let (w, h) = (16, 12);
            let cfa = cfa_fill(w, h, pattern, 0.8, 0.5, 0.2);
            let rgb = vng_debayer_f32(&cfa, w, h, pattern);
            assert_eq!(rgb.len(), 3 * w * h);
            for (plane, want) in [(0, 0.8f32), (1, 0.5), (2, 0.2)] {
                for v in &rgb[plane * w * h..(plane + 1) * w * h] {
                    assert!(
                        (v - want).abs() < 1e-6,
                        "{pattern:?} plane {plane}: {v} != {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn horizontal_ramp_stays_monotonic_and_bounded() {
        // A luminance ramp (all channels equal, linear in x) must reconstruct to
        // values within the local input range — no over/undershoot beyond neighbors.
        let (w, h) = (32, 16);
        let cfa: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
        let rgb = vng_debayer_f32(&cfa, w, h, BayerPattern::Rggb);
        for plane in 0..3 {
            for y in 2..h - 2 {
                for x in 3..w - 3 {
                    let v = rgb[plane * w * h + y * w + x];
                    let lo = (x as f32 - 2.0) / w as f32;
                    let hi = (x as f32 + 2.0) / w as f32;
                    assert!(
                        v >= lo - 1e-5 && v <= hi + 1e-5,
                        "plane {plane} ({x},{y}): {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn negatives_pass_through() {
        let (w, h) = (8, 8);
        let cfa = cfa_fill(w, h, BayerPattern::Rggb, -0.1, -0.1, -0.1);
        let rgb = vng_debayer_f32(&cfa, w, h, BayerPattern::Rggb);
        assert!(rgb.iter().all(|v| (*v - -0.1).abs() < 1e-6));
    }
}
