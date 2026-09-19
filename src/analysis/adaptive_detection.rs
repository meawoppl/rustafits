//! Adaptive multi-level star detection — the detector used by plate solving.
//!
//! A single global threshold either drowns in the diffuse light of a bright
//! nebula/galaxy or thresholds the diffuse light itself as one giant blob, so
//! it under-detects exactly the long-focal-length / extended-object frames
//! the solver most needs to handle. This detector instead deepens
//! adaptively:
//!
//!  1. **Falling-threshold ladder with an occupancy mask.** Bright stars are
//!     found first at a high, star-count-derived level and their `3·HFD`
//!     disks marked claimed; progressively lower thresholds then add only
//!     fainter, not-yet-claimed stars. The deepest pass recomputes
//!     background/noise on a 12×N mesh, so faint stars sitting *on top of*
//!     nebulosity clear a local threshold instead of a global one.
//!  2. **Saturation-aware level clip** `level = max(3.5σ, level − bg − 1)` so
//!     a flat-topped saturated core still presents supra-threshold pixels
//!     and is detected/centroided rather than vanishing.
//!  3. **MAD-annulus local background + shrink-to-symmetry flux-weighted
//!     centroid**, with a ≥2/4 cross-neighbour hot-pixel gate and
//!     `snr>10`, `hfd∈(hfd_min,30]` accept tests, for clean centroids on a
//!     contaminated/gradient background.

use super::detection::DetectedStar;

/// One detected star (superset of what the solver consumes).
struct Star {
    x: f32,
    y: f32,
    peak: f32,
    flux: f32,
    hfd: f32,
    snr: f32,
}

/// Histogram-mode background + sigma-clipped global noise σ. The histogram
/// mode is robust to the bright extended light of a nebula/galaxy (which a
/// mean would chase); the noise is a 3σ-clipped RMS about that background.
pub(crate) fn background_and_noise(
    lum: &[f32],
    width: usize,
    height: usize,
) -> (f32, f32) {
    // Integer histogram, bins 1..=64999 (ignore 0 and ≥65000). Built as
    // per-row partial `u32` histograms (`par_chunks`) merged by an
    // element-wise integer add: `u32` addition is associative and
    // commutative, so the merged counts are bit-for-bit identical to the
    // old single-threaded build no matter how rayon splits the rows — no
    // per-row alignment is even required for that guarantee, it is just a
    // natural chunk granularity for an image.
    //
    // `sum`/`n` stay their OWN sequential, single-pass loop below — `sum`
    // is an `f64` running total, and float addition is NOT associative, so
    // parallel partial sums (however they are combined back together)
    // are not guaranteed to reproduce the exact sequential total `mean`
    // below is computed from. Splitting the sequential pass out of the
    // (now parallel) histogram build costs one extra linear scan of
    // `lum`, which is cheap next to the histogram build it replaces
    // (sequential, cache-hostile random writes into a 65000-entry array).
    use rayon::prelude::*;
    let hist: Vec<u32> = lum
        .par_chunks(width.max(1))
        .fold(
            || vec![0u32; 65000],
            |mut acc, row| {
                for &v in row {
                    if v <= 0.0 || v >= 65000.0 {
                        continue;
                    }
                    let b = v.round() as usize;
                    if b >= 1 && b < 65000 {
                        acc[b] += 1;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0u32; 65000],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        );
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for &v in lum {
        if v <= 0.0 || v >= 65000.0 {
            continue;
        }
        let b = v.round() as usize;
        if b >= 1 && b < 65000 {
            sum += v as f64;
            n += 1;
        }
    }
    if n == 0 {
        return (0.0, 1.0);
    }
    let mut mode = 1usize;
    let mut best = 0u32;
    for (b, &c) in hist.iter().enumerate().take(65000).skip(1) {
        if c > best {
            best = c;
            mode = b;
        }
    }
    let mean = (sum / n as f64) as f32;
    // Strange low-value spike → use the mean instead.
    let backgr = if mean > 1.5 * mode as f32 { mean } else { mode as f32 };

    // Sigma-clipped RMS about the background, sparse global sample.
    let mut step = (height as f32 / 71.0).round() as usize;
    if step < 1 {
        step = 1;
    }
    if step % 2 == 0 {
        step += 1;
    }
    let mut sd = 0.0f32;
    for iter in 0..7 {
        let mut acc = 0.0f64;
        let mut cnt = 0u64;
        let mut y = 0;
        while y < height {
            let mut x = 0;
            while x < width {
                let v = lum[y * width + x];
                if v != 0.0 && v < 2.0 * backgr {
                    let d = v - backgr;
                    if iter == 0 || d.abs() <= 3.0 * sd {
                        acc += (d as f64) * (d as f64);
                        cnt += 1;
                    }
                }
                x += step;
            }
            y += step;
        }
        let sd_new = if cnt > 0 {
            (acc / cnt as f64).sqrt() as f32
        } else {
            sd.max(1.0)
        };
        let converged = iter > 0 && (sd - sd_new).abs() < 0.05 * sd_new.max(1e-6);
        sd = sd_new.max(1e-6);
        if converged {
            break;
        }
    }
    (backgr, sd.max(1e-6))
}

/// How the two detection levels the falling-threshold ladder starts from
/// are chosen. `RankBudget` is the historical rule (levels at the
/// intensities below which `6·max_stars` and `24·max_stars` brightest
/// pixels lie — a fixed bright-pixel budget, blind to sky brightness);
/// `NoiseRelative` puts them at `background + k·noise`, with `noise`
/// measured on the data this detector was handed; `Absolute` takes the two
/// levels from the caller verbatim, in ADU above background.
///
/// The three are not just different numbers: `RankBudget` keeps the full
/// four-arm ladder (both budget levels, then a fixed `30·noise` peak level,
/// then a per-tile adaptive pass), which descends until `max_stars` stars
/// are found. The two caller-chosen variants run ONLY the two requested
/// levels — the deeper arms would refill the population up to the cap and
/// undo the caller's threshold choice.
///
/// `Absolute` exists for a caller that pre-filters the image it hands in:
/// a filter that attenuates stars usually attenuates the noise as well, so
/// a level derived from the FILTERED data moves with the stars and the
/// filter cancels itself. Such a caller measures the noise on the original
/// and passes the levels it wants.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DetectionLevels {
    #[default]
    RankBudget,
    NoiseRelative { k1: f32, k2: f32 },
    Absolute { above_bg_1: f32, above_bg_2: f32 },
}

/// Count-driven `star_level` / `star_level2` with the saturation-aware clip.
/// Levels are picked so the brightest ~`6·max_stars` / `24·max_stars` pixels
/// fall above them, then pulled below the saturation ceiling (`− bg − 1`) so
/// flat-topped cores still register.
fn star_levels(
    lum: &[f32],
    width: usize,
    backgr: f32,
    sd: f32,
    max_stars: usize,
) -> (f32, f32) {
    // Same per-row partial-histogram + integer-reduce shape as
    // `background_and_noise` — this function has no `sum`/`n` accumulator
    // at all, so unlike that one there is nothing to keep sequential.
    use rayon::prelude::*;
    let hist: Vec<u32> = lum
        .par_chunks(width.max(1))
        .fold(
            || vec![0u32; 65536],
            |mut acc, row| {
                for &v in row {
                    if v > 0.0 && v < 65535.0 {
                        acc[v.round() as usize] += 1;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0u32; 65536],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        );
    let factor = (6 * max_stars) as u64;
    let factor2 = (24 * max_stars) as u64;
    let mut cum = 0u64;
    let mut level1 = backgr;
    let mut level2 = backgr;
    let mut got1 = false;
    for b in (1..65535).rev() {
        cum += hist[b] as u64;
        if !got1 && cum >= factor {
            level1 = b as f32;
            got1 = true;
        }
        if cum >= factor2 {
            level2 = b as f32;
            break;
        }
    }
    let floor = (3.5 * sd).max(1.0);
    let clip = |lvl: f32| (lvl - backgr - 1.0).max(floor);
    (clip(level1), clip(level2))
}

/// Sigma-clipped (background, noise) over an explicit sample slice.
fn local_bg_noise(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 1.0);
    }
    let mut mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let mut sd = 1.0f32;
    for it in 0..5 {
        let mut acc = 0.0f64;
        let mut cnt = 0u64;
        for &v in samples {
            if it == 0 || (v - mean).abs() <= 3.0 * sd {
                acc += (v - mean) as f64 * (v - mean) as f64;
                cnt += 1;
            }
        }
        if cnt == 0 {
            break;
        }
        let sd_new = (acc / cnt as f64).sqrt() as f32;
        // recompute clipped mean
        let mut msum = 0.0f64;
        let mut mcnt = 0u64;
        for &v in samples {
            if (v - mean).abs() <= 3.0 * sd_new.max(1e-6) {
                msum += v as f64;
                mcnt += 1;
            }
        }
        if mcnt > 0 {
            mean = (msum / mcnt as f64) as f32;
        }
        let converged = it > 0 && (sd - sd_new).abs() < 0.1 * sd_new.max(1e-6);
        sd = sd_new.max(1e-6);
        if converged {
            break;
        }
    }
    (mean, sd.max(1e-6))
}

/// Selection median: the value `select_nth_unstable_by` places at index
/// `len/2` is, by definition, the same value a full sort would place there
/// (duplicates make the position ambiguous, never the VALUE at it) — same
/// odd/even rule as before (no averaging either way: this always reads a
/// single element at `len/2`, truncating division). `total_cmp` is a
/// genuine total order (unlike `partial_cmp`), so it never panics; a
/// standalone 500k-trial fuzz harness (same rustc, all-finite `f32` inputs
/// incl. duplicate/tied values, reported alongside this task) found it
/// returns the exact same value as the old
/// `sort_by(|a,b| a.partial_cmp(b).unwrap_or(Equal))[len/2]` in every
/// trial. A NaN-containing input is NOT the same story:
/// `partial_cmp().unwrap_or(Equal)` is not a total order, and on this
/// toolchain `sort_by` with it PANICS on a large fraction of
/// NaN-containing inputs (the same harness measured ~73% of 200k
/// single-NaN trials) — a pre-existing, Task-4-unrelated defect in the old
/// code, not a behaviour this function preserves.
/// `select_nth_unstable_by(total_cmp)` never panics on any input. See the
/// report for the measurement and why `hfd_at`'s own fixture keeps its one
/// in-image NaN pixel out of `median()`'s input (`ann`/`devs`) rather than
/// attempting to pin the old code's crash.
fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mid = v.len() / 2;
    let (_, elem, _) = v.select_nth_unstable_by(mid, f32::total_cmp);
    *elem
}

/// Half-width of the window `hfd_at` ever reads: the annulus band's outer
/// radius (`RS_INITIAL + ANNULUS_W`). `rs` may only shrink from
/// `RS_INITIAL` after this point (the centroid loop's shrink-to-symmetry
/// retry) and `r_ap` only grows up to the (possibly shrunk) `rs` — so this
/// bound covers every pixel every loop in `hfd_at` ever touches, and the
/// window can be copied out of `lum` exactly once.
const RS_INITIAL: i32 = 14;
const ANNULUS_W: i32 = 3;
const HFD_HALF: i32 = RS_INITIAL + ANNULUS_W;
const HFD_SCRATCH_DIM: usize = (2 * HFD_HALF + 1) as usize;

/// `(dx*dx + dy*dy) as f32` for every offset in the `hfd_at` scratch window,
/// precomputed once at program start (`static`, not `const` — a `const` is
/// a value that may be re-materialized at every use site; `HFD_R2` is one
/// 4.9 KB table at one fixed address, computed once by `build_hfd_r2_table`
/// and shared, fix round 1 I5). Both radius tests in `hfd_at` computed
/// `((dx * dx + dy * dy) as f32).sqrt()` — `i32` multiply/add, THEN cast to
/// `f32`, THEN `sqrt` — and compared the sqrt against a non-squared radius
/// (`rs`/`rs+annulus_w`, or `r_ap`/`r_ap+1`); this table caches exactly the
/// `(i32 sum) as f32` step (itself exact: the largest value here, `578`,
/// is far inside `f32`'s 24-bit exact-integer range) and `.sqrt()` is still
/// called on the looked-up value at each use site, so no comparison changes
/// from a squared-vs-squared form — the boundary rounding the brief warns
/// about never enters the picture.
const fn build_hfd_r2_table() -> [[f32; HFD_SCRATCH_DIM]; HFD_SCRATCH_DIM] {
    let mut t = [[0.0f32; HFD_SCRATCH_DIM]; HFD_SCRATCH_DIM];
    let mut iy = 0usize;
    while iy < HFD_SCRATCH_DIM {
        let dy = iy as i32 - HFD_HALF;
        let mut ix = 0usize;
        while ix < HFD_SCRATCH_DIM {
            let dx = ix as i32 - HFD_HALF;
            t[iy][ix] = (dx * dx + dy * dy) as f32;
            ix += 1;
        }
        iy += 1;
    }
    t
}
static HFD_R2: [[f32; HFD_SCRATCH_DIM]; HFD_SCRATCH_DIM] = build_hfd_r2_table();

#[inline]
fn hfd_r2(dx: i32, dy: i32) -> f32 {
    HFD_R2[(dy + HFD_HALF) as usize][(dx + HFD_HALF) as usize]
}

/// One `hfd_at` call's window, copied out of `lum` exactly once.
enum HfdScratch {
    /// The whole 35×35 window is inside the image — every offset is a
    /// real pixel, unconditionally. Built with ONE contiguous-slice copy
    /// per row (35 memcpys of 35 floats — `lum`'s row-major layout makes a
    /// window row exactly `HFD_SCRATCH_DIM` contiguous floats) rather than
    /// 1225 individually bounds-checked scalar reads. This is the path
    /// almost every candidate takes (the border ring is `HFD_HALF` = 17 px
    /// wide), and it is the one the first cut of this scratch got wrong:
    /// pre-filling a `[[f32::NAN; 35]; 35]` (and an all-`false` bounds
    /// array) before overwriting nearly all of it turned out to be pure
    /// waste the compiler did not elide, measured as a net SLOWDOWN on
    /// `register_probe` (see the report) — this variant does no fill at
    /// all, only real writes.
    Interior([[f32; HFD_SCRATCH_DIM]; HFD_SCRATCH_DIM]),
    /// The window reaches past an edge. `val` is the raw pixel value
    /// (`f32::NAN` sentinel for an out-of-bounds cell) and `in_bounds` is
    /// the SAME per-offset bound check the old code ran inline in every
    /// one of its five loops, kept as its own flag rather than inferred
    /// from `val`'s NaN-ness: a real in-image NaN pixel is a legal
    /// in-bounds value the old bound checks always let through, and it
    /// must keep being let through here.
    Border {
        val: [[f32; HFD_SCRATCH_DIM]; HFD_SCRATCH_DIM],
        in_bounds: [[bool; HFD_SCRATCH_DIM]; HFD_SCRATCH_DIM],
    },
}

impl HfdScratch {
    fn build(lum: &[f32], width: usize, height: usize, sx: usize, sy: usize) -> Self {
        let half = HFD_HALF as usize;
        if sx >= half && sy >= half && sx + half < width && sy + half < height {
            let x0 = sx - half;
            let y0 = sy - half;
            let mut val = [[0.0f32; HFD_SCRATCH_DIM]; HFD_SCRATCH_DIM];
            for (iy, row) in val.iter_mut().enumerate() {
                let src = (y0 + iy) * width + x0;
                row.copy_from_slice(&lum[src..src + HFD_SCRATCH_DIM]);
            }
            return HfdScratch::Interior(val);
        }
        let mut val = [[f32::NAN; HFD_SCRATCH_DIM]; HFD_SCRATCH_DIM];
        let mut in_bounds = [[false; HFD_SCRATCH_DIM]; HFD_SCRATCH_DIM];
        for dy in -HFD_HALF..=HFD_HALF {
            let y = sy as i32 + dy;
            if y < 0 || y as usize >= height {
                continue;
            }
            let iy = (dy + HFD_HALF) as usize;
            let row_base = y as usize * width;
            for dx in -HFD_HALF..=HFD_HALF {
                let x = sx as i32 + dx;
                if x < 0 || x as usize >= width {
                    continue;
                }
                let ix = (dx + HFD_HALF) as usize;
                in_bounds[iy][ix] = true;
                val[iy][ix] = lum[row_base + x as usize];
            }
        }
        HfdScratch::Border { val, in_bounds }
    }

    /// `Some(raw pixel value)` when `(sx+dx, sy+dy)` was in bounds — exactly
    /// the condition every old per-loop bound check gated on — `None`
    /// otherwise, for the caller to `continue` on precisely as before.
    /// `Interior` never needs the bounds lookup at all: by construction
    /// every offset there is a real, in-bounds pixel.
    #[inline]
    fn get(&self, dx: i32, dy: i32) -> Option<f32> {
        let iy = (dy + HFD_HALF) as usize;
        let ix = (dx + HFD_HALF) as usize;
        match self {
            HfdScratch::Interior(val) => Some(val[iy][ix]),
            HfdScratch::Border { val, in_bounds } => {
                if in_bounds[iy][ix] {
                    Some(val[iy][ix])
                } else {
                    None
                }
            }
        }
    }
}

/// Half-flux-diameter + flux-weighted centroid at a seed pixel. Uses a
/// MAD-estimated local annulus background and a shrink-to-symmetry box so
/// the centroid stays clean on a contaminated/gradient background and
/// blended/extended detections are rejected.
/// Returns `(cx, cy, hfd, flux, peak, snr)`.
#[allow(clippy::too_many_arguments)]
fn hfd_at(
    lum: &[f32],
    width: usize,
    height: usize,
    sx: usize,
    sy: usize,
) -> Option<(f32, f32, f32, f32, f32, f32)> {
    let scratch = HfdScratch::build(lum, width, height, sx, sy);
    let mut rs: i32 = RS_INITIAL;

    // Local background = median of the rs..rs+annulus_w annulus; σ via MAD.
    let mut ann: Vec<f32> = Vec::new();
    for dy in -HFD_HALF..=HFD_HALF {
        for dx in -HFD_HALF..=HFD_HALF {
            let rr = hfd_r2(dx, dy).sqrt();
            if rr > RS_INITIAL as f32 && rr <= (RS_INITIAL + ANNULUS_W) as f32 {
                if let Some(v) = scratch.get(dx, dy) {
                    ann.push(v);
                }
            }
        }
    }
    if ann.len() < 8 {
        return None;
    }
    let star_bg = median(&mut ann);
    let mut devs: Vec<f32> = ann.iter().map(|&v| (v - star_bg).abs()).collect();
    let sd_bg = (1.4826 * median(&mut devs)).max(1.0);

    // Flux-weighted centroid + shrink-to-symmetry box.
    let (cx, cy);
    let (mut sum_val, mut sum_x, mut sum_y);
    loop {
        sum_val = 0.0f64;
        sum_x = 0.0f64;
        sum_y = 0.0f64;
        let mut signal = 0u32;
        for dy in -rs..=rs {
            for dx in -rs..=rs {
                let Some(raw) = scratch.get(dx, dy) else {
                    continue;
                };
                let val = raw - star_bg;
                if val > 3.0 * sd_bg {
                    let x = sx as i32 + dx;
                    let y = sy as i32 + dy;
                    sum_val += val as f64;
                    sum_x += val as f64 * x as f64;
                    sum_y += val as f64 * y as f64;
                    signal += 1;
                }
            }
        }
        if sum_val <= 12.0 * sd_bg as f64 || signal <= 1 {
            return None;
        }
        let box_px = ((2 * rs + 1) * (2 * rs + 1)) as f32;
        let boxed = signal as f32 >= (2.0 / 9.0) * box_px;
        if boxed || rs <= 4 {
            cx = (sum_x / sum_val) as f32;
            cy = (sum_y / sum_val) as f32;
            break;
        }
        rs -= if rs <= 4 { 1 } else { 2 };
    }

    // Aperture: grow until the radial signal falls to ≤10% of peak.
    let mut peak = 0.0f32;
    for dy in -rs..=rs {
        for dx in -rs..=rs {
            let Some(raw) = scratch.get(dx, dy) else {
                continue;
            };
            let val = raw - star_bg;
            if val > peak {
                peak = val;
            }
        }
    }
    if peak <= 0.0 {
        return None;
    }
    let mut r_ap = 1i32;
    while r_ap < rs {
        // Mean signal in the (r_ap, r_ap+1] annulus.
        let mut s = 0.0f32;
        let mut c = 0u32;
        for dy in -(r_ap + 1)..=(r_ap + 1) {
            for dx in -(r_ap + 1)..=(r_ap + 1) {
                let rr = hfd_r2(dx, dy).sqrt();
                if rr > r_ap as f32 && rr <= (r_ap + 1) as f32 {
                    if let Some(raw) = scratch.get(dx, dy) {
                        s += raw - star_bg;
                        c += 1;
                    }
                }
            }
        }
        let m = if c > 0 { s / c as f32 } else { 0.0 };
        if m <= 0.1 * peak {
            break;
        }
        r_ap += 1;
    }

    // HFD (Miyashita) over the r_ap box; flux & snr.
    let mut sum_v = 0.0f64;
    let mut sum_vr = 0.0f64;
    let mut flux = 0.0f64;
    for dy in -r_ap..=r_ap {
        for dx in -r_ap..=r_ap {
            let Some(raw) = scratch.get(dx, dy) else {
                continue;
            };
            let x = sx as i32 + dx;
            let y = sy as i32 + dy;
            let val = (raw - star_bg).max(0.0);
            let r = (((x as f32 - cx).powi(2)) + ((y as f32 - cy).powi(2))).sqrt();
            sum_v += val as f64;
            sum_vr += val as f64 * r as f64;
            flux += val as f64;
        }
    }
    if sum_v <= 0.0 {
        return None;
    }
    let hfd = ((2.0 * sum_vr / sum_v) as f32).max(0.8);
    let flux = flux as f32;
    let snr = if flux >= 1.0 {
        flux / (flux + std::f32::consts::PI * (r_ap as f32).powi(2) * sd_bg * sd_bg).sqrt()
    } else {
        0.0
    };
    Some((cx, cy, hfd, flux, peak, snr))
}

/// Stamp a filled disk of radius `r` into the occupancy mask.
fn stamp(mask: &mut [u8], width: usize, height: usize, cx: f32, cy: f32, r: f32) {
    let r = r.max(1.0);
    let ri = r.ceil() as i32;
    let cxi = cx.round() as i32;
    let cyi = cy.round() as i32;
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            if ((dx * dx + dy * dy) as f32) <= r * r {
                let x = cxi + dx;
                let y = cyi + dy;
                if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
                    mask[y as usize * width + x as usize] = 1;
                }
            }
        }
    }
}

/// Scan a rectangular region at a fixed detection level, appending accepted
/// stars and masking their disks. `bg`/`noise` are the reference background
/// and global noise for the hot-pixel cross test.
#[allow(clippy::too_many_arguments)]
fn scan_region(
    lum: &[f32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    bg: f32,
    detection_level: f32,
    noise: f32,
    hfd_min: f32,
    mask: &mut [u8],
    out: &mut Vec<Star>,
    budget: usize,
) {
    let cross = 4.0 * noise;
    for y in y0.max(1)..y1.min(height.saturating_sub(1)) {
        for x in x0.max(1)..x1.min(width.saturating_sub(1)) {
            let i = y * width + x;
            if mask[i] != 0 {
                continue;
            }
            let v = lum[i] - bg;
            if v <= detection_level {
                continue;
            }
            // ≥2 of 4 cross-neighbours above bg+4σ (hot-pixel rejection).
            let n = ((lum[i - 1] - bg > cross) as u32)
                + ((lum[i + 1] - bg > cross) as u32)
                + ((lum[i - width] - bg > cross) as u32)
                + ((lum[i + width] - bg > cross) as u32);
            if n < 2 {
                continue;
            }
            if let Some((cx, cy, hfd, flux, peak, snr)) =
                hfd_at(lum, width, height, x, y)
            {
                if hfd > hfd_min && hfd <= 30.0 && snr > 10.0 {
                    let mi = (cy.round() as i32).clamp(0, height as i32 - 1) as usize
                        * width
                        + (cx.round() as i32).clamp(0, width as i32 - 1) as usize;
                    if mask[mi] == 0 {
                        stamp(mask, width, height, cx, cy, (3.0 * hfd).round());
                        out.push(Star {
                            x: cx,
                            y: cy,
                            peak,
                            flux,
                            hfd,
                            snr,
                        });
                        // Safety bound only — the star cap is applied AFTER a
                        // completed scan (flux sort + truncate). Returning at
                        // max_stars here truncated dense frames in row-major
                        // scan order: cocoon.fits came back as a horizontal
                        // stripe (all 500 stars at y < 1470 of 4176).
                        if out.len() >= budget {
                            return;
                        }
                    }
                }
            }
        }
    }
}

/// Adaptive multi-level detection. Returns stars sorted brightest-first
/// (by flux), trimmed to `max_stars`.
///
/// `precomputed_bg_noise`: a caller that already computed
/// `background_and_noise(lum, width, height)` over this exact buffer (same
/// values, unmodified in between) may pass that pair here to skip the
/// second, identical computation. `None` recomputes it internally — the
/// only correct choice for any caller that cannot prove the pair is over
/// the same buffer at the same point (a prefiltered detection copy, an
/// in-place edit, or simply not having computed it already).
pub fn detect_stars_adaptive(
    lum: &[f32],
    width: usize,
    height: usize,
    max_stars: usize,
    hfd_min: f32,
    levels: DetectionLevels,
    precomputed_bg_noise: Option<(f32, f32)>,
) -> Vec<(DetectedStar, f32)> {
    if width < 8 || height < 8 {
        return Vec::new();
    }
    let max_stars = max_stars.max(8);
    // Per-scan safety bound: well above any real frame's star count at the
    // ladder levels, only there to bound memory on pathological input.
    let scan_budget = (max_stars * 64).max(8192);
    let (bg, noise) =
        precomputed_bg_noise.unwrap_or_else(|| background_and_noise(lum, width, height));
    // `scan_region` compares `lum[i] - bg` against its level, so both
    // arms below are ABOVE-background levels: `k·noise` here is the
    // absolute level `bg + k·noise`. No saturation clip is needed on the
    // noise-relative arm — a few-σ level can never land above the
    // saturation ceiling the way a rank-derived one can.
    let (star_level, star_level2) = match levels {
        DetectionLevels::RankBudget => star_levels(lum, width, bg, noise, max_stars),
        DetectionLevels::NoiseRelative { k1, k2 } => (k1 * noise, k2 * noise),
        DetectionLevels::Absolute {
            above_bg_1,
            above_bg_2,
        } => (above_bg_1, above_bg_2),
    };
    // Both caller-chosen variants run the two requested arms and stop.
    let caller_levels = !matches!(levels, DetectionLevels::RankBudget);

    let mut mask = vec![0u8; width * height];
    let mut stars: Vec<Star> = Vec::with_capacity(max_stars);

    // retries 4 → 1, stop once we have enough or the ladder is exhausted.
    // A caller-chosen level stops after the two arms it asked for (4 and 3):
    // the fixed `30·noise` arm and the per-tile adaptive pass exist to keep
    // descending until `max_stars` is reached, which would refill the
    // population regardless of the threshold the caller asked for.
    let deepest = if caller_levels { 3i32 } else { 1i32 };
    let mut retries = 4i32;
    while retries >= deepest && stars.len() < max_stars {
        match retries {
            4 => {
                // The `> 30·noise` guard is the rank-budget ladder's own:
                // a budget level that low means the budget already reaches
                // the noise, and arm 2's fixed level covers it. A
                // caller-chosen level is never skipped.
                if star_level > 30.0 * noise || caller_levels {
                    scan_region(
                        lum, width, height, 0, 0, width, height, bg, star_level,
                        noise, hfd_min, &mut mask, &mut stars, scan_budget,
                    );
                }
            }
            3 => {
                if star_level2 > 30.0 * noise || caller_levels {
                    scan_region(
                        lum, width, height, 0, 0, width, height, bg,
                        star_level2, noise, hfd_min, &mut mask, &mut stars,
                        scan_budget,
                    );
                }
            }
            2 => {
                scan_region(
                    lum, width, height, 0, 0, width, height, bg,
                    30.0 * noise, noise, hfd_min, &mut mask, &mut stars,
                    scan_budget,
                );
            }
            _ => {
                // retry 1: per-tile adaptive (12 columns on the long axis).
                let raster = 12usize;
                let (nx, ny) = if width >= height {
                    (raster, (raster * height / width).max(1))
                } else {
                    ((raster * width / height).max(1), raster)
                };
                let tw = width.div_ceil(nx);
                let th = height.div_ceil(ny);
                for ty in 0..ny {
                    for tx in 0..nx {
                        let x0 = tx * tw;
                        let y0 = ty * th;
                        let x1 = (x0 + tw).min(width);
                        let y1 = (y0 + th).min(height);
                        if x1 <= x0 || y1 <= y0 {
                            continue;
                        }
                        // Local bg/noise from a strided tile sample.
                        let mut samp: Vec<f32> = Vec::new();
                        let sstep = (((x1 - x0).max(y1 - y0)) / 64).max(1);
                        let mut yy = y0;
                        while yy < y1 {
                            let mut xx = x0;
                            while xx < x1 {
                                samp.push(lum[yy * width + xx]);
                                xx += sstep;
                            }
                            yy += sstep;
                        }
                        let (lbg, lnoise) = local_bg_noise(&samp);
                        scan_region(
                            lum, width, height, x0, y0, x1, y1, lbg,
                            7.0 * lnoise, lnoise.max(noise), hfd_min, &mut mask,
                            &mut stars, scan_budget,
                        );
                    }
                }
            }
        }
        retries -= 1;
    }

    // Brightest-first; trim to max_stars.
    stars.sort_by(|a, b| {
        b.flux.partial_cmp(&a.flux).unwrap_or(std::cmp::Ordering::Equal)
    });
    stars.truncate(max_stars);

    // Shape, for every survivor. The fast path used to leave `theta` and
    // `eccentricity` at zero, which made a trail indistinguishable from a star
    // to everything downstream — a plate solver would happily build quads out
    // of streak fragments and "solve" a wind-shaken frame at a wildly wrong
    // scale. Measured on a 26 MP frame this costs a fraction of a millisecond
    // for 600 stars: one pass over an 11×11 stamp each.
    let bg_at = |_x: usize, _y: usize| bg;
    let threshold = 3.0 * noise;
    stars
        .into_iter()
        .map(|s| {
            let cx_i = s.x.round() as i32;
            let cy_i = s.y.round() as i32;
            // The stamp has to be bigger than the thing it measures: a window
            // narrower than the star sees only its core and reports it round.
            // Scale with the detected size (HFD), clamped so a hot pixel still
            // gets a usable window and a bloated blob does not cost a scan of
            // the frame.
            let want = ((2.0 * s.hfd).ceil() as i32).clamp(STAMP_RADIUS_MIN, STAMP_RADIUS_MAX);
            let stamp_r = want
                .min(cx_i)
                .min(cy_i)
                .min(width as i32 - 1 - cx_i)
                .min(height as i32 - 1 - cy_i)
                .max(0);
            let (theta, eccentricity) = if stamp_r >= 2 {
                crate::analysis::detection::shape_from_moments(
                    lum, width, cx_i, cy_i, stamp_r, threshold, &bg_at,
                )
            } else {
                (0.0, 0.0)
            };
            (
                DetectedStar {
                    x: s.x,
                    y: s.y,
                    peak: s.peak,
                    flux: s.flux,
                    area: (s.hfd * s.hfd).max(1.0) as usize,
                    theta,
                    eccentricity,
                },
                s.snr,
            )
        })
        .collect()
}

/// Half-width bounds of the stamp the shape moments are measured over. The
/// window follows the star's own size (2 × HFD) between these.
const STAMP_RADIUS_MIN: i32 = 4;
const STAMP_RADIUS_MAX: i32 = 24;

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixture shared by the D7/D11 bit-identity pins below: a deterministic
    /// synthetic frame with five `hfd_at` candidates —
    /// a star near the image border/corner (offset 12,12: the `hfd_at`
    /// window reaches `HFD_HALF = 17` px in every direction, so this seed's
    /// window spans well past both edges), two stars of different radii
    /// (sigma 1.6 vs 3.2, at 80,60 and 160,60), a saturated star (240,60,
    /// clipped to 60000 ADU), and a star with a genuine in-image NaN pixel
    /// 5,5 px from its center (80,160 → NaN at 85,165) — inside the rs=14
    /// centroid/peak/aperture disk (dist ≈7.07) but OUTSIDE the (14,17]
    /// background annulus `median()` reads, so the fixture does not trip
    /// the pre-existing (Task-4-unrelated) panic in `f32::sort_by` with a
    /// `partial_cmp().unwrap_or(Equal)` comparator over NaN-containing
    /// input (see the report: `select_nth_unstable_by(total_cmp)` does not
    /// have this failure mode, `sort_by` does, on this toolchain).
    fn hfd_pin_fixture() -> (Vec<f32>, usize, usize) {
        let width = 300usize;
        let height = 220usize;
        let mut data = vec![300.0_f32; width * height];

        add_star(&mut data, width, 12.0, 12.0, 6000.0, 1.6);
        add_star(&mut data, width, 80.0, 60.0, 5000.0, 1.6);
        add_star(&mut data, width, 160.0, 60.0, 5000.0, 3.2);
        add_star(&mut data, width, 240.0, 60.0, 200000.0, 2.0);
        add_star(&mut data, width, 80.0, 160.0, 5000.0, 1.6);
        data[165 * width + 85] = f32::NAN;

        let mut rng = 24601u64;
        for v in data.iter_mut() {
            if v.is_nan() {
                continue;
            }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v += ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0;
        }
        for v in data.iter_mut() {
            if !v.is_nan() && *v > 60000.0 {
                *v = 60000.0;
            }
        }
        (data, width, height)
    }

    /// D7 RED/GREEN oracle: the exact `hfd_at` outputs recorded from the
    /// pre-Task-4 code (per-loop bound checks straight into `lum`, a full
    /// `sort_by` median) on [`hfd_pin_fixture`]. The scratch-window +
    /// `R2`-table + `select_nth_unstable_by` rewrite must reproduce every
    /// bit.
    #[test]
    fn hfd_at_matches_pre_task4_recording() {
        let (data, width, height) = hfd_pin_fixture();
        let seeds: [(usize, usize); 5] = [(12, 12), (80, 60), (160, 60), (240, 60), (80, 160)];
        let expected: [Option<(f32, f32, f32, f32, f32, f32)>; 5] = [
            Some((
                12.000035, 11.999964, 3.7141128, 91521.09, 6000.586, 302.36023,
            )),
            Some((80.00001, 60.0012, 3.714019, 76257.55, 4998.896, 275.84518)),
            Some((
                159.99959, 60.000435, 7.6192484, 309840.22, 4999.7036, 555.84424,
            )),
            Some((
                240.00005, 60.000004, 5.872633, 3278450.8, 59700.074, 1810.5519,
            )),
            Some((
                80.000046, 159.99963, 3.7132785, 76249.07, 4996.823, 275.84732,
            )),
        ];
        for (i, (sx, sy)) in seeds.into_iter().enumerate() {
            let got = hfd_at(&data, width, height, sx, sy);
            assert_eq!(
                got, expected[i],
                "hfd_at({sx}, {sy}) diverged from the pre-Task-4 recording"
            );
        }
    }

    /// D11 RED/GREEN oracle: `background_and_noise` and `star_levels`
    /// recorded from the pre-Task-4 (sequential-histogram) code on
    /// [`hfd_pin_fixture`]. The counts must stay integer-exact under the
    /// parallel per-row histogram + reduce.
    #[test]
    fn histograms_match_pre_task4_recording() {
        let (data, width, height) = hfd_pin_fixture();
        let bgn = background_and_noise(&data, width, height);
        assert_eq!(
            bgn,
            (299.0, 2.0045419),
            "background_and_noise diverged from the pre-Task-4 recording"
        );
        let sl = star_levels(&data, width, bgn.0, bgn.1, 24);
        assert_eq!(
            sl,
            (3385.0, 148.0),
            "star_levels diverged from the pre-Task-4 recording"
        );
    }

    fn add_star(data: &mut [f32], width: usize, x: f32, y: f32, amp: f32, sigma: f32) {
        let height = data.len() / width;
        let r = (4.0 * sigma).ceil() as i32;
        let inv = 1.0 / (2.0 * sigma * sigma);
        for dy in -r..=r {
            for dx in -r..=r {
                let px = x as i32 + dx;
                let py = y as i32 + dy;
                if px >= 1 && (px as usize) < width - 1 && py >= 1 && (py as usize) < height - 1 {
                    let ddx = px as f32 - x;
                    let ddy = py as f32 - y;
                    data[py as usize * width + px as usize] +=
                        amp * (-(ddx * ddx + ddy * ddy) * inv).exp();
                }
            }
        }
    }

    /// The max_stars cap must select the BRIGHTEST stars frame-wide, not the
    /// first ones in scan order. Regression: on dense frames the row-major
    /// scan filled the cap from the top of the image and returned — the fast
    /// detector only ever saw a horizontal stripe (cocoon.fits: all 500
    /// stars at y < 1470 of 4176, 3% overlap with the slow path's top-100).
    #[test]
    fn cap_keeps_brightest_not_first_in_scan_order() {
        // Dense-frame regression (cocoon.fits): one bright wide halo eats the
        // brightest-pixel budget the ladder levels derive from, the working
        // level drops below hundreds of ordinary stars, and the row-major
        // scan fills the max_stars cap from the TOP of the frame and returns
        // — the fast detector only ever saw a horizontal stripe (all 500
        // cocoon stars at y < 1470 of 4176; 3% overlap with the slow path's
        // top-100). The cap must keep the brightest stars frame-wide.
        let width = 512;
        let height = 512;
        let mut data = vec![100.0_f32; width * height];
        // Wide bright halo: keeps levels 1-2 high (they only ever see it).
        add_star(&mut data, width, 256.0, 40.0, 30000.0, 12.0);
        // 10×18 grid of EQUAL ordinary stars over the whole frame — far more
        // than max_stars, all admitted by the same ladder level.
        for gy in 0..18 {
            for gx in 0..10 {
                add_star(
                    &mut data, width,
                    30.0 + gx as f32 * 48.0,
                    85.0 + gy as f32 * 21.0,
                    6000.0,
                    1.6,
                );
            }
        }
        // A row of clearly BRIGHTER stars at the very bottom — scanned last,
        // below the halo-driven levels 1-2, above the ordinary grid.
        for gx in 0..8 {
            add_star(&mut data, width, 50.0 + gx as f32 * 56.0, 486.0, 12000.0, 1.6);
        }
        // Deterministic mild noise so levels are sane.
        let mut rng = 12345u64;
        for v in data.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v += ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0;
        }

        let max_stars = 24;
        let stars = detect_stars_adaptive(
            &data,
            width,
            height,
            max_stars,
            1.0,
            DetectionLevels::RankBudget,
            None,
        );
        assert!(stars.len() >= 16, "expected detections, got {}", stars.len());
        let bright_bottom = stars.iter().filter(|(s, _)| s.y > 470.0).count();
        assert!(
            bright_bottom >= 6,
            "scan-order bias: only {}/8 of the brightest (bottom-row) stars made the cap; got {} stars, y range [{:.0},{:.0}]",
            bright_bottom,
            stars.len(),
            stars.iter().map(|(s, _)| s.y).fold(f32::MAX, f32::min),
            stars.iter().map(|(s, _)| s.y).fold(f32::MIN, f32::max),
        );
    }

    /// The caller-chosen levels must actually decide the population: at
    /// `5σ`/`2.5σ` both the bright and the faint half of a synthetic field
    /// are found; at `20σ`/`10σ` the faint half sits below both levels and
    /// only the bright half survives. `RankBudget` with a budget larger
    /// than the field reproduces the deep answer, which is exactly the
    /// sky-blind behaviour `NoiseRelative` exists to replace.
    #[test]
    fn noise_relative_levels_decide_the_population() {
        let (width, height) = (512usize, 512usize);
        let mut data = vec![1000.0_f32; width * height];
        // 20×20 grid: alternate bright (peak 300) and faint (peak 60)
        // stars, 200 of each, well clear of the borders.
        let mut bright = 0;
        let mut faint = 0;
        for gy in 0..20 {
            for gx in 0..20 {
                let x = 16.0 + gx as f32 * 24.0;
                let y = 16.0 + gy as f32 * 24.0;
                if (gx + gy) % 2 == 0 {
                    add_star(&mut data, width, x, y, 300.0, 1.6);
                    bright += 1;
                } else {
                    add_star(&mut data, width, x, y, 60.0, 1.6);
                    faint += 1;
                }
            }
        }
        assert_eq!((bright, faint), (200, 200));
        // Deterministic Gaussian-ish noise, σ = 8 ADU (sum of 4 uniforms,
        // whose own σ is 1/√12 per term). σ has to leave the FAINT half
        // above the detector's OWN accept gate (`snr > 10` on the aperture
        // flux) — at σ = 20 a peak-60 star scores ≈ 6 and never survives
        // any level, so the fixture would test the gate, not the levels.
        const NOISE: f32 = 8.0;
        let mut rng = 987_654_321u64;
        for v in data.iter_mut() {
            let mut acc = 0.0f32;
            for _ in 0..4 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                acc += (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            }
            *v += acc * NOISE / (1.0 / 12.0f32 * 4.0).sqrt();
        }

        let max_stars = 24576;
        let deep = detect_stars_adaptive(
            &data,
            width,
            height,
            max_stars,
            1.0,
            DetectionLevels::NoiseRelative { k1: 5.0, k2: 2.5 },
            None,
        );
        assert!(
            deep.len() >= 380,
            "5σ/2.5σ should reach the faint half: {}",
            deep.len()
        );

        let shallow = detect_stars_adaptive(
            &data,
            width,
            height,
            max_stars,
            1.0,
            DetectionLevels::NoiseRelative { k1: 20.0, k2: 10.0 },
            None,
        );
        assert!(
            shallow.len() <= 220,
            "20σ/10σ is above the faint half's peak: {}",
            shallow.len()
        );
        assert!(
            shallow.len() >= 150,
            "20σ/10σ must still find the bright half: {}",
            shallow.len()
        );

        // `Absolute` takes the caller's two levels verbatim, in ADU above
        // background — nothing is multiplied by the detector's own noise.
        // 200/100 sits between the faint half's peak (60) and the bright
        // half's (300); 40/20 is under both.
        let bright_only = detect_stars_adaptive(
            &data,
            width,
            height,
            max_stars,
            1.0,
            DetectionLevels::Absolute {
                above_bg_1: 200.0,
                above_bg_2: 100.0,
            },
            None,
        );
        assert!(
            bright_only.len() <= 220 && bright_only.len() >= 150,
            "an absolute 200/100 level keeps the bright half only: {}",
            bright_only.len()
        );
        let both = detect_stars_adaptive(
            &data,
            width,
            height,
            max_stars,
            1.0,
            DetectionLevels::Absolute {
                above_bg_1: 40.0,
                above_bg_2: 20.0,
            },
            None,
        );
        assert!(
            both.len() >= 380,
            "an absolute 40/20 level reaches the faint half: {}",
            both.len()
        );
        // And with NOISE = 8 the two spellings agree: `Absolute{5σ, 2.5σ}`
        // is exactly what `NoiseRelative{5, 2.5}` computes for itself.
        assert_eq!(
            deep.len(),
            detect_stars_adaptive(
                &data,
                width,
                height,
                max_stars,
                1.0,
                DetectionLevels::Absolute {
                    above_bg_1: 5.0 * NOISE,
                    above_bg_2: 2.5 * NOISE,
                },
                None,
            )
            .len(),
            "absolute k·σ must reproduce the noise-relative answer"
        );

        // The rank budget is blind to the caller's threshold: a budget
        // larger than the whole field puts both its levels at the noise
        // floor and the ladder's deeper arms run anyway, so it lands near
        // the DEEP answer no matter what threshold a caller wanted (331 of
        // 400 as measured — the per-tile arm's own local-noise estimate is
        // inflated inside the star-dense tiles, which costs the rest).
        let budget = detect_stars_adaptive(
            &data,
            width,
            height,
            max_stars,
            1.0,
            DetectionLevels::RankBudget,
            None,
        );
        assert!(
            budget.len() >= 300 && budget.len() > shallow.len(),
            "the budget ladder descends past the faint half regardless of any \
             threshold: {} (shallow {})",
            budget.len(),
            shallow.len()
        );
    }

    /// A caller-supplied `(bg, noise)` pair, computed the same way over the
    /// same buffer, must reproduce exactly what the detector computes for
    /// itself when passed `None` — the whole point of D3 (skip the caller's
    /// duplicate `background_and_noise` call) is that this substitution is
    /// invisible in the output.
    #[test]
    fn precomputed_bg_noise_matches_internal_computation() {
        let (width, height) = (256usize, 256usize);
        let mut data = vec![500.0_f32; width * height];
        // A real-shaped field: a grid of stars over flat sky, same fixture
        // style as the tests above.
        for gy in 0..8 {
            for gx in 0..8 {
                let x = 16.0 + gx as f32 * 30.0;
                let y = 16.0 + gy as f32 * 30.0;
                add_star(&mut data, width, x, y, 4000.0, 1.8);
            }
        }
        let mut rng = 42u64;
        for v in data.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v += ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 10.0;
        }

        let max_stars = 128;
        let with_none = detect_stars_adaptive(
            &data,
            width,
            height,
            max_stars,
            1.0,
            DetectionLevels::RankBudget,
            None,
        );
        let bg_noise = background_and_noise(&data, width, height);
        let with_precomputed = detect_stars_adaptive(
            &data,
            width,
            height,
            max_stars,
            1.0,
            DetectionLevels::RankBudget,
            Some(bg_noise),
        );

        assert!(!with_none.is_empty(), "fixture should detect stars");

        // `DetectedStar` derives neither `PartialEq` nor `Debug`; compare
        // its fields (plus the paired SNR) as plain tuples instead of
        // widening the struct's derives just for this test.
        let flatten =
            |v: &[(DetectedStar, f32)]| -> Vec<(f32, f32, f32, f32, usize, f32, f32, f32)> {
                v.iter()
                    .map(|(s, snr)| {
                        (s.x, s.y, s.peak, s.flux, s.area, s.theta, s.eccentricity, *snr)
                    })
                    .collect()
            };
        assert_eq!(
            flatten(&with_none),
            flatten(&with_precomputed),
            "a caller-supplied (bg, noise) pair must reproduce the internally \
             computed one exactly"
        );
    }
}

#[cfg(all(test, feature = "debug-pipeline"))]
mod diag {
    use super::*;

    #[test]
    #[ignore]
    fn cocoon_top_structure() {
        let path = "tests/cocoon.fits";
        if !std::path::Path::new(path).exists() { return; }
        let (meta, pixels) = crate::formats::read_image(std::path::Path::new(path)).unwrap();
        let (lum, w, h, _, _) = crate::analysis::prepare_luminance(&meta, &pixels, true);
        let stars =
            detect_stars_adaptive(&lum, w, h, 8000, 0.8, DetectionLevels::RankBudget, None);
        eprintln!("total {}", stars.len());
        eprintln!("top 25 by flux (x y flux area~hfd2 snr):");
        for (s, snr) in stars.iter().take(25) {
            eprintln!("  {:7.1} {:7.1} {:10.0} {:5} {:7.1}", s.x, s.y, s.flux, s.area, snr);
        }
        // hfd distribution among top 500 vs rest
        let hfd = |a: usize| (a as f32).sqrt();
        let mut top: Vec<f32> = stars.iter().take(500).map(|(s, _)| hfd(s.area)).collect();
        let mut rest: Vec<f32> = stars.iter().skip(500).map(|(s, _)| hfd(s.area)).collect();
        top.sort_by(|a, b| a.total_cmp(b)); rest.sort_by(|a, b| a.total_cmp(b));
        eprintln!("hfd p10/p50/p90 top500: {:.1}/{:.1}/{:.1}  rest: {:.1}/{:.1}/{:.1}",
            top[top.len()/10], top[top.len()/2], top[top.len()*9/10],
            rest[rest.len()/10], rest[rest.len()/2], rest[rest.len()*9/10]);
        // surface-brightness ranking: flux/hfd^2 = flux/area
        let mut by_sb: Vec<&(DetectedStar, f32)> = stars.iter().collect();
        by_sb.sort_by(|a, b| (b.0.flux / b.0.area as f32).total_cmp(&(a.0.flux / a.0.area as f32)));
        eprintln!("top 25 by flux/area (x y flux area snr):");
        for e in by_sb.iter().take(25) {
            eprintln!("  {:7.1} {:7.1} {:10.0} {:5} {:7.1}", e.0.x, e.0.y, e.0.flux, e.0.area, e.1);
        }
    }
}
