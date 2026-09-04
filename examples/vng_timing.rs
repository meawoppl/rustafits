//! Timing harness for `vng_debayer_f32` — measures a single call against a
//! synthetic 26 MP (6248x4176) RGGB mosaic, matching the code review's
//! baseline benchmark so before/after numbers are directly comparable.
//!
//! Release only (the serial code is unusably slow in debug):
//!   cargo run --release --example vng_timing
//!
//! Prints one line: `width height seconds`.

use astroimage::processing::vng::vng_debayer_f32;
use astroimage::BayerPattern;

/// Deterministic pseudo-random CFA fill (xorshift64*), so the timed input is
/// reproducible across runs and machines without depending on a `rand` crate.
fn pseudo_random_mosaic(w: usize, h: usize, seed: u64) -> Vec<f32> {
    let mut state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
    (0..w * h)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let frac = ((state >> 40) & 0xFFFF) as f32 / 0xFFFF as f32; // [0, 1]
            frac * 65535.0 // FITS-u16-ish dynamic range
        })
        .collect()
}

fn main() {
    let (width, height) = (6248usize, 4176usize);
    let data = pseudo_random_mosaic(width, height, 0x5EED_1272_ABCD_EF01);

    let start = std::time::Instant::now();
    let out = vng_debayer_f32(&data, width, height, BayerPattern::Rggb);
    let elapsed = start.elapsed();

    // Keep the result alive through the timed region without adding to it.
    std::hint::black_box(&out);

    println!("{width} {height} {:.3}", elapsed.as_secs_f64());
}
