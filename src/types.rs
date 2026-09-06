#[allow(dead_code)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum BayerPattern {
    None,
    Rggb,
    Bggr,
    Gbrg,
    Grbg,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum DataType {
    Uint16,
    Float32,
}

pub enum PixelData {
    Uint16(Vec<u16>),
    Float32(Vec<f32>),
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct ImageMetadata {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub dtype: DataType,
    pub bayer_pattern: BayerPattern,
    pub flip_vertical: bool,
}

#[allow(dead_code)]
pub struct ProcessedImage {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub is_color: bool,
    /// Number of channels in `data`: 3 = RGB, 4 = RGBA.
    pub channels: u8,
    /// Whether the image was vertically flipped during processing.
    pub flip_vertical: bool,
}

#[allow(dead_code)]
pub struct ProcessConfig {
    pub downscale_factor: usize,
    pub jpeg_quality: u8,
    pub apply_debayer: bool,
    pub preview_mode: bool,
    pub auto_stretch: bool,
    /// Output RGBA (4 bytes/pixel) instead of RGB (3 bytes/pixel).
    pub rgba_output: bool,
    /// Debayer a CFA frame with the full-resolution gradient method
    /// ([`crate::processing::vng`]) instead of the super-pixel one. Keeps the
    /// native pixel grid — four times the pixels, and roughly a hundred times
    /// the debayer cost — so it is opt-in and meant for one frame a user is
    /// actually looking at, not for bulk conversion.
    pub vng_debayer: bool,
}
