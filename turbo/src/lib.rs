use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyReadwriteArray1,
    PyReadwriteArray2, PyReadwriteArray3, PyReadwriteArray4, PyReadwriteArray5,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

#[derive(Clone, Copy)]
enum ResizeAlgorithm {
    Nearest,
    Bilinear,
    Area,
}

impl ResizeAlgorithm {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "nearest" => Ok(Self::Nearest),
            "bilinear" => Ok(Self::Bilinear),
            "area" => Ok(Self::Area),
            _ => Err(PyValueError::new_err(
                "algorithm must be 'nearest', 'bilinear', or 'area'",
            )),
        }
    }
}

const MASKED_SAMPLE: usize = u32::MAX as usize;
const INDEXED_AREA_CHANNEL_BITS: u32 = 21;
const INDEXED_AREA_CHANNEL_MASK: u64 = (1 << INDEXED_AREA_CHANNEL_BITS) - 1;
const INDEXED_HISTORY_CAPACITY: usize = 4;
const INDEXED_TILE_SIZE: usize = 12;
const INDEXED_TILE_SAMPLES: usize = 8;
const INDEXED_SHARED_TILE_CAPACITY: usize = 16;
const INDEXED_TILE_COLUMNS: usize = 84_usize.div_ceil(INDEXED_TILE_SIZE);
const INDEXED_TILE_ROWS: usize = 84_usize.div_ceil(INDEXED_TILE_SIZE);
const INDEXED_TILE_COUNT: usize = INDEXED_TILE_COLUMNS * INDEXED_TILE_ROWS;
const INDEXED_TILE_WORDS: usize = INDEXED_TILE_COUNT.div_ceil(64);
const NATIVE_ERROR_CAPACITY: usize = 2048;

type NativeClearError = unsafe extern "C" fn(*mut c_void);
type NativeCopyError = unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> usize;

fn native_error_detail(context: usize, copy_error: NativeCopyError) -> String {
    let mut buffer = [0_u8; NATIVE_ERROR_CAPACITY];
    let copied = unsafe { copy_error(context as *mut c_void, buffer.as_mut_ptr(), buffer.len()) }
        .min(buffer.len());
    if copied == 0 {
        return "native diagnostic unavailable".to_owned();
    }
    String::from_utf8_lossy(&buffer[..copied]).into_owned()
}
const INDEXED_BACKGROUND_CAPACITY: usize = 256;

fn indexed_background_prefill_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("ENV_VIZDOOM_TURBO_DISABLE_BACKGROUND_PREFILL").is_none())
}

fn greatest_common_divisor(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct AreaSample {
    offset: u32,
    weight: u32,
}

struct AreaPixel {
    sample_start: u32,
    sample_end: u32,
}

struct IndexedAreaPixel {
    base_offset: u32,
    sample_start: u32,
    sample_end: u32,
    masked_weight: u32,
}

struct IndexedTile {
    source_y_start: usize,
    source_y_end: usize,
    source_x_start: usize,
    source_x_end: usize,
    output_y_start: usize,
    output_y_end: usize,
    output_x_start: usize,
    output_x_end: usize,
    fingerprint_offsets: [u32; INDEXED_TILE_SAMPLES],
}

struct ImagePlan {
    raw_h: usize,
    raw_w: usize,
    source_h: usize,
    source_w: usize,
    out_h: usize,
    out_w: usize,
    out_c: usize,
    crop: [usize; 4],
    mask_crop: bool,
    crop_fill: u8,
    algorithm: ResizeAlgorithm,
    nearest_y: Vec<usize>,
    nearest_x: Vec<usize>,
    linear_y: Vec<(usize, usize, f64)>,
    linear_x: Vec<(usize, usize, f64)>,
    area_divisor: u64,
    indexed_packed: bool,
    area_samples: Vec<AreaSample>,
    area_pixels: Vec<AreaPixel>,
    indexed_area_samples: Vec<AreaSample>,
    indexed_area_pixels: Vec<IndexedAreaPixel>,
    indexed_tiles: Vec<IndexedTile>,
}

impl ImagePlan {
    #[allow(clippy::too_many_arguments)]
    fn new(
        raw_h: usize,
        raw_w: usize,
        out_h: usize,
        out_w: usize,
        out_c: usize,
        crop: [usize; 4],
        mask_crop: bool,
        crop_fill: u8,
        algorithm: ResizeAlgorithm,
    ) -> Self {
        let source_h = if mask_crop {
            raw_h
        } else {
            raw_h - crop[0] - crop[1]
        };
        let source_w = if mask_crop {
            raw_w
        } else {
            raw_w - crop[2] - crop[3]
        };
        let area_y = Self::area_axis(source_h, out_h);
        let area_x = Self::area_axis(source_w, out_w);
        let mut area_samples = Vec::new();
        let mut area_pixels = Vec::new();
        if matches!(algorithm, ResizeAlgorithm::Area) {
            area_pixels.reserve(out_h * out_w);
            for y_samples in &area_y {
                for x_samples in &area_x {
                    let sample_start = area_samples.len();
                    for &(source_y, y_weight) in y_samples {
                        for &(source_x, x_weight) in x_samples {
                            let weight = y_weight * x_weight;
                            let (raw_y, raw_x) = if mask_crop {
                                (source_y, source_x)
                            } else {
                                (source_y + crop[0], source_x + crop[2])
                            };
                            let masked = mask_crop
                                && (raw_y < crop[0]
                                    || raw_y >= raw_h - crop[1]
                                    || raw_x < crop[2]
                                    || raw_x >= raw_w - crop[3]);
                            area_samples.push(AreaSample {
                                offset: if masked {
                                    MASKED_SAMPLE as u32
                                } else {
                                    (raw_y * raw_w + raw_x) as u32
                                },
                                weight: weight as u32,
                            });
                        }
                    }
                    area_pixels.push(AreaPixel {
                        sample_start: sample_start as u32,
                        sample_end: area_samples.len() as u32,
                    });
                }
            }
        }
        let indexed_exact = raw_h == 240 && raw_w == 320 && out_h == 84 && out_w == 84;
        let area_weight_quantum = if indexed_exact {
            area_samples
                .iter()
                .fold((source_h * source_w) as u32, |divisor, sample| {
                    greatest_common_divisor(divisor, sample.weight)
                })
        } else {
            1
        };
        if area_weight_quantum > 1 {
            for sample in &mut area_samples {
                sample.weight /= area_weight_quantum;
            }
        }
        let (indexed_area_samples, indexed_area_pixels) = if indexed_exact {
            let mut indexed_samples = Vec::new();
            let mut indexed_pixels = Vec::with_capacity(area_pixels.len());
            let mut patterns = HashMap::<Vec<AreaSample>, (u32, u32)>::new();
            for pixel in &area_pixels {
                let samples = &area_samples[pixel.sample_start as usize..pixel.sample_end as usize];
                let base_offset = samples
                    .iter()
                    .find(|sample| sample.offset != MASKED_SAMPLE as u32)
                    .map_or(0, |sample| sample.offset);
                let pattern = samples
                    .iter()
                    .filter(|sample| sample.offset != MASKED_SAMPLE as u32)
                    .map(|sample| AreaSample {
                        offset: sample.offset - base_offset,
                        weight: sample.weight,
                    })
                    .collect::<Vec<_>>();
                let (sample_start, sample_end) =
                    *patterns.entry(pattern.clone()).or_insert_with(|| {
                        let sample_start = indexed_samples.len() as u32;
                        indexed_samples.extend(pattern);
                        (sample_start, indexed_samples.len() as u32)
                    });
                indexed_pixels.push(IndexedAreaPixel {
                    base_offset,
                    sample_start,
                    sample_end,
                    masked_weight: samples
                        .iter()
                        .filter(|sample| sample.offset == MASKED_SAMPLE as u32)
                        .map(|sample| sample.weight)
                        .sum(),
                });
            }
            (indexed_samples, indexed_pixels)
        } else {
            (Vec::new(), Vec::new())
        };
        let indexed_tiles = if indexed_exact {
            let source_y_offset = if mask_crop { 0 } else { crop[0] };
            let source_x_offset = if mask_crop { 0 } else { crop[2] };
            (0..INDEXED_TILE_COUNT)
                .map(|tile| {
                    let tile_y = tile / INDEXED_TILE_COLUMNS;
                    let tile_x = tile % INDEXED_TILE_COLUMNS;
                    let output_y_start = tile_y * INDEXED_TILE_SIZE;
                    let output_y_end = ((tile_y + 1) * INDEXED_TILE_SIZE).min(out_h);
                    let output_x_start = tile_x * INDEXED_TILE_SIZE;
                    let output_x_end = ((tile_x + 1) * INDEXED_TILE_SIZE).min(out_w);
                    let source_y_start = output_y_start * source_h / out_h + source_y_offset;
                    let source_y_end = (output_y_end * source_h).div_ceil(out_h) + source_y_offset;
                    let source_x_start = output_x_start * source_w / out_w + source_x_offset;
                    let source_x_end = (output_x_end * source_w).div_ceil(out_w) + source_x_offset;
                    let width = source_x_end - source_x_start;
                    let area = (source_y_end - source_y_start) * width;
                    let mut fingerprint_offsets = [0; INDEXED_TILE_SAMPLES];
                    for (sample, offset) in fingerprint_offsets.iter_mut().enumerate() {
                        let position = sample * area / INDEXED_TILE_SAMPLES;
                        *offset = ((source_y_start + position / width) * raw_w
                            + source_x_start
                            + position % width) as u32;
                    }
                    IndexedTile {
                        source_y_start,
                        source_y_end,
                        source_x_start,
                        source_x_end,
                        output_y_start,
                        output_y_end,
                        output_x_start,
                        output_x_end,
                        fingerprint_offsets,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            raw_h,
            raw_w,
            source_h,
            source_w,
            out_h,
            out_w,
            out_c,
            crop,
            mask_crop,
            crop_fill,
            algorithm,
            nearest_y: Self::nearest_axis(source_h, out_h),
            nearest_x: Self::nearest_axis(source_w, out_w),
            linear_y: Self::linear_axis(source_h, out_h),
            linear_x: Self::linear_axis(source_w, out_w),
            area_divisor: source_h as u64 * source_w as u64 / u64::from(area_weight_quantum),
            indexed_packed: source_h as u64 * source_w as u64 / u64::from(area_weight_quantum)
                * 255
                <= INDEXED_AREA_CHANNEL_MASK,
            area_samples,
            area_pixels,
            indexed_area_samples,
            indexed_area_pixels,
            indexed_tiles,
        }
    }

    fn nearest_axis(source: usize, output: usize) -> Vec<usize> {
        (0..output)
            .map(|coordinate| coordinate * source / output)
            .collect()
    }

    fn linear_axis(source: usize, output: usize) -> Vec<(usize, usize, f64)> {
        (0..output)
            .map(|coordinate| {
                let position = ((coordinate as f64 + 0.5) * source as f64 / output as f64 - 0.5)
                    .clamp(0.0, (source - 1) as f64);
                let low = position.floor() as usize;
                (low, (low + 1).min(source - 1), position - low as f64)
            })
            .collect()
    }

    fn area_axis(source: usize, output: usize) -> Vec<Vec<(usize, u64)>> {
        (0..output)
            .map(|coordinate| {
                let start = coordinate * source;
                let end = (coordinate + 1) * source;
                let first_source = start / output;
                let source_end = end.div_ceil(output).min(source);
                (first_source..source_end)
                    .map(|source_coordinate| {
                        let source_start = source_coordinate * output;
                        let source_end = (source_coordinate + 1) * output;
                        let overlap = end.min(source_end).saturating_sub(start.max(source_start));
                        (source_coordinate, overlap as u64)
                    })
                    .collect()
            })
            .collect()
    }

    fn supports_indexed_area(&self) -> bool {
        matches!(self.algorithm, ResizeAlgorithm::Area)
            && self.out_c == 1
            && self.raw_w == 320
            && self.raw_h == 240
            && self.out_w == 84
            && self.out_h == 84
    }

    #[inline]
    fn source_rgb(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        source_y: usize,
        source_x: usize,
    ) -> [u8; 3] {
        let (raw_y, raw_x) = if self.mask_crop {
            (source_y, source_x)
        } else {
            (source_y + self.crop[0], source_x + self.crop[2])
        };
        if self.mask_crop
            && (raw_y < self.crop[0]
                || raw_y >= self.raw_h - self.crop[1]
                || raw_x < self.crop[2]
                || raw_x >= self.raw_w - self.crop[3])
        {
            return [self.crop_fill; 3];
        }
        let offset = (raw_y * self.raw_w + raw_x) * 3;
        let mut rgb = [current[offset], current[offset + 1], current[offset + 2]];
        if let Some(prior) = previous {
            rgb[0] = rgb[0].max(prior[offset]);
            rgb[1] = rgb[1].max(prior[offset + 1]);
            rgb[2] = rgb[2].max(prior[offset + 2]);
        }
        rgb
    }

    #[inline]
    fn nearest_rgb(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        let source_y = self.nearest_y[out_y];
        let source_x = self.nearest_x[out_x];
        self.source_rgb(current, previous, source_y, source_x)
    }

    #[inline]
    fn bilinear_rgb(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        let (y0, y1, wy) = self.linear_y[out_y];
        let (x0, x1, wx) = self.linear_x[out_x];
        let p00 = self.source_rgb(current, previous, y0, x0);
        let p01 = self.source_rgb(current, previous, y0, x1);
        let p10 = self.source_rgb(current, previous, y1, x0);
        let p11 = self.source_rgb(current, previous, y1, x1);
        let mut result = [0_u8; 3];
        for channel in 0..3 {
            let top = p00[channel] as f64 * (1.0 - wx) + p01[channel] as f64 * wx;
            let bottom = p10[channel] as f64 * (1.0 - wx) + p11[channel] as f64 * wx;
            result[channel] = (top * (1.0 - wy) + bottom * wy).round() as u8;
        }
        result
    }

    #[inline]
    fn area_rgb(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        match (self.mask_crop, previous) {
            (false, None) => self.area_rgb_with(current, previous, out_y, out_x, |offset| {
                [current[offset], current[offset + 1], current[offset + 2]]
            }),
            (false, Some(prior)) => self.area_rgb_with(current, previous, out_y, out_x, |offset| {
                [
                    current[offset].max(prior[offset]),
                    current[offset + 1].max(prior[offset + 1]),
                    current[offset + 2].max(prior[offset + 2]),
                ]
            }),
            (true, None) => self.area_rgb_with(current, previous, out_y, out_x, |offset| {
                if offset == MASKED_SAMPLE {
                    [self.crop_fill; 3]
                } else {
                    [current[offset], current[offset + 1], current[offset + 2]]
                }
            }),
            (true, Some(prior)) => self.area_rgb_with(current, previous, out_y, out_x, |offset| {
                if offset == MASKED_SAMPLE {
                    [self.crop_fill; 3]
                } else {
                    [
                        current[offset].max(prior[offset]),
                        current[offset + 1].max(prior[offset + 1]),
                        current[offset + 2].max(prior[offset + 2]),
                    ]
                }
            }),
        }
    }

    #[inline(always)]
    fn area_rgb_with<F>(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        out_y: usize,
        out_x: usize,
        mut rgb_at: F,
    ) -> [u8; 3]
    where
        F: FnMut(usize) -> [u8; 3],
    {
        let pixel = &self.area_pixels[out_y * self.out_w + out_x];
        let mut sums = [0_u64; 3];
        for sample in &self.area_samples[pixel.sample_start as usize..pixel.sample_end as usize] {
            let rgb = rgb_at(if sample.offset == MASKED_SAMPLE as u32 {
                MASKED_SAMPLE
            } else {
                sample.offset as usize * 3
            });
            let weight = u64::from(sample.weight);
            sums[0] += u64::from(rgb[0]) * weight;
            sums[1] += u64::from(rgb[1]) * weight;
            sums[2] += u64::from(rgb[2]) * weight;
        }
        let integer_result = [
            ((sums[0] + self.area_divisor / 2) / self.area_divisor) as u8,
            ((sums[1] + self.area_divisor / 2) / self.area_divisor) as u8,
            ((sums[2] + self.area_divisor / 2) / self.area_divisor) as u8,
        ];
        if sums
            .iter()
            .any(|sum| (sum % self.area_divisor) * 2 == self.area_divisor)
        {
            self.area_rgb_float(current, previous, out_y, out_x)
        } else {
            integer_result
        }
    }

    #[cold]
    fn area_rgb_float(
        &self,
        current: &[u8],
        previous: Option<&[u8]>,
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        let y_start = out_y as f64 * self.source_h as f64 / self.out_h as f64;
        let y_end = (out_y + 1) as f64 * self.source_h as f64 / self.out_h as f64;
        let x_start = out_x as f64 * self.source_w as f64 / self.out_w as f64;
        let x_end = (out_x + 1) as f64 * self.source_w as f64 / self.out_w as f64;
        let mut sums = [0.0_f64; 3];
        let mut weight_sum = 0.0_f64;
        for source_y in y_start.floor() as usize..(y_end.ceil() as usize).min(self.source_h) {
            let y_weight =
                (y_end.min(source_y as f64 + 1.0) - y_start.max(source_y as f64)).max(0.0);
            for source_x in x_start.floor() as usize..(x_end.ceil() as usize).min(self.source_w) {
                let x_weight =
                    (x_end.min(source_x as f64 + 1.0) - x_start.max(source_x as f64)).max(0.0);
                let weight = y_weight * x_weight;
                let rgb = self.source_rgb(current, previous, source_y, source_x);
                for channel in 0..3 {
                    sums[channel] += rgb[channel] as f64 * weight;
                }
                weight_sum += weight;
            }
        }
        [
            (sums[0] / weight_sum).round() as u8,
            (sums[1] / weight_sum).round() as u8,
            (sums[2] / weight_sum).round() as u8,
        ]
    }

    #[inline]
    fn area_indexed_rgb(
        &self,
        current: &[u8],
        palette: &[u8],
        palette_rgb: &[u64; 256],
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        if self.indexed_packed {
            self.area_indexed_rgb_from_packed(
                current,
                palette,
                out_y,
                out_x,
                self.area_indexed_packed_rgb(current, palette_rgb, out_y, out_x),
            )
        } else {
            self.area_indexed_rgb_unpacked(current, palette, out_y, out_x)
        }
    }

    #[inline]
    fn area_indexed_rgb_unpacked(
        &self,
        current: &[u8],
        palette: &[u8],
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        let pixel = &self.indexed_area_pixels[out_y * self.out_w + out_x];
        let indexed_current = &current[pixel.base_offset as usize..];
        let samples =
            &self.indexed_area_samples[pixel.sample_start as usize..pixel.sample_end as usize];
        let fill_weight = u64::from(pixel.masked_weight);
        let mut sums = [u64::from(self.crop_fill) * fill_weight; 3];
        for sample in samples {
            let palette_index =
                usize::from(unsafe { *indexed_current.get_unchecked(sample.offset as usize) });
            let palette_offset = palette_index * 3;
            let weight = u64::from(sample.weight);
            sums[0] += u64::from(unsafe { *palette.get_unchecked(palette_offset) }) * weight;
            sums[1] += u64::from(unsafe { *palette.get_unchecked(palette_offset + 1) }) * weight;
            sums[2] += u64::from(unsafe { *palette.get_unchecked(palette_offset + 2) }) * weight;
        }
        self.round_indexed_rgb(current, palette, out_y, out_x, sums)
    }

    #[inline(always)]
    fn area_indexed_packed_rgb(
        &self,
        current: &[u8],
        palette_rgb: &[u64; 256],
        out_y: usize,
        out_x: usize,
    ) -> u64 {
        let pixel = &self.indexed_area_pixels[out_y * self.out_w + out_x];
        let indexed_current = &current[pixel.base_offset as usize..];
        let samples =
            &self.indexed_area_samples[pixel.sample_start as usize..pixel.sample_end as usize];
        let mut packed_rgb = [0_u64; 4];
        let mut chunks = samples.chunks_exact(4);
        for chunk in &mut chunks {
            for (lane, packed) in packed_rgb.iter_mut().enumerate() {
                let sample = unsafe { chunk.get_unchecked(lane) };
                let palette_index =
                    usize::from(unsafe { *indexed_current.get_unchecked(sample.offset as usize) });
                let weight = u64::from(sample.weight);
                *packed += unsafe { *palette_rgb.get_unchecked(palette_index) } * weight;
            }
        }
        for (lane, sample) in chunks.remainder().iter().enumerate() {
            let palette_index =
                usize::from(unsafe { *indexed_current.get_unchecked(sample.offset as usize) });
            let weight = u64::from(sample.weight);
            packed_rgb[lane] += unsafe { *palette_rgb.get_unchecked(palette_index) } * weight;
        }
        let packed_fill = u64::from(self.crop_fill)
            | (u64::from(self.crop_fill) << INDEXED_AREA_CHANNEL_BITS)
            | (u64::from(self.crop_fill) << (INDEXED_AREA_CHANNEL_BITS * 2));
        packed_rgb.into_iter().sum::<u64>() + packed_fill * u64::from(pixel.masked_weight)
    }

    #[inline(always)]
    fn area_indexed_rgb_from_packed(
        &self,
        current: &[u8],
        palette: &[u8],
        out_y: usize,
        out_x: usize,
        packed_rgb: u64,
    ) -> [u8; 3] {
        let sums = [
            packed_rgb & INDEXED_AREA_CHANNEL_MASK,
            (packed_rgb >> INDEXED_AREA_CHANNEL_BITS) & INDEXED_AREA_CHANNEL_MASK,
            packed_rgb >> (INDEXED_AREA_CHANNEL_BITS * 2),
        ];
        self.round_indexed_rgb(current, palette, out_y, out_x, sums)
    }

    #[inline(always)]
    fn round_indexed_rgb(
        &self,
        current: &[u8],
        palette: &[u8],
        out_y: usize,
        out_x: usize,
        sums: [u64; 3],
    ) -> [u8; 3] {
        let adjusted = [
            sums[0] + self.area_divisor / 2,
            sums[1] + self.area_divisor / 2,
            sums[2] + self.area_divisor / 2,
        ];
        let rounded = [
            adjusted[0] / self.area_divisor,
            adjusted[1] / self.area_divisor,
            adjusted[2] / self.area_divisor,
        ];
        if adjusted
            .iter()
            .zip(rounded)
            .any(|(sum, value)| value * self.area_divisor == *sum)
        {
            self.area_indexed_rgb_float(current, palette, out_y, out_x)
        } else {
            [rounded[0] as u8, rounded[1] as u8, rounded[2] as u8]
        }
    }

    #[cold]
    fn area_indexed_rgb_float(
        &self,
        current: &[u8],
        palette: &[u8],
        out_y: usize,
        out_x: usize,
    ) -> [u8; 3] {
        let y_start = out_y as f64 * self.source_h as f64 / self.out_h as f64;
        let y_end = (out_y + 1) as f64 * self.source_h as f64 / self.out_h as f64;
        let x_start = out_x as f64 * self.source_w as f64 / self.out_w as f64;
        let x_end = (out_x + 1) as f64 * self.source_w as f64 / self.out_w as f64;
        let mut sums = [0.0_f64; 3];
        let mut weight_sum = 0.0_f64;
        for source_y in y_start.floor() as usize..(y_end.ceil() as usize).min(self.source_h) {
            let y_weight =
                (y_end.min(source_y as f64 + 1.0) - y_start.max(source_y as f64)).max(0.0);
            for source_x in x_start.floor() as usize..(x_end.ceil() as usize).min(self.source_w) {
                let x_weight =
                    (x_end.min(source_x as f64 + 1.0) - x_start.max(source_x as f64)).max(0.0);
                let weight = y_weight * x_weight;
                let (raw_y, raw_x) = if self.mask_crop {
                    (source_y, source_x)
                } else {
                    (source_y + self.crop[0], source_x + self.crop[2])
                };
                let masked = self.mask_crop
                    && (raw_y < self.crop[0]
                        || raw_y >= self.raw_h - self.crop[1]
                        || raw_x < self.crop[2]
                        || raw_x >= self.raw_w - self.crop[3]);
                if masked {
                    for sum in &mut sums {
                        *sum += self.crop_fill as f64 * weight;
                    }
                } else {
                    let palette_offset = usize::from(current[raw_y * self.raw_w + raw_x]) * 3;
                    sums[0] += palette[palette_offset] as f64 * weight;
                    sums[1] += palette[palette_offset + 1] as f64 * weight;
                    sums[2] += palette[palette_offset + 2] as f64 * weight;
                }
                weight_sum += weight;
            }
        }
        [
            (sums[0] / weight_sum).round() as u8,
            (sums[1] / weight_sum).round() as u8,
            (sums[2] / weight_sum).round() as u8,
        ]
    }

    #[inline(always)]
    fn grayscale(rgb: [u8; 3]) -> u8 {
        ((u32::from(rgb[0]) * 77 + u32::from(rgb[1]) * 150 + u32::from(rgb[2]) * 29 + 128) >> 8)
            as u8
    }

    fn write_frame(&self, current: &[u8], previous: Option<&[u8]>, output: &mut [u8]) {
        match (self.algorithm, self.out_c) {
            (ResizeAlgorithm::Nearest, 1) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        output[out_y * self.out_w + out_x] =
                            Self::grayscale(self.nearest_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            (ResizeAlgorithm::Bilinear, 1) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        output[out_y * self.out_w + out_x] =
                            Self::grayscale(self.bilinear_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            (ResizeAlgorithm::Area, 1) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        output[out_y * self.out_w + out_x] =
                            Self::grayscale(self.area_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            (ResizeAlgorithm::Nearest, 3) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        let offset = (out_y * self.out_w + out_x) * 3;
                        output[offset..offset + 3]
                            .copy_from_slice(&self.nearest_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            (ResizeAlgorithm::Bilinear, 3) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        let offset = (out_y * self.out_w + out_x) * 3;
                        output[offset..offset + 3]
                            .copy_from_slice(&self.bilinear_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            (ResizeAlgorithm::Area, 3) => {
                for out_y in 0..self.out_h {
                    for out_x in 0..self.out_w {
                        let offset = (out_y * self.out_w + out_x) * 3;
                        output[offset..offset + 3]
                            .copy_from_slice(&self.area_rgb(current, previous, out_y, out_x));
                    }
                }
            }
            _ => unreachable!("output channel count is validated at construction"),
        }
    }

    fn write_indexed_frame(&self, current: &[u8], palette: &[u8], output: &mut [u8]) {
        let palette_rgb = Self::packed_palette(palette);
        for out_y in 0..self.out_h {
            for out_x in 0..self.out_w {
                output[out_y * self.out_w + out_x] = Self::grayscale(self.area_indexed_rgb(
                    current,
                    palette,
                    &palette_rgb,
                    out_y,
                    out_x,
                ));
            }
        }
    }

    #[inline]
    fn packed_palette(palette: &[u8]) -> [u64; 256] {
        let mut palette_rgb = [0_u64; 256];
        for (palette_index, packed) in palette_rgb.iter_mut().enumerate() {
            let offset = palette_index * 3;
            *packed = u64::from(palette[offset])
                | (u64::from(palette[offset + 1]) << INDEXED_AREA_CHANNEL_BITS)
                | (u64::from(palette[offset + 2]) << (INDEXED_AREA_CHANNEL_BITS * 2));
        }
        palette_rgb
    }

    #[inline]
    fn indexed_tile_fingerprint(&self, current: &[u8], tile: usize) -> u64 {
        let descriptor = unsafe { self.indexed_tiles.get_unchecked(tile) };
        let mut fingerprint = 0x9e37_79b9_7f4a_7c15_u64 ^ tile as u64;
        for &offset in &descriptor.fingerprint_offsets {
            fingerprint ^= u64::from(unsafe { *current.get_unchecked(offset as usize) })
                .wrapping_mul(0xbf58_476d_1ce4_e5b9);
            fingerprint = fingerprint
                .rotate_left(11)
                .wrapping_mul(0x94d0_49bb_1331_11eb);
        }
        fingerprint
    }

    #[inline]
    fn indexed_frame_fingerprint(current: &[u8]) -> u64 {
        let mut fingerprint = 0x243f_6a88_85a3_08d3_u64;
        for sample in 0..64 {
            let offset = sample * current.len() / 64;
            fingerprint ^= u64::from(unsafe { *current.get_unchecked(offset) })
                .wrapping_mul(0x9e37_79b9_7f4a_7c15);
            fingerprint = fingerprint
                .rotate_left(13)
                .wrapping_mul(0xbf58_476d_1ce4_e5b9);
        }
        fingerprint
    }

    #[inline]
    fn indexed_tile_equal(&self, current: &[u8], previous: &[u8], tile: usize) -> bool {
        let descriptor = unsafe { self.indexed_tiles.get_unchecked(tile) };
        let width = descriptor.source_x_end - descriptor.source_x_start;
        for y in descriptor.source_y_start..descriptor.source_y_end {
            let start = y * self.raw_w + descriptor.source_x_start;
            if current[start..start + width] != previous[start..start + width] {
                return false;
            }
        }
        true
    }

    #[inline]
    fn indexed_tile_is_dirty(dirty_tiles: &[u64; INDEXED_TILE_WORDS], tile: usize) -> bool {
        dirty_tiles[tile / 64] & (1_u64 << (tile % 64)) != 0
    }

    fn copy_indexed_frame_source(
        &self,
        current: &[u8],
        destination: &mut [u8],
        dirty_tiles: Option<&[u64; INDEXED_TILE_WORDS]>,
    ) {
        let Some(dirty) = dirty_tiles else {
            destination.copy_from_slice(current);
            return;
        };
        for tile_y in 0..INDEXED_TILE_ROWS {
            let mut tile_x = 0;
            while tile_x < INDEXED_TILE_COLUMNS {
                let tile = tile_y * INDEXED_TILE_COLUMNS + tile_x;
                if !Self::indexed_tile_is_dirty(dirty, tile) {
                    tile_x += 1;
                    continue;
                }
                let run_start = tile_x;
                tile_x += 1;
                while tile_x < INDEXED_TILE_COLUMNS
                    && Self::indexed_tile_is_dirty(dirty, tile_y * INDEXED_TILE_COLUMNS + tile_x)
                {
                    tile_x += 1;
                }
                let first = unsafe {
                    self.indexed_tiles
                        .get_unchecked(tile_y * INDEXED_TILE_COLUMNS + run_start)
                };
                let last = unsafe {
                    self.indexed_tiles
                        .get_unchecked(tile_y * INDEXED_TILE_COLUMNS + tile_x - 1)
                };
                let width = last.source_x_end - first.source_x_start;
                for y in first.source_y_start..first.source_y_end {
                    let start = y * self.raw_w + first.source_x_start;
                    destination[start..start + width]
                        .copy_from_slice(&current[start..start + width]);
                }
            }
        }
    }

    fn indexed_cached_frame_equal(
        &self,
        current: &[u8],
        previous: &[u8],
        current_background: Option<(u64, &[u64; INDEXED_TILE_WORDS])>,
        previous_sparse: bool,
        previous_background_token: u64,
        previous_dirty_tiles: &[u64; INDEXED_TILE_WORDS],
    ) -> bool {
        if !previous_sparse {
            return current == previous;
        }
        let Some((token, dirty_tiles)) = current_background else {
            return false;
        };
        if token != previous_background_token || dirty_tiles != previous_dirty_tiles {
            return false;
        }
        for tile_y in 0..INDEXED_TILE_ROWS {
            let mut tile_x = 0;
            while tile_x < INDEXED_TILE_COLUMNS {
                let tile = tile_y * INDEXED_TILE_COLUMNS + tile_x;
                if !Self::indexed_tile_is_dirty(dirty_tiles, tile) {
                    tile_x += 1;
                    continue;
                }
                let run_start = tile_x;
                tile_x += 1;
                while tile_x < INDEXED_TILE_COLUMNS
                    && Self::indexed_tile_is_dirty(
                        dirty_tiles,
                        tile_y * INDEXED_TILE_COLUMNS + tile_x,
                    )
                {
                    tile_x += 1;
                }
                let first = unsafe {
                    self.indexed_tiles
                        .get_unchecked(tile_y * INDEXED_TILE_COLUMNS + run_start)
                };
                let last = unsafe {
                    self.indexed_tiles
                        .get_unchecked(tile_y * INDEXED_TILE_COLUMNS + tile_x - 1)
                };
                let width = last.source_x_end - first.source_x_start;
                for y in first.source_y_start..first.source_y_end {
                    let start = y * self.raw_w + first.source_x_start;
                    if current[start..start + width] != previous[start..start + width] {
                        return false;
                    }
                }
            }
        }
        true
    }

    #[inline]
    fn indexed_shared_tile_equal(&self, current: &[u8], tile: usize, cached: &[u8]) -> bool {
        let descriptor = unsafe { self.indexed_tiles.get_unchecked(tile) };
        let width = descriptor.source_x_end - descriptor.source_x_start;
        for (row, y) in (descriptor.source_y_start..descriptor.source_y_end).enumerate() {
            let current_start = y * self.raw_w + descriptor.source_x_start;
            let cached_start = row * width;
            if current[current_start..current_start + width]
                != cached[cached_start..cached_start + width]
            {
                return false;
            }
        }
        true
    }

    fn capture_indexed_shared_tile_into(
        &self,
        current: &[u8],
        palette: &[u8],
        output: &[u8],
        tile: usize,
        fingerprint: u64,
        entry: &mut IndexedSharedTileEntry,
    ) {
        let descriptor = unsafe { self.indexed_tiles.get_unchecked(tile) };
        let source_width = descriptor.source_x_end - descriptor.source_x_start;
        let source_height = descriptor.source_y_end - descriptor.source_y_start;
        entry.fingerprint = fingerprint;
        entry.palette.copy_from_slice(palette);
        entry.source.clear();
        entry.source.reserve(source_width * source_height);
        for y in descriptor.source_y_start..descriptor.source_y_end {
            let start = y * self.raw_w + descriptor.source_x_start;
            entry
                .source
                .extend_from_slice(&current[start..start + source_width]);
        }
        let output_width = descriptor.output_x_end - descriptor.output_x_start;
        let output_height = descriptor.output_y_end - descriptor.output_y_start;
        entry.output.clear();
        entry.output.reserve(output_width * output_height);
        for out_y in descriptor.output_y_start..descriptor.output_y_end {
            let start = out_y * self.out_w + descriptor.output_x_start;
            entry
                .output
                .extend_from_slice(&output[start..start + output_width]);
        }
    }

    #[inline]
    fn copy_indexed_shared_tile_output(&self, cached: &[u8], output: &mut [u8], tile: usize) {
        let descriptor = unsafe { self.indexed_tiles.get_unchecked(tile) };
        let width = descriptor.output_x_end - descriptor.output_x_start;
        for (row, out_y) in (descriptor.output_y_start..descriptor.output_y_end).enumerate() {
            let output_start = out_y * self.out_w + descriptor.output_x_start;
            let cached_start = row * width;
            output[output_start..output_start + width]
                .copy_from_slice(&cached[cached_start..cached_start + width]);
        }
    }

    fn write_indexed_tile(
        &self,
        current: &[u8],
        palette: &[u8],
        palette_rgb: &[u64; 256],
        output: &mut [u8],
        tile: usize,
    ) {
        let descriptor = unsafe { self.indexed_tiles.get_unchecked(tile) };
        for out_y in descriptor.output_y_start..descriptor.output_y_end {
            for out_x in descriptor.output_x_start..descriptor.output_x_end {
                let output_offset = out_y * self.out_w + out_x;
                output[output_offset] = Self::grayscale(self.area_indexed_rgb(
                    current,
                    palette,
                    palette_rgb,
                    out_y,
                    out_x,
                ));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_indexed_tile_shared(
        &self,
        current: &[u8],
        palette: &[u8],
        palette_rgb: &[u64; 256],
        output: &mut [u8],
        tile: usize,
        fingerprint: u64,
        shared_tile: &RwLock<IndexedSharedTileCache>,
    ) {
        {
            let shared = shared_tile
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(entry) = shared.entries.iter().find(|entry| {
                entry.fingerprint == fingerprint
                    && entry.palette.as_slice() == palette
                    && self.indexed_shared_tile_equal(current, tile, &entry.source)
            }) {
                self.copy_indexed_shared_tile_output(&entry.output, output, tile);
                return;
            }
        }
        self.write_indexed_tile(current, palette, palette_rgb, output, tile);
        let mut shared = shared_tile
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if shared.entries.iter().any(|entry| {
            entry.fingerprint == fingerprint
                && entry.palette.as_slice() == palette
                && self.indexed_shared_tile_equal(current, tile, &entry.source)
        }) {
            return;
        }
        if shared.entries.len() < INDEXED_SHARED_TILE_CAPACITY {
            let descriptor = unsafe { self.indexed_tiles.get_unchecked(tile) };
            let source_area = (descriptor.source_x_end - descriptor.source_x_start)
                * (descriptor.source_y_end - descriptor.source_y_start);
            let output_area = (descriptor.output_x_end - descriptor.output_x_start)
                * (descriptor.output_y_end - descriptor.output_y_start);
            let mut entry = IndexedSharedTileEntry {
                fingerprint,
                palette: vec![0; palette.len()],
                source: Vec::with_capacity(source_area),
                output: Vec::with_capacity(output_area),
            };
            self.capture_indexed_shared_tile_into(
                current,
                palette,
                output,
                tile,
                fingerprint,
                &mut entry,
            );
            shared.entries.push(entry);
        } else {
            let cursor = shared.cursor;
            self.capture_indexed_shared_tile_into(
                current,
                palette,
                output,
                tile,
                fingerprint,
                &mut shared.entries[cursor],
            );
            shared.cursor = (cursor + 1) % INDEXED_SHARED_TILE_CAPACITY;
        }
    }

    fn write_indexed_frame_cached(
        &self,
        current: &[u8],
        palette: &[u8],
        cache: &mut IndexedAreaCache,
        output: &mut [u8],
        shared_tiles: Option<&[RwLock<IndexedSharedTileCache>]>,
        background: Option<(usize, u64, [u64; INDEXED_TILE_WORDS])>,
    ) {
        let background_frame = background
            .as_ref()
            .map(|(_, token, dirty_tiles)| (*token, dirty_tiles));
        if !cache.valid || cache.palette != palette {
            cache.last_frame_result = 0;
            cache.palette_rgb = Self::packed_palette(palette);
            self.copy_indexed_frame_source(
                current,
                &mut cache.frame,
                background_frame.map(|(_, dirty_tiles)| dirty_tiles),
            );
            cache.frame_fingerprint = Self::indexed_frame_fingerprint(current);
            cache.frame_sparse = background_frame.is_some();
            cache.frame_background_token = background_frame.map_or(0, |(token, _)| token);
            cache.frame_dirty_tiles =
                background_frame.map_or([0; INDEXED_TILE_WORDS], |(_, dirty_tiles)| *dirty_tiles);
            cache.palette.copy_from_slice(palette);
            cache.history.clear();
            cache.history_cursor = 0;
            for entry in &mut cache.backgrounds {
                entry.valid_tiles = [0; INDEXED_TILE_WORDS];
            }
            for output_offset in 0..cache.output.len() {
                let out_y = output_offset / self.out_w;
                let out_x = output_offset % self.out_w;
                cache.output[output_offset] = Self::grayscale(self.area_indexed_rgb(
                    current,
                    palette,
                    &cache.palette_rgb,
                    out_y,
                    out_x,
                ));
            }
            for tile in 0..INDEXED_TILE_COUNT {
                cache.tile_fingerprints[tile] = self.indexed_tile_fingerprint(current, tile);
            }
            cache.valid = true;
            if let Some((slot, token, dirty_tiles)) = background {
                let entry = &mut cache.backgrounds[slot];
                if entry.token != token {
                    entry.token = token;
                    entry.valid_tiles = [0; INDEXED_TILE_WORDS];
                }
                if entry.output.len() != cache.output.len() {
                    entry.output.resize(cache.output.len(), 0);
                }
                for tile in 0..INDEXED_TILE_COUNT {
                    let word = tile / 64;
                    let bit = 1_u64 << (tile % 64);
                    if dirty_tiles[word] & bit == 0 {
                        let descriptor = unsafe { self.indexed_tiles.get_unchecked(tile) };
                        for out_y in descriptor.output_y_start..descriptor.output_y_end {
                            let start = out_y * self.out_w + descriptor.output_x_start;
                            let end = out_y * self.out_w + descriptor.output_x_end;
                            entry.output[start..end].copy_from_slice(&cache.output[start..end]);
                        }
                        entry.valid_tiles[word] |= bit;
                    }
                }
            }
            output.copy_from_slice(&cache.output);
            return;
        }
        let frame_fingerprint = Self::indexed_frame_fingerprint(current);
        if (!cache.frame_sparse
            || background_frame.is_some_and(|(token, dirty_tiles)| {
                token == cache.frame_background_token && dirty_tiles == &cache.frame_dirty_tiles
            }))
            && frame_fingerprint == cache.frame_fingerprint
            && self.indexed_cached_frame_equal(
                current,
                &cache.frame,
                background_frame,
                cache.frame_sparse,
                cache.frame_background_token,
                &cache.frame_dirty_tiles,
            )
        {
            cache.last_frame_result = 1;
            output.copy_from_slice(&cache.output);
            return;
        }
        if let Some(snapshot_index) = cache.history.iter().position(|snapshot| {
            (!snapshot.frame_sparse
                || background_frame.is_some_and(|(token, dirty_tiles)| {
                    token == snapshot.frame_background_token
                        && dirty_tiles == &snapshot.frame_dirty_tiles
                }))
                && snapshot.frame_fingerprint == frame_fingerprint
                && self.indexed_cached_frame_equal(
                    current,
                    &snapshot.frame,
                    background_frame,
                    snapshot.frame_sparse,
                    snapshot.frame_background_token,
                    &snapshot.frame_dirty_tiles,
                )
        }) {
            cache.last_frame_result = 2;
            let snapshot = &mut cache.history[snapshot_index];
            std::mem::swap(&mut cache.frame, &mut snapshot.frame);
            std::mem::swap(&mut cache.output, &mut snapshot.output);
            std::mem::swap(
                &mut cache.tile_fingerprints,
                &mut snapshot.tile_fingerprints,
            );
            std::mem::swap(
                &mut cache.frame_fingerprint,
                &mut snapshot.frame_fingerprint,
            );
            std::mem::swap(&mut cache.frame_sparse, &mut snapshot.frame_sparse);
            std::mem::swap(
                &mut cache.frame_background_token,
                &mut snapshot.frame_background_token,
            );
            std::mem::swap(
                &mut cache.frame_dirty_tiles,
                &mut snapshot.frame_dirty_tiles,
            );
            output.copy_from_slice(&cache.output);
            return;
        }

        let history_index = if cache.history.len() < INDEXED_HISTORY_CAPACITY {
            cache.history.push(IndexedAreaSnapshot {
                frame: vec![0; cache.frame.len()],
                output: vec![0; cache.output.len()],
                tile_fingerprints: vec![0; INDEXED_TILE_COUNT],
                frame_fingerprint: 0,
                frame_sparse: false,
                frame_background_token: 0,
                frame_dirty_tiles: [0; INDEXED_TILE_WORDS],
            });
            cache.history.len() - 1
        } else {
            let index = cache.history_cursor;
            cache.history_cursor = (cache.history_cursor + 1) % INDEXED_HISTORY_CAPACITY;
            index
        };
        cache.last_frame_result = 3;
        {
            let snapshot = &mut cache.history[history_index];
            std::mem::swap(&mut cache.frame, &mut snapshot.frame);
            std::mem::swap(&mut cache.output, &mut snapshot.output);
            std::mem::swap(
                &mut cache.tile_fingerprints,
                &mut snapshot.tile_fingerprints,
            );
            std::mem::swap(
                &mut cache.frame_fingerprint,
                &mut snapshot.frame_fingerprint,
            );
            std::mem::swap(&mut cache.frame_sparse, &mut snapshot.frame_sparse);
            std::mem::swap(
                &mut cache.frame_background_token,
                &mut snapshot.frame_background_token,
            );
            std::mem::swap(
                &mut cache.frame_dirty_tiles,
                &mut snapshot.frame_dirty_tiles,
            );
        }

        let background_prefilled = background.is_some_and(|(slot, token, dirty_tiles)| {
            let entry = &cache.backgrounds[slot];
            let all_clean_tiles_valid = (0..INDEXED_TILE_COUNT).all(|tile| {
                Self::indexed_tile_is_dirty(&dirty_tiles, tile)
                    || entry.valid_tiles[tile / 64] & (1_u64 << (tile % 64)) != 0
            });
            if indexed_background_prefill_enabled() && entry.token == token && all_clean_tiles_valid
            {
                cache.output.copy_from_slice(&entry.output);
                true
            } else {
                false
            }
        });

        for tile in 0..INDEXED_TILE_COUNT {
            let background_tile = background.and_then(|(slot, token, dirty_tiles)| {
                let word = tile / 64;
                let bit = 1_u64 << (tile % 64);
                (dirty_tiles[word] & bit == 0).then_some((slot, token, word, bit))
            });
            if let Some((slot, token, word, bit)) = background_tile {
                if background_prefilled {
                    cache.tile_fingerprints[tile] = 0;
                    continue;
                }
                let entry = &mut cache.backgrounds[slot];
                if entry.token != token {
                    entry.token = token;
                    entry.valid_tiles = [0; INDEXED_TILE_WORDS];
                }
                if entry.valid_tiles[word] & bit != 0 {
                    let descriptor = unsafe { self.indexed_tiles.get_unchecked(tile) };
                    for out_y in descriptor.output_y_start..descriptor.output_y_end {
                        let start = out_y * self.out_w + descriptor.output_x_start;
                        let end = out_y * self.out_w + descriptor.output_x_end;
                        cache.output[start..end].copy_from_slice(&entry.output[start..end]);
                    }
                    cache.tile_fingerprints[tile] = 0;
                    continue;
                }
            }
            let fingerprint = self.indexed_tile_fingerprint(current, tile);
            let cached_snapshot = cache.history.iter().position(|snapshot| {
                (!snapshot.frame_sparse
                    || Self::indexed_tile_is_dirty(&snapshot.frame_dirty_tiles, tile))
                    && snapshot.tile_fingerprints[tile] == fingerprint
                    && self.indexed_tile_equal(current, &snapshot.frame, tile)
            });
            let descriptor = unsafe { self.indexed_tiles.get_unchecked(tile) };
            let output_y_start = descriptor.output_y_start;
            let output_y_end = descriptor.output_y_end;
            let output_x_start = descriptor.output_x_start;
            let output_x_end = descriptor.output_x_end;
            if let Some(snapshot_index) = cached_snapshot {
                let snapshot = &cache.history[snapshot_index];
                for out_y in output_y_start..output_y_end {
                    let start = out_y * self.out_w + output_x_start;
                    let end = out_y * self.out_w + output_x_end;
                    cache.output[start..end].copy_from_slice(&snapshot.output[start..end]);
                }
            } else {
                let mut shared_processed = false;
                if let Some(tiles) = shared_tiles {
                    self.write_indexed_tile_shared(
                        current,
                        palette,
                        &cache.palette_rgb,
                        &mut cache.output,
                        tile,
                        fingerprint,
                        &tiles[tile],
                    );
                    shared_processed = true;
                }
                if !shared_processed {
                    self.write_indexed_tile(
                        current,
                        palette,
                        &cache.palette_rgb,
                        &mut cache.output,
                        tile,
                    );
                }
            }
            cache.tile_fingerprints[tile] = fingerprint;
            if let Some((slot, token, word, bit)) = background_tile {
                let entry = &mut cache.backgrounds[slot];
                if entry.token != token {
                    entry.token = token;
                    entry.valid_tiles = [0; INDEXED_TILE_WORDS];
                }
                if entry.output.len() != cache.output.len() {
                    entry.output.resize(cache.output.len(), 0);
                }
                let descriptor = unsafe { self.indexed_tiles.get_unchecked(tile) };
                for out_y in descriptor.output_y_start..descriptor.output_y_end {
                    let start = out_y * self.out_w + descriptor.output_x_start;
                    let end = out_y * self.out_w + descriptor.output_x_end;
                    entry.output[start..end].copy_from_slice(&cache.output[start..end]);
                }
                entry.valid_tiles[word] |= bit;
            }
        }
        self.copy_indexed_frame_source(
            current,
            &mut cache.frame,
            background_frame.map(|(_, dirty_tiles)| dirty_tiles),
        );
        cache.frame_fingerprint = frame_fingerprint;
        cache.frame_sparse = background_frame.is_some();
        cache.frame_background_token = background_frame.map_or(0, |(token, _)| token);
        cache.frame_dirty_tiles =
            background_frame.map_or([0; INDEXED_TILE_WORDS], |(_, dirty_tiles)| *dirty_tiles);
        output.copy_from_slice(&cache.output);
    }
}

#[derive(Clone, Copy)]
enum ObservationLayout {
    Hwc,
    Chw,
}

impl ObservationLayout {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "hwc" => Ok(Self::Hwc),
            "chw" => Ok(Self::Chw),
            _ => Err(PyValueError::new_err("layout must be 'hwc' or 'chw'")),
        }
    }
}

#[repr(align(64))]
struct IndexedAreaCache {
    valid: bool,
    frame: Vec<u8>,
    frame_fingerprint: u64,
    frame_sparse: bool,
    frame_background_token: u64,
    frame_dirty_tiles: [u64; INDEXED_TILE_WORDS],
    palette: Vec<u8>,
    palette_rgb: [u64; 256],
    output: Vec<u8>,
    tile_fingerprints: Vec<u64>,
    history: Vec<IndexedAreaSnapshot>,
    history_cursor: usize,
    backgrounds: Vec<IndexedBackgroundEntry>,
    last_frame_result: u8,
}

struct IndexedAreaSnapshot {
    frame: Vec<u8>,
    output: Vec<u8>,
    tile_fingerprints: Vec<u64>,
    frame_fingerprint: u64,
    frame_sparse: bool,
    frame_background_token: u64,
    frame_dirty_tiles: [u64; INDEXED_TILE_WORDS],
}

struct IndexedBackgroundEntry {
    token: u64,
    valid_tiles: [u64; INDEXED_TILE_WORDS],
    output: Vec<u8>,
}

struct IndexedSharedTileEntry {
    fingerprint: u64,
    palette: Vec<u8>,
    source: Vec<u8>,
    output: Vec<u8>,
}

struct IndexedSharedTileCache {
    entries: Vec<IndexedSharedTileEntry>,
    cursor: usize,
}

impl IndexedSharedTileCache {
    fn new() -> Self {
        Self {
            entries: Vec::with_capacity(INDEXED_SHARED_TILE_CAPACITY),
            cursor: 0,
        }
    }
}

impl IndexedAreaCache {
    fn new(raw_frame_size: usize, output_frame_size: usize) -> Self {
        Self {
            valid: false,
            frame: vec![0; raw_frame_size],
            frame_fingerprint: 0,
            frame_sparse: false,
            frame_background_token: 0,
            frame_dirty_tiles: [0; INDEXED_TILE_WORDS],
            palette: vec![0; 256 * 3],
            palette_rgb: [0; 256],
            output: vec![0; output_frame_size],
            tile_fingerprints: vec![0; INDEXED_TILE_COUNT],
            history: Vec::with_capacity(INDEXED_HISTORY_CAPACITY),
            history_cursor: 0,
            last_frame_result: 0,
            backgrounds: (0..INDEXED_BACKGROUND_CAPACITY)
                .map(|_| IndexedBackgroundEntry {
                    token: 0,
                    valid_tiles: [0; INDEXED_TILE_WORDS],
                    output: Vec::new(),
                })
                .collect(),
        }
    }
}

#[pyclass]
struct ImageProcessor {
    num_envs: usize,
    frame_stack: usize,
    layout: ObservationLayout,
    plan: ImagePlan,
    pool: ThreadPool,
    indexed_area_caches: Vec<Mutex<IndexedAreaCache>>,
    indexed_shared_tiles: Vec<RwLock<IndexedSharedTileCache>>,
    shared_tile_cache: bool,
    frame_sequence: bool,
    async_reset_threshold: usize,
    discrete_action_cache: Mutex<(Vec<i64>, usize)>,
}

impl ImageProcessor {
    fn validate_arrays(
        &self,
        current_shape: &[usize],
        stack_shape: &[usize],
        heads_shape: &[usize],
        output_shape: &[usize],
        previous_shape: Option<&[usize]>,
    ) -> PyResult<()> {
        let expected_current = [self.num_envs, self.plan.raw_h, self.plan.raw_w, 3];
        let expected_stack = [
            self.num_envs,
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.num_envs,
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.num_envs,
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if current_shape != expected_current {
            return Err(PyValueError::new_err(format!(
                "current must have shape {expected_current:?}"
            )));
        }
        if stack_shape != expected_stack {
            return Err(PyValueError::new_err(format!(
                "stack must have shape {expected_stack:?}"
            )));
        }
        if heads_shape != [self.num_envs] {
            return Err(PyValueError::new_err(format!(
                "heads must have shape ({},)",
                self.num_envs
            )));
        }
        if output_shape != expected_output {
            return Err(PyValueError::new_err(format!(
                "output must have shape {expected_output:?}"
            )));
        }
        if let Some(shape) = previous_shape
            && shape != expected_current
        {
            return Err(PyValueError::new_err(format!(
                "previous must have shape {expected_current:?}"
            )));
        }
        Ok(())
    }

    #[inline]
    fn write_observation(&self, stack: &[u8], head: usize, output: &mut [u8]) {
        let pixels = self.plan.out_h * self.plan.out_w;
        let frame_size = pixels * self.plan.out_c;
        match self.layout {
            ObservationLayout::Chw => {
                for output_slot in 0..self.frame_stack {
                    let source_slot = (head + 1 + output_slot) % self.frame_stack;
                    let source = &stack[source_slot * frame_size..(source_slot + 1) * frame_size];
                    if self.plan.out_c == 1 {
                        output[output_slot * pixels..(output_slot + 1) * pixels]
                            .copy_from_slice(source);
                    } else {
                        for channel in 0..self.plan.out_c {
                            let output_start = (output_slot * self.plan.out_c + channel) * pixels;
                            for pixel in 0..pixels {
                                output[output_start + pixel] =
                                    source[pixel * self.plan.out_c + channel];
                            }
                        }
                    }
                }
            }
            ObservationLayout::Hwc => {
                let stacked_channels = self.plan.out_c * self.frame_stack;
                for pixel in 0..pixels {
                    for output_slot in 0..self.frame_stack {
                        let source_slot = (head + 1 + output_slot) % self.frame_stack;
                        let source_start = source_slot * frame_size + pixel * self.plan.out_c;
                        let output_start = pixel * stacked_channels + output_slot * self.plan.out_c;
                        output[output_start..output_start + self.plan.out_c]
                            .copy_from_slice(&stack[source_start..source_start + self.plan.out_c]);
                    }
                }
            }
        }
    }
}

#[pymethods]
impl ImageProcessor {
    #[new]
    #[pyo3(signature = (
        num_envs,
        raw_height,
        raw_width,
        out_height,
        out_width,
        out_channels,
        crop,
        mask_crop,
        crop_fill,
        algorithm,
        frame_stack,
        layout,
        num_threads,
        optimized_profile=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        raw_height: usize,
        raw_width: usize,
        out_height: usize,
        out_width: usize,
        out_channels: usize,
        crop: Vec<usize>,
        mask_crop: bool,
        crop_fill: u8,
        algorithm: &str,
        frame_stack: usize,
        layout: &str,
        num_threads: usize,
        optimized_profile: bool,
    ) -> PyResult<Self> {
        if num_envs == 0
            || raw_height == 0
            || raw_width == 0
            || out_height == 0
            || out_width == 0
            || frame_stack == 0
            || num_threads == 0
        {
            return Err(PyValueError::new_err(
                "dimensions, frame_stack, and num_threads must be positive",
            ));
        }
        if !matches!(out_channels, 1 | 3) {
            return Err(PyValueError::new_err("out_channels must be one or three"));
        }
        if crop.len() != 4 {
            return Err(PyValueError::new_err(
                "crop must contain top, bottom, left, right",
            ));
        }
        if crop[0] + crop[1] >= raw_height || crop[2] + crop[3] >= raw_width {
            return Err(PyValueError::new_err(
                "crop must preserve at least one source pixel",
            ));
        }
        let plan = ImagePlan::new(
            raw_height,
            raw_width,
            out_height,
            out_width,
            out_channels,
            [crop[0], crop[1], crop[2], crop[3]],
            mask_crop,
            crop_fill,
            ResizeAlgorithm::parse(algorithm)?,
        );
        let available_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let worker_threads = std::env::var("ENV_VIZDOOM_TURBO_POOL_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or_else(|| {
                let requested_threads = if optimized_profile {
                    num_threads.min(4)
                } else {
                    num_threads
                };
                requested_threads.min(num_envs).min(available_threads)
            })
            .min(num_threads)
            .min(num_envs);
        let pool = ThreadPoolBuilder::new()
            .num_threads(worker_threads)
            .thread_name(|index| format!("env-vizdoom-turbo-image-{index}"))
            .build()
            .map_err(|error| {
                PyRuntimeError::new_err(format!(
                    "failed to create native image worker pool: {error}"
                ))
            })?;
        let indexed_area_caches = if plan.indexed_tiles.is_empty() {
            Vec::new()
        } else {
            (0..num_envs)
                .map(|_| {
                    Mutex::new(IndexedAreaCache::new(
                        raw_height * raw_width,
                        out_height * out_width,
                    ))
                })
                .collect()
        };
        let indexed_shared_tiles = if plan.indexed_tiles.is_empty() {
            Vec::new()
        } else {
            (0..INDEXED_TILE_COUNT)
                .map(|_| RwLock::new(IndexedSharedTileCache::new()))
                .collect()
        };
        Ok(Self {
            num_envs,
            frame_stack,
            layout: ObservationLayout::parse(layout)?,
            plan,
            pool,
            indexed_area_caches,
            indexed_shared_tiles,
            shared_tile_cache: (optimized_profile
                && std::env::var_os("ENV_VIZDOOM_TURBO_DISABLE_SHARED_TILE_CACHE").is_none())
                || std::env::var_os("ENV_VIZDOOM_TURBO_SHARED_TILE_CACHE").is_some(),
            frame_sequence: optimized_profile
                || std::env::var_os("ENV_VIZDOOM_TURBO_FRAME_SEQUENCE").is_some(),
            async_reset_threshold: std::env::var("ENV_VIZDOOM_TURBO_ASYNC_RESET_THRESHOLD")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value <= num_envs)
                .unwrap_or(16),
            discrete_action_cache: Mutex::new((Vec::new(), 0)),
        })
    }

    fn frame_cache_results(&self) -> Vec<u8> {
        self.indexed_area_caches
            .iter()
            .map(|cache| {
                cache
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .last_frame_result
            })
            .collect()
    }

    fn prepare_discrete_actions_into(
        &self,
        actions: PyReadonlyArray1<'_, i64>,
        table: PyReadonlyArray2<'_, f64>,
        mut output: PyReadwriteArray2<'_, f64>,
    ) -> PyResult<()> {
        if actions.shape() != [self.num_envs]
            || table.shape().len() != 2
            || output.shape() != [self.num_envs, table.shape()[1]]
        {
            return Err(PyValueError::new_err(
                "discrete action arrays have invalid shapes",
            ));
        }
        let action_count = table.shape()[0];
        let action_width = table.shape()[1];
        let actions = actions.as_slice()?;
        let table = table.as_slice()?;
        let output = output.as_slice_mut()?;
        let output_address = output.as_ptr() as usize;
        let mut cache = self
            .discrete_action_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache.1 == output_address && cache.0.as_slice() == actions {
            return Ok(());
        }
        for (lane, &action) in actions.iter().enumerate() {
            if action < 0 || action as usize >= action_count {
                return Err(PyValueError::new_err(format!(
                    "actions must be in [0, {}]",
                    action_count - 1
                )));
            }
            let source = action as usize * action_width;
            let destination = lane * action_width;
            output[destination..destination + action_width]
                .copy_from_slice(&table[source..source + action_width]);
        }
        cache.0.clear();
        cache.0.extend_from_slice(actions);
        cache.1 = output_address;
        Ok(())
    }

    #[pyo3(signature = (current, stack, heads, output, previous=None))]
    fn step_into(
        &self,
        py: Python<'_>,
        current: PyReadonlyArray4<'_, u8>,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
        previous: Option<PyReadonlyArray4<'_, u8>>,
    ) -> PyResult<()> {
        self.validate_arrays(
            current.shape(),
            stack.shape(),
            heads.shape(),
            output.shape(),
            previous.as_ref().map(|array| array.shape()),
        )?;
        let current_data = current.as_slice()?;
        let previous_data = previous
            .as_ref()
            .map(|array| array.as_slice())
            .transpose()?;
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let raw_frame_size = self.plan.raw_h * self.plan.raw_w * 3;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        let stack_lane_size = image_frame_size * self.frame_stack;
        let output_lane_size = image_frame_size * self.frame_stack;
        py.detach(|| {
            self.pool.install(|| {
                heads_data
                    .par_iter_mut()
                    .zip(stack_data.par_chunks_mut(stack_lane_size))
                    .zip(output_data.par_chunks_mut(output_lane_size))
                    .zip(current_data.par_chunks(raw_frame_size))
                    .enumerate()
                    .for_each(
                        |(lane, (((head, stack_lane), output_lane), current_frame))| {
                            let new_head = (*head as usize + 1) % self.frame_stack;
                            let destination = &mut stack_lane
                                [new_head * image_frame_size..(new_head + 1) * image_frame_size];
                            let prior = previous_data.map(|data| {
                                &data[lane * raw_frame_size..(lane + 1) * raw_frame_size]
                            });
                            self.plan.write_frame(current_frame, prior, destination);
                            *head = new_head as i64;
                            self.write_observation(stack_lane, new_head, output_lane);
                        },
                    );
            });
        });
        Ok(())
    }

    #[pyo3(signature = (current, stack, heads, output, previous=None))]
    fn step_frames_into(
        &self,
        py: Python<'_>,
        current: Vec<PyReadonlyArray3<'_, u8>>,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
        previous: Option<Vec<PyReadonlyArray3<'_, u8>>>,
    ) -> PyResult<()> {
        let expected_frame = [self.plan.raw_h, self.plan.raw_w, 3];
        let expected_batch = [self.num_envs, self.plan.raw_h, self.plan.raw_w, 3];
        self.validate_arrays(
            &expected_batch,
            stack.shape(),
            heads.shape(),
            output.shape(),
            None,
        )?;
        if current.len() != self.num_envs
            || current.iter().any(|frame| frame.shape() != expected_frame)
        {
            return Err(PyValueError::new_err(format!(
                "current must contain {} frames with shape {expected_frame:?}",
                self.num_envs
            )));
        }
        if let Some(prior) = previous.as_ref()
            && (prior.len() != self.num_envs
                || prior.iter().any(|frame| frame.shape() != expected_frame))
        {
            return Err(PyValueError::new_err(format!(
                "previous must contain {} frames with shape {expected_frame:?}",
                self.num_envs
            )));
        }
        let current_data = current
            .iter()
            .map(|frame| frame.as_slice().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let previous_data = previous
            .as_ref()
            .map(|frames| {
                frames
                    .iter()
                    .map(|frame| frame.as_slice().map_err(PyErr::from))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        let stack_lane_size = image_frame_size * self.frame_stack;
        let output_lane_size = image_frame_size * self.frame_stack;
        py.detach(|| {
            self.pool.install(|| {
                heads_data
                    .par_iter_mut()
                    .zip(stack_data.par_chunks_mut(stack_lane_size))
                    .zip(output_data.par_chunks_mut(output_lane_size))
                    .enumerate()
                    .for_each(|(lane, ((head, stack_lane), output_lane))| {
                        let new_head = (*head as usize + 1) % self.frame_stack;
                        let destination = &mut stack_lane
                            [new_head * image_frame_size..(new_head + 1) * image_frame_size];
                        let prior = previous_data.as_ref().map(|frames| frames[lane]);
                        self.plan
                            .write_frame(current_data[lane], prior, destination);
                        *head = new_head as i64;
                        self.write_observation(stack_lane, new_head, output_lane);
                    });
            });
        });
        Ok(())
    }

    fn step_lane_into(
        &self,
        py: Python<'_>,
        current: PyReadonlyArray3<'_, u8>,
        mut stack: PyReadwriteArray4<'_, u8>,
        mut head: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray3<'_, u8>,
    ) -> PyResult<()> {
        let expected_current = [self.plan.raw_h, self.plan.raw_w, 3];
        let expected_stack = [
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if current.shape() != expected_current {
            return Err(PyValueError::new_err(format!(
                "current must have shape {expected_current:?}"
            )));
        }
        if stack.shape() != expected_stack {
            return Err(PyValueError::new_err(format!(
                "stack must have shape {expected_stack:?}"
            )));
        }
        if head.shape() != [1] {
            return Err(PyValueError::new_err("head must have shape (1,)"));
        }
        if output.shape() != expected_output {
            return Err(PyValueError::new_err(format!(
                "output must have shape {expected_output:?}"
            )));
        }

        let current_data = current.as_slice()?;
        let stack_data = stack.as_slice_mut()?;
        let head_data = head.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        py.detach(|| {
            let new_head = (head_data[0] as usize + 1) % self.frame_stack;
            let destination =
                &mut stack_data[new_head * image_frame_size..(new_head + 1) * image_frame_size];
            self.plan.write_frame(current_data, None, destination);
            head_data[0] = new_head as i64;
            self.write_observation(stack_data, new_head, output_data);
        });
        Ok(())
    }

    fn step_indexed_lane_into(
        &self,
        py: Python<'_>,
        current: PyReadonlyArray2<'_, u8>,
        palette: PyReadonlyArray2<'_, u8>,
        mut stack: PyReadwriteArray4<'_, u8>,
        mut head: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray3<'_, u8>,
    ) -> PyResult<()> {
        if !self.plan.supports_indexed_area() {
            return Err(PyValueError::new_err(
                "indexed preprocessing requires 320x240 to 84x84 area-resize grayscale with crop removal or masking",
            ));
        }
        if current.shape() != [self.plan.raw_h, self.plan.raw_w] {
            return Err(PyValueError::new_err(
                "current has an invalid indexed shape",
            ));
        }
        if palette.shape() != [256, 3] {
            return Err(PyValueError::new_err("palette must have shape (256, 3)"));
        }
        let expected_stack = [
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if stack.shape() != expected_stack || head.shape() != [1] {
            return Err(PyValueError::new_err("stack or head has an invalid shape"));
        }
        if output.shape() != expected_output {
            return Err(PyValueError::new_err("output has an invalid shape"));
        }

        let current_data = current.as_slice()?;
        let palette_data = palette.as_slice()?;
        let stack_data = stack.as_slice_mut()?;
        let head_data = head.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w;
        py.detach(|| {
            let new_head = (head_data[0] as usize + 1) % self.frame_stack;
            let destination =
                &mut stack_data[new_head * image_frame_size..(new_head + 1) * image_frame_size];
            self.plan
                .write_indexed_frame(current_data, palette_data, destination);
            head_data[0] = new_head as i64;
            self.write_observation(stack_data, new_head, output_data);
        });
        Ok(())
    }

    #[pyo3(signature = (
        context,
        start_address,
        finish_address,
        frame_address,
        palette_address,
        error_clear_address,
        error_copy_address,
        stack,
        heads,
        output,
        terminal_indexed,
        terminal_palettes,
        background_address=None,
        reset_start_address=None,
        reset_seeds=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn step_native_batch_into(
        &self,
        py: Python<'_>,
        context: usize,
        start_address: usize,
        finish_address: usize,
        frame_address: usize,
        palette_address: usize,
        error_clear_address: usize,
        error_copy_address: usize,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
        mut terminal_indexed: PyReadwriteArray3<'_, u8>,
        mut terminal_palettes: PyReadwriteArray3<'_, u8>,
        background_address: Option<usize>,
        reset_start_address: Option<usize>,
        reset_seeds: Option<PyReadonlyArray1<'_, u32>>,
    ) -> PyResult<bool> {
        if !self.plan.supports_indexed_area() {
            return Err(PyValueError::new_err(
                "native batch preprocessing requires 320x240 to 84x84 area-resize grayscale with crop removal or masking",
            ));
        }
        let expected_stack = [
            self.num_envs,
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.num_envs,
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.num_envs,
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if stack.shape() != expected_stack || heads.shape() != [self.num_envs] {
            return Err(PyValueError::new_err("stack or heads has an invalid shape"));
        }
        if output.shape() != expected_output
            || terminal_indexed.shape() != [self.num_envs, self.plan.raw_h, self.plan.raw_w]
            || terminal_palettes.shape() != [self.num_envs, 256, 3]
        {
            return Err(PyValueError::new_err("output has an invalid shape"));
        }
        if reset_seeds
            .as_ref()
            .is_some_and(|values| values.shape() != [self.num_envs])
            || reset_start_address.is_some() != reset_seeds.is_some()
        {
            return Err(PyValueError::new_err(
                "native reset-start address and lane seeds must be supplied together",
            ));
        }

        type StartAll = unsafe extern "C" fn(*mut c_void) -> u64;
        type StepLane = unsafe extern "C" fn(*mut c_void, usize) -> u64;
        type StartResetLane = unsafe extern "C" fn(*mut c_void, usize, u32) -> u32;
        type BufferLane = unsafe extern "C" fn(*mut c_void, usize) -> *const u8;
        type BackgroundDataLane = unsafe extern "C" fn(*mut c_void, usize) -> *const u64;
        let start_all: StartAll = unsafe { std::mem::transmute(start_address) };
        let finish_lane: StepLane = unsafe { std::mem::transmute(finish_address) };
        let clear_error: NativeClearError = unsafe { std::mem::transmute(error_clear_address) };
        let copy_error: NativeCopyError = unsafe { std::mem::transmute(error_copy_address) };
        let frame_lane: BufferLane = unsafe { std::mem::transmute(frame_address) };
        let palette_lane: BufferLane = unsafe { std::mem::transmute(palette_address) };
        let background_data_lane: Option<BackgroundDataLane> =
            background_address.map(|address| unsafe { std::mem::transmute(address) });
        let start_reset_lane: Option<StartResetLane> =
            reset_start_address.map(|address| unsafe { std::mem::transmute(address) });
        let reset_seed_data = reset_seeds
            .as_ref()
            .map(|values| values.as_slice())
            .transpose()?;
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let terminal_indexed_data = terminal_indexed.as_slice_mut()?;
        let terminal_palette_data = terminal_palettes.as_slice_mut()?;
        let raw_frame_size = self.plan.raw_h * self.plan.raw_w;
        let palette_size = 256 * 3;
        let image_frame_size = self.plan.out_h * self.plan.out_w;
        let stack_lane_size = self.frame_stack * image_frame_size;
        let output_lane_size = self.frame_stack * image_frame_size;
        let failed = AtomicBool::new(false);
        let terminal = AtomicBool::new(false);
        let finished_lanes = AtomicUsize::new(0);
        let pending_resets = (0..self.num_envs)
            .map(|_| AtomicBool::new(false))
            .collect::<Vec<_>>();

        unsafe { clear_error(context as *mut c_void) };

        let start_pending_resets = || {
            if let (Some(start_reset), Some(lane_seeds)) = (start_reset_lane, reset_seed_data) {
                for (lane, pending) in pending_resets.iter().enumerate() {
                    if pending.swap(false, Ordering::AcqRel)
                        && unsafe {
                            start_reset(
                                context as *mut c_void,
                                lane,
                                *lane_seeds.get_unchecked(lane),
                            )
                        } & 4
                            != 0
                    {
                        failed.store(true, Ordering::Relaxed);
                    }
                }
            }
        };

        let process_lane = |lane: usize,
                            stack_lane: &mut [u8],
                            head: &mut i64,
                            output_lane: &mut [u8],
                            terminal_indexed_lane: &mut [u8],
                            terminal_palette_lane: &mut [u8]| {
            let status = unsafe { finish_lane(context as *mut c_void, lane) };
            if status & 4 != 0 {
                failed.store(true, Ordering::Relaxed);
                return;
            }
            if status & 3 != 0 {
                let frame = unsafe {
                    std::slice::from_raw_parts(
                        frame_lane(context as *mut c_void, lane),
                        raw_frame_size,
                    )
                };
                let palette = unsafe {
                    std::slice::from_raw_parts(
                        palette_lane(context as *mut c_void, lane),
                        palette_size,
                    )
                };
                terminal_indexed_lane.copy_from_slice(frame);
                terminal_palette_lane.copy_from_slice(palette);
                pending_resets[lane].store(true, Ordering::Release);
            }
            let completed = finished_lanes.fetch_add(1, Ordering::AcqRel) + 1;
            if completed >= self.async_reset_threshold {
                start_pending_resets();
            }
            if failed.load(Ordering::Relaxed) {
                return;
            }
            let old_head = *head as usize;
            let new_head = (old_head + 1) % self.frame_stack;
            let destination = new_head * image_frame_size;
            if status & 3 != 0 {
                terminal.store(true, Ordering::Relaxed);
                let source = old_head * image_frame_size;
                if source < destination {
                    let (before, after) = stack_lane.split_at_mut(destination);
                    after[..image_frame_size]
                        .copy_from_slice(&before[source..source + image_frame_size]);
                } else if destination < source {
                    let (before, after) = stack_lane.split_at_mut(source);
                    before[destination..destination + image_frame_size]
                        .copy_from_slice(&after[..image_frame_size]);
                }
            } else if self.frame_sequence && status & 8 != 0 {
                let cache = self.indexed_area_caches[lane]
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                stack_lane[destination..destination + image_frame_size]
                    .copy_from_slice(&cache.output);
            } else {
                let frame = unsafe {
                    std::slice::from_raw_parts(
                        frame_lane(context as *mut c_void, lane),
                        self.plan.raw_h * self.plan.raw_w,
                    )
                };
                let palette = unsafe {
                    std::slice::from_raw_parts(palette_lane(context as *mut c_void, lane), 256 * 3)
                };
                let mut cache = self.indexed_area_caches[lane]
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let background = background_data_lane.and_then(|data_lane| {
                    let background_hit = status & (1_u64 << 52) != 0;
                    background_hit.then(|| {
                        let slot = ((status >> 44) & 255) as usize;
                        let data = unsafe {
                            std::slice::from_raw_parts(
                                data_lane(context as *mut c_void, lane),
                                1 + INDEXED_TILE_WORDS,
                            )
                        };
                        let token = data[0];
                        let mut dirty_tiles = [0_u64; INDEXED_TILE_WORDS];
                        dirty_tiles.copy_from_slice(&data[1..]);
                        (slot, token, dirty_tiles)
                    })
                });
                self.plan.write_indexed_frame_cached(
                    frame,
                    palette,
                    &mut cache,
                    &mut stack_lane[destination..destination + image_frame_size],
                    self.shared_tile_cache
                        .then_some(self.indexed_shared_tiles.as_slice()),
                    background,
                );
            }
            *head = new_head as i64;
            self.write_observation(stack_lane, new_head, output_lane);
        };

        py.detach(|| {
            let started = unsafe { start_all(context as *mut c_void) } & 4 == 0;
            if !started {
                failed.store(true, Ordering::Relaxed);
                return;
            }
            self.pool.install(|| {
                stack_data
                    .chunks_mut(stack_lane_size)
                    .zip(heads_data.iter_mut())
                    .zip(output_data.chunks_mut(output_lane_size))
                    .zip(terminal_indexed_data.chunks_mut(raw_frame_size))
                    .zip(terminal_palette_data.chunks_mut(palette_size))
                    .enumerate()
                    .par_bridge()
                    .for_each(
                        |(
                            lane,
                            (
                                (((stack_lane, head), output_lane), terminal_indexed_lane),
                                terminal_palette_lane,
                            ),
                        )| {
                            process_lane(
                                lane,
                                stack_lane,
                                head,
                                output_lane,
                                terminal_indexed_lane,
                                terminal_palette_lane,
                            );
                        },
                    );
            });
        });
        if failed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err(format!(
                "native Doom lane step failed: {}",
                native_error_detail(context, copy_error)
            )));
        }
        Ok(terminal.load(Ordering::Relaxed))
    }

    #[pyo3(signature = (
        context,
        frame_address,
        palette_address,
        error_clear_address,
        error_copy_address,
        mask,
        stack,
        heads,
        output,
        reset_address=None,
        seeds=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn reset_native_batch_into(
        &self,
        py: Python<'_>,
        context: usize,
        frame_address: usize,
        palette_address: usize,
        error_clear_address: usize,
        error_copy_address: usize,
        mask: PyReadonlyArray1<'_, bool>,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
        reset_address: Option<usize>,
        seeds: Option<PyReadonlyArray1<'_, u32>>,
    ) -> PyResult<()> {
        if !self.plan.supports_indexed_area() {
            return Err(PyValueError::new_err(
                "native batch preprocessing requires 320x240 to 84x84 area-resize grayscale with crop removal or masking",
            ));
        }
        let expected_stack = [
            self.num_envs,
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.num_envs,
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.num_envs,
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if mask.shape() != [self.num_envs]
            || stack.shape() != expected_stack
            || heads.shape() != [self.num_envs]
            || output.shape() != expected_output
        {
            return Err(PyValueError::new_err(
                "native reset arrays have invalid shapes",
            ));
        }
        if seeds
            .as_ref()
            .is_some_and(|values| values.shape() != [self.num_envs])
            || reset_address.is_some() != seeds.is_some()
        {
            return Err(PyValueError::new_err(
                "native reset address and lane seeds must be supplied together",
            ));
        }

        type BufferLane = unsafe extern "C" fn(*mut c_void, usize) -> *const u8;
        type ResetLane = unsafe extern "C" fn(*mut c_void, usize, u32) -> u32;
        let frame_lane: BufferLane = unsafe { std::mem::transmute(frame_address) };
        let palette_lane: BufferLane = unsafe { std::mem::transmute(palette_address) };
        let clear_error: NativeClearError = unsafe { std::mem::transmute(error_clear_address) };
        let copy_error: NativeCopyError = unsafe { std::mem::transmute(error_copy_address) };
        let reset_lane: Option<ResetLane> =
            reset_address.map(|address| unsafe { std::mem::transmute(address) });
        let mask_data = mask.as_slice()?;
        let seed_data = seeds.as_ref().map(|values| values.as_slice()).transpose()?;
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w;
        let stack_lane_size = self.frame_stack * image_frame_size;
        let output_lane_size = self.frame_stack * image_frame_size;
        let failed = AtomicBool::new(false);

        unsafe { clear_error(context as *mut c_void) };

        py.detach(|| {
            self.pool.install(|| {
                stack_data
                    .par_chunks_mut(stack_lane_size)
                    .zip(heads_data.par_iter_mut())
                    .zip(output_data.par_chunks_mut(output_lane_size))
                    .zip(mask_data.par_iter())
                    .enumerate()
                    .for_each(|(lane, (((stack_lane, head), output_lane), selected))| {
                        if *selected {
                            if let (Some(reset), Some(lane_seeds)) = (reset_lane, seed_data)
                                && unsafe {
                                    reset(
                                        context as *mut c_void,
                                        lane,
                                        *lane_seeds.get_unchecked(lane),
                                    )
                                } & 4
                                    != 0
                            {
                                failed.store(true, Ordering::Relaxed);
                                return;
                            }
                            let frame = unsafe {
                                std::slice::from_raw_parts(
                                    frame_lane(context as *mut c_void, lane),
                                    self.plan.raw_h * self.plan.raw_w,
                                )
                            };
                            let palette = unsafe {
                                std::slice::from_raw_parts(
                                    palette_lane(context as *mut c_void, lane),
                                    256 * 3,
                                )
                            };
                            let mut cache = self.indexed_area_caches[lane]
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                            self.plan.write_indexed_frame_cached(
                                frame,
                                palette,
                                &mut cache,
                                &mut stack_lane[..image_frame_size],
                                self.shared_tile_cache
                                    .then_some(self.indexed_shared_tiles.as_slice()),
                                None,
                            );
                            for slot in 1..self.frame_stack {
                                stack_lane.copy_within(..image_frame_size, slot * image_frame_size);
                            }
                            *head = 0;
                        }
                        self.write_observation(stack_lane, *head as usize, output_lane);
                    });
            });
        });
        if failed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err(format!(
                "native Doom lane reset failed: {}",
                native_error_detail(context, copy_error)
            )));
        }
        Ok(())
    }

    fn reset_indexed_lane_into(
        &self,
        py: Python<'_>,
        current: PyReadonlyArray2<'_, u8>,
        palette: PyReadonlyArray2<'_, u8>,
        mut stack: PyReadwriteArray4<'_, u8>,
        mut head: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray3<'_, u8>,
    ) -> PyResult<()> {
        if !self.plan.supports_indexed_area() {
            return Err(PyValueError::new_err(
                "indexed preprocessing requires 320x240 to 84x84 area-resize grayscale with crop removal or masking",
            ));
        }
        if current.shape() != [self.plan.raw_h, self.plan.raw_w] {
            return Err(PyValueError::new_err(
                "current has an invalid indexed shape",
            ));
        }
        if palette.shape() != [256, 3] {
            return Err(PyValueError::new_err("palette must have shape (256, 3)"));
        }
        let expected_stack = [
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if stack.shape() != expected_stack || head.shape() != [1] {
            return Err(PyValueError::new_err("stack or head has an invalid shape"));
        }
        if output.shape() != expected_output {
            return Err(PyValueError::new_err("output has an invalid shape"));
        }

        let current_data = current.as_slice()?;
        let palette_data = palette.as_slice()?;
        let stack_data = stack.as_slice_mut()?;
        let head_data = head.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w;
        py.detach(|| {
            self.plan.write_indexed_frame(
                current_data,
                palette_data,
                &mut stack_data[..image_frame_size],
            );
            for slot in 1..self.frame_stack {
                stack_data.copy_within(..image_frame_size, slot * image_frame_size);
            }
            head_data[0] = 0;
            self.write_observation(stack_data, 0, output_data);
        });
        Ok(())
    }

    fn repeat_last_lane_into(
        &self,
        py: Python<'_>,
        mut stack: PyReadwriteArray4<'_, u8>,
        mut head: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray3<'_, u8>,
    ) -> PyResult<()> {
        let expected_stack = [
            self.frame_stack,
            self.plan.out_h,
            self.plan.out_w,
            self.plan.out_c,
        ];
        let expected_output = match self.layout {
            ObservationLayout::Hwc => [
                self.plan.out_h,
                self.plan.out_w,
                self.plan.out_c * self.frame_stack,
            ],
            ObservationLayout::Chw => [
                self.plan.out_c * self.frame_stack,
                self.plan.out_h,
                self.plan.out_w,
            ],
        };
        if stack.shape() != expected_stack || head.shape() != [1] {
            return Err(PyValueError::new_err("stack or head has an invalid shape"));
        }
        if output.shape() != expected_output {
            return Err(PyValueError::new_err("output has an invalid shape"));
        }

        let stack_data = stack.as_slice_mut()?;
        let head_data = head.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        py.detach(|| {
            let old_head = head_data[0] as usize;
            let new_head = (old_head + 1) % self.frame_stack;
            let source = old_head * image_frame_size;
            let destination = new_head * image_frame_size;
            if source < destination {
                let (before, after) = stack_data.split_at_mut(destination);
                after[..image_frame_size]
                    .copy_from_slice(&before[source..source + image_frame_size]);
            } else if destination < source {
                let (before, after) = stack_data.split_at_mut(source);
                before[destination..destination + image_frame_size]
                    .copy_from_slice(&after[..image_frame_size]);
            }
            head_data[0] = new_head as i64;
            self.write_observation(stack_data, new_head, output_data);
        });
        Ok(())
    }

    fn reset_into(
        &self,
        py: Python<'_>,
        current: PyReadonlyArray4<'_, u8>,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
        reset_mask: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<()> {
        self.validate_arrays(
            current.shape(),
            stack.shape(),
            heads.shape(),
            output.shape(),
            None,
        )?;
        if reset_mask.shape() != [self.num_envs] {
            return Err(PyValueError::new_err(format!(
                "reset_mask must have shape ({},)",
                self.num_envs
            )));
        }
        let current_data = current.as_slice()?;
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let mask_data = reset_mask.as_slice()?;
        let raw_frame_size = self.plan.raw_h * self.plan.raw_w * 3;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        let stack_lane_size = image_frame_size * self.frame_stack;
        let output_lane_size = image_frame_size * self.frame_stack;
        py.detach(|| {
            self.pool.install(|| {
                heads_data
                    .par_iter_mut()
                    .zip(stack_data.par_chunks_mut(stack_lane_size))
                    .zip(output_data.par_chunks_mut(output_lane_size))
                    .zip(current_data.par_chunks(raw_frame_size))
                    .enumerate()
                    .for_each(
                        |(lane, (((head, stack_lane), output_lane), current_frame))| {
                            if mask_data[lane] {
                                self.plan.write_frame(
                                    current_frame,
                                    None,
                                    &mut stack_lane[..image_frame_size],
                                );
                                for slot in 1..self.frame_stack {
                                    stack_lane
                                        .copy_within(..image_frame_size, slot * image_frame_size);
                                }
                                *head = 0;
                            }
                            self.write_observation(stack_lane, *head as usize, output_lane);
                        },
                    );
            });
        });
        Ok(())
    }

    fn reset_frames_into(
        &self,
        py: Python<'_>,
        current: Vec<PyReadonlyArray3<'_, u8>>,
        mut stack: PyReadwriteArray5<'_, u8>,
        mut heads: PyReadwriteArray1<'_, i64>,
        mut output: PyReadwriteArray4<'_, u8>,
        reset_mask: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<()> {
        let expected_frame = [self.plan.raw_h, self.plan.raw_w, 3];
        let expected_batch = [self.num_envs, self.plan.raw_h, self.plan.raw_w, 3];
        self.validate_arrays(
            &expected_batch,
            stack.shape(),
            heads.shape(),
            output.shape(),
            None,
        )?;
        if current.len() != self.num_envs
            || current.iter().any(|frame| frame.shape() != expected_frame)
        {
            return Err(PyValueError::new_err(format!(
                "current must contain {} frames with shape {expected_frame:?}",
                self.num_envs
            )));
        }
        if reset_mask.shape() != [self.num_envs] {
            return Err(PyValueError::new_err(format!(
                "reset_mask must have shape ({},)",
                self.num_envs
            )));
        }
        let current_data = current
            .iter()
            .map(|frame| frame.as_slice().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let stack_data = stack.as_slice_mut()?;
        let heads_data = heads.as_slice_mut()?;
        let output_data = output.as_slice_mut()?;
        let mask_data = reset_mask.as_slice()?;
        let image_frame_size = self.plan.out_h * self.plan.out_w * self.plan.out_c;
        let stack_lane_size = image_frame_size * self.frame_stack;
        let output_lane_size = image_frame_size * self.frame_stack;
        py.detach(|| {
            self.pool.install(|| {
                heads_data
                    .par_iter_mut()
                    .zip(stack_data.par_chunks_mut(stack_lane_size))
                    .zip(output_data.par_chunks_mut(output_lane_size))
                    .enumerate()
                    .for_each(|(lane, ((head, stack_lane), output_lane))| {
                        if mask_data[lane] {
                            self.plan.write_frame(
                                current_data[lane],
                                None,
                                &mut stack_lane[..image_frame_size],
                            );
                            for slot in 1..self.frame_stack {
                                stack_lane.copy_within(..image_frame_size, slot * image_frame_size);
                            }
                            *head = 0;
                        }
                        self.write_observation(stack_lane, *head as usize, output_lane);
                    });
            });
        });
        Ok(())
    }
}

#[pyclass]
struct ActionHistory {
    action_width: usize,
    lanes: Vec<Vec<f64>>,
}

#[pymethods]
impl ActionHistory {
    #[new]
    fn new(num_envs: usize, action_width: usize) -> PyResult<Self> {
        if num_envs == 0 || action_width == 0 {
            return Err(PyValueError::new_err(
                "num_envs and action_width must be positive",
            ));
        }
        Ok(Self {
            action_width,
            lanes: vec![Vec::new(); num_envs],
        })
    }

    fn append(&mut self, actions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let shape = actions.shape();
        if shape != [self.lanes.len(), self.action_width] {
            return Err(PyValueError::new_err(format!(
                "actions must have shape ({}, {})",
                self.lanes.len(),
                self.action_width
            )));
        }
        let values = actions.as_slice()?;
        for (lane, action) in self.lanes.iter_mut().zip(values.chunks(self.action_width)) {
            lane.extend_from_slice(action);
        }
        Ok(())
    }

    fn clear(&mut self, mask: PyReadonlyArray1<'_, bool>) -> PyResult<()> {
        if mask.shape() != [self.lanes.len()] {
            return Err(PyValueError::new_err(format!(
                "mask must have shape ({},)",
                self.lanes.len()
            )));
        }
        for (lane, &selected) in self.lanes.iter_mut().zip(mask.as_slice()?) {
            if selected {
                lane.clear();
            }
        }
        Ok(())
    }

    fn replace_lane(&mut self, lane: usize, actions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        if lane >= self.lanes.len() {
            return Err(PyIndexError::new_err("lane is out of range"));
        }
        let shape = actions.shape();
        if shape.len() != 2 || shape[1] != self.action_width {
            return Err(PyValueError::new_err(format!(
                "actions must have shape (steps, {})",
                self.action_width
            )));
        }
        self.lanes[lane] = actions.as_slice()?.to_vec();
        Ok(())
    }

    fn lane(&self, lane: usize) -> PyResult<Vec<Vec<f64>>> {
        if lane >= self.lanes.len() {
            return Err(PyIndexError::new_err("lane is out of range"));
        }
        Ok(self.lanes[lane]
            .chunks(self.action_width)
            .map(<[f64]>::to_vec)
            .collect())
    }
}

#[pyfunction]
#[pyo3(signature = (current, output, crop, mask_crop, crop_fill, algorithm, previous=None))]
#[allow(clippy::too_many_arguments)]
fn preprocess_into(
    py: Python<'_>,
    current: PyReadonlyArray4<'_, u8>,
    mut output: PyReadwriteArray4<'_, u8>,
    crop: Vec<usize>,
    mask_crop: bool,
    crop_fill: u8,
    algorithm: &str,
    previous: Option<PyReadonlyArray4<'_, u8>>,
) -> PyResult<()> {
    let current_shape = current.shape();
    let output_shape = output.shape();
    if current_shape.len() != 4
        || current_shape[3] != 3
        || output_shape.len() != 4
        || output_shape[0] != current_shape[0]
        || !matches!(output_shape[3], 1 | 3)
    {
        return Err(PyValueError::new_err(
            "current must be NHWC RGB and output must be NHWC with one or three channels",
        ));
    }
    if crop.len() != 4 {
        return Err(PyValueError::new_err(
            "crop must contain top, bottom, left, right",
        ));
    }
    let raw_h = current_shape[1];
    let raw_w = current_shape[2];
    if crop[0] + crop[1] >= raw_h || crop[2] + crop[3] >= raw_w {
        return Err(PyValueError::new_err(
            "crop must preserve at least one source pixel",
        ));
    }
    if let Some(prior) = previous.as_ref()
        && prior.shape() != current_shape
    {
        return Err(PyValueError::new_err(
            "previous must have the same shape as current",
        ));
    }
    let plan = ImagePlan::new(
        raw_h,
        raw_w,
        output_shape[1],
        output_shape[2],
        output_shape[3],
        [crop[0], crop[1], crop[2], crop[3]],
        mask_crop,
        crop_fill,
        ResizeAlgorithm::parse(algorithm)?,
    );
    let current_data = current.as_slice()?;
    let previous_data = previous
        .as_ref()
        .map(|value| value.as_slice())
        .transpose()?;
    let output_data = output.as_slice_mut()?;
    let raw_frame_size = raw_h * raw_w * 3;
    let output_frame_size = plan.out_h * plan.out_w * plan.out_c;
    py.detach(|| {
        output_data
            .par_chunks_mut(output_frame_size)
            .zip(current_data.par_chunks(raw_frame_size))
            .enumerate()
            .for_each(|(lane, (output_frame, current_frame))| {
                let prior = previous_data
                    .map(|data| &data[lane * raw_frame_size..(lane + 1) * raw_frame_size]);
                plan.write_frame(current_frame, prior, output_frame);
            });
    });
    Ok(())
}

#[pymodule]
fn _env_vizdoom_turbo(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ActionHistory>()?;
    module.add_class::<ImageProcessor>()?;
    module.add_function(wrap_pyfunction!(preprocess_into, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn copy_test_error(
        _context: *mut c_void,
        destination: *mut u8,
        capacity: usize,
    ) -> usize {
        let message = b"phase=finish lane=7: Doom process exited unexpectedly";
        let copied = message.len().min(capacity.saturating_sub(1));
        unsafe {
            std::ptr::copy_nonoverlapping(message.as_ptr(), destination, copied);
            *destination.add(copied) = 0;
        }
        copied
    }

    #[test]
    fn native_error_detail_copies_callback_message() {
        assert_eq!(
            native_error_detail(0, copy_test_error),
            "phase=finish lane=7: Doom process exited unexpectedly"
        );
    }
}
