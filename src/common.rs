pub mod context;
pub mod math;
pub mod point_set3d;

use std::char::MAX;
use crate::decoder::{Image, PatchOrientation, Video};
use crate::decoder::PatchOrientation::{Default, MRot180, MRot270, MRot90, Mirror, Rot180, Rot270, Rot90, Swap};

#[derive(Debug, Clone, Copy)]
pub(crate) enum CodecGroup {
    // AvcProgressiveHigh = 0,
    // HevcMain10 = 1,
    // Hevc444 = 2,
    // VvcMain10 = 3,
    // Mp4Ra = 127,
}

#[derive(Default, Debug, Clone, Copy)]
pub(crate) enum ColorFormat {
    Unknown,
    _Rgb444,
    // Yuv444,
    #[default]
    Yuv420,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum CodecId {
    JMAPP = 0,
    HMAPP = 1,
    SHMAPP = 2,
    JMLIB = 3,
    HMLIB = 4,
    VTMLIB = 5,
    FFMPEG = 6,
    UNKNOWN_CODEC = 255
}

pub(crate) type VideoOccupancyMap = Video<u8>;
pub(crate) type VideoGeometry = Video<u16>;
pub(crate) type VideoAttribute = Video<u16>;
pub(crate) type ImageOccupancyMap = Image<u8>;

// Global variables
pub(crate) const INFINITE_DEPTH: i16 = i16::MAX;
pub(crate) const INFINITE_NUMBER: i64 = i64::MAX;
pub(crate) const INVALID_PATCH_INDEX: i32 = -1;
pub(crate) const UNDEFINED_INDEX: u32 = u32::MAX;
pub(crate) const PRINT_DETAILED_INFO: bool = false;
pub(crate) const INTERMEDIATE_LAYER_INDEX: usize = 100;
pub(crate) const NEIGHBOR_THRESHOLD: usize = 4;
pub(crate) const ORIENTATION_VERTICAL: [PatchOrientation; 8] = [
    Default,
    Swap,
    Rot180,
    Mirror,
    MRot180,
    Rot270,
    MRot90,
    Rot90
];

// ZICO: dunno why this is in different order
pub(crate) const ORIENTATION_HORIZONTAL: [PatchOrientation; 8] = [
    Swap,
    Default,
    Rot270,
    MRot90,
    Rot90,
    Rot180,
    Mirror,
    MRot180
];

pub(crate) type Range = (i32, i32);
pub(crate) struct Tile {
    minU: i32,
    maxU: i32,
    minV: i32,
    maxV: i32,
}

impl Tile {
    pub fn new() {
        Self {minU: -1, maxU: -1, minV: -1, maxV: -1};
    }
}