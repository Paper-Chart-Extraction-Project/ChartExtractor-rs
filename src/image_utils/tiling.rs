use std::fmt;

/// A set of custom errors for more informative error handling.
#[derive(Debug)]
pub enum TilingError {
    InvalidTileSize {
        tile_width: u32,
        tile_height: u32,
        image_width: u32
        image_height: u32,
    },
    InvalidOverlapRatio {
        image_width: u32,
        image_height: u32,
        overlap_ratio: OverlapRatio
    }
}

pub enum OverlapRatio {
    ONE_HALF,
    ONE_THIRD,
    ONE_FOURTH,
    ONE_FIFTH
}
