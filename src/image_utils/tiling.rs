use std::fmt;

/// A set of custom errors for more informative error handling.
#[derive(Debug)]
pub enum TilingError {
    InvalidTileSize {
        tile_size: u32,
        image_width: u32,
        image_height: u32,
    },
    IncompatibleProportionWithTileSize {
        tile_size: u32,
        overlap_proportion: OverlapProportion
    },
    UnevenImageDivision {
        image_height: u32,
        image_width: u32,
        tile_size: u32,
        overlap_proportion: OverlapProportion
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OverlapProportion {
    ONE_HALF = 2,
    ONE_THIRD = 3,
    ONE_FOURTH = 4,
    ONE_FIFTH = 5
}

pub fn validate_tiling_parameters(
    proportion: OverlapProportion,
    tile_size: u32,
    image_width: u32,
    image_height: u32
) -> Option<TilingError> {
    if tile_size > image_width || tile_size > image_height {
        return Some(TilingError::InvalidTileSize {tile_size, image_width, image_height});
    }

    let tile_cleanly_divides = tile_size % (proportion as u32) == 0;
    if !tile_cleanly_divides {
        return Some(TilingError::IncompatibleProportionWithTileSize {
            tile_size,
            overlap_proportion: proportion
        });
    }
    
    let tiles_fit_cleanly_laterally = image_width % (tile_size / (proportion as u32)) == 0;
    let tiles_fit_cleanly_vertically = image_height % (tile_size / (proportion as u32)) == 0;
    
    if !tiles_fit_cleanly_laterally || !tiles_fit_cleanly_vertically {
        return Some(TilingError::UnevenImageDivision {
            image_height,
            image_width,
            tile_size,
            overlap_proportion: proportion
        })
    }
    None
}
