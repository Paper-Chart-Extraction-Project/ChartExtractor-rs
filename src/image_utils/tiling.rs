use ndarray::{Array, ArrayBase, Axis, Dim, OwnedRepr, ViewRepr, s};
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
        overlap_proportion: OverlapProportion,
    },
    UnevenImageDivision {
        image_height: u32,
        image_width: u32,
        tile_size: u32,
        overlap_proportion: OverlapProportion,
    },
}

impl fmt::Display for TilingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TilingError::InvalidTileSize {
                tile_size,
                image_width,
                image_height,
            } => {
                if tile_size > image_width {
                    return write!(
                        f,
                        "Failed to tile image, tile size ({}) > image width ({}).",
                        tile_size, image_width
                    );
                } else if tile_size > image_height {
                    return write!(
                        f,
                        "Failed to tile image, tile size ({}) > image height ({}).",
                        tile_size, image_height
                    );
                } else {
                    panic!();
                }
            }
            TilingError::IncompatibleProportionWithTileSize {
                tile_size,
                overlap_proportion,
            } => {
                write!(
                    f,
                    "Failed to tile image, overlap proportion does not evenly divide tile size."
                )
            }
            TilingError::UnevenImageDivision {
                image_width,
                image_height,
                tile_size,
                overlap_proportion,
            } => {
                write!(
                    f,
                    "Failed to tile image, the tiles do not evenly divide the image given the \
                    overlap proportion.",
                )
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OverlapProportion {
    OneHalf = 2,
    OneThird = 3,
    OneFourth = 4,
    OneFifth = 5,
}

pub fn validate_tiling_parameters(
    proportion: OverlapProportion,
    tile_size: u32,
    image_width: u32,
    image_height: u32,
) -> Option<TilingError> {
    if tile_size > image_width || tile_size > image_height {
        return Some(TilingError::InvalidTileSize {
            tile_size,
            image_width,
            image_height,
        });
    }

    let tile_cleanly_divides = tile_size % (proportion as u32) == 0;
    if !tile_cleanly_divides {
        return Some(TilingError::IncompatibleProportionWithTileSize {
            tile_size,
            overlap_proportion: proportion,
        });
    }

    let tiles_fit_cleanly_laterally = image_width % (tile_size / (proportion as u32)) == 0;
    let tiles_fit_cleanly_vertically = image_height % (tile_size / (proportion as u32)) == 0;

    if !tiles_fit_cleanly_laterally || !tiles_fit_cleanly_vertically {
        return Some(TilingError::UnevenImageDivision {
            image_height,
            image_width,
            tile_size,
            overlap_proportion: proportion,
        });
    }
    None
}

/// Tiles an image by returning a vector of immutable views into the image.
pub fn tile_image(
    image: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    tile_size: u32,
    proportion: OverlapProportion,
) -> Result<Vec<Vec<ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>>>, TilingError> {
    let image_width = image.shape()[2] as u32;
    let image_height = image.shape()[3] as u32;
    if let Some(e) = validate_tiling_parameters(proportion, tile_size, image_width, image_height) {
        return Err(e);
    }
    let mut tiles: Vec<Vec<ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>>> = Vec::new();
    let stride = tile_size / (proportion as u32);
    let num_rows = image_height / stride;
    let num_columns = image_width / stride;

    for row_ix in 0..num_rows - 1 {
        let mut row_of_tiles: Vec<ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>> = vec![];
        let start_row = (row_ix * stride) as usize;
        let end_row = ((row_ix + 1) * stride) as usize;
        for col_ix in 0..num_columns - 1 {
            let start_col = (col_ix * stride) as usize;
            let end_col = ((col_ix + 1) * stride) as usize;
            let tile = image.slice(s![.., .., start_row..end_row, start_col..end_col]);
            row_of_tiles.push(tile);
        }
        tiles.push(row_of_tiles);
    }
    Ok(tiles)
}
