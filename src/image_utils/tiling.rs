use ndarray::{ArrayBase, Dim, OwnedRepr, ViewRepr, s};
use std::fmt;

/// A set of custom errors for more informative error handling.
#[derive(Debug, PartialEq)]
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
                    "Failed to tile image, overlap proportion ({}) does not evenly divide \
                    tile size ({}).",
                    overlap_proportion,
                    tile_size
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
                    "Failed to tile image, the tile size ({}) does not evenly divide the image's\
                    width ({}) and height ({}) given the overlap proportion ({}).",
                    tile_size,
                    image_width,
                    image_height,
                    overlap_proportion
                )
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Fraction {
    numerator: u32,
    denominator: u32
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverlapProportion {
    OneHalf = Fraction { 1_u32, 2_u32 },
    OneThird = Fraction { 1_u32, 3_u32 },
    OneFourth = Fraction { 1_u32, 4_u32 },
    OneFifth = Fraction { 1_u32, 5_u32 },
}

impl fmt::Display for OverlapProportion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}/{}", *self.numerator as u32, *self.denominator as u32)
    }
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
        let end_row = start_row + (tile_size as usize);
        for col_ix in 0..num_columns - 1 {
            let start_col = (col_ix * stride) as usize;
            let end_col = start_col + (tile_size as usize);
            let tile = image.slice(s![.., .., start_row..end_row, start_col..end_col]);
            row_of_tiles.push(tile);
        }
        tiles.push(row_of_tiles);
    }
    Ok(tiles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_utils::image_conversion::convert_array_view_to_rgb_image;
    use crate::image_utils::image_io::{read_image_as_array4, read_image_as_rgb8};
    use std::path::Path;

    #[test]
    fn tile_with_invalid_tile_size_for_width() {
        let validation =
            validate_tiling_parameters(OverlapProportion::OneHalf, 10_u32, 8_u32, 12_u32);
        assert_eq!(
            validation,
            Some(TilingError::InvalidTileSize {
                tile_size: 10_u32,
                image_width: 8_u32,
                image_height: 12_u32
            })
        );
    }

    #[test]
    fn tile_with_invalid_tile_size_for_height() {
        let validation =
            validate_tiling_parameters(OverlapProportion::OneHalf, 10_u32, 12_u32, 8_u32);
        assert_eq!(
            validation,
            Some(TilingError::InvalidTileSize {
                tile_size: 10_u32,
                image_width: 12_u32,
                image_height: 8_u32
            })
        );
    }

    #[test]
    fn tile_with_invalid_tile_size_for_both_dimensions() {
        let validation =
            validate_tiling_parameters(OverlapProportion::OneHalf, 10_u32, 8_u32, 8_u32);
        assert_eq!(
            validation,
            Some(TilingError::InvalidTileSize {
                tile_size: 10_u32,
                image_width: 8_u32,
                image_height: 8_u32
            })
        );
    }

    #[test]
    fn tile_with_tile_size_proportion_mismatch() {
        let validation =
            validate_tiling_parameters(OverlapProportion::OneHalf, 17_u32, 68_u32, 68_u32);
        assert_eq!(
            validation,
            Some(TilingError::IncompatibleProportionWithTileSize {
                tile_size: 17_u32,
                overlap_proportion: OverlapProportion::OneHalf
            })
        );
    }

    #[test]
    fn tile_with_uneven_tile_division_left_right() {
        let validation =
            validate_tiling_parameters(OverlapProportion::OneHalf, 8_u32, 18_u32, 20_u32);
        assert_eq!(
            validation,
            Some(TilingError::UnevenImageDivision {
                image_width: 18_u32,
                image_height: 20_u32,
                tile_size: 8_u32,
                overlap_proportion: OverlapProportion::OneHalf
            })
        );
    }

    #[test]
    fn tile_with_uneven_tile_division_top_down() {
        let validation =
            validate_tiling_parameters(OverlapProportion::OneHalf, 8_u32, 20_u32, 18_u32);
        assert_eq!(
            validation,
            Some(TilingError::UnevenImageDivision {
                image_width: 20_u32,
                image_height: 18_u32,
                tile_size: 8_u32,
                overlap_proportion: OverlapProportion::OneHalf
            })
        );
    }

    #[test]
    fn test_tiling() {
        let img = read_image_as_array4(Path::new("./data/test_data/test_image.png"));
        let tiles = tile_image(&img, 2, OverlapProportion::OneHalf).unwrap();
        for (row_ix, row) in tiles.iter().enumerate() {
            for (col_ix, tile) in row.iter().enumerate() {
                let rgb_tile = convert_array_view_to_rgb_image(*tile);
                let filepath_to_true_tile = format!(
                    "./data/test_data/test_image_tile_{row}_{col}.png",
                    row = row_ix,
                    col = col_ix
                );
                let true_rgb_tile = read_image_as_rgb8(Path::new(&filepath_to_true_tile));
                assert_eq!(rgb_tile, true_rgb_tile);
            }
        }
    }
}
