use crate::image_utils::image_conversion::convert_array_view_to_rgb_image;
use crate::image_utils::padding::pad_right_bottom_img_rbg8;
use image::RgbImage;
use ndarray::{ArrayBase, Dim, OwnedRepr, ViewRepr, s};
use serde::{Deserialize, Serialize};
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
                    overlap_proportion, tile_size
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
                    tile_size, image_width, image_height, overlap_proportion
                )
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct OverlapProportion {
    pub numerator: u32,
    pub denominator: u32,
}

impl fmt::Display for OverlapProportion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

impl From<OverlapProportion> for f32 {
    fn from(overlap_proportion: OverlapProportion) -> f32 {
        overlap_proportion.numerator as f32 / overlap_proportion.denominator as f32
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

    let tile_cleanly_divides = (tile_size * proportion.numerator) % proportion.denominator == 0;
    if !tile_cleanly_divides {
        return Some(TilingError::IncompatibleProportionWithTileSize {
            tile_size,
            overlap_proportion: proportion,
        });
    }

    let stride: u32 = (tile_size * proportion.numerator) / proportion.denominator;
    let tiles_fit_cleanly_laterally = (image_width - tile_size) % stride == 0;
    let tiles_fit_cleanly_vertically = (image_height - tile_size) % stride == 0;

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
    let stride: u32 = (tile_size * proportion.numerator) / proportion.denominator;
    let num_rows = ((image_height - tile_size) / stride) + 1;
    let num_columns = ((image_width - tile_size) / stride) + 1;

    let mut tiles: Vec<Vec<ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>>> = Vec::new();
    for row_ix in 0..num_rows {
        let mut row_of_tiles: Vec<ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>> = vec![];
        let start_row = (row_ix * stride) as usize;
        let end_row = start_row + (tile_size as usize);
        for col_ix in 0..num_columns {
            let start_col = (col_ix * stride) as usize;
            let end_col = start_col + (tile_size as usize);
            let tile = image.slice(s![.., .., start_row..end_row, start_col..end_col]);
            row_of_tiles.push(tile);
        }
        tiles.push(row_of_tiles);
    }
    Ok(tiles)
}

/// Pads an image to the smallest size that is larger than the image's original
/// size if it cannot be tiled with the tiling parameters supplied.
pub fn pad_image_to_fit_tiling_params(
    image: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    tile_size: u32,
    proportion: OverlapProportion,
) -> RgbImage {
    let image = image.view();
    let image_width: u32 = image.shape()[2] as u32;
    let image_height: u32 = image.shape()[3] as u32;
    let params_are_valid: bool =
        validate_tiling_parameters(proportion, tile_size, image_width, image_height).is_none();
    if params_are_valid {
        return convert_array_view_to_rgb_image(image);
    } else {
        let (new_width, new_height) = find_smallest_img_size_large_enough_to_tile(
            image_width,
            image_height,
            tile_size,
            proportion,
        );
        let rgb_image: RgbImage = convert_array_view_to_rgb_image(image);
        return pad_right_bottom_img_rbg8(rgb_image, new_width, new_height).unwrap();
    }
}

/// Helper fn for pad_image_to_fit_tiling_params that determines the smallest
/// image size that is larger than the current image which can be tiled with
/// the given tiling parameters
fn find_smallest_img_size_large_enough_to_tile(
    original_image_width: u32,
    original_image_height: u32,
    tile_size: u32,
    overlap_proportion: OverlapProportion,
) -> (u32, u32) {
    let proportion_f32: f32 = overlap_proportion.into();
    let image_width: f32 = original_image_width as f32;
    let image_height: f32 = original_image_height as f32;
    let new_width: u32 = {
        let multiplier: f32 = (image_width / ((tile_size as f32) * proportion_f32)).ceil();
        (multiplier * tile_size as f32 * proportion_f32) as u32
    };
    let new_height: u32 = {
        let multiplier: f32 = (image_height / ((tile_size as f32) * proportion_f32)).ceil();
        (multiplier * tile_size as f32 * proportion_f32) as u32
    };
    (new_width, new_height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_utils::image_conversion::convert_array_view_to_rgb_image;
    use crate::image_utils::image_io::{read_image_as_array4, read_image_as_rgb8};
    use std::path::Path;

    const ONE_HALF: OverlapProportion = OverlapProportion {
        numerator: 1_u32,
        denominator: 2_u32,
    };
    const TWO_FIFTHS: OverlapProportion = OverlapProportion {
        numerator: 2_u32,
        denominator: 5_u32,
    };

    #[test]
    fn tile_with_invalid_tile_size_for_width() {
        let validation = validate_tiling_parameters(ONE_HALF, 10_u32, 8_u32, 12_u32);
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
        let validation = validate_tiling_parameters(ONE_HALF, 10_u32, 12_u32, 8_u32);
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
        let validation = validate_tiling_parameters(ONE_HALF, 10_u32, 8_u32, 8_u32);
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
        let validation = validate_tiling_parameters(ONE_HALF, 17_u32, 68_u32, 68_u32);
        assert_eq!(
            validation,
            Some(TilingError::IncompatibleProportionWithTileSize {
                tile_size: 17_u32,
                overlap_proportion: ONE_HALF
            })
        );
    }

    #[test]
    fn tile_with_uneven_tile_division_left_right() {
        let validation = validate_tiling_parameters(ONE_HALF, 8_u32, 18_u32, 20_u32);
        assert_eq!(
            validation,
            Some(TilingError::UnevenImageDivision {
                image_width: 18_u32,
                image_height: 20_u32,
                tile_size: 8_u32,
                overlap_proportion: ONE_HALF
            })
        );
    }

    #[test]
    fn tile_with_uneven_tile_division_top_down() {
        let validation = validate_tiling_parameters(ONE_HALF, 8_u32, 20_u32, 18_u32);
        assert_eq!(
            validation,
            Some(TilingError::UnevenImageDivision {
                image_width: 20_u32,
                image_height: 18_u32,
                tile_size: 8_u32,
                overlap_proportion: ONE_HALF
            })
        );
    }

    #[test]
    fn test_tiling() {
        let img = read_image_as_array4(Path::new("./data/test_data/test_image.png"));
        let tiles = tile_image(&img, 2, ONE_HALF).unwrap();
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

    #[test]
    fn test_find_smallest_img_size_large_enough_to_tile_tile_size_larger_than_width() {
        let image_width: u32 = 1250;
        let image_height: u32 = 2000;
        let tile_size: u32 = 1500;
        let proportion: OverlapProportion = OverlapProportion {
            numerator: 1,
            denominator: 2,
        };
        let padding_params = find_smallest_img_size_large_enough_to_tile(
            image_width,
            image_height,
            tile_size,
            proportion,
        );
        assert_eq!(padding_params, (1500, 2250));
    }

    #[test]
    fn test_find_smallest_img_size_large_enough_to_tile_tile_size_larger_than_height() {
        let image_width: u32 = 2000;
        let image_height: u32 = 1250;
        let tile_size: u32 = 1500;
        let proportion: OverlapProportion = OverlapProportion {
            numerator: 1,
            denominator: 2,
        };
        let padding_params = find_smallest_img_size_large_enough_to_tile(
            image_width,
            image_height,
            tile_size,
            proportion,
        );
        assert_eq!(padding_params, (2250, 1500));
    }

    #[test]
    fn test_find_smallest_img_size_large_enough_to_tile_tile_size_larger_than_image() {
        let image_width: u32 = 1250;
        let image_height: u32 = 1250;
        let tile_size: u32 = 1500;
        let proportion: OverlapProportion = OverlapProportion {
            numerator: 1,
            denominator: 1,
        };
        let padding_params = find_smallest_img_size_large_enough_to_tile(
            image_width,
            image_height,
            tile_size,
            proportion,
        );
        assert_eq!(padding_params, (1500, 1500));
    }

    #[test]
    fn test_find_smallest_img_size_large_enough_to_tile_no_padding_needed() {
        let image_width: u32 = 1250;
        let image_height: u32 = 1250;
        let tile_size: u32 = 500;
        let proportion: OverlapProportion = OverlapProportion {
            numerator: 1,
            denominator: 2,
        };
        let padding_params = find_smallest_img_size_large_enough_to_tile(
            image_width,
            image_height,
            tile_size,
            proportion,
        );
        assert_eq!(padding_params, (image_width, image_height));
    }

    #[test]
    fn test_find_smallest_img_size_large_enough_to_tile_standard_usage() {
        let image_width: u32 = 1200;
        let image_height: u32 = 1200;
        let tile_size: u32 = 500;
        let proportion: OverlapProportion = OverlapProportion {
            numerator: 1,
            denominator: 2,
        };
        let padding_params = find_smallest_img_size_large_enough_to_tile(
            image_width,
            image_height,
            tile_size,
            proportion,
        );
        assert_eq!(padding_params, (1250, 1250));
    }
}
