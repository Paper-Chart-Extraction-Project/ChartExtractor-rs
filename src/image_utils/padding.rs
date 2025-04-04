use image::{Rgb, RgbImage};
use std::fmt;

/// A set of custom errors for more informative error handling.
#[derive(Debug)]
pub enum ImagePaddingError {
    InvalidWidth {original_width: u32, new_width: u32},
    InvalidHeight {original_height: u32, new_height: u32},
    InvalidDimensions {original_width: u32, new_width: u32, original_height: u32, new_height: u32}
}

impl fmt::Display for ImagePaddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImagePaddingError::InvalidWidth{original_width, new_width} => {
                write!(
                    f,
                    "Failed to pad image, new width ({}) < original width ({}).",
                    new_width, original_width
                )
            },
            ImagePaddingError::InvalidHeight{original_height, new_height} => {
                write!(
                    f,
                    "Failed to pad image, new height ({}) < original height ({}).",
                    new_height, original_height
                )
            }
            ImagePaddingError::InvalidDimensions{
                original_width, new_width, original_height, new_height
            } => {
                write!(
                    f,
                    "Failed to pad image, new width ({}) < original width ({}) \
                    and new height({}) < original height({})",
                    new_width, original_width, new_height, original_height
                )
            }
        }
    }
}

fn validate_padding_parameters(
    original_width: u32, original_height: u32, new_width: u32, new_height: u32
) -> Option<ImagePaddingError> {
    let new_width_is_too_small = new_width < original_width;
    let new_height_is_too_small = new_height < original_height;
    if new_width_is_too_small && new_height_is_too_small {
        return Some(ImagePaddingError::InvalidDimensions {
            original_width, new_width, original_height, new_height
        });
    } else if new_width_is_too_small {
        return Some(ImagePaddingError::InvalidWidth {original_width, new_width});
    } else if new_height_is_too_small {
        return Some(ImagePaddingError::InvalidHeight {original_height, new_height});
    }
    None
}

/// Pads an rgb8 image by adding pixels to the right and bottom of the image.
pub fn pad_right_bottom_img_rbg8(
    original_image: RgbImage, new_width: u32, new_height: u32
) -> Result<RgbImage, ImagePaddingError> {
    let (original_width, original_height) = original_image.dimensions();
    let params_are_valid = validate_padding_parameters(
        original_width, original_height, new_width, new_height
    );
    if let Some(e) = params_are_valid {
        return Err(e);
    }

    let mut padded_image: RgbImage = RgbImage::new(new_width, new_height);
    for pixel in original_image.enumerate_pixels() {
        let x = pixel.0;
        let y = pixel.1;
        let [r, g, b] = original_image.get_pixel_checked(x, y).unwrap().0;
        padded_image.put_pixel(x, y, Rgb([r, g, b]));
    }
    Ok(padded_image)
}
