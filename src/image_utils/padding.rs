use image::{Rgb, RgbImage};
use std::fmt;

/// A set of custom errors for more informative error handling.
#[derive(Debug, PartialEq)]
pub enum ImagePaddingError {
    InvalidWidth {
        original_width: u32,
        new_width: u32,
    },
    InvalidHeight {
        original_height: u32,
        new_height: u32,
    },
    InvalidDimensions {
        original_width: u32,
        new_width: u32,
        original_height: u32,
        new_height: u32,
    },
}

impl fmt::Display for ImagePaddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImagePaddingError::InvalidWidth {
                original_width,
                new_width,
            } => {
                write!(
                    f,
                    "Failed to pad image, new width ({}) < original width ({}).",
                    new_width, original_width
                )
            }
            ImagePaddingError::InvalidHeight {
                original_height,
                new_height,
            } => {
                write!(
                    f,
                    "Failed to pad image, new height ({}) < original height ({}).",
                    new_height, original_height
                )
            }
            ImagePaddingError::InvalidDimensions {
                original_width,
                new_width,
                original_height,
                new_height,
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
    original_width: u32,
    original_height: u32,
    new_width: u32,
    new_height: u32,
) -> Option<ImagePaddingError> {
    let new_width_is_too_small = new_width < original_width;
    let new_height_is_too_small = new_height < original_height;
    if new_width_is_too_small && new_height_is_too_small {
        return Some(ImagePaddingError::InvalidDimensions {
            original_width,
            new_width,
            original_height,
            new_height,
        });
    } else if new_width_is_too_small {
        return Some(ImagePaddingError::InvalidWidth {
            original_width,
            new_width,
        });
    } else if new_height_is_too_small {
        return Some(ImagePaddingError::InvalidHeight {
            original_height,
            new_height,
        });
    }
    None
}

/// Pads an rgb8 image by adding pixels to the right and bottom of the image.
pub fn pad_right_bottom_img_rbg8(
    original_image: RgbImage,
    new_width: u32,
    new_height: u32,
) -> Result<RgbImage, ImagePaddingError> {
    let (original_width, original_height) = original_image.dimensions();
    let params_are_valid =
        validate_padding_parameters(original_width, original_height, new_width, new_height);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_utils::image_io::read_image_as_rgb8;
    use image::Rgb;
    use std::path::Path;
    
    fn read_test_image() -> RgbImage {
        read_image_as_rgb8(Path::new("./data/test_data/test_image.png"))
    }

    #[test]
    fn validate_padding_parameters_invalid_width() {
        let invalid_width_result = validate_padding_parameters(10_u32, 10_u32, 5_u32, 15_u32);
        assert_eq!(
            invalid_width_result,
            Some(ImagePaddingError::InvalidWidth { original_width: 10_u32, new_width: 5_u32 })
        );
    }

    #[test]
    fn validate_padding_parameters_invalid_height() {
        let invalid_height_result = validate_padding_parameters(10_u32, 10_u32, 15_u32, 5_u32);
        assert_eq!(
            invalid_height_result,
            Some(ImagePaddingError::InvalidHeight { original_height: 10_u32, new_height: 5_u32 })
        );
    }

    #[test]
    fn validate_padding_parameters_invalid_dimensions() {
        let invalid_dimensions_result = validate_padding_parameters(10_u32, 10_u32, 5_u32, 5_u32);
        assert_eq!(
            invalid_dimensions_result,
            Some(ImagePaddingError::InvalidDimensions {
                original_width: 10_u32,
                original_height: 10_u32,
                new_width: 5_u32,
                new_height: 5_u32
            })
        );
    }

    #[test]
    fn validate_padding_parameters_valid() {
        let valid_result = validate_padding_parameters(10_u32, 10_u32, 15_u32, 15_u32);
        assert_eq!(valid_result, None);
    }

    #[test]
    fn pad_right_bottom() {
        let unpadded_img = read_test_image();
        let img = pad_right_bottom_img_rbg8(unpadded_img, 4, 4).unwrap();
        assert_eq!(img.get_pixel(0, 0), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(1, 0), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(2, 0), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(3, 0), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(0, 1), &Rgb([255, 0, 0]));
        assert_eq!(img.get_pixel(1, 1), &Rgb([0, 255, 0]));
        assert_eq!(img.get_pixel(2, 1), &Rgb([0, 0, 255]));
        assert_eq!(img.get_pixel(3, 1), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(0, 2), &Rgb([255, 255, 255]));
        assert_eq!(img.get_pixel(1, 2), &Rgb([255, 255, 255]));
        assert_eq!(img.get_pixel(2, 2), &Rgb([255, 255, 255]));
        assert_eq!(img.get_pixel(3, 2), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(0, 3), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(1, 3), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(2, 3), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(3, 3), &Rgb([0, 0, 0]));
    }
}
 
