use crate::image_utils::image_conversion::convert_rgb_image_to_owned_array;
use image::{self, RgbImage};
use ndarray::{ArrayBase, Dim, OwnedRepr};
use std::path::Path;

pub fn read_image_as_rgb8(filepath: &Path) -> RgbImage {
    image::open(filepath).unwrap().into_rgb8()
}

pub fn read_image_as_array4(filepath: &Path) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> {
    let img = read_image_as_rgb8(filepath);
    return convert_rgb_image_to_owned_array(img);
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn read_test_data_as_rgb8() {
        let img = read_image_as_rgb8(Path::new("./data/test_data/test_image.png"));
        assert_eq!(img.get_pixel(0, 0), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(1, 0), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(2, 0), &Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(0, 1), &Rgb([255, 0, 0]));
        assert_eq!(img.get_pixel(1, 1), &Rgb([0, 255, 0]));
        assert_eq!(img.get_pixel(2, 1), &Rgb([0, 0, 255]));
        assert_eq!(img.get_pixel(0, 2), &Rgb([255, 255, 255]));
        assert_eq!(img.get_pixel(1, 2), &Rgb([255, 255, 255]));
        assert_eq!(img.get_pixel(2, 2), &Rgb([255, 255, 255]));
    }

    #[test]
    fn read_test_data_as_array4() {
        let img = read_image_as_array4(Path::new("./data/test_data/test_image.png"));
        // Array4s for images are arrays of images. Here we load 1 image.
        // The dimensions for these arrays encode (image, channel, row, column).
        // Each line below tests one pixel by getting all its channels into a tuple of three
        // elements.
        assert_eq!(
            (img[[0, 0, 0, 0]], img[[0, 1, 0, 0]], img[[0, 2, 0, 0]]),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(
            (img[[0, 0, 0, 1]], img[[0, 1, 0, 1]], img[[0, 2, 0, 1]]),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(
            (img[[0, 0, 0, 2]], img[[0, 1, 0, 2]], img[[0, 2, 0, 2]]),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(
            (img[[0, 0, 1, 0]], img[[0, 1, 1, 0]], img[[0, 2, 1, 0]]),
            (1.0, 0.0, 0.0)
        );
        assert_eq!(
            (img[[0, 0, 1, 1]], img[[0, 1, 1, 1]], img[[0, 2, 1, 1]]),
            (0.0, 1.0, 0.0)
        );
        assert_eq!(
            (img[[0, 0, 1, 2]], img[[0, 1, 1, 2]], img[[0, 2, 1, 2]]),
            (0.0, 0.0, 1.0)
        );
        assert_eq!(
            (img[[0, 0, 2, 0]], img[[0, 1, 2, 0]], img[[0, 2, 2, 0]]),
            (1.0, 1.0, 1.0)
        );
        assert_eq!(
            (img[[0, 0, 2, 1]], img[[0, 1, 2, 1]], img[[0, 2, 2, 1]]),
            (1.0, 1.0, 1.0)
        );
        assert_eq!(
            (img[[0, 0, 2, 2]], img[[0, 1, 2, 2]], img[[0, 2, 2, 2]]),
            (1.0, 1.0, 1.0)
        );
    }
}
