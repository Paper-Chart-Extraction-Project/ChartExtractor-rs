use image::{self, RgbImage};
use crate::image_utils::image_conversion::convert_rgb_image_to_owned_array;
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use std::path::Path;

pub fn read_image_as_rgb8(filepath: &Path) -> RgbImage {
    image::open(filepath).unwrap().into_rgb8()
}

pub fn read_image_as_array4(filepath: &Path) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> {
    let img = read_image_as_rgb8(filepath);
    return convert_rgb_image_to_owned_array(img);
}
