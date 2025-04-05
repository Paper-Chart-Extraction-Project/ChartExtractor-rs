use image::{self, RgbImage};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use std::path::Path;

pub fn read_image_as_array4(filepath: &Path) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> {
    let img = image::open(filepath).unwrap().into_rgb8();
    let mut input = Array::zeros((1, 3, img.width() as usize, img.height() as usize));
    for pixel in img.enumerate_pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }
    input
}

pub fn read_image_as_rgb8(filepath: &Path) -> RgbImage {
    image::open(filepath).unwrap().into_rgb8()
}
