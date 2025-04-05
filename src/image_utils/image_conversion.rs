use image::{self, Rgb, RgbImage};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};

pub fn ArrayView_to_RgbImage(image_array: ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>) -> RgbImage {
    let image_width = image_array.shape()[2] as u32;
    let image_height = image_array.shape()[3] as u32;
    
    let mut rgb_image = RgbImage::new(image_width, image_height);
    for y in 0..image_height {
        for x in 0..image_width {
            let r = (image_array[[0, 0, y, x]] * 255.0).round().min(255.0).max(0.0) as u8;
            let g = (image_array[[0, 1, y, x]] * 255.0).round().min(255.0).max(0.0) as u8;
            let b = (image_array[[0, 2, y, x]] * 255.0).round().min(255.0).max(0.0) as u8;
            rgb_image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    rgb_image
}
