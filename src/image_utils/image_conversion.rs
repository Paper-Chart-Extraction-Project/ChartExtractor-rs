use image::{self, Rgb, RgbImage};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr, ViewRepr};

pub fn convert_array_view_to_rgb_image(
    image_array: ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>,
) -> RgbImage {
    let image_width = image_array.shape()[2] as u32;
    let image_height = image_array.shape()[3] as u32;

    let mut rgb_image = RgbImage::new(image_width, image_height);
    for y in 0..image_height {
        for x in 0..image_width {
            let r = (image_array[[0, 0, y as usize, x as usize]] * 255.0)
                .round()
                .min(255.0)
                .max(0.0) as u8;
            let g = (image_array[[0, 1, y as usize, x as usize]] * 255.0)
                .round()
                .min(255.0)
                .max(0.0) as u8;
            let b = (image_array[[0, 2, y as usize, x as usize]] * 255.0)
                .round()
                .min(255.0)
                .max(0.0) as u8;
            println!("r: {:?}, g: {:?}, b: {:?}", r, g, b);
            rgb_image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    rgb_image
}

pub fn convert_rgb_image_to_owned_array(
    rgb_image: RgbImage,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> {
    let mut image_array = Array::zeros((
        1,
        3,
        rgb_image.width() as usize,
        rgb_image.height() as usize,
    ));
    for pixel in rgb_image.enumerate_pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b] = pixel.2.0;
        image_array[[0, 0, y, x]] = (r as f32) / 255.;
        image_array[[0, 1, y, x]] = (g as f32) / 255.;
        image_array[[0, 2, y, x]] = (b as f32) / 255.;
    }
    image_array
}
