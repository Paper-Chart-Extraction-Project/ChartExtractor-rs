use image::{Rgb, RgbImage};

/// Pads an rgb8 image by adding pixels to the right and bottom of the image.
pub fn pad_right_bottom_img_rbg8(
    original_image: RgbImage, new_width: u32, new_height: u32
) -> RgbImage {
    let mut padded_image: RgbImage = RgbImage::new(new_width, new_height);
    for pixel in original_image.enumerate_pixels() {
        let x = pixel.0;
        let y = pixel.1;
        let [r, g, b] = original_image.get_pixel_checked(x, y).unwrap().0;
        padded_image.put_pixel(x, y, Rgb([r, g, b]));
    }
    padded_image
}
