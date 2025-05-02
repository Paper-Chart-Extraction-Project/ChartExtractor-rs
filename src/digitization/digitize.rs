use crate::annotations::detection::Detection;
use crate::annotations::bounding_box::BoundingBox;
use crate::digitization::chart::Chart;
use crate::image_utils::image_io::read_image_as_array4;
use std::path::Path;


struct DigitzationParameters { }

pub fn digitize(
    preop_postop_image_filepath: &Path,
    intraop_image_filepath: &Path,
    parameters: DigitzationParameters
) -> Result<Chart, &'static str> {
    let preop_postop_image = read_image_as_array4(preop_postop_image_filepath);
    let intraop_image = read_image_as_array4(intraop_image_filepath);

    Err("")
}

