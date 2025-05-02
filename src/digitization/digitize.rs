use crate::annotations::detection::Detection;
use crate::annotations::bounding_box::BoundingBox;
use crate::digitization::chart::Chart;
use crate::image_utils::image_io::read_image_as_array4;
use crate::image_utils::tiling::OverlapProportion;
use std::path::Path;


struct DigitzationParameters {
    document_landmark_model_path: Path,
    document_landmark_tile_size: u32,
    document_landmark_overlap_proportion: OverlapProportion
}

pub fn digitize(
    preop_postop_image_filepath: &Path,
    intraop_image_filepath: &Path,
    parameters: DigitzationParameters
) -> Result<Chart, &'static str> {
    let preop_postop_image = read_image_as_array4(preop_postop_image_filepath);
    let intraop_image = read_image_as_array4(intraop_image_filepath);

    Err("")
}

fn detect_document_landmarks() -> Vec<Detection<BoundingBox>> {
    vec![]
}
