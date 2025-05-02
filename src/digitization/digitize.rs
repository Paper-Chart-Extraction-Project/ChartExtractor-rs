use crate::annotations::detection::Detection;
use crate::annotations::bounding_box::BoundingBox;
use crate::digitization::chart::Chart;
use crate::image_utils::image_io::read_image_as_array4;
use crate::image_utils::tiling::OverlapProportion;
use crate::object_detection::object_detection_utils::tile_and_predict;
use crate::object_detection::yolov11_bounding_box::Yolov11BoundingBox;
use ndarray::{ArrayBase, Dim, ViewRepr};
use std::path::Path;


struct BoundingBoxModelParameters {
    model_path: Box<Path>,
    class_names_path: Box<Path>,
    input_width: usize,
    input_height: usize,
    tile_size: u32,
    overlap_proportion: OverlapProportion,
    confidence_threshold: f32,
    nms_threshold: f32,
}

struct DigitzationParameters {
    document_landmark_model_parameters: BoundingBoxModelParameters
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

fn run_yolov11_bounding_box_model(
    image: ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>,
    model_parameters: BoundingBoxModelParameters,
) -> Vec<Detection<BoundingBox>> {
    vec![]
}
