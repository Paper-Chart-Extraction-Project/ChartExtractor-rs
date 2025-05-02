use crate::annotations::detection::Detection;
use crate::annotations::bounding_box::BoundingBox;
use crate::digitization::chart::Chart;
use crate::image_utils::image_conversion::convert_rgb_image_to_owned_array;
use crate::image_utils::image_io::read_image_as_array4;
use crate::image_utils::tiling::{OverlapProportion, pad_image_to_fit_tiling_params};
use crate::object_detection::object_detection_utils::{read_classes_txt_file, tile_and_predict};
use crate::object_detection::yolov11_bounding_box::Yolov11BoundingBox;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use std::path::Path;


struct BoundingBoxModelParameters {
    name: String,
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
    image: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    model_parameters: BoundingBoxModelParameters,
    use_adaptive_padding: bool,
) -> Vec<Detection<BoundingBox>> {
    let class_names: Vec<String> = read_classes_txt_file(
        &model_parameters.class_names_path
    ).unwrap();
    let model: Yolov11BoundingBox = Yolov11BoundingBox::new(
        &model_parameters.model_path,
        class_names,
        model_parameters.input_width,
        model_parameters.input_height,
        model_parameters.name
    ).unwrap();
    if use_adaptive_padding {
        let image = pad_image_to_fit_tiling_params(
            &image,
            model_parameters.tile_size,
            model_parameters.overlap_proportion,
        );
        let image = convert_rgb_image_to_owned_array(image);
    }
    tile_and_predict(
        &model,
        image,
        model_parameters.tile_size,
        model_parameters.overlap_proportion,
        model_parameters.confidence_threshold,
        model_parameters.nms_threshold
    ).unwrap()
}
