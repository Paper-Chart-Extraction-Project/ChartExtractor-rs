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


pub struct BoundingBoxModelParameters<'a> {
    pub name: String,
    pub model_path: &'a Path,
    pub class_names_path: &'a Path,
    pub input_width: usize,
    pub input_height: usize,
    pub tile_size: u32,
    pub overlap_proportion: OverlapProportion,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
}

struct DigitzationParameters<'a> {
    intraop_document_landmark_model_parameters: BoundingBoxModelParameters<'a>,
    preop_postop_document_landmark_model_parameters: BoundingBoxModelParameters<'a>,
    handwritten_numbers_model_parameters: BoundingBoxModelParameters<'a>,
}

pub fn digitize(
    preop_postop_image_filepath: &Path,
    intraop_image_filepath: &Path,
    parameters: DigitzationParameters,
    use_adaptive_padding: bool
) -> Result<Chart, &'static str> {
    let preop_postop_image = read_image_as_array4(preop_postop_image_filepath);
    let intraop_image = read_image_as_array4(intraop_image_filepath);
    let intraop_document_landmarks = run_yolov11_bounding_box_model(
        &intraop_image,
        &parameters.intraop_document_landmark_model_parameters,
        use_adaptive_padding,
    );
    let preop_postop_document_landmarks = run_yolov11_bounding_box_model(
        &intraop_image,
        &parameters.preop_postop_document_landmark_model_parameters,
        use_adaptive_padding,
    );
    let intraop_handwritten_numbers = run_yolov11_bounding_box_model(
        &intraop_image,
        &parameters.handwritten_numbers_model_parameters,
        use_adaptive_padding,
    );
    let preop_postop_handwritten_numbers = run_yolov11_bounding_box_model(
        &intraop_image,
        &parameters.handwritten_numbers_model_parameters,
        use_adaptive_padding,
    );
    Err("")
}

pub fn run_yolov11_bounding_box_model(
    image: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    model_parameters: &BoundingBoxModelParameters,
    use_adaptive_padding: bool,
) -> Vec<Detection<BoundingBox>> {
    let image = image.clone();
    let class_names: Vec<String> = read_classes_txt_file(
        &model_parameters.class_names_path
    ).unwrap();
    let model: Yolov11BoundingBox = Yolov11BoundingBox::new(
        &model_parameters.model_path,
        class_names,
        model_parameters.input_width,
        model_parameters.input_height,
        model_parameters.name.clone(),
    ).unwrap();
    if use_adaptive_padding {
        let padded_image = pad_image_to_fit_tiling_params(
            &image,
            model_parameters.tile_size,
            model_parameters.overlap_proportion,
        );
        let padded_image = convert_rgb_image_to_owned_array(padded_image);
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
