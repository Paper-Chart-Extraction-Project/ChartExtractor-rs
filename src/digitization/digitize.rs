use crate::annotations::bounding_box::{BoundingBox, BoundingBoxGeometry};
use crate::annotations::detection::Detection;
use crate::annotations::point::Point;
use crate::digitization::chart::Chart;
use crate::image_utils::image_conversion::convert_rgb_image_to_owned_array;
use crate::image_utils::image_io::read_image_as_array4;
use crate::image_utils::tiling::{OverlapProportion, pad_image_to_fit_tiling_params};
use crate::object_detection::object_detection_utils::{read_classes_txt_file, tile_and_predict};
use crate::object_detection::yolov11_bounding_box::Yolov11BoundingBox;
use crate::registration::coherent_point_drift::CoherentPointDriftTransform;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use std::collections::HashMap;
use std::fmt::Display;
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
    checkbox_model_parameters: BoundingBoxModelParameters<'a>,
    intraop_document_landmarks_centroids: HashMap<String, Point>,
    preop_postop_document_landmarks_centroids: HashMap<String, Point>,
    intraop_checkboxes_centroids: HashMap<String, Point>,
    preop_postop_checkboxes_centroids: HashMap<String, Point>,
    intraop_number_boxes_centroids: HashMap<String, Point>,
    preop_postop_number_boxes_centroids: HashMap<String, Point>,
}

pub fn digitize(
    preop_postop_image_filepath: &Path,
    intraop_image_filepath: &Path,
    parameters: DigitzationParameters,
    use_adaptive_padding: bool,
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
    let intraop_checkboxes = run_yolov11_bounding_box_model(
        &intraop_image,
        &parameters.checkbox_model_parameters,
        use_adaptive_padding,
    );
    let preop_postop_checkboxes = run_yolov11_bounding_box_model(
        &preop_postop_image,
        &parameters.checkbox_model_parameters,
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
    let class_names: Vec<String> =
        read_classes_txt_file(&model_parameters.class_names_path).unwrap();
    let model: Yolov11BoundingBox = Yolov11BoundingBox::new(
        &model_parameters.model_path,
        class_names,
        model_parameters.input_width,
        model_parameters.input_height,
        model_parameters.name.clone(),
    )
    .unwrap();
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
        model_parameters.nms_threshold,
    )
    .unwrap()
}

/// Uses cpd to match the detections to a perfect, scanned version of the chart
/// then removes all detections which don't find a match, and further removes
/// all of the detections who do match, but whose predicted class does not match
/// the class of its match.
pub fn filter_detections_with_cpd<T: BoundingBoxGeometry + Display>(
    ground_truth_centroids: HashMap<String, Point>,
    detections: Vec<Detection<T>>,
    lambda: f32,
    beta: f32,
    weight_of_uniform_dist: f32,
    tolerance: f32,
    max_iterations: u32,
    debug: bool,
) -> Vec<Detection<T>> {
    let detections_as_points = detections
        .iter()
        .map(|d| d.annotation.center())
        .collect::<Vec<Point>>();

    let pairs = ground_truth_centroids
        .into_iter()
        .collect::<Vec<(String, Point)>>();
    let gt_centroid_classes = pairs.iter().map(|p| p.0.clone()).collect::<Vec<String>>();
    let gt_centroids_as_points = pairs.iter().map(|p| p.1.clone()).collect::<Vec<Point>>();

    let mut cpd: CoherentPointDriftTransform = CoherentPointDriftTransform::from_point_vectors(
        gt_centroids_as_points,
        detections_as_points,
        lambda,
        beta,
        Some(weight_of_uniform_dist),
        Some(tolerance),
        Some(max_iterations),
        Some(debug),
    );
    cpd.register();
    let matches: Vec<(usize, usize)> = cpd.generate_matching();
    let indexes_of_matched_detections: Vec<usize> = matches
        .clone()
        .into_iter()
        .map(|(ix, _)| ix)
        .collect::<Vec<usize>>();

    // filter all points from detections whose index is not in the 0 index of
    // any tuple in the cpd transform, and whose class string is not equal
    // to the class string of the centroid key.
    let filtered_detections: Vec<Detection<T>> = detections
        .into_iter()
        .enumerate()
        .filter(|(ix, det)| {
            indexes_of_matched_detections.contains(ix)
                && *det.annotation.category() == gt_centroid_classes[*ix]
        })
        .map(|(_, det)| det)
        .collect::<Vec<Detection<T>>>();
    filtered_detections
}
