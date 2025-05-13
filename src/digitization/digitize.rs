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
use crate::registration::thin_plate_splines::TpsTransform;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Serialize)]
pub struct BoundingBoxModelParameters {
    pub name: String,
    pub model_path: PathBuf,
    pub class_names_path: PathBuf,
    pub input_width: usize,
    pub input_height: usize,
    pub tile_size: u32,
    pub overlap_proportion: OverlapProportion,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
}

/// All the parameters for doing coherent point drift.
#[derive(Debug, Deserialize, Serialize)]
pub struct CpdParameters {
    pub lambda: f32,
    pub beta: f32,
    pub weight_of_uniform_dist: f32,
    pub tolerance: f32,
    pub max_iterations: u32,
    pub debug: bool,
}

pub struct DigitzationParameters {
    // document model parameters
    intraop_document_landmark_model_parameters: BoundingBoxModelParameters,
    preop_postop_document_landmark_model_parameters: BoundingBoxModelParameters,
    handwritten_numbers_model_parameters: BoundingBoxModelParameters,
    checkbox_model_parameters: BoundingBoxModelParameters,
    // paths to centroids.
    intraop_document_landmarks_centroids: HashMap<String, Point>,
    preop_postop_document_landmarks_centroids: HashMap<String, Point>,
    intraop_checkboxes_centroids: HashMap<String, Point>,
    preop_postop_checkboxes_centroids: HashMap<String, Point>,
    intraop_number_boxes_centroids: HashMap<String, Point>,
    preop_postop_number_boxes_centroids: HashMap<String, Point>,
    // parameters for coherent point drift.
    intraop_document_landmarks_cpd_parameters: CpdParameters,
    preop_postop_document_landmarks_cpd_parameters: CpdParameters,
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

    let filtered_intraop_document_landmarks = filter_detections_with_cpd(
        intraop_document_landmarks,
        parameters.intraop_document_landmarks_centroids,
        parameters.intraop_document_landmarks_cpd_parameters
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
pub fn filter_detections_with_cpd<T: BoundingBoxGeometry + Display+ std::fmt::Debug>(
    detections: Vec<Detection<T>>,
    ground_truth_centroids: HashMap<String, Point>,
    cpd_params: CpdParameters,
) -> Vec<Detection<T>> where T: Clone {
    let detections_as_points = detections
        .iter()
        .map(|d| d.annotation.center())
        .collect::<Vec<Point>>();

    let pairs = ground_truth_centroids.clone()
        .into_iter()
        .collect::<Vec<(String, Point)>>();
    let gt_centroid_classes = pairs.iter().map(|p| p.0.clone()).collect::<Vec<String>>();
    let gt_centroids_as_points = pairs.iter().map(|p| p.1.clone()).collect::<Vec<Point>>();

    let mut cpd: CoherentPointDriftTransform = CoherentPointDriftTransform::from_point_vectors(
        gt_centroids_as_points,
        detections_as_points,
        cpd_params.lambda,
        cpd_params.beta,
        Some(cpd_params.weight_of_uniform_dist),
        Some(cpd_params.tolerance),
        Some(cpd_params.max_iterations),
        Some(cpd_params.debug),
    );
    cpd.register();
    let matches: Vec<(usize, usize)> = cpd.generate_matching();
    let mut filtered_detections: Vec<Detection<T>> = Vec::new();
    for (src_ix, tar_ix) in matches.clone().into_iter() {
        if *detections[src_ix].annotation.category() == pairs[tar_ix].0 {
            filtered_detections.push(detections[src_ix].clone());
        }
    }
    filtered_detections
}

fn create_tps_transform(
    source_detections: Vec<Detection<BoundingBox>>,
    target_centroids: HashMap<String, Point>,
) -> TpsTransform {
    let mut source_hashmap: HashMap<String, Point> = HashMap::new();
    for det in source_detections.into_iter() {
        source_hashmap.insert(det.annotation.category().clone(), det.annotation.center());
    }

    let mut source_points: Vec<Point> = Vec::new();
    let mut target_points: Vec<Point> = Vec::new();
    for (class, centroid) in target_centroids.into_iter() {
        if source_hashmap.contains_key(&class) {
            source_points.push(source_hashmap.get(&class).unwrap().clone());
            target_points.push(centroid.clone());
        }
    }

    TpsTransform::new(source_points, target_points)
}
