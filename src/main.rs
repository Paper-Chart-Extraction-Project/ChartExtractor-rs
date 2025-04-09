mod annotations;
mod image_utils;
mod object_detection;
use image_utils::image_io::read_image_as_array4;
use image_utils::tiling::OverlapProportion;
use object_detection::yolov11::{Yolov11, read_classes_txt_file, tile_and_predict};
use serde_json;
use std::error::Error;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let model_path = Path::new("./data/models/yolo11n.onnx");
    let classes_path = Path::new("./data/model_metadata/coco-classes.txt");

    if !model_path.exists() {
        return Err(format!(
            "Model path does not exist, or cannot be read: {:?}",
            model_path
        )
        .into());
    }
    if !classes_path.exists() {
        return Err(format!(
            "Classes path does not exist, or cannot be read: {:?}",
            classes_path
        )
        .into());
    }
    let model = Yolov11::new(
        model_path,
        read_classes_txt_file(classes_path).unwrap(),
        640,
        640,
        "yolov11n onnx".to_string(),
    )
    .unwrap();
    let img = read_image_as_array4(Path::new("./data/images/people_on_street.jpg"));
    let now = Instant::now();
    let preds = tile_and_predict(
        &model,
        img,
        640,
        OverlapProportion { numerator: 1_u32, denominator: 2_u32 },
        0.5_f32,
        0.1_f32,
    )
    .unwrap();
    println!("Time elapsed: {:?}", now.elapsed());
    println!("{}", serde_json::to_string(&preds).unwrap());
    Ok(())
}
