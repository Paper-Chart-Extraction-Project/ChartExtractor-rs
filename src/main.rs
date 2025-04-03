mod annotations;
mod object_detection;
mod image_utils;
use image_utils::image_io::read_image_as_array4;
use ndarray::s;
use object_detection::yolov11::{read_classes_txt_file, Yolov11};
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let model_path = Path::new("./data/models/yolo11n.onnx");
    let classes_path = Path::new("./data/model_metadata/coco-classes.txt");

    if !model_path.exists() {
        return Err(
            format!("Model path does not exist, or cannot be read: {:?}", model_path).into()
        );
    }
    if !classes_path.exists() {
        return Err(
            format!("Classes path does not exist, or cannot be read: {:?}", classes_path).into()
        );
    }
    let model = Yolov11::new(
        model_path,
        read_classes_txt_file(classes_path).unwrap(),
        640,
        640,
        "yolov11n onnx".to_string()
    ).unwrap();
    let img = read_image_as_array4(Path::new("./data/images/test.jpg"));
    // This implementation only takes views into arrays as input.
    // Check yolov11.rs for detail.
    let view_into_img = img.slice(s![.., .., 0..640, 0..640]);
    println!("Preds: {:?}", model.run_inference(view_into_img, 0.5_f32));
    Ok(())
}
