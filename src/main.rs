mod annotations;
mod object_detection;
mod image_utils;
use image_utils::image_io::read_image_as_array4;
use ndarray::s;
use object_detection::yolov11::{read_classes_txt_file, Yolov11};
use std::path::Path;

fn main() {
    let model_path = Path::new("./data/models/yolo11n.onnx");
    let classes_path = Path::new("./data/model_metadata/coco-classes.txt");
    let yv11 = Yolov11::new(
        model_path,
        read_classes_txt_file(classes_path).unwrap(),
        640,
        640,
        "yolov11n onnx".to_string()
    ).unwrap();
    let img = read_image_as_array4(Path::new("./data/images/test.jpg"));
    let view_into_img = img.slice(s![.., .., 0..640, 0..640]);
    println!("Preds: {:?}", yv11.run_inference(view_into_img, 0.5_f32));
}
