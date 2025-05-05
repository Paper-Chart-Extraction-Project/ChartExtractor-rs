use crate::annotations::point::Point;
use serde_json::{Value, from_reader};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Reads a HashMap of named points (the names are the keys) from a json file.
pub fn read_centroids_from_json(filepath: &Path) -> HashMap<String, Point> {
    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(file);
    let mut centroids: HashMap<String, Point> = HashMap::new();
    let centroids_json: serde_json::Value = serde_json::from_reader(reader).unwrap();
    if let serde_json::Value::Object(map) = centroids_json {
        for (key, value) in map.iter() {
            let name: String = key.as_str().to_string();
            let centroid: Point = Point {
                x: value[0].as_f64().unwrap() as f32,
                y: value[1].as_f64().unwrap() as f32,
            };
            centroids.insert(name, centroid);
        }
    }
    centroids
}
