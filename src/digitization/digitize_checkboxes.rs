use crate::annotations::detection::Detection;
use crate::annotations::point::Point;
use crate::annotations::bounding_box::{BoundingBox, BoundingBoxGeometry};
use std::collections::HashMap;


fn find_min_distance_key(map: &HashMap<String, Point>, new_point: Point) -> Option<(String, f32)> {
    fn dist(p1: Point, p2: Point) -> f32 {
        ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt()
    }
    let min_key = map.iter()
        .map(|(key, value)| (key, dist(new_point, *value)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(key, _)| key.clone());
    if min_key.is_some() {
        return Some((min_key.clone().unwrap(), dist(new_point, *map.get(&min_key.unwrap()).unwrap())));
    } else {
        return None;
    }
}

pub fn digitize_checkboxes(
    checkbox_detections: Vec<Detection<BoundingBox>>,
    checkbox_centroids: HashMap<String, Point>
) -> HashMap<String, bool> {
    let mut checkbox_statuses: HashMap<String, bool> = HashMap::new();
    for ckbx_det in checkbox_detections.into_iter() {
        let min_key_and_dist: Option<(String, f32)> = find_min_distance_key(
            &checkbox_centroids,
            ckbx_det.annotation.center()
        );
        if min_key_and_dist.is_some() {
            let (min_key, dist) = min_key_and_dist.unwrap();
            println!("{:?}, {:?}", min_key, dist);
            let status = false;
            checkbox_statuses.insert(min_key.clone(), status);
        }
    }
    checkbox_statuses
}
