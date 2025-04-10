use crate::annotations::point::Point;
use imageproc::geometric_transformations::Projection;


/// Computes the homography transformation.
///
/// A homography is a projective transformation that, given two cameras,
/// will transform the view of the first camera into the second camera.
/// Our project uses the homography to transform an off-angle photo into
/// the coordinates of a perfect, scan-like version of the image.
///
/// source_points and destination_points must be exactly four points.
pub fn compute_homography_transformation(
    source_points: Vec<Point>,
    destination_points: Vec<Point>
) -> Option<Projection> {
    if source_points.len() != 4 || destination_points.len() != 4 {
        return None
    }
    let from_points: [(f32, f32); 4] = source_points
        .iter()
        .map(|p| (p.x, p.y))
        .collect::<Vec<(f32, f32)>>()
        .try_into()
        .unwrap_or_else(
            |v: Vec<(f32, f32)>| panic!("Expected Vec of length {} but found {}.", 4, v.len())
        );
    let to_points: [(f32, f32); 4] = destination_points
        .iter()
        .map(|p| (p.x, p.y))
        .collect::<Vec<(f32, f32)>>()
        .try_into()
        .unwrap_or_else(
            |v: Vec<(f32, f32)>| panic!("Expected Vec of length {} but found {}.", 4, v.len())
        );
    Projection::from_control_points(from_points, to_points)
}
