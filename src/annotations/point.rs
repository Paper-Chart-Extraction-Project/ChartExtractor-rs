use std::fmt;

/// A struct representing a simple point.
#[derive(Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point {{ x: {}, y: {} }}", self.x, self.y)
    }
}
