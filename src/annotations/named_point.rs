use crate::annotations::point::Point;
use std::fmt;

struct NamedPoint {
    pub name: String,
    point: Point
}

impl NamedPoint {
    pub fn new(name: String, x: f32, y: f32) -> Self {
        NamedPoint { name: name, point: Point { x, y } }
    }

    pub fn x(&self) -> f32 {
        self.point.x
    }

    pub fn y(&self) -> f32 {
        self.point.y
    }
}

impl fmt::Display for NamedPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NamedPoint {{ name: {}, x: {}, y: {} }}", self.name, self.point.x, self.point.y)
    }
}
