use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash;

/// A struct representing a simple point.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point {{ x: {}, y: {} }}", self.x, self.y)
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        fn normalize(val: f32) -> u32 {
            // deals with the issue of -0 and 0 having different
            // bit representations.
            if val == 0.0 {
                0_f32.to_bits()
            } else {
                val.to_bits()
            }
        }
        normalize(self.x) == normalize(other.x) && normalize(self.y) == normalize(other.y)
    }
}

impl Eq for Point {}

impl hash::Hash for Point {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        // Convert f32 to raw bits for deterministic hashing.
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}
