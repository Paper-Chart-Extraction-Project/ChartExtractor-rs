use std::fmt;

/// A set of custom errors for more informative error handling.
#[derive(Debug)]
pub enum TilingError {
    InvalidWidth {
        original_width: u32,
        new_width: u32,
    },
    InvalidHeight {
        original_height: u32,
        new_height: u32,
    },
    InvalidDimensions {
        original_width: u32,
        new_width: u32,
        original_height: u32,
        new_height: u32,
    },
}
