use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Reads a file with the class names into a vector so that the number ids
/// which come directly from the ORT inference session can be given meaning.
pub fn read_classes_txt_file(filepath: &Path) -> io::Result<Vec<String>> {
    BufReader::new(File::open(filepath)?).lines().collect()
}
