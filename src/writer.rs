use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::codec::PointSet3;

// http://gamma.cs.unc.edu/POWERPLANT/papers/ply.pdf

pub enum Format {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

pub(crate) struct PlyWriter {
    pc: PointSet3,
    format: Format,
}

impl PlyWriter {
    pub fn new(pc: PointSet3, format: Format) -> Self {
        Self { pc, format }
    }

    pub fn write(&self, path: &Path) {
        let mut output = File::create(path).unwrap();
        self.write_header(&mut output);
        self.write_body(&mut output);
    }

    fn write_header(&self, file: &mut File) {
        writeln!(file, "ply");
        match self.format {
            Format::Ascii => {
                writeln!(file, "format ascii 1.0");
            }
            Format::BinaryLittleEndian => {
                writeln!(file, "format binary_little_endian 1.0");
            }
            Format::BinaryBigEndian => {
                writeln!(file, "format binary_big_endian 1.0");
            }
        }
        writeln!(file, "element vertex {}", self.pc.point_count());
        writeln!(file, "property uint x");
        writeln!(file, "property uint y");
        writeln!(file, "property uint z");
        if self.pc.with_colors {
            writeln!(file, "property uchar red");
            writeln!(file, "property uchar green");
            writeln!(file, "property uchar blue");
        }
        writeln!(file, "element face 0");
        writeln!(file, "property list uint8 int32 vertex_index");
        writeln!(file, "end_header");
    }

    fn write_body(&self, file: &mut File) {
        for i in 0..self.pc.point_count() {
            let pos = &self.pc.positions[i];
            write!(file, "{} {} {}", pos.x, pos.y, pos.z);
            if self.pc.with_colors {
                let color = &self.pc.colors[i];
                write!(file, " {} {} {}", color.x, color.y, color.z);
            }
            writeln!(file, "");
        }
    }
}
