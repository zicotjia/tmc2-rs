use cgmath::{point1, Vector3};
use unzip3::Unzip3;
use crate::encoder::Vector3D;

pub type Point3D = Vector3<u16>;
pub type Color3B = Vector3<u8>;

type Color16bit = Vector3<u16>;
pub type Normal3D = Vector3<f64>;
// type Matrix3D = Matrix3<usize>;

// Set of point clouds
#[derive(Debug, Default)]
pub struct PointSet3 {
    // NOTE: IF YOU UPDATE THIS STRUCT, dont forget to update resize and append point set.
    pub positions: Vec<Point3D>,
    pub colors: Vec<Color3B>,
    pub(crate) colors16bit: Vec<Color16bit>,
    // boundary_point_types: Vec<u16>,
    pub(crate) point_patch_indexes: Vec<(usize, usize)>,
    // parent_point_index: Vec<usize>,
    /// Only if PCC_SAVE_POINT_TYPE is true
    // types: Vec<u8>,
    // reflectances: Vec<u16>,
    pub normals: Vec<Normal3D>,
    pub with_normals: bool,
    pub with_colors: bool,
    // with_reflectances: bool,
}

impl From<Vec<vivotk::formats::pointxyzrgba::PointXyzRgba>> for PointSet3 {
    fn from(value: Vec<vivotk::formats::pointxyzrgba::PointXyzRgba>) -> Self {
        let (positions, colors): (Vec<Point3D>, Vec<Color3B>) = value
            .into_iter()
            .map(|point| (
                Point3D {x : point.x as u16, y: point.y as u16, z: point.z as u16},
                Color3B {x: point.r, y: point.g, z: point.b})
            ).unzip();
        PointSet3 { positions, colors, colors16bit: vec![], point_patch_indexes: vec![], normals: vec![], with_normals: false, with_colors: true }
    }
}

// ZICO: Beware VVTK use f32 for normals while tmc2 use f64
impl From<Vec<vivotk::formats::pointxyzrgbanormal::PointXyzRgbaNormal>> for PointSet3 {
    fn from(value: Vec<vivotk::formats::pointxyzrgbanormal::PointXyzRgbaNormal>) -> Self {
        // ZICO: Alpha is unused for now
        let (positions, colors, normals): (Vec<Point3D>, Vec<Color3B>, Vec<Normal3D>) = value
            .into_iter()
            .map(|point| (
                Point3D {x : point.x as u16, y: point.y as u16, z: point.z as u16},
                Color3B {x: point.r, y: point.g, z: point.b},
                Normal3D {x: point.nx as f64, y: point.ny as f64, z: point.z as f64})
            ).unzip3();
        PointSet3 { positions, colors, colors16bit: vec![], point_patch_indexes: vec![], normals, with_normals: true, with_colors: true }
    }
}

impl PointSet3 {
    #[inline]
    pub fn point_count(&self) -> usize {
        self.positions.len()
    }

    /// add point to PointSet, and allocate the rest of the structure and returns its index
    pub(crate) fn add_point(&mut self, position: Point3D) -> usize {
        self.positions.push(position);
        if self.with_colors {
            self.colors.push(Color3B::new(127, 127, 127));
            self.colors16bit.push(Color16bit::new(0, 0, 0));
        }
        if self.with_normals {
            self.normals.push(Normal3D::new(0.0, 0.0, 0.0));
        }
        self.point_patch_indexes.push((0, 0));
        self.positions.len() - 1
    }

    pub(crate) fn add_point_from_vector_3d(&mut self, position: Vector3D) -> usize {
        self.positions.push(Point3D::new(position.x as u16, position.y as u16, position.z as u16));
        if self.with_colors {
            self.colors.push(Color3B::new(127, 127, 127));
            self.colors16bit.push(Color16bit::new(0, 0, 0));
        }
        if self.with_normals {
            self.normals.push(Normal3D::new(0.0, 0.0, 0.0));
        }
        self.point_patch_indexes.push((0, 0));
        self.positions.len() - 1
    }

    /// set PointSet to use color
    pub(crate) fn add_colors(&mut self) {
        self.with_colors = true;
        self.reserve(self.point_count());
    }

    pub(crate) fn append_point_set(&mut self, pointset: PointSet3) -> usize {
        self.positions.extend(pointset.positions.iter());
        self.colors.extend(pointset.colors.iter());
        self.colors16bit.extend(pointset.colors16bit.iter());
        self.point_patch_indexes
            .extend(pointset.point_patch_indexes.iter());
        self.normals.extend(pointset.normals.iter());

        // SKIP: self.resize(self.point_count()). for what?
        self.point_count()
    }

    pub(crate) fn reserve(&mut self, size: usize) {
        self.positions.reserve(size);
        if self.with_colors {
            self.colors.reserve(size);
            self.colors16bit.reserve(size);
        }
        if self.with_normals {
            self.normals.reserve(size);
        }
        self.point_patch_indexes.reserve(size);
    }

    #[inline]
    /// add color to PointSet
    pub(crate) fn _set_color(&mut self, index: usize, color: Color3B) {
        assert!(self.with_colors && index < self.colors.len());
        self.colors[index] = color;
    }

    pub(crate) fn convert_yuv16_to_rgb8(&mut self) {
        assert!(self.with_colors);
        assert!(self.colors16bit.len() == self.point_count());
        for (i, color16bit) in self.colors16bit.iter().enumerate() {
            self.colors[i] = convert_yuv10_to_rgb8(color16bit);
        }
    }

    pub(crate) fn copy_rgb16_to_rgb8(&mut self) {
        assert!(self.with_colors);
        assert!(self.colors16bit.len() == self.point_count());
        for (i, color16bit) in self.colors16bit.iter().enumerate() {
            self.colors[i] = Vector3 {
                x: color16bit.x as u8,
                y: color16bit.y as u8,
                z: color16bit.z as u8,
            };
        }
    }

    pub fn add_normals(&mut self, normal: Vec<Normal3D>) {
        assert_eq!(self.point_count(), normal.len());
        self.with_normals = true;
        self.normals = normal;
    }

    pub fn len(&self) -> usize {
        assert!(!self.with_colors || self.positions.len() == self.colors.len());
        self.positions.len()
    }
}


/// Convert YUV10bit to RGB8bit
fn convert_yuv10_to_rgb8(color16: &Vector3<u16>) -> Vector3<u8> {
    // yuv16 to rgb8
    let clamp = |x: f64| -> u8 {
        if x < 0. {
            0
        } else if x > 255. {
            255
        } else {
            x as u8
        }
    };

    let offset = 512.;
    let scale = 1023.;

    let Vector3 { x: y, y: u, z: v } = *color16;
    let y = y as f64;
    let u = u as f64;
    let v = v as f64;
    let r = y + 1.57480 * (v - offset);
    let g = y - 0.18733 * (u - offset) - (0.46813 * (v - offset));
    let b = y + 1.85563 * (u - offset);
    let r = clamp((r / scale * 255.).floor());
    let g = clamp((g / scale * 255.).floor());
    let b = clamp((b / scale * 255.).floor());
    Vector3 { x: r, y: g, z: b }
}


#[cfg(test)]
mod tests {
    use cgmath::assert_relative_eq;
    use vivotk::formats::pointxyzrgba::PointXyzRgba;
    use vivotk::formats::pointxyzrgbanormal::PointXyzRgbaNormal;
    use super::*;

    #[test]
    fn test_vectorOfPointsConversionToPointSet() {
        let vectorOfStructs: Vec<PointXyzRgba> = Vec::from(
            [PointXyzRgba { x: 10.0, y: 0.0, z: 0.0, r: 1, g: 0, b: 0, a: 0 },
                PointXyzRgba { x: 0.0, y: 10.0, z: 0.0, r: 0, g: 2, b: 0, a: 0 },
                PointXyzRgba { x: 0.0, y: 0.0, z: 10.0, r: 0, g: 0, b: 3, a: 3 }]
        );
        // COuld be improved to account for conversion of f32 to u16
        let structOfVectors = PointSet3::from(vectorOfStructs);
        assert_eq!(structOfVectors.positions[0], Point3D { x: 10, y: 0, z: 0 });
        assert_eq!(structOfVectors.positions[1], Point3D { x: 0, y: 10, z: 0 });
        assert_eq!(structOfVectors.positions[2], Point3D { x: 0, y: 0, z: 10 });
        assert_eq!(structOfVectors.colors[0], Color3B { x: 1, y: 0, z: 0 });
        assert_eq!(structOfVectors.colors[1], Color3B { x: 0, y: 2, z: 0 });
        assert_eq!(structOfVectors.colors[2], Color3B { x: 0, y: 0, z: 3 });

        #[test]
        fn test_vectorOfPointsWithNormalConversionToPointSet() {
            let vectorOfStructs: Vec<PointXyzRgbaNormal> = Vec::from(
                [PointXyzRgbaNormal { x: 10.0, y: 0.0, z: 0.0, r: 1, g: 0, b: 0, a: 0, nx: 0.0, ny: 0.0, nz: 1.0 },
                    PointXyzRgbaNormal { x: 0.0, y: 10.0, z: 0.0, r: 0, g: 2, b: 0, a: 0, nx: 0.0, ny: 2.0, nz: 0.0 },
                    PointXyzRgbaNormal { x: 0.0, y: 0.0, z: 10.0, r: 0, g: 0, b: 3, a: 3, nx: 3.0, ny: 0.0, nz: 0.0 }]
            );
            // COuld be improved to account for conversion of f32 to u16
            let structOfVectors = PointSet3::from(vectorOfStructs);
            assert_eq!(structOfVectors.positions[0], Point3D { x: 10, y: 0, z: 0 });
            assert_eq!(structOfVectors.positions[1], Point3D { x: 0, y: 10, z: 0 });
            assert_eq!(structOfVectors.positions[2], Point3D { x: 0, y: 0, z: 10 });
            assert_eq!(structOfVectors.colors[0], Color3B { x: 1, y: 0, z: 0 });
            assert_eq!(structOfVectors.colors[1], Color3B { x: 0, y: 2, z: 0 });
            assert_eq!(structOfVectors.colors[2], Color3B { x: 0, y: 0, z: 3 });
            assert_relative_eq!(structOfVectors.normals[0], Normal3D {x: 0.0, y: 0.0, z: 1.0});
            assert_relative_eq!(structOfVectors.normals[1], Normal3D {x: 0.0, y: 2.0, z: 0.0});
            assert_relative_eq!(structOfVectors.normals[2], Normal3D {x: 3.0, y: 0.0, z: 0.0});
        }
    }
}
