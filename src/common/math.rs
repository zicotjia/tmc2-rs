// Some math stuff for encoding purpose

use std::f64::EPSILON;
use std::fmt::Debug;
use cgmath::{InnerSpace, Matrix, Matrix3, Quaternion, Vector3};
use cgmath::num_traits::float::FloatCore;
use serde::{Deserialize, Serialize};
use vivotk::formats::PointCloud;
use vivotk::formats::pointxyzrgba::PointXyzRgba;
// use super::pointxyzrgba::PointXyzRgba;

// ZICO: Can make this generic
#[derive(Clone)]
pub struct BoundingBox {
    // min values for x, y, z
    pub min: Vector3<f64>,
    // max values for x, y, z
    pub max: Vector3<f64>,
}

impl BoundingBox {
    pub fn contains(&self, point: Vector3<f64>) -> bool {
        !( point.x < self.min.x || point.x > self.max.x || point.y < self.min.y || point.y> self.max.y ||
            point.z < self.min.z || point.z > self.max.z)
    }

    pub fn merge(&self, other: BoundingBox) -> bool {
        unimplemented!("Box merging not implemented")
    }

    pub fn add(&mut self, point: Vector3<f64>) {
        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);
        self.min.z = self.min.z.min(point.z);
        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
        self.max.z = self.max.z.max(point.z);
    }
    pub fn intersects(&self, other: BoundingBox) -> bool {
        unimplemented!("Box intersection not implemented")
    }

    pub fn fully_contains_box(&self, other: BoundingBox) -> bool {
        unimplemented!("Box contains box not implemented")
    }

    pub fn fully_contains_point(&self, point: &Vector3<i16>) -> bool {
        self.max.x >= point.x as f64  && self.min.x <= point.x as f64
        && self.max.y >= point.y as f64 && self.min.y <= point.y as f64
        && self.max.z >= point.z as f64 && self.min.z <= point.z as f64
    }
}

impl Default for BoundingBox {
    fn default() -> Self {
        Self {
            min: Vector3::new(f64::MAX, f64::MAX, f64::MAX),
            max: Vector3::new(f64::MIN, f64::MIN, f64::MIN)
        }
    }
}

impl From<Vec<Vector3<f64>>> for BoundingBox {
    fn from(value: Vec<Vector3<f64>>) -> Self {
        let mut b_box = BoundingBox::default();
        for point in value {b_box.add(point)}
        b_box
    }
}

impl From<PointCloud<PointXyzRgba>> for BoundingBox {
    fn from(point_cloud: PointCloud<PointXyzRgba>) -> Self {
        let mut b_box = BoundingBox::default();
        for point in point_cloud.points {
            b_box.add( Vector3 {
                x: point.x as f64,
                y: point.y as f64,
                z: point.z as f64,
            })
        };
        b_box
    }
}

impl From<Vec<PointXyzRgba>> for BoundingBox {
    fn from(point_cloud: Vec<PointXyzRgba>) -> Self {
        let mut b_box = BoundingBox::default();
        for point in point_cloud { b_box.add( Vector3 {
            x: point.x as f64,
            y: point.y as f64,
            z: point.z as f64,
        }) };
        b_box
    }
}

impl Debug for BoundingBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BoundingBox {{")?;
        writeln!(f, "min: {:?}", self.min)?;
        writeln!(f, "max: {:?}", self.max)?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

// Diagonalize a matrix
pub(crate) fn diagonalize(a: &Matrix3<f64>) -> (Matrix3<f64>, Matrix3<f64>) {
    let maxsteps = 24;
    let mut quat = Quaternion::from_sv(1.0, cgmath::Vector3::new(0.0, 0.0, 0.0)); // Quaternion identity

    let mut q: Matrix3<f64> = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut d: Matrix3<f64> = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for _ in 0..maxsteps {
        // Convert quaternion to matrix using cgmath
        q = Matrix3::from(quat);

        // Multiply AQ = A * Q
        let aq = a * q;

        // D = Q.transpose() * AQ
        d = q.transpose() * aq;

        let o = [d[1][2], d[0][2], d[0][1]];
        let m = [o[0].abs(), o[1].abs(), o[2].abs()];

        // Find the index of the largest element of the off-diagonal
        let k0 = if m[0] > m[1] && m[0] > m[2] {
            0
        } else if m[1] > m[2] {
            1
        } else {
            2
        };

        let k1 = (k0 + 1) % 3;
        let k2 = (k0 + 2) % 3;

        if o[k0].abs() < EPSILON {
            break; // diagonal already
        }

        let theta = (d[k2][k2] - d[k1][k1]) / (2.0 * o[k0]);
        let sgn = if theta > 0.0 { 1.0 } else { -1.0 };
        let theta = theta * sgn;
        let t = sgn / (theta + if theta < 1.0e6 { (theta * theta + 1.0).sqrt() } else { theta });
        let c = 1.0 / (t * t + 1.0).sqrt();

        if (c - 1.0).abs() < EPSILON {
            break;
        }

        let mut jr_quat = Quaternion::new(0.0, 0.0, 0.0, 0.0);
        jr_quat.v[k0] = sgn * ((1.0 - c) / 2.0).sqrt();
        jr_quat.v[k0] *= -1.0; // account for quaternion convention
        jr_quat.s = (1.0 - jr_quat.v[k0] * jr_quat.v[k0]).sqrt();

        // Update quaternion
        quat = quat * jr_quat;
        quat = quat.normalize();
    }
    (q, d)
}


#[cfg(test)]
mod tests {
    use std::cmp;
    use approx::assert_relative_eq;
    use cgmath::{assert_ulps_eq, relative_eq, SquareMatrix};
    use cgmath::num_traits::float::FloatCore;
    use vivotk::pcd::read_pcd_file;
    use super::*;

    fn create_sut(vec: Vec<Vector3<f64>>) -> BoundingBox {
        BoundingBox::from(vec)
    }

    #[test]
    fn test_add() {
        let vec: Vec<Vector3<f64>> = Vec::from(
            [Vector3 {x: 10.0, y: 0.0, z: 0.0 },
                Vector3 {x: 0.0, y: 10.0, z: 0.0 },
                Vector3 {x: 0.0, y: 0.0, z: 10.0 },
                Vector3 {x: -10.0, y: 0.0, z: 0.0 },
                Vector3 {x: 0.0, y: -10.0, z: 0.0 },
                Vector3 {x: 0.0, y: 0.0, z: -10.0 }]
        );
        let b_box = create_sut(vec);
        assert!(relative_eq!(b_box.min.x, -10.0, epsilon = f64::EPSILON));
        assert!(relative_eq!(b_box.min.y, -10.0, epsilon = f64::EPSILON));
        assert!(relative_eq!(b_box.min.z, -10.0, epsilon = f64::EPSILON));
        assert!(relative_eq!(b_box.max.x, 10.0, epsilon = f64::EPSILON));
        assert!(relative_eq!(b_box.max.y, 10.0, epsilon = f64::EPSILON));
        assert!(relative_eq!(b_box.max.z, 10.0, epsilon = f64::EPSILON));
    }

    #[test]
    fn test_contain_point() {
        let vec: Vec<Vector3<f64>> = Vec::from(
            [Vector3 {x: 10.0, y: 0.0, z: 0.0 },
                Vector3 {x: 0.0, y: 10.0, z: 0.0 },
                Vector3 {x: 0.0, y: 0.0, z: 10.0 },
                Vector3 {x: -10.0, y: 0.0, z: 0.0 },
                Vector3 {x: 0.0, y: -10.0, z: 0.0 },
                Vector3 {x: 0.0, y: 0.0, z: -10.0 }]
        );
        let b_box = create_sut(vec);
        let point_inside = Vector3 {x: 0.0, y: 0.0, z: 0.0 };
        let point_outside =  Vector3 {x: 20.0, y: 0.0, z: 0.0 };
        assert_eq!(b_box.contains(point_inside), true);
        assert_eq!(b_box.contains(point_outside), false);
    }

    #[test]
    fn test_from() {
        let pcd = read_pcd_file("test_files/pcd/ascii.pcd");
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut min_z = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        let mut max_z = f64::MIN;
        let mut point_cloud = PointCloud::<PointXyzRgba>::from(pcd.unwrap());
        for point in &point_cloud.points {
            min_x = min_x.min(point.x as f64);
            min_y = min_y.min(point.y as f64);
            min_z = min_z.min(point.z as f64);
            max_x = max_x.max(point.x as f64);
            max_y = max_y.max(point.y as f64);
            max_z = max_z.max(point.z as f64);
        }
        let b_box = BoundingBox::from(point_cloud);
        assert!(relative_eq!(b_box.min.x, min_x, epsilon = f64::EPSILON));
        assert!(relative_eq!(b_box.min.y, min_y, epsilon = f64::EPSILON));
        assert!(relative_eq!(b_box.min.z, min_z, epsilon = f64::EPSILON));
        assert!(relative_eq!(b_box.max.x, max_x, epsilon = f64::EPSILON));
        assert!(relative_eq!(b_box.max.y, max_y, epsilon = f64::EPSILON));
        assert!(relative_eq!(b_box.max.z, max_z, epsilon = f64::EPSILON));
    }

    #[test]
    fn test_diagonalize() {
        // Test matrix (symmetric)
        let matrix = Matrix3::new(
            4.0, 1.0, 1.0,
            1.0, 3.0, 1.0,
            1.0, 1.0, 2.0
        );

        let (q, d) = diagonalize(&matrix);

        let reconstructed = q * d * q.invert().unwrap();

        let reconstructed_d = q.invert().unwrap() * matrix * q;

        for i in 0..3 {
            for j in 0..3 {
                println!("i: {} j: {} val: {}", i, j, reconstructed_d[i][j]);
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                println!("i: {} j: {} val: {}", i, j, d[i][j]);
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix[i][j] - reconstructed[i][j]).abs() < 0.000001, "Reconstructed matrix element at ({}, {}) is incorrect", i, j);
                assert!((d[i][j] - reconstructed_d[i][j]).abs() < 0.000001, "diagonal matrix element at ({}, {}) is incorrect", i, j);
            }
        }
    }
}