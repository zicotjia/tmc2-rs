// Some math stuff for encoding purpose

use std::fmt::Debug;
use cgmath::{Vector3};
use cgmath::num_traits::float::FloatCore;
use serde::{Deserialize, Serialize};
use crate::formats::PointCloud;
// use super::pointxyzrgba::PointXyzRgba;

#[derive(Clone)]
pub struct BoundingBox {
    // min values for x, y, z
    pub min: Vector3<f32>,
    // max values for x, y, z
    pub max: Vector3<f32>,

}

// impl BoundingBox {
//     pub fn contains(&self, point: PointXyzRgba) -> bool {
//         !( point.x < self.min.x || point.x > self.max.x || point.y < self.min.y || point.y> self.max.y ||
//             point.z < self.min.z || point.z > self.max.z)
//     }
//
//     pub fn merge(&self, other: BoundingBox) -> bool {
//         unimplemented!("Box merging not implemented")
//     }
//
//     pub fn add(&mut self, point: PointXyzRgba) {
//         self.min.x = self.min.x.min(point.x);
//         self.min.y = self.min.y.min(point.y);
//         self.min.z = self.min.z.min(point.z);
//         self.max.x = self.max.x.max(point.x);
//         self.max.y = self.max.y.max(point.y);
//         self.max.z = self.max.z.max(point.z);
//     }
//     pub fn intersects(&self, other: BoundingBox) -> bool {
//         unimplemented!("Box intersection not implemented")
//     }
//
//     pub fn fullyContainsBox(&self, other: BoundingBox) -> bool {
//         unimplemented!("Box contains box not implemented")
//     }
//
//     pub fn fullyContainsPoint(&self, point: PointXyzRgba) -> bool {
//         unimplemented!("Box contains point not implemented")
//     }
// }
//
// impl Default for BoundingBox {
//     fn default() -> Self {
//         Self {
//             min: Vector3::new(f32::MAX, f32::MAX, f32::MAX),
//             max: Vector3::new(f32::MIN, f32::MIN, f32::MIN)
//         }
//     }
// }
//
// impl From<PointCloud<PointXyzRgba>> for BoundingBox {
//     fn from(point_cloud: PointCloud<PointXyzRgba>) -> Self {
//         let mut b_box = BoundingBox::default();
//         for point in point_cloud.points { b_box.add(point) };
//         b_box
//     }
// }
//
// impl From<Vec<PointXyzRgba>> for BoundingBox {
//     fn from(point_cloud: Vec<PointXyzRgba>) -> Self {
//         let mut b_box = BoundingBox::default();
//         for point in point_cloud { b_box.add(point) };
//         b_box
//     }
// }
//
// impl Debug for BoundingBox {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         writeln!(f, "BoundingBox {{")?;
//         writeln!(f, "min: {:?}", self.min)?;
//         writeln!(f, "max: {:?}", self.max)?;
//         writeln!(f, "}}")?;
//         Ok(())
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use std::cmp;
//     use cgmath::{assert_ulps_eq, relative_eq};
//     use cgmath::num_traits::float::FloatCore;
//     use crate::pcd::read_pcd_file;
//     use super::*;
//
//     fn create_sut(vec: Vec<PointXyzRgba>) -> BoundingBox {
//         BoundingBox::from(vec)
//     }
//
//     #[test]
//     fn test_add() {
//         let vec: Vec<PointXyzRgba> = Vec::from(
//             [PointXyzRgba {x: 10.0, y: 0.0, z: 0.0, r: 0, g: 0, b: 0, a: 0 },
//                 PointXyzRgba {x: 0.0, y: 10.0, z: 0.0, r: 0, g: 0, b: 0, a: 0 },
//                 PointXyzRgba {x: 0.0, y: 0.0, z: 10.0, r: 0, g: 0, b: 0, a: 0 },
//                 PointXyzRgba {x: -10.0, y: 0.0, z: 0.0, r: 0, g: 0, b: 0, a: 0 },
//                 PointXyzRgba {x: 0.0, y: -10.0, z: 0.0, r: 0, g: 0, b: 0, a: 0 },
//                 PointXyzRgba {x: 0.0, y: 0.0, z: -10.0, r: 0, g: 0, b: 0, a: 0 }]
//         );
//         let b_box = create_sut(vec);
//         assert!(relative_eq!(b_box.min.x, -10.0, epsilon = f32::EPSILON));
//         assert!(relative_eq!(b_box.min.y, -10.0, epsilon = f32::EPSILON));
//         assert!(relative_eq!(b_box.min.z, -10.0, epsilon = f32::EPSILON));
//         assert!(relative_eq!(b_box.max.x, 10.0, epsilon = f32::EPSILON));
//         assert!(relative_eq!(b_box.max.y, 10.0, epsilon = f32::EPSILON));
//         assert!(relative_eq!(b_box.max.z, 10.0, epsilon = f32::EPSILON));
//     }
//
//     #[test]
//     fn test_contain_point() {
//         let vec: Vec<PointXyzRgba> = Vec::from(
//             [PointXyzRgba {x: 10.0, y: 0.0, z: 0.0, r: 0, g: 0, b: 0, a: 0 },
//                 PointXyzRgba {x: 0.0, y: 10.0, z: 0.0, r: 0, g: 0, b: 0, a: 0 },
//                 PointXyzRgba {x: 0.0, y: 0.0, z: 10.0, r: 0, g: 0, b: 0, a: 0 },
//                 PointXyzRgba {x: -10.0, y: 0.0, z: 0.0, r: 0, g: 0, b: 0, a: 0 },
//                 PointXyzRgba {x: 0.0, y: -10.0, z: 0.0, r: 0, g: 0, b: 0, a: 0 },
//                 PointXyzRgba {x: 0.0, y: 0.0, z: -10.0, r: 0, g: 0, b: 0, a: 0 }]
//         );
//         let b_box = create_sut(vec);
//         let point_inside = PointXyzRgba {x: 0.0, y: 0.0, z: 0.0, r: 0, g: 0, b: 0, a: 0 };
//         let point_outside =  PointXyzRgba {x: 20.0, y: 0.0, z: 0.0, r: 0, g: 0, b: 0, a: 0 };
//         assert_eq!(b_box.contains(point_inside), true);
//         assert_eq!(b_box.contains(point_outside), false);
//     }
//
//     #[test]
//     fn test_from() {
//         let pcd = read_pcd_file("test_files/pcd/ascii.pcd");
//         let mut min_x = f32::MAX;
//         let mut min_y = f32::MAX;
//         let mut min_z = f32::MAX;
//         let mut max_x = f32::MIN;
//         let mut max_y = f32::MIN;
//         let mut max_z = f32::MIN;
//         let mut point_cloud = PointCloud::<PointXyzRgba>::from(pcd.unwrap());
//         for point in &point_cloud.points {
//             min_x = min_x.min(point.x);
//             min_y = min_y.min(point.y);
//             min_z = min_z.min(point.z);
//             max_x = max_x.max(point.x);
//             max_y = max_y.max(point.y);
//             max_z = max_z.max(point.z);
//         }
//         let b_box = BoundingBox::from(point_cloud);
//         assert!(relative_eq!(b_box.min.x, min_x, epsilon = f32::EPSILON));
//         assert!(relative_eq!(b_box.min.y, min_y, epsilon = f32::EPSILON));
//         assert!(relative_eq!(b_box.min.z, min_z, epsilon = f32::EPSILON));
//         assert!(relative_eq!(b_box.max.x, max_x, epsilon = f32::EPSILON));
//         assert!(relative_eq!(b_box.max.y, max_y, epsilon = f32::EPSILON));
//         assert!(relative_eq!(b_box.max.z, max_z, epsilon = f32::EPSILON));
//     }
// }