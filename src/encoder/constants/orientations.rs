use cgmath::Vector3;
use crate::encoder::Vector3D;

const X_POSITIVE: Vector3D = Vector3D::new(1.0, 0.0, 0.0);
const Y_POSITIVE: Vector3D = Vector3D::new(0.0, 1.0, 0.0);
const Z_POSITIVE: Vector3D = Vector3D::new(0.0, 0.0, 1.0);
const X_NEGATIVE: Vector3D = Vector3D::new(-1.0, 0.0, 0.0);
const Y_NEGATIVE: Vector3D = Vector3D::new(0.0, -1.0, 0.0);
const Z_NEGATIVE: Vector3D = Vector3D::new(0.0, 0.0, -1.0);

// The rest to be added later
pub const orientations6Count: usize = 6;
pub const orientations6: [Vector3D; 6] = [
    X_POSITIVE,
    X_NEGATIVE,
    Y_POSITIVE,
    Y_NEGATIVE,
    Z_POSITIVE,
    Z_NEGATIVE
];

