use std::path::PathBuf;
use cgmath::Vector3;

mod encoder_parameters;
pub mod normals_generator;
pub mod patch_segmenter;
pub mod kd_tree;
pub mod constants;
mod encoder;

pub type Vector3D = Vector3<f64>;