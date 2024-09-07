use std::path::PathBuf;
use cgmath::Vector3;

mod encoder_parameters;
mod normals_generator;
pub mod vvtk_normal_estimation;
pub mod patch_segmenter;
pub mod kd_tree;
mod constants;

type Vector3D = Vector3<f64>;