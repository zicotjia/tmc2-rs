use std::fs;
use std::time::Instant;
use vivotk::formats::PointCloud;
use vivotk::formats::pointxyzrgba::PointXyzRgba;
use vivotk::pcd::{read_pcd_file, read_pcd_header};
use tmc2rs::common;
use tmc2rs::common::point_set3d::{Point3D, PointSet3};
use tmc2rs::encoder::patch_segmenter::PatchSegmenter;

// For quick testing
fn main() {
    test_point_set_conversion();
}

fn read_pcd_test() {
    let file_path = "../pointClouds/longdress/Pcd";

    let mut paths = fs::read_dir(file_path).unwrap();

    paths.next();

    let sample_pcd_file_path = paths.next().unwrap().unwrap().path();

    println!("First path: {:?}", sample_pcd_file_path);

    let start = Instant::now(); // Start timing

    let header = read_pcd_header(sample_pcd_file_path.clone());
    let ptcl = read_pcd_file(sample_pcd_file_path.clone());

    let duration = start.elapsed(); // Stop timing

    let ptcl = PointCloud::<PointXyzRgba>::from(ptcl.unwrap());

    println!("Time taken by read_pcd: {:?}", duration);

    // let ph = ph.unwrap();
    println!("Pcd header: {:#?}", header);
    println!("Pcd content: {:#?}", ptcl.points.len());
}

fn test_point_set_conversion() {
    let file_path = "../pointClouds/longdress/Pcd";

    let mut paths = fs::read_dir(file_path).unwrap();

    paths.next();

    let sample_pcd_file_path = paths.next().unwrap().unwrap().path();

    println!("First path: {:?}", sample_pcd_file_path);

    let start = Instant::now(); // Start timing

    let header = read_pcd_header(sample_pcd_file_path.clone());
    let ptcl = read_pcd_file(sample_pcd_file_path.clone());

    let duration = start.elapsed(); // Stop timing
    println!("Time taken by read_pcd: {:?}", duration);

    let start = Instant::now(); // Start timing

    let ptcl = PointCloud::<PointXyzRgba>::from(ptcl.unwrap());

    let ptcl_with_normal = tmc2rs::encoder::vvtk_normal_estimation::perform_normal_estimation(&ptcl, 16);

    let point_cloud = PointSet3::from(ptcl_with_normal.points);

    let duration = start.elapsed(); // Stop timing
    println!("Time taken by point_cloud conversion: {:?}", duration);

    // let ph = ph.unwrap();
    println!("number if points: {:#?}", point_cloud.positions.len());
}