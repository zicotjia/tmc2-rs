use std::fs;
use std::time::Instant;
use env_logger::Env;
use vivotk::formats::PointCloud;
use vivotk::formats::pointxyzrgba::PointXyzRgba;
use vivotk::pcd::{read_pcd_file, read_pcd_header};
use tmc2rs::common::point_set3d::{Point3D, PointSet3};
use tmc2rs::encoder::kd_tree::PCCKdTree;
use tmc2rs::encoder::patch_segmenter::{PatchSegmenter, PatchSegmenterParams};

// For quick testing
fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();
    test_patch_segmenter();
    // test_point_initial_segmentation();
}
fn test_patch_segmenter() {
    let file_path = "../pointClouds/longdress/Pcd";
    let mut paths = fs::read_dir(file_path).unwrap();
    paths.next();
    let sample_pcd_file_path = "./test_files/pcd/longdress_vox10_1051.pcd";
    // let sample_pcd_file_path = "./test_files/pcd/ascii2.pcd";
    // let sample_pcd_file_path = paths.next().unwrap().unwrap().path();
    println!("First path: {:?}", sample_pcd_file_path);

    let start = Instant::now(); // Start timing
    let header = read_pcd_header(sample_pcd_file_path.clone());
    let ptcl = read_pcd_file(sample_pcd_file_path.clone());
    let duration = start.elapsed(); // Stop timing
    println!("Time taken by read_pcd: {:?}", duration);

    let start = Instant::now(); // Start timing
    let ptcl = PointCloud::<PointXyzRgba>::from(ptcl.unwrap());
    let mut point_cloud = PointSet3::from(ptcl.points);
    let duration = start.elapsed(); // Stop timing
    println!("Time taken by point_cloud conversion: {:?}", duration);

    println!("Point cloud size: {}", point_cloud.point_count());

    let start = Instant::now(); // Start timing
    let mut kd_tree = PCCKdTree::new();
    kd_tree.build_from_point_set(&point_cloud);
    let duration = start.elapsed(); // Stop timing
    println!("Time taken to build kd tree : {:?}", duration);
    let patch_segmenter_params = PatchSegmenterParams::new_grid_based();
    // let patch_segmenter_params = PatchSegmenterParams::new();

    let start = Instant::now(); // Start timing
    PatchSegmenter::compute(point_cloud,
                            0,
                            patch_segmenter_params,
                            &mut vec![],
                            &mut 0.0
    );
    let duration = start.elapsed(); // Stop timing
    println!("Time taken compute : {:?}", duration);
}