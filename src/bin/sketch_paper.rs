use std::fs;
use std::time::Instant;
use env_logger::Env;
use vivotk::formats::PointCloud;
use vivotk::formats::pointxyzrgba::PointXyzRgba;
use vivotk::pcd::{read_pcd_file, read_pcd_header};
use tmc2rs::common;
use tmc2rs::common::point_set3d::{Point3D, PointSet3};
use tmc2rs::encoder::constants::orientations::{orientations6, orientations6Count};
use tmc2rs::encoder::kd_tree::PCCKdTree;
use tmc2rs::encoder::patch_segmenter::{PatchSegmenter, PatchSegmenterParams};
use tmc2rs::encoder::Vector3D;

// For quick testing
fn main() {
    test_patch_segmenter();
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

fn test_point_initial_segmentation() {
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

    let start = Instant::now(); // Start timing

    let axis_weights = Vector3D { x: 1.0, y: 1.0, z: 1.0 };
    let partition = PatchSegmenter::initial_segmentation(
        &point_cloud, &orientations6, orientations6Count, &axis_weights
    );

    let duration = start.elapsed(); // Stop timing
    println!("Time taken by initial segmentation: {:?}", duration);


    // let ph = ph.unwrap();
    println!("number if points: {:#?}", partition.len());
}

fn test_build_kd_tree() {
    let file_path = "../pointClouds/longdress/Pcd";

    let mut paths = fs::read_dir(file_path).unwrap();

    paths.next();

    //let sample_pcd_file_path = paths.next().unwrap().unwrap().path();
    let sample_pcd_file_path = "./test_files/pcd/ascii.pcd";

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

    let start = Instant::now(); // Start timing

    let mut kd_tree = PCCKdTree::new();
    kd_tree.build_from_point_set(&point_cloud);


    let duration = start.elapsed(); // Stop timing
    println!("Time taken to build kd tree : {:?}", duration);


    // let ph = ph.unwrap();
    println!("number if points: {:#?}", point_cloud.len());
}

fn test_point_compute_adj_info() {
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

    let start = Instant::now(); // Start timing
    let axis_weights = Vector3D { x: 1.0, y: 1.0, z: 1.0 };
    let partition = PatchSegmenter::initial_segmentation(
        &point_cloud, &orientations6, orientations6Count, &axis_weights
    );
    let duration = start.elapsed(); // Stop timing
    println!("Time taken by initial segmentation: {:?}", duration);

    let start = Instant::now(); // Start timing
    let mut kd_tree = PCCKdTree::new();
    kd_tree.build_from_point_set(&point_cloud);
    let duration = start.elapsed(); // Stop timing
    println!("Time taken to build kd tree : {:?}", duration);

    let start = Instant::now(); // Start timing
    let adj_matrix = PatchSegmenter::compute_adjacency_info(
        &point_cloud,
        &kd_tree,
        3
    );
    let duration = start.elapsed();
    println!("Time taken to compute adj info: {:?}", duration);
    // let ph = ph.unwrap();
    println!("number of points: {:#?}", partition.len());
}

fn test_patch_segmenter() {
    let file_path = "../pointClouds/longdress/Pcd";
    let mut paths = fs::read_dir(file_path).unwrap();
    paths.next();
    let sample_pcd_file_path = "./test_files/pcd/ascii2.pcd";
    // let sample_pcd_file_path = paths.next().unwrap().unwrap().path();
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

    let start = Instant::now(); // Start timing
    let axis_weights = Vector3D { x: 1.0, y: 1.0, z: 1.0 };
    let partition = PatchSegmenter::initial_segmentation(
        &point_cloud, &orientations6, orientations6Count, &axis_weights
    );
    let duration = start.elapsed(); // Stop timing
    println!("Time taken by initial segmentation: {:?}", duration);

    let start = Instant::now(); // Start timing
    let mut kd_tree = PCCKdTree::new();
    kd_tree.build_from_point_set(&point_cloud);
    let duration = start.elapsed(); // Stop timing
    println!("Time taken to build kd tree : {:?}", duration);

    let patch_segmenter_params = PatchSegmenterParams::default();

    let start = Instant::now(); // Start timing
    PatchSegmenter::compute(&point_cloud,
                            0,
                            patch_segmenter_params,
                            &mut vec![],
                            &mut 0.0
    );
    let duration = start.elapsed(); // Stop timing
    println!("Time taken compute : {:?}", duration);
}