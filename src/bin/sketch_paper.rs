use std::fs;
use std::time::Instant;
use cgmath::Vector3;
use env_logger::Env;
use vivotk::formats::PointCloud;
use vivotk::formats::pointxyzrgba::PointXyzRgba;
use vivotk::pcd::{read_pcd_file, read_pcd_header};
use tmc2rs::common;
use tmc2rs::common::point_set3d::{Point3D, PointSet3};
use tmc2rs::encoder::constants::orientations::{orientations6, orientations6Count};
use tmc2rs::encoder::kd_tree::PCCKdTree;
use tmc2rs::encoder::normals_generator::{NormalsGenerator3, NormalsGenerator3Parameters, NormalsGeneratorOrientation};
use tmc2rs::encoder::patch_segmenter::{PatchSegmenter, PatchSegmenterParams};
use tmc2rs::encoder::Vector3D;

// For quick testing
fn main() {
    // env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();
    test_patch_segmenter();
    // test_point_initial_segmentation();
}

fn read_pcd_test() {
    let file_path = "../pointClouds/longdress_pcd/Pcd";

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
    let file_path = "../pointClouds/longdress_pcd/Pcd";

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
    let sample_pcd_file_path = "./test_files/pcd/longdress_vox10_1051.pcd";

    println!("First path: {:?}", sample_pcd_file_path);

    let start = Instant::now(); // Start timing

    let header = read_pcd_header(sample_pcd_file_path.clone());
    let ptcl = read_pcd_file(sample_pcd_file_path.clone());

    let start = Instant::now(); // Start timing
    let ptcl = PointCloud::<PointXyzRgba>::from(ptcl.unwrap());
    let mut point_cloud = PointSet3::from(ptcl.points);
    let duration = start.elapsed(); // Stop timing
    println!("Time taken by point_cloud conversion: {:?}", duration);

    let start = Instant::now(); // Start timing
    let mut kd_tree = PCCKdTree::new();
    kd_tree.build_from_point_set(&point_cloud);
    let duration = start.elapsed(); // Stop timing
    println!("Time taken to build kd tree : {:?}", duration);

    let start = Instant::now(); // Start timing
    let param = NormalsGenerator3Parameters {
        view_point: Vector3 {x: 0.0, y: 0.0, z: 0.0},
        radius_normal_smoothing: 0.0,
        radius_normal_estimation: 0.0,
        radius_normal_orientation: 0.0,
        weight_normal_smoothing: 0.0,
        number_of_nearest_neighbors_in_normal_smoothing: 0,
        number_of_nearest_neighbors_in_normal_estimation: point_cloud.point_count(),
        number_of_nearest_neighbors_in_normal_orientation: 0,
        number_of_iterations_in_normal_smoothing: 0,
        orientation_strategy: NormalsGeneratorOrientation::PCC_NORMALS_GENERATOR_ORIENTATION_VIEW_POINT,
        store_eigenvalues: false,
        store_number_of_nearest_neighbors_in_normal_estimation: true,
        store_centroids: false,
    };
    let mut normal_generator = NormalsGenerator3::init(point_cloud.point_count(), &param);
    let normals = normal_generator.compute_normals(&point_cloud, &kd_tree, &param);
    let duration = start.elapsed(); // Stop timing
    println!("Time taken to generate normal: {:?}", duration);

    point_cloud.add_normals(normals);

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

    let start = Instant::now(); // Start timing
    let mut kd_tree = PCCKdTree::new();
    kd_tree.build_from_point_set(&point_cloud);
    let duration = start.elapsed(); // Stop timing
    println!("Time taken to build kd tree : {:?}", duration);

    let start = Instant::now(); // Start timing
    let param = NormalsGenerator3Parameters {
        view_point: Vector3 {x: 0.0, y: 0.0, z: 0.0},
        radius_normal_smoothing: 0.0,
        radius_normal_estimation: 0.0,
        radius_normal_orientation: 0.0,
        weight_normal_smoothing: 0.0,
        number_of_nearest_neighbors_in_normal_smoothing: 0,
        number_of_nearest_neighbors_in_normal_estimation: point_cloud.point_count(),
        number_of_nearest_neighbors_in_normal_orientation: 0,
        number_of_iterations_in_normal_smoothing: 0,
        orientation_strategy: NormalsGeneratorOrientation::PCC_NORMALS_GENERATOR_ORIENTATION_VIEW_POINT,
        store_eigenvalues: false,
        store_number_of_nearest_neighbors_in_normal_estimation: true,
        store_centroids: false,
    };
    let mut normal_generator = NormalsGenerator3::init(point_cloud.point_count(), &param);
    let normals = normal_generator.compute_normals(&point_cloud, &kd_tree, &param);

    point_cloud.add_normals(normals);
    let duration = start.elapsed(); // Stop timing
    println!("Time taken to generate normal: {:?}", duration);

    let start = Instant::now(); // Start timing
    let axis_weights = Vector3D { x: 1.0, y: 1.0, z: 1.0 };
    let partition = PatchSegmenter::initial_segmentation(
        &point_cloud, &orientations6, orientations6Count, &axis_weights
    );
    let duration = start.elapsed(); // Stop timing
    println!("Time taken by initial segmentation: {:?}", duration);


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