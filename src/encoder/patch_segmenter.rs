use std::cmp::{min, PartialEq};
use std::collections::HashMap;
use std::ffi::c_double;
use std::time::Instant;
use cgmath::{InnerSpace, Vector3};
use fast_math::log2;
use kdtree::KdTree;
use log::debug;
use nalgebra::geometry;
use num_traits::{abs, Zero};
use num_traits::real::Real;
use crate::common::{INFINITE_DEPTH, INFINITE_NUMBER, INVALID_PATCH_INDEX};
use crate::common::math::BoundingBox;
use crate::common::point_set3d::{Color3B, Point3D, PointSet3};
use crate::decoder::Patch;
use crate::encoder::Vector3D;
use crate::encoder::constants::orientations::{orientations6, orientations6Count};
use crate::encoder::kd_tree::{NNResult, PCCKdTree};
use rayon::prelude::*;
use crate::encoder::normals_generator::{NormalsGenerator3, NormalsGenerator3Parameters, NormalsGeneratorOrientation};

type Voxels = HashMap<u64, Vec<usize>>;

pub struct PatchSegmenterParams {
    pub grid_based_segmentation: bool,
    pub voxel_dimension_grid_based_segmentation: usize,
    pub nn_normal_estimation: usize,
    pub normal_orientation: usize,
    pub grid_based_refine_segmentation: bool,
    pub max_nn_count_refine_segmentation: usize,
    pub iteration_count_refine_segmentation: usize,
    pub voxel_dimension_refine_segmentation: usize,
    pub search_radius_refine_segmentation: usize,
    pub occupancy_resolution: usize,
    pub enable_patch_splitting: bool,
    pub max_patch_size: usize,
    pub quantizer_size_x: usize,
    pub quantizer_size_y: usize,
    pub min_point_count_per_cc_patch_segmentation: usize,
    pub max_nn_count_patch_segmentation: usize,
    pub surface_thickness: usize,
    pub eom_fix_bit_count: usize,
    pub eom_single_layer_mode: bool,
    pub map_count_minus_1: usize,
    pub min_level: usize,
    pub max_allowed_depth: usize,
    pub max_allowed_dist_2_raw_points_detection: f64,
    pub max_allowed_dist_2_raw_points_selection: f64,
    pub lambda_refine_segmentation: f64,
    pub use_enhanced_occupancy_map_code: bool,
    pub absolute_d1: bool,
    pub create_sub_point_cloud: bool,
    pub surface_separation: bool,
    pub weight_normal: Vector3D,
    pub additional_projection_plane_mode: usize,
    pub partial_additional_projection_plane: f64,
    pub geometry_bit_depth_2d: usize,
    pub geometry_bit_depth_3d: usize,
    pub patch_expansion: bool,
    pub high_gradient_separation: bool,
    pub min_gradient: f64,
    pub min_num_high_gradient_points: usize,
    pub enable_point_cloud_partitioning: bool,
    pub roi_bounding_box_min_x: Vec<i32>,
    pub roi_bounding_box_max_x: Vec<i32>,
    pub roi_bounding_box_min_y: Vec<i32>,
    pub roi_bounding_box_max_y: Vec<i32>,
    pub roi_bounding_box_min_z: Vec<i32>,
    pub roi_bounding_box_max_z: Vec<i32>,
    pub num_tiles_hor: i32,
    pub tile_height_to_width_ratio: f64,
    pub num_cuts_along_1st_longest_axis: i32,
    pub num_cuts_along_2nd_longest_axis: i32,
    pub num_cuts_along_3rd_longest_axis: i32,
}

impl PatchSegmenterParams {
    pub fn new() -> Self {
        PatchSegmenterParams {
            grid_based_segmentation: false,
            voxel_dimension_grid_based_segmentation: 2,
            nn_normal_estimation: 16,
            normal_orientation: 1,
            grid_based_refine_segmentation: false,
            max_nn_count_refine_segmentation: 256,
            iteration_count_refine_segmentation: 100,
            voxel_dimension_refine_segmentation: 4,
            search_radius_refine_segmentation: 192,
            occupancy_resolution: 16,
            enable_patch_splitting: false,
            max_patch_size: 1024,
            quantizer_size_x: 1 << 4,
            quantizer_size_y: 1 << 4,
            min_point_count_per_cc_patch_segmentation: 16,
            max_nn_count_patch_segmentation: 16,
            surface_thickness: 4,
            eom_fix_bit_count: 0,
            eom_single_layer_mode: false,
            map_count_minus_1: 1,
            min_level: 64,
            max_allowed_depth: 0,
            max_allowed_dist_2_raw_points_detection: 9.0,
            max_allowed_dist_2_raw_points_selection: 9.0,
            lambda_refine_segmentation: 3.0,
            use_enhanced_occupancy_map_code: false,
            absolute_d1: false,
            create_sub_point_cloud: false,
            surface_separation: false,
            weight_normal: Vector3 { x: 1.0, y: 1.0, z: 1.0 },
            additional_projection_plane_mode: 0,
            partial_additional_projection_plane: 0.0,
            geometry_bit_depth_2d: 8,
            geometry_bit_depth_3d: 10,
            patch_expansion: false,
            high_gradient_separation: false,
            min_gradient: 15.0,
            min_num_high_gradient_points: 256,
            enable_point_cloud_partitioning: false,
            roi_bounding_box_min_x: vec![],
            roi_bounding_box_max_x: vec![],
            roi_bounding_box_min_y: vec![],
            roi_bounding_box_max_y: vec![],
            roi_bounding_box_min_z: vec![],
            roi_bounding_box_max_z: vec![],
            num_tiles_hor: 2,
            tile_height_to_width_ratio: 1.0,
            num_cuts_along_1st_longest_axis: 0,
            num_cuts_along_2nd_longest_axis: 0,
            num_cuts_along_3rd_longest_axis: 0,
        }
    }
    pub fn new_grid_based() -> Self {
        PatchSegmenterParams {
            grid_based_segmentation: true,
            voxel_dimension_grid_based_segmentation: 2,
            nn_normal_estimation: 16,
            normal_orientation: 1,
            grid_based_refine_segmentation: true,
            max_nn_count_refine_segmentation: 384,
            iteration_count_refine_segmentation: 5,
            voxel_dimension_refine_segmentation: 2,
            search_radius_refine_segmentation: 128,
            occupancy_resolution: 1,
            enable_patch_splitting: false,
            max_patch_size: 1024,
            quantizer_size_x: 1 << 4,
            quantizer_size_y: 1 << 4,
            min_point_count_per_cc_patch_segmentation: 16,
            max_nn_count_patch_segmentation: 16,
            surface_thickness: 4,
            eom_fix_bit_count: 0,
            eom_single_layer_mode: false,
            map_count_minus_1: 1,
            min_level: 64,
            max_allowed_depth: 0,
            max_allowed_dist_2_raw_points_detection: 9.0,
            max_allowed_dist_2_raw_points_selection: 9.0,
            lambda_refine_segmentation: 3.0,
            use_enhanced_occupancy_map_code: false,
            absolute_d1: false,
            create_sub_point_cloud: false,
            surface_separation: false,
            weight_normal: Vector3 { x: 1.0, y: 1.0, z: 1.0 },
            additional_projection_plane_mode: 0,
            partial_additional_projection_plane: 0.0,
            geometry_bit_depth_2d: 8,
            geometry_bit_depth_3d: 10,
            patch_expansion: false,
            high_gradient_separation: false,
            min_gradient: 15.0,
            min_num_high_gradient_points: 256,
            enable_point_cloud_partitioning: false,
            roi_bounding_box_min_x: vec![],
            roi_bounding_box_max_x: vec![],
            roi_bounding_box_min_y: vec![],
            roi_bounding_box_max_y: vec![],
            roi_bounding_box_min_z: vec![],
            roi_bounding_box_max_z: vec![],
            num_tiles_hor: 2,
            tile_height_to_width_ratio: 1.0,
            num_cuts_along_1st_longest_axis: 0,
            num_cuts_along_2nd_longest_axis: 0,
            num_cuts_along_3rd_longest_axis: 0,
        }
    }
}
impl Default for PatchSegmenterParams {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Default)]
pub struct PatchSegmenter {
    pub nb_thread: usize,
    pub box_min_depths: Vec<Patch>,
    pub box_max_depths: Vec<Patch>
}
impl PatchSegmenter {
    pub fn compute(
        mut geometry: PointSet3,
        frame_index: usize,
        params: PatchSegmenterParams,
        sub_point_cloud: &mut Vec<PointSet3>,
        distance_src_rec: &mut f32
    ) {
        // Determine Orientation
        // ZICO: For now assume only the base 6 projection plane
        let orientations = orientations6;
        let orientations_count = orientations6Count;
        let mut geometry_vox;
        let voxels;

        // ZICO: For now assumes gridBasedSegmentation is off
        if params.grid_based_segmentation {
            println!("Convert points to voxels");
            (geometry_vox, voxels) = Self::convert_points_to_voxels(&geometry, params.geometry_bit_depth_3d, params.voxel_dimension_grid_based_segmentation);
            println!("Done");
        } else {
            voxels = Voxels::new();
            geometry_vox = geometry.clone();
        }

        // Build KDTree
        println!("Computing normals for original point cloud... ");
        let mut kd_tree = PCCKdTree::new();
        kd_tree.build_from_point_set(&geometry_vox);

        if !geometry_vox.with_normals {
            let normal_gen_params = NormalsGenerator3Parameters {
                view_point: Vector3::zero(),
                radius_normal_smoothing: f64::MAX,
                radius_normal_estimation: f64::MAX,
                radius_normal_orientation: f64::MAX,
                weight_normal_smoothing: f64::MAX,
                number_of_nearest_neighbors_in_normal_smoothing: params.nn_normal_estimation,
                number_of_nearest_neighbors_in_normal_estimation: params.nn_normal_estimation,
                number_of_nearest_neighbors_in_normal_orientation: params.nn_normal_estimation,
                number_of_iterations_in_normal_smoothing: 0,
                orientation_strategy: NormalsGeneratorOrientation::PCC_NORMALS_GENERATOR_ORIENTATION_NONE,// params.nn_normal_estimation
                store_eigenvalues: false,
                store_number_of_nearest_neighbors_in_normal_estimation: false,
                store_centroids: false,
            };
            let normal_generator = NormalsGenerator3::init(geometry_vox.point_count(), &normal_gen_params);
            let normals = normal_generator.compute_normals(&geometry_vox, &kd_tree, &normal_gen_params);
            geometry_vox.add_normals(normals);
        }
        println!("Done");
        assert!(geometry_vox.with_normals);

        println!("Computing initial segmentation...");
        let mut partition: Vec<usize>;
        // Initial Segmentation
        if params.additional_projection_plane_mode == 0 {
            partition = Self::initial_segmentation(&geometry_vox, &orientations, orientations_count, &params.weight_normal)
        } else {
            unimplemented!("initialSegmentation for additional projection plane")
        }
        println!("Done");

        // ZICO: For now lets skip all segmentation refinement
        if params.grid_based_refine_segmentation {
            println!("Refining segmentation (Grid-Based)");
            Self::refine_segmentation_grid_based(
                &geometry_vox, &orientations, orientations_count, params.max_nn_count_refine_segmentation,
                params.lambda_refine_segmentation, params.iteration_count_refine_segmentation,
                params.voxel_dimension_refine_segmentation, params.search_radius_refine_segmentation, &mut partition
            )
        } else {
            println!("Refining segmentation");
            Self::refine_segmentation(
                &geometry_vox, &kd_tree, &orientations, orientations_count, params.max_nn_count_refine_segmentation,
                params.lambda_refine_segmentation, params.iteration_count_refine_segmentation, &mut partition
            )
        }
        println!("Done");

        if params.grid_based_segmentation {
            println!("Applying Voxels data to point");
            Self::apply_voxels_data_to_points(geometry.point_count(), params.geometry_bit_depth_3d,
                                              params.voxel_dimension_grid_based_segmentation, &voxels, &mut geometry_vox, &mut partition);
            println!("Done");
            kd_tree.build_from_point_set(&geometry);
        }

        geometry.normals = geometry_vox.normals;
        // ZICO: Implement Segment Patches
        println!("Patch Segmentation");
        let patches = Self::segment_patches(
            &geometry, &kd_tree, &params, &mut partition, sub_point_cloud, distance_src_rec,
            &orientations, orientations_count
        );
        println!("Done");

    }

    pub fn convert_points_to_voxels(
        source: &PointSet3,
        geo_bits: usize, // 10
        vox_dim: usize // 2
    ) -> (PointSet3, Voxels) {
        let geo_bits2 = geo_bits << 1;
        let mut vox_dim_shift = 0;
        let mut i = vox_dim;
        while i > 1  {
            vox_dim_shift += 1;
            i >>= 1;
        }
        let vox_dim_half = vox_dim >> 1;
        let mut voxel_point_set = PointSet3::default();
        voxel_point_set.reserve(source.len());
        let mut voxels = Voxels::new();

        let sub_to_ind = |x: u64, y: u64, z: u64| -> u64 {
            x + (y << geo_bits) + (z << geo_bits2)
        };

        // C++ version params is usize, but in this code they cast to uint64 ??
        for (i, point) in source.positions.iter().enumerate() {
            let x0 = ((point.x as usize) + vox_dim_half) >> vox_dim_shift;
            let y0  = ((point.y as usize) + vox_dim_half) >> vox_dim_shift;
            let z0 = ((point.z as usize) + vox_dim_half) >> vox_dim_shift;
            // Associate each points with a voxel
            let p = sub_to_ind(x0 as u64, y0 as u64, z0 as u64);
            if !voxels.contains_key(&p) {
                voxels.entry(p).or_insert_with(|| Vec::with_capacity(vox_dim * vox_dim << 1));
                let j = voxel_point_set.add_point(Vector3 { x: x0 as i16, y: y0 as i16, z: z0 as i16 });
                if source.with_colors {
                    voxel_point_set.add_colors();
                    voxel_point_set.colors[j] = source.colors[i]
                }
            };
            // Definitely will exist
            voxels.get_mut(&p).unwrap().push(i);
        }
        (voxel_point_set, voxels)
    }

    pub fn apply_voxels_data_to_points(
        point_count: usize,
        geo_bits: usize,
        vox_dim: usize,
        voxels: &Voxels,
        source_vox: &mut PointSet3,
        partitions: &mut Vec<usize>
    ) {
        let mut partitions_tmp = vec![0; point_count];
        let mut normals_tmp = vec![Vector3D::zero(); point_count];

        let normals = &source_vox.normals;

        let geo_bits2 = geo_bits << 1;
        let mut vox_dim_shift = 0;
        let mut i = vox_dim;
        while i > 1 {
            vox_dim_shift += 1;
            i >>= 1;
        }
        let sub_to_ind = |point: Point3D| -> u64 {
            point[0] as u64 + ((point[1] as u64) << geo_bits) + ((point[2] as u64) << geo_bits2)
        };

        for i in 0..source_vox.len() {
            let partition = partitions[i];
            let normal = &normals[i];
            let pos = &source_vox.positions[i];
            let indices = voxels.get(&sub_to_ind(*pos));
            if let Some(indices) = indices {
                for &index in indices {
                    partitions_tmp[index] = partition;
                    normals_tmp[index] = *normal;
                }
            }
        }

        // Swap partitions and normals
        *partitions = partitions_tmp;
        source_vox.normals = normals_tmp;
    }

    // Segment into partitions based on the normal
    pub fn initial_segmentation(
        geometry: &PointSet3,
        orientations: &[Vector3D; 6], //Hardcoded to 6 for now
        orientations_count: usize,
        axis_weight: &Vector3D
    ) -> Vec<usize> {
        let mut partition = Vec::new();
        let point_count = geometry.point_count();
        partition.reserve(point_count);
        let mut weight_value: [f64; 18] = [1.0; 18];
        weight_value[0] = axis_weight[0];
        weight_value[1] = axis_weight[1];
        weight_value[2] = axis_weight[2];
        weight_value[3] = axis_weight[0];
        weight_value[4] = axis_weight[1];
        weight_value[5] = axis_weight[2];

        #[cfg(feature = "use_rayon")]
        {
            use rayon::prelude::*;
            partition.resize(point_count, 0);
            partition
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, part)| {
                    let normal = geometry.normals[i];
                    let mut cluster_index = 0;
                    let mut best_score = normal.dot(orientations[0]);
                    for j in 1..orientations_count {
                        let score = normal.dot(orientations[j]) * weight_value[j];
                        if score > best_score {
                            best_score = score;
                            cluster_index = j;
                        }
                    }
                    *part = cluster_index;
            });
        }

        // let mut m: HashMap<i32, usize> = HashMap::new();
        // for i in &partition {
        //     *m.entry(*i as i32).or_default() += 1;
        // }
        // for i in m.into_iter() {
        //     println!("Freq of {} is {}", i.0, i.1);
        // }

        // ZICO: Can be made functional
        #[cfg(not(feature = "use_rayon"))]
        {
            for i in 0..point_count {
                let normal = geometry.normals[i];
                let mut cluster_index = 0;
                let mut best_score = normal.dot(orientations[0]);
                for j in 1..orientations_count {
                    let score = normal.dot(orientations[j]) * weight_value[j];
                    if score > best_score {
                        best_score = score;
                        cluster_index = j;
                    }
                }
                partition.push(cluster_index);
            }
        }
        partition
    }

    pub fn refine_segmentation_grid_based(
        points: &PointSet3,
        orientations: &[Vector3D; 6],
        orientation_count: usize,
        max_nn_count: usize,
        lambda: f64,
        iteration_count: usize,
        vox_dim: usize,
        search_radius: usize,
        partition: &mut Vec<usize>
    ) {
        let point_count = points.point_count();
        let mut geo_max = points.positions[0][0];

        for i in 0..point_count {
            let pos = &points.positions[i];
            geo_max = geo_max.max(pos[0]).max(pos[1]).max(pos[2]);
        }

        let mut geo_range = 1;
        let mut i = geo_max - 1;
        while i != 0i16 {
            geo_range <<= 1;
            i >>= 1;
        }

        let mut vox_dim_shift = 0;
        let mut i = vox_dim;
        while i > 1 {
            vox_dim_shift += 1;
            i >>= 1;
        }

        let grid_dim = geo_range >> vox_dim_shift;
        let mut grid_dim_shift = 0;
        i = grid_dim;
        while i > 1 {
            grid_dim_shift += 1;
            i >>= 1;
        }
        let grid_dim_shift_sqr = grid_dim_shift << 1;
        let vox_dim_half = vox_dim >> 1;

        let sub_to_ind = |x: usize, y: usize, z: usize| -> usize {
            (x + (y << grid_dim_shift) + (z << grid_dim_shift_sqr)) as usize
        };

        let mut grid_centers = PointSet3::default();
        let mut grid_with_point_indices: HashMap<usize, PointIndicesOfGridCell> = HashMap::new();
        let mut grid_with_attributes: HashMap<usize, AttributeOfGridCell> = HashMap::new();

        let MAX_NUM_POINTS_IN_ONE_VOX: u32 = (vox_dim * vox_dim * vox_dim) as u32;

        for i in 0..point_count {
            let pos = &points.positions[i];
            let x0 = ((pos[0] as usize + vox_dim_half) >> vox_dim_shift);
            let y0 = ((pos[1] as usize + vox_dim_half) >> vox_dim_shift);
            let z0 = ((pos[2] as usize + vox_dim_half) >> vox_dim_shift);

            let p = sub_to_ind(x0, y0, z0);

            if !grid_with_point_indices.contains_key(&p) {
                grid_centers.add_point(Point3D::new(x0 as i16, y0 as i16, z0 as i16));
                grid_with_point_indices.insert(p, PointIndicesOfGridCell::with_capacity(MAX_NUM_POINTS_IN_ONE_VOX as usize));
                grid_with_attributes.insert(p, AttributeOfGridCell::with_orientation_count(orientation_count));
            }
            grid_with_point_indices.get_mut(&p).unwrap().add_point_index(i as u32);
        }

        let total_num_of_vox = grid_centers.point_count();

        // Pre-processing steps
        let mut point_indices_of_vox = Vec::with_capacity(total_num_of_vox);
        let mut attribute_of_vox = Vec::with_capacity(total_num_of_vox);

        for i in 0..total_num_of_vox {
            let p = &grid_centers.positions[i];
            let mut p_pi = grid_with_point_indices.remove(&sub_to_ind(p.x as usize, p.y as usize, p.z as usize)).unwrap();
            let mut p_attr = grid_with_attributes.remove(&sub_to_ind(p.x as usize, p.y as usize, p.z as usize)).unwrap();

            p_attr.set_num_of_total_points(p_pi.get_point_count() as u8);

            // 1st voxel classification
            p_attr.update_scores(p_pi.get_point_indices(), partition);

            point_indices_of_vox.push(p_pi);
            attribute_of_vox.push(p_attr);
        }

        // Search adjacent voxels within the search radius
        let mut kd_tree = PCCKdTree::new();
        kd_tree.build_from_point_set(&grid_centers);
        let vox_search_radius = search_radius >> vox_dim_shift;
        const MAX_NEIGHBOR_COUNT: usize = std::u16::MAX as usize;

        let mut adj = compute_adjacency_info_in_radius(&grid_centers, &kd_tree, MAX_NEIGHBOR_COUNT, vox_search_radius);

        // Candidates for the indirect edge voxels
        let mut adj_dev: Vec<Vec<u32>> = vec![Vec::new(); total_num_of_vox];
        let idv_search_range = if vox_dim >= 4 { 1 } else { 2 };

        // Pre-processing steps
        let mut weights: Vec<f64> = Vec::with_capacity(total_num_of_vox);

        for i in 0..total_num_of_vox {
            let p = grid_centers.positions[i];
            adj_dev[i].reserve(128);

            let mut nn_point_count = 0;
            let mut current_adj_of_i = &mut adj[i];
            let mut iter = current_adj_of_i.iter();

            while let Some(&neighbor) = iter.next() {
                // 2nd voxel classification
                let q = grid_centers.positions[neighbor];
                let x_abs = (p.x - q.x).abs();
                let y_abs = (p.y - q.y).abs();
                let z_abs = (p.z - q.z).abs();
                if x_abs <= idv_search_range && y_abs <= idv_search_range && z_abs <= idv_search_range {
                    adj_dev[i].push(neighbor as u32);
                }
                nn_point_count += point_indices_of_vox[neighbor].get_point_count();
                if nn_point_count >= max_nn_count {
                    break;
                }
            }

            // Pre-computing weights
            weights.push(lambda / nn_point_count as f64);

            // Remove points from the adjacent list if there are more than max_nn_count
            if iter.len() > 0 { current_adj_of_i.truncate(nn_point_count); }
        }

        let mut scores = vec![0.0; orientation_count];
        let mut score_smooth = vec![0; orientation_count];

        let mut iter = 0;
        loop {
            for i in 0..total_num_of_vox {
                // If the current voxel belongs to N-EV (No edge-voxel), then refining steps are skipped.
                let edge_of_i = attribute_of_vox[i].get_edge();
                if edge_of_i == VoxEdge::NoEdge { continue; }
                score_smooth.fill(0);

                let current_adj_of_i = &adj[i];
                for &j in current_adj_of_i {
                    let score_smooth_of_adj = attribute_of_vox[j].get_score_smooth();
                    for k in 0..orientation_count { score_smooth[k] += score_smooth_of_adj[k]; }
                }

                // 2nd voxel classification (indirect edge-voxel)
                let mut ppi_of_score_smooth: usize = 0;
                let mut max_score: u16 = 0;
                for (index, score) in score_smooth.iter().enumerate() {
                    if *score > max_score {
                        ppi_of_score_smooth = index;
                        max_score = *score;
                    }
                }
                for &j in &adj_dev[i] {
                    let j = j as usize;
                    let edge_of_adj = attribute_of_vox[j].get_edge();
                    let ppi = attribute_of_vox[j].get_ppi() as usize;
                    if edge_of_adj == VoxEdge::NoEdge && ppi != ppi_of_score_smooth { attribute_of_vox[j].update_edge(VoxEdge::IndirectEdge); }
                }

                if edge_of_i != VoxEdge::MDirectEdge {
                    let valid_num_of_scores = orientation_count - score_smooth.iter().filter(|&&x| x == 0).count();
                    let vox_ppi = attribute_of_vox[i].get_ppi() as usize;
                    if valid_num_of_scores == 1 && score_smooth[vox_ppi] > 0 { continue; }
                }

                let p_i = point_indices_of_vox[i].get_point_indices();
                for &j in p_i {
                    let j = j as usize;
                    let normal = points.normals[j];
                    for k in 0..orientation_count {
                        scores[k] = normal.dot(orientations[k]) + weights[i] * score_smooth[k] as f64;
                    }
                    let max_index: usize = 0;
                    let mut max_score: f64 = f64::MIN;
                    for (index, score) in scores.iter().enumerate() {
                        if *score > max_score {
                            ppi_of_score_smooth = index;
                            max_score = *score;
                        }
                    }
                    partition[j as usize] = max_index
                }
                attribute_of_vox[i].set_updated_flag();
            }

            // Restart the values of score smooth
            for i in 0..total_num_of_vox {
                attribute_of_vox[i].update_scores(point_indices_of_vox[i].get_point_indices(), partition);
            }
            iter += 1;

            // Check the iteration condition to break the loop
            if iter >= iteration_count {
                break;
            }
        }
    }

    pub fn refine_segmentation(
        points: &PointSet3,
        kd_tree: &PCCKdTree,
        orientations: &[Vector3D; 6],
        orientations_count: usize,
        max_nn_count: usize,
        lambda: f64,
        iteration_count: usize,
        partition: &mut Vec<usize>
    ) {
        let adj = Self::compute_adjacency_info(points, kd_tree, max_nn_count);
        let point_count = points.point_count();
        let weight = lambda / max_nn_count as f64;
        let mut temp_partition: Vec<usize> = vec![0; point_count];

        #[cfg(feature = "use_rayon")]
        {
            use rayon::prelude::*;

            for _k in 0..iteration_count {
                let temp_scores_smooth: Vec<Vec<usize>> = (0..point_count)
                    .into_par_iter()
                    .map(|i| {
                        let mut score_smooth = vec![0; orientations_count];
                        for &neighbor in &adj[i] {
                            score_smooth[partition[neighbor]] += 1;
                        }
                        score_smooth
                    })
                    .collect();

                temp_partition
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, cluster_index)| {
                        let normal = points.normals[i];
                        let mut best_cluster_index = *cluster_index;
                        let mut best_score = 0.0;
                        let score_smooth = &temp_scores_smooth[i]; // Immutable access

                        for j in 0..orientations_count {
                            let score_normal = normal.dot(orientations[j]); // assuming dot product method
                            let score = score_normal + weight * score_smooth[j] as f64;

                            if score > best_score {
                                best_score = score;
                                best_cluster_index = j;
                            }
                        }

                        *cluster_index = best_cluster_index;
                    });

                // 3. Swap partitions
                std::mem::swap(partition, &mut temp_partition);
            }
        }
        
        #[cfg(not(feature = "use_rayon"))]
        {
            let mut scores_smooth: Vec<Vec<usize>> = vec![vec![0; orientations_count]; point_count];
            for _k in 0..iteration_count {
                // Sequential section to process scoresSmooth
                for i in 0..point_count {
                    let score_smooth = &mut scores_smooth[i];
                    score_smooth.fill(0);
                    for &neighbor in &adj[i] {
                        score_smooth[partition[neighbor]] += 1;
                    }
                }

                // Sequential section to update partition based on normals and scores
                for i in 0..point_count {
                    let normal = points.normals[i];
                    let mut cluster_index = partition[i];
                    let mut best_score = 0.0;
                    let score_smooth = &scores_smooth[i];

                    for j in 0..orientations_count {
                        let score_normal = normal.dot(orientations[j]); // assuming dot product method
                        let score = score_normal + weight * score_smooth[j] as f64;

                        if score > best_score {
                            best_score = score;
                            cluster_index = j;
                        }
                    }

                    temp_partition[i] = cluster_index;
                }

                // Swap partitions
                std::mem::swap(partition, &mut temp_partition);
            }

            // Clear scoresSmooth vectors
            for vector in &mut scores_smooth {
                vector.clear();
            }
        }
    }

    pub fn segment_patches(
        points: &PointSet3,
        kd_tree: &PCCKdTree,
        params: &PatchSegmenterParams,
        partition: &mut Vec<usize>,
        sub_point_cloud: &mut Vec<PointSet3>,
        distance_source_rec: &mut f32,
        orientations: &[Vector3D; 6],
        orientation_count: usize
    ) -> Vec<Patch>  {
        // ZICO: C++ version pass alot of stuff from the frame context
        // Might have to do that here too but for now lets create then return
        let mut patches: Vec<Patch> = Vec::with_capacity(256);
        let mut resampled = PointSet3::default();
        let mut patch_partition: Vec<usize> = vec![];
        let mut resampled_patch_partition: Vec<usize> = vec![];
        let mut raw_points: Vec<usize> = vec![];

        let max_nn_count = params.max_nn_count_patch_segmentation;
        let min_point_count_per_cc = params.min_point_count_per_cc_patch_segmentation;
        let occupancy_resolution = params.occupancy_resolution;
        let quantizer_size_x = params.quantizer_size_x;
        let quantizer_size_y = params.quantizer_size_y;
        let max_allowed_dist2_raw_points_detection = params.max_allowed_dist_2_raw_points_detection;
        let max_allowed_dist2_raw_points_selection = params.max_allowed_dist_2_raw_points_selection;
        let eom_single_layer_mode = params.eom_single_layer_mode;
        let eom_fix_bit_count = params.eom_fix_bit_count;
        let surface_thickness = params.surface_thickness;
        let max_allowed_depth = params.max_allowed_depth;
        let min_level = params.min_level;
        let use_enhanced_occupancy_map_code = params.use_enhanced_occupancy_map_code;
        let create_sub_point_cloud = params.create_sub_point_cloud;
        let absolute_d1 = params.absolute_d1;
        let use_surface_separation = params.surface_separation;
        // let additional_projection_axis = params.additional_projection_plane_mode;
        let geometry_bit_depth_2d = params.geometry_bit_depth_2d;
        let geometry_bit_depth_3d = params.geometry_bit_depth_3d;
        let patch_expansion_enabled = params.patch_expansion;
        let high_gradient_separation = params.high_gradient_separation;
        // let min_gradient = params.min_gradient;
        // let min_num_high_gradient_points = params.min_num_high_gradient_points;
        let enable_point_cloud_partitioning = params.enable_point_cloud_partitioning;

        // let mut roi_bounding_box_min_x = params.roi_bounding_box_min_x.clone();
        // let mut roi_bounding_box_max_x = params.roi_bounding_box_max_x.clone();
        // let mut roi_bounding_box_min_y = params.roi_bounding_box_min_y.clone();
        // let mut roi_bounding_box_max_y = params.roi_bounding_box_max_y.clone();
        // let mut roi_bounding_box_min_z = params.roi_bounding_box_min_z.clone();
        // let mut roi_bounding_box_max_z = params.roi_bounding_box_max_z.clone();
        //
        // let mut num_cuts_along_1st_longest_axis = params.num_cuts_along_1st_longest_axis;
        // let mut num_cuts_along_2nd_longest_axis = params.num_cuts_along_2nd_longest_axis;
        // let mut num_cuts_along_3rd_longest_axis = params.num_cuts_along_3rd_longest_axis;

        let point_count = points.point_count();

        patch_partition.resize(point_count, 0);
        resampled_patch_partition.reserve(point_count);
        let mut frame_pcc_color: Vec<Color3B> = Vec::new();
        frame_pcc_color.reserve(point_count);
        assert!(&points.with_colors);
        // Add colors
        points.colors.iter().for_each(|color| frame_pcc_color.push(color.clone()));


        let mut num_d0_points = 0;
        let mut num_d1_points = 0;

        // Compute adjacency info
        let mut adj: Vec<Vec<usize>> = Vec::new();                     // Adjacency list
        // let mut adj_dist: Vec<Vec<f64>> = Vec::new();                  // Adjacency distance matrix
        // let mut flag_exp: Vec<bool> = Vec::new();                      // Flag for expansion (Boolean flags)
        // let mut num_rois: i32 = 0;                                     // Number of Regions of Interest (ROIs)
        // let mut num_chunks: i32 = 0;                                   // Number of chunks (or segments)
        // let mut points_chunks: Vec<PointSet3> = Vec::new();         // Vector of point cloud chunks
        // let mut points_index_chunks: Vec<Vec<usize>> = Vec::new();     // Vector of point indices for each chunk
        // let mut point_count_chunks: Vec<usize> = Vec::new();           // Vector of point counts for each chunk
        // let mut kdtree_chunks: Vec<PCCKdTree> = Vec::new();            // Vector of KD-Trees for each chunk
        // let mut bounding_box_chunks: Vec<BoundingBox> = Vec::new();       // Vector of bounding boxes for each chunk
        // let mut adj_chunks: Vec<Vec<Vec<usize>>> = Vec::new();         // Adjacency list for each chunk

        // ZICO: for now patch_expansion and partitioning is disabled
        // Maybe implement partitioning cause its more memory efficient
        if patch_expansion_enabled {
            unimplemented!("Compute Adjacency Info Dist");
        } else {
            if enable_point_cloud_partitioning {
                unimplemented!("Point Cloud Partitioning");
            } else {
                let adj_start = Instant::now();
                adj = Self::compute_adjacency_info(
                    &points,
                    &kd_tree,
                    max_nn_count
                );
                let adj_duration = adj_start.elapsed();
                debug!("Time taken to compute adj info: {:?}", adj_duration)
            }
        }
        // Extract Patches
        let mut raw_points_distance: Vec<f64> = vec![f64::MAX; point_count];
        raw_points = (0..point_count).collect();

        // let raw_points_chunks: Vec<Vec<usize>>;
        // let raw_points_distance_chunks: Vec<Vec<f64>>;
        if enable_point_cloud_partitioning {
            unimplemented!("implement point cloud partitioning")
        }

        sub_point_cloud.clear();
        // ZICO: What is A, B here?
        let mut mean_pab: f64 = 0.0;
        let mut mean_yab: f64 = 0.0;
        let mut mean_uab: f64 = 0.0;
        let mut mean_vab: f64 = 0.0;
        let mut mean_pba: f64 = 0.0;
        let mut mean_yba: f64 = 0.0;
        let mut mean_uba: f64 = 0.0;
        let mut mean_vba: f64 = 0.0;
        let test_source_num: usize = 0;
        let test_reconstructed_num: usize = 0;
        let mut number_of_eom: usize = 0;

        // Must go through all raw points
        // Maybe made into function


        // pub fn process_raw_points(
        //     raw_points: &Vec<f64>,
        //
        // )
        /// Process Raw Points
        ///
        while !raw_points.is_empty() {
            // This algo go through all raw points. Group them into connected components
            // until no more raw points

            /// Generate Connected Components
            ///
            let cc_start = Instant::now(); // Start timing
            let mut connected_components: Vec<Vec<usize>> = Vec::new();
            if !enable_point_cloud_partitioning {
                let mut fifo: Vec<usize> = Vec::with_capacity(point_count);

                let mut flags: Vec<bool> = vec![false; point_count];
                for i in &raw_points { flags[*i] = true }

                connected_components.reserve(256);

                // ZICO: Must we go through all points for each connected components?
                println!("num raw points: {}", raw_points.len());
                for index in &raw_points {
                    let index = *index;

                    if flags[index] && raw_points_distance[index] > max_allowed_dist2_raw_points_detection {
                        flags[index] = false;
                        let connected_components_index = connected_components.len();
                        // There are 6 partitions for now
                        let cluster_index = partition[index];
                        connected_components.resize(connected_components_index + 1, vec![]);
                        let mut connected_component = &mut connected_components[connected_components_index];
                        fifo.push(index);
                        connected_component.push(index);
                        while !fifo.is_empty() {
                            let current = fifo.pop().unwrap();
                            for neighbour in &adj[current] {
                                let neighbour = *neighbour;
                                if partition[neighbour] == cluster_index && flags[neighbour] && neighbour != current {
                                    flags[neighbour] = false;
                                    fifo.push(neighbour);
                                    connected_component.push(neighbour);
                                }
                            }
                        }
                        if connected_component.len() < min_point_count_per_cc {
                            // Drop CC
                            connected_components.resize(connected_components_index, vec![]);
                        } else {
                            // Add CC
                            // println!("\t\t CC {} -> {}", connected_components_index, connected_components.len());
                        }
                    }
                }
            } else {
                unimplemented!("cloud partitioning")
            }
            let cc_duration = cc_start.elapsed();
            // debug!("Time taken to create connected components: {:?}", cc_duration);
            if connected_components.is_empty() {break;}
            if high_gradient_separation {
                unimplemented!("separate high gradient points")
            }
            if patch_expansion_enabled {
                unimplemented!("patch expansion enabled")
            }

            /// Patches generation
            let patch_start = Instant::now();
            for connected_component in connected_components.iter() {
                let patch_index = patches.len();

                patches.resize(patch_index + 1, Patch::default());
                let mut patch = patches.get_mut(patch_index).unwrap();

                let mut d0_count_per_patch: usize = 0;
                let mut d1_count_per_patch: usize = 0;
                let mut eom_count_per_patch: usize = 0;
                patch.set_index(patch_index);

                // ZICO: Skip EOM field, let's minimally update Patch struct
                let cluster_index = partition[connected_component[0]];
                let is_additional_projection_plane = cluster_index > 5;

                // ZICO: Skip additional projection plane logic
                patch.set_view_id(cluster_index as u8);
                patch.set_best_match_idx(INVALID_PATCH_INDEX);

                patch.cur_gpa_patch_data.initialize();
                patch.pre_gpa_patch_data.initialize();

                if params.enable_patch_splitting {
                    unimplemented!("patch splitting")
                }

                let projection_direction_type: i16 = -2 * (patch.projection_mode as i16) + 1;

                if patch_expansion_enabled {
                    unimplemented!("patch expansion")
                }

                let mut bounding_box = BoundingBox::default();

                // ZICO: Maybe make this default
                for i in 0..=2 {
                    bounding_box.max[i] = 0.0;
                }

                for &index in connected_component.iter() {
                    // patch_partition map point index to the patch it belongs to
                    // Why patch_index + 1? for 1-indexing?
                    patch_partition[index] = patch_index + 1;
                    if is_additional_projection_plane {
                        unimplemented!("Additional Projection Plane");
                    }
                    let point = &points.positions[index];
                    // ZICO: this can be made a method of bounding box
                    for k in 0..=2 {
                        if (point[k] as f64) < bounding_box.min[k] { bounding_box.min[k] = (point[k] as f64).floor() }
                        if (point[k] as f64) > bounding_box.max[k] { bounding_box.max[k] = (point[k] as f64).ceil() }
                    }
                }

                if enable_point_cloud_partitioning {
                    unimplemented!("Enable Point Cloud Partitioning")
                }
                let normal_axis = patch.axes.0 as usize;
                let tangent_axis = patch.axes.1 as usize;
                let bitangent_axis = patch.axes.2 as usize;
                patch.size_u = 1 + (bounding_box.max[tangent_axis].ceil() - bounding_box.min[tangent_axis].floor()) as usize;
                patch.size_v = 1 + (bounding_box.max[bitangent_axis].ceil() - bounding_box.min[bitangent_axis].floor()) as usize;
                patch.uv1.0 = bounding_box.min[tangent_axis] as usize;
                patch.uv1.1 = bounding_box.min[bitangent_axis] as usize;
                patch.d1 = if patch.projection_mode == 0 { INFINITE_DEPTH as usize } else { 0 };
                patch.depth.0.resize(patch.size_u * patch.size_v, INFINITE_DEPTH);
                patch.depth_0pc_idx.resize(patch.size_u * patch.size_v, INFINITE_NUMBER);
                if use_enhanced_occupancy_map_code {
                    unimplemented!("Enhanced Occupancy Map")
                }
                patch.occupancy_resolution = occupancy_resolution;
                patch.size_uv0 = (0, 0);
                patch._size_2d_in_pixel = (0, 0);
                // Can parallelize
                for i in connected_component {
                    let i = *i;
                    // let pointTmp = points.positions[i];
                    if is_additional_projection_plane {
                        unimplemented!("Additional Projection Plane")
                    }
                    let point = points.positions[i];
                    // C++ version round down, but the values are all int
                    let d = point[normal_axis] as i16;
                    let u = point[tangent_axis] as usize - patch.uv1.0 ;
                    let v = point[bitangent_axis] as usize - patch.uv1.1;
                    assert!(u >= 0 && u < patch.size_u);
                    assert!(v >= 0 && v < patch.size_v);
                    let p = v * patch.size_u + u;
                    let is_valid_point = if patch.projection_mode == 0 { patch.depth.0[p] > d }
                    else { patch.depth.0[p] == INFINITE_DEPTH || patch.depth.0[p] < d};

                    /// Fill Depth 0
                    if is_valid_point {
                        let mut minD0 = patch.d1;
                        let mut maxD0 = patch.d1;
                        patch.depth.0[p] = d;
                        patch.depth_0pc_idx[p] = i as i64;
                        patch._size_2d_in_pixel.0 = patch._size_2d_in_pixel.0.max(u);
                        patch._size_2d_in_pixel.1 = patch._size_2d_in_pixel.1.max(v);
                        patch.size_uv0.0 = patch.size_uv0.0.max(u / patch.occupancy_resolution);
                        patch.size_uv0.1 = patch.size_uv0.1.max(v / patch.occupancy_resolution);
                        minD0 = minD0.min(d as usize);
                        maxD0 = maxD0.max(d as usize);
                        if patch.projection_mode == 0 {
                            patch.d1 = minD0 / min_level * min_level;
                        } else {
                            patch.d1 = (maxD0 as f64 / min_level as f64 * min_level as f64).ceil() as usize;
                        }
                    }
                } // 1

                patch._size_2d_in_pixel.0 += 1;
                patch._size_2d_in_pixel.1 += 1;
                let no_quantized_patch_size_2d_x_in_pixel = patch._size_2d_in_pixel.0;
                let no_quantized_patch_size_2d_y_in_pixel = patch._size_2d_in_pixel.1;
                if quantizer_size_x != 0 {
                    patch._size_2d_in_pixel.0 = (no_quantized_patch_size_2d_x_in_pixel as f64 / quantizer_size_x as f64 * quantizer_size_x as f64).ceil() as usize;
                }
                if quantizer_size_y != 0 {
                    patch._size_2d_in_pixel.1 = (no_quantized_patch_size_2d_y_in_pixel as f64 / quantizer_size_y as f64 * quantizer_size_y as f64).ceil() as usize;
                }
                patch.size_uv0.0 += 1;
                patch.size_uv0.1 += 1;
                patch.occupancy.resize(patch.size_uv0.0 * patch.size_uv0.1, false);

                /// filter depth
                // let mut peakPerBlock: Vec<i16> = Vec::new();
                // let peak_number = if patch.projection_mode == 0 { INFINITE_DEPTH } else { 0 };
                // peakPerBlock.resize(patch.size_uv0.0 * patch.size_uv0.1, peak_number);
                //
                // // C++ version iterate through i64 but the patch.size and others use size_t
                // for v in 0..patch.size_v {
                //     for u in 0..patch.size_u {
                //         let p = v * patch.size_u + u;
                //         let depth0 = patch.depth.0[p];
                //         if depth0 == INFINITE_DEPTH {
                //             continue;
                //         }
                //
                //         let u0 = u / patch.occupancy_resolution;
                //         let v0 = v / patch.occupancy_resolution;
                //         let p0 = v0 * patch.size_uv0.0 + u0;
                //
                //         if patch.projection_mode == 0 {
                //             peakPerBlock[p0] = peakPerBlock[p0].min(depth0);
                //         } else {
                //             peakPerBlock[p0] = peakPerBlock[p0].max(depth0);
                //         }
                //     }
                // }
                //
                // // debug!("Do Depth refinement");
                // // debug!("u: {}, v: {}", patch.size_u, patch.size_v);
                // for v in 0..patch.size_v  {
                //     for u in 0..patch.size_u {
                //         let p = v * patch.size_u + u;
                //         let depth0 = patch.depth.0[p];
                //
                //         if depth0 == INFINITE_DEPTH {
                //             continue;
                //         }
                //
                //         let u0 = u / patch.occupancy_resolution;
                //         let v0 = v / patch.occupancy_resolution;
                //         let p0 = v0 * patch.size_uv0.0 + u0;
                //
                //         let tmp_a = (depth0 - peakPerBlock[p0]).abs();
                //         let tmp_b = surface_thickness as i16 + projection_direction_type * depth0;
                //         let tmp_c = projection_direction_type * patch.d1 as i16 + max_allowed_depth as i16;
                //
                //         // ZICO: This code is problematic all depth is made infinite, is it the tmp calculation?
                //         // I think we can ignore this for now
                //         // if depth0 != INFINITE_DEPTH {
                //         //     if tmp_a > 32 || tmp_b > tmp_c {
                //         //         patch.depth.0[p] = INFINITE_DEPTH;
                //         //         patch.depth_0pc_idx[p] = INFINITE_NUMBER;
                //         //     }
                //         // }
                //
                //
                //         if depth0 != INFINITE_DEPTH {
                //             if tmp_a > 32 {
                //                 patch.depth.0[p] = INFINITE_DEPTH;
                //                 patch.depth_0pc_idx[p] = INFINITE_NUMBER;
                //             }
                //         }
                //     }
                // }

                if eom_single_layer_mode {
                    unimplemented!("eom single layer mode");
                } else {
                    patch.depth.1 = patch.depth.0.clone();
                    let patch_surface_thickness = surface_thickness;
                    if use_surface_separation {
                        unimplemented!("Use surface separation")
                    }

                    /// Fill Depth 1
                    if patch_surface_thickness > 0 {
                        for i in connected_component {
                            let i = *i;
                            // let pointTmp = points.positions[i];
                            if is_additional_projection_plane {
                                unimplemented!("Additional Projection Plane")
                            }
                            let point = points.positions[i];
                            // C++ version round down, but the values are all int
                            let d = point[normal_axis] as i16;
                            let u = point[tangent_axis] as usize - patch.uv1.0;
                            let v = point[bitangent_axis] as usize - patch.uv1.1;
                            assert!(u >= 0 && u < patch.size_u);
                            assert!(v >= 0 && v < patch.size_v);
                            let p = v * patch.size_u + u;
                            let depth_0 = patch.depth.0[p];
                            let delta_d = projection_direction_type * (d - depth_0);
                            // ZICO: C++ version do this instead of depth0 >= INFINITE DEPTH
                            if !(depth_0 < INFINITE_DEPTH) { continue };
                            let is_color_similar = Self::colorSimilarity(
                                &frame_pcc_color[i],
                                &frame_pcc_color[patch.depth_0pc_idx[p] as usize], 128);
                            if depth_0 < INFINITE_DEPTH
                                && delta_d <= patch_surface_thickness as i16
                                && delta_d >= 0 && is_color_similar {
                                if (projection_direction_type * (d - patch.depth.1[p])) > 0 {
                                    patch.depth.1[p] = d;
                                }
                                if use_enhanced_occupancy_map_code {
                                    unimplemented!("enhanced occupancy map")
                                }
                            }
                            if patch.projection_mode == 0 && patch.depth.1[p] < patch.depth.0[p]
                                || patch.projection_mode == 1 && patch.depth.1[p] > patch.depth.0[p] {
                                println!(
                                    "ERROR: d1({}) and d0({}) for projection mode[{}]",
                                    patch.depth.1[p],
                                    patch.depth.0[p],
                                    patch.projection_mode
                                );
                            }
                        }
                    }
                }

                patch.size_d = 0;
                let mut rec: PointSet3 = PointSet3::default();
                // ZICO: C++ version resize to 0, dunno whats the advantage
                let mut point_count: Vec<usize> = Vec::new();
                point_count.resize(3, 0);

                let resample_start = Instant::now();
                Self::resample_point_cloud(
                    &mut point_count, &mut resampled, &mut resampled_patch_partition, patch,
                    patch_index, params.map_count_minus_1 > 0, surface_thickness,
                    eom_fix_bit_count, is_additional_projection_plane, use_enhanced_occupancy_map_code,
                    geometry_bit_depth_3d, create_sub_point_cloud
                );
                let resample_duration = resample_start.elapsed();
                // debug!("Time taken to resample point cloud: {:?}", resample_duration);

                d0_count_per_patch = point_count[0];
                d1_count_per_patch = point_count[1];
                eom_count_per_patch = point_count[2];

                patch.size_d_pixel = patch.size_d;
                patch.size_d = min((1 << min(geometry_bit_depth_2d, geometry_bit_depth_3d)) - 1, patch.size_d);
                let bit_depth_d = min(geometry_bit_depth_3d, geometry_bit_depth_2d) - log2(min_level as f32) as usize;
                let max_dd_plus_1 = 1 << bit_depth_d;
                let mut quant_dd = if patch.size_d == 0 { 0 } else { (patch.size_d - 1) / min_level + 1};
                quant_dd = min(quant_dd, max_dd_plus_1 - 1);
                patch.size_d = if quant_dd == 0 { 0 } else { quant_dd * min_level - 1};

                if create_sub_point_cloud {
                    unimplemented!("Sub point cloud")
                }
                if eom_single_layer_mode { d1_count_per_patch = 0 }
                patch.eom_and_d1_count = eom_count_per_patch;
                if use_enhanced_occupancy_map_code {
                    unimplemented!("Enhanced Occupancy Map")
                }
                patch.d0_count = d0_count_per_patch;
                number_of_eom += eom_count_per_patch - d1_count_per_patch;
                num_d0_points += d0_count_per_patch;
                num_d1_points += d1_count_per_patch;
                if use_enhanced_occupancy_map_code {
                    unimplemented!("Enhanced Occupancy Map")
                }
                // println!(
                //     "\t\t Patch {} ->(d1,u1,v1)=({}, {}, {})(dd,du,dv)=({}, {}, {}), Normal: {}, Direction: {}, EOM: {}",
                //     patch_index, patch.d1, patch.uv1.0, patch.uv1.1, patch.size_d,
                //     patch.size_u, patch.size_v, patch.axes.0 as usize, patch.projection_mode, patch.eom_and_d1_count
                // );
            }
            let resample_tree_start = Instant::now();
            let mut kd_tree_resampled = PCCKdTree::new();
            kd_tree_resampled.build_from_point_set(&resampled);
            let resample_tree_duration = resample_tree_start.elapsed();;
            debug!("Time taken to generate resample kd tree: {:?}", resample_tree_duration);

            raw_points.clear();
            // ZICO: this one check the distance diff between resampled and original then read if too far
            // Should double check if nanoflann use square distance
            let resample_tree_query_start = Instant::now();
            for i in 0..point_count {
                // Compare with original version how close is it
                let result = kd_tree_resampled.search(&points.positions[i], 1);
                let dist2 = result.squared_dists[0];
                raw_points_distance[i] = dist2;
                if dist2 > max_allowed_dist2_raw_points_selection { raw_points.push(i); }
            }
            let resample_tree_query_duration = resample_tree_query_start.elapsed();
            debug!("Time taken to query resample kd tree: {:?}", resample_tree_query_duration);

            let patch_duration = patch_start.elapsed();
            debug!("Time taken to generate patch: {:?}", patch_duration);

            if enable_point_cloud_partitioning {
                unimplemented!("Point Cloud Partitioning")
            }
            println!(" # patches {}", patches.len());
            println!(" # resampled {}", resampled.point_count());
            println!(" # raw points {}", raw_points.len());
            if use_enhanced_occupancy_map_code {
                println!(" # EOM points {}", number_of_eom);
            }
        }
        // ZICO: C++ version did a conversion from f64 to f32
        *distance_source_rec = (mean_yab + mean_uab + mean_vab + mean_yba + mean_uba + mean_vba) as f32;
        patches
    }

    pub fn resample_point_cloud(
        point_count: &mut Vec<usize>,
        resampled: &mut PointSet3,
        resampled_patch_partition: &mut Vec<usize>,
        patch: &mut Patch,
        patch_index: usize,
        multiple_maps: bool,
        surface_thickness: usize,
        eom_fix_bit_count: usize,
        is_additional_projection_plane: bool,
        use_enhanced_occupancy_map_code: bool,
        geometry_bit_depth_3d: usize,
        create_sub_point_cloud: bool,
    ) {
        let normal_axis = patch.axes.0 as usize;
        let tangent_axis = patch.axes.1 as usize;
        let bitangent_axis = patch.axes.2 as usize;
        point_count.resize(3, 0);
        let mut d0_count_per_patch = 0;
        let mut d1_count_per_patch = 0;
        let mut eom_count_per_patch = 0;

        // projection = 0 -> 1, projection = 1 -> -1
        let projection_type_indication = -2 * (patch.projection_mode as i16) + 1;

        for v in 0..patch.size_v {
            for u in 0..patch.size_u {
                let p = v * patch.size_u + u;
                if patch.depth.0[p] < INFINITE_DEPTH {
                    let depth0 = patch.depth.0[p];

                    // Add depth0
                    let u0 = u / patch.occupancy_resolution;
                    let v0 = v / patch.occupancy_resolution;
                    let p0 = v0 * patch.size_uv0.0 + u0;
                    assert!(u0 < patch.size_uv0.0);
                    assert!(v0 < patch.size_uv0.1);

                    patch.occupancy[p0] = true;

                    // ZICO: Should prolly make this Point3D
                    let mut point = Vector3D::new(0.0, 0.0, 0.0);
                    point[normal_axis] = depth0 as f64;
                    point[tangent_axis] = u as f64 + patch.uv1.0 as f64;
                    point[bitangent_axis] = v as f64 + patch.uv1.1 as f64;

                    if is_additional_projection_plane {
                        unimplemented!("Additional Projection Plane")
                        // let point_tmp = Self::iconvert(
                        //     patch.axis_of_additional_plane,
                        //     geometry_bit_depth_3d,
                        //     &point,
                        // ).unwrap();
                        // resampled.add_point_from_vector_3d(point_tmp);
                        // resampled_patch_partition.push(patch_index);
                        // if create_sub_point_cloud {
                        //     rec.add_point_from_vector_3d(point_tmp);
                        // }
                    } else {
                        resampled.add_point_from_vector_3d(point);
                        resampled_patch_partition.push(patch_index);
                        if create_sub_point_cloud {
                            unimplemented!("Create Sub Point Cloud")
                            // rec.add_point_from_vector_3d(point);
                        }
                    }

                    d0_count_per_patch += 1;

                    // Add EOM
                    let mut point_eom = point.clone();
                    if use_enhanced_occupancy_map_code && patch.depth_eom[p] != 0 {
                        unimplemented!("Use Enhanced Occupancy Map")
                    //     let n = if multiple_maps {
                    //         surface_thickness
                    //     } else {
                    //         eom_fix_bit_count
                    //     };
                    //
                    //     for i in 0..n {
                    //         if patch.depth_eom[p] & (1 << i) != 0 {
                    //             let n_delta_d_cur = (i + 1) as i16;
                    //             point_eom[normal_axis] =
                    //                 depth0 as f64 + projection_type_indication as f64 * (n_delta_d_cur as f64);
                    //
                    //             if point_eom[normal_axis] != point[normal_axis] {
                    //                 if is_additional_projection_plane {
                    //                     let mut point_tmp = Self::iconvert(
                    //                         patch.axis_of_additional_plane,
                    //                         geometry_bit_depth_3d,
                    //                         &point_eom,
                    //                     ).unwrap();
                    //                     resampled.add_point_from_vector_3d(point_tmp);
                    //                     resampled_patch_partition.push(patch_index);
                    //                 } else {
                    //                     resampled.add_point_from_vector_3d(point_eom.clone());
                    //                     resampled_patch_partition.push(patch_index);
                    //                 }
                    //                 eom_count_per_patch += 1;
                    //             }
                    //         }
                    //     }
                    }

                    // Add depth1
                    if !(use_enhanced_occupancy_map_code && !multiple_maps) {
                        let depth1 = patch.depth.1[p];
                        point[normal_axis] = depth1 as f64;
                        if patch.depth.0[p] != patch.depth.1[p] {
                            d1_count_per_patch += 1;
                        }
                        if is_additional_projection_plane {
                            // let point_tmp = Self::iconvert(
                            //     patch.axis_of_additional_plane,
                            //     geometry_bit_depth_3d,
                            //     &point,
                            // ).unwrap();
                            // resampled.add_point_from_vector_3d(point_tmp);
                            // resampled_patch_partition.push(patch_index);
                            // if create_sub_point_cloud {
                            //     unimplemented!("Create Sub PointCloud");
                            //     rec.add_point_from_vector_3d(point_tmp);
                            // }
                        } else if !use_enhanced_occupancy_map_code
                            || point_eom[normal_axis] != point[normal_axis]
                        {
                            resampled.add_point_from_vector_3d(point.clone());
                            resampled_patch_partition.push(patch_index);
                            // if create_sub_point_cloud {
                            //     rec.add_point_from_vector_3d(point);
                            // }
                        }
                        assert!(
                            (patch.depth.0[p] as i32 - patch.depth.1[p] as i32).abs() <= surface_thickness as i32
                        );
                    }

                    // Adjust depth(0), depth(1)
                    patch.depth.0[p] = projection_type_indication * (patch.depth.0[p] as i16 - patch.d1 as i16);
                    patch.size_d = patch.size_d.max(patch.depth.0[p] as usize);

                    if !(use_enhanced_occupancy_map_code && !multiple_maps) {
                        patch.depth.1[p] = projection_type_indication * (patch.depth.1[p] as i16 - patch.d1 as i16);
                        patch.size_d = patch.size_d.max(patch.depth.1[p] as usize);
                    }
                }
            }
        }

        point_count[0] = d0_count_per_patch;
        point_count[1] = d1_count_per_patch;
        point_count[2] = eom_count_per_patch;
    }

    pub fn iconvert(axis: u8, lod: usize, input: &Vector3D) -> Result<Vector3D, &'static str> {
        let shif: f64 = (1 << (lod - 1)) as f64 - 1.0;
        let (output_x, output_y, output_z) = match axis {
            1 => {
                let output_x = (input.x - input.z + shif) / 2.0;
                let output_y = input.y;
                let output_z = (input.x + input.z - shif) / 2.0;
                (output_x, output_y, output_z)
            },
            2 => {
                let output_x = input.x;
                let output_y = (input.z + input.y - shif) / 2.0;
                let output_z = (input.z - input.y + shif) / 2.0;
                (output_x, output_y, output_z)
            },
            3 => {
                let output_x = (input.y + input.x - shif) / 2.0;
                let output_y = (input.y - input.x + shif) / 2.0;
                let output_z = input.z;
                (output_x, output_y, output_z)
            },
            _ => return Err("Invalid axis. Axis must be 1, 2, or 3."),
        };
        Ok(Vector3D::new(output_x, output_y, output_z))
    }

    // Connect points with their nearest neighbour
    pub fn compute_adjacency_info(
        point_cloud: &PointSet3,
        kd_tree: &PCCKdTree,
        max_NN_count: usize
    ) -> Vec<Vec<usize>> {
        let point_count = point_cloud.point_count();
        let mut adj_matrix: Vec<Vec<usize>> = Vec::new();
        adj_matrix.resize(point_count, vec![]);

        #[cfg(feature = "use_rayon")]
        {
            adj_matrix
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, adj_list)| {
                    let point = &point_cloud.positions[index];
                    let result = kd_tree.search(point, max_NN_count);
                    *adj_list = result.indices;
                });
        }


        #[cfg(not(feature = "use_rayon"))]
        for (index, point) in point_cloud.positions.iter().enumerate() {
            let result = kd_tree.search(point, max_NN_count);
            adj_matrix[index] = result.indices;
        }

        adj_matrix
    }

    pub fn colorSimilarity(color_d1_candidate: &Color3B, color_d0: &Color3B, threshold: u8) -> bool {
        (color_d0[0] as i16 - color_d1_candidate[0] as i16).abs() < threshold as i16 &&
            (color_d0[1] as i16 - color_d1_candidate[1] as i16).abs() < threshold as i16 &&
            (color_d0[2] as i16 - color_d1_candidate[2] as i16).abs() < threshold as i16
    }
}

fn compute_adjacency_info_in_radius(
    point_cloud: &PointSet3,
    kd_tree: &PCCKdTree,
    max_nn_count: usize,
    radius: usize
) -> Vec<Vec<usize>> {
    let point_count = point_cloud.point_count();
    let mut adj_matrix: Vec<Vec<usize>> = Vec::new();
    adj_matrix.resize(point_count, vec![]);

    #[cfg(feature = "use_rayon")]
    {
        adj_matrix
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, adj_list)| {
                let point = &point_cloud.positions[index];
                let result = kd_tree.searchRadius(point, max_nn_count, radius as f64);
                *adj_list = result.indices;
            });
    }


    #[cfg(not(feature = "use_rayon"))]
    for (index, point) in point_cloud.positions.iter().enumerate() {
        let result = kd_tree.searchRadius(point, max_nn_count, radius as f64);
        adj_matrix[index] = result.indices;
    }

    adj_matrix
}


type PointIndicesVector = Vec<u32>;
type ScoresVector = Vec<u16>;

struct PointIndicesOfGridCell {
    point_indices: PointIndicesVector
}

impl PointIndicesOfGridCell {
    fn with_capacity(max_points: usize) -> Self { Self { point_indices: Vec::with_capacity(max_points), } }
    fn get_point_indices(&self) -> &Vec<u32> { &self.point_indices }
    fn add_point_index(&mut self, index: u32) { self.point_indices.push(index); }
    fn get_point_count(&self) -> usize { self.point_indices.len() }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum VoxEdge {
    NoEdge       = 0,  // one ppi-value in a voxel
    IndirectEdge = 1,  // adjacent voxels of M_DIRECT_EDGE, S_DIRECT_EDGE
    MDirectEdge  = 2,  // multiple points && more than two ppi-values in a voxel
    SDirectEdge  = 3,  // single-point in a voxel, considered as a direct edge-voxel
}

struct AttributeOfGridCell {
    score_smooth: ScoresVector,
    vox_edge: VoxEdge,
    update_flag: bool,
    total_points: u8,
    vox_ppi: u8
}


impl AttributeOfGridCell {
    // Default constructor
    pub fn new() -> Self {
        Self {
            score_smooth: Vec::new(),
            vox_edge: VoxEdge::NoEdge,
            update_flag: true,
            total_points: 0,
            vox_ppi: 0,
        }
    }

    // Constructor with orientation count
    pub fn with_orientation_count(orientation_count: usize) -> Self {
        Self {
            score_smooth: vec![0; orientation_count],
            vox_edge: VoxEdge::NoEdge,
            update_flag: true,
            total_points: 0,
            vox_ppi: 0,
        }
    }

    pub fn get_score_smooth(&self) -> &ScoresVector {
        &self.score_smooth
    }

    #[inline]
    pub fn update_scores(
        &mut self,
        point_indices: &PointIndicesVector,
        partitions: &[usize],
    ) {
        self.score_smooth.iter_mut().for_each(|s| *s = 0);
        for &j in point_indices { self.score_smooth[partitions[j as usize]] += 1; }
        // Update voxel type (1st voxel classification)
        self.update_voxel_type_with_scores();
    }

    pub fn get_edge(&self) -> VoxEdge { self.vox_edge }
    pub fn get_ppi(&self) -> u8 { self.vox_ppi }
    pub fn update_edge(&mut self, vox_edge: VoxEdge) { self.vox_edge = vox_edge; }
    pub fn set_updated_flag(&mut self) { self.update_flag = true; }

    pub fn update_voxel_type_with_scores(&mut self) {
        if !self.update_flag { return; }

        // Update edge type
        if self.vox_edge != VoxEdge::SDirectEdge {
            let uniformity_idx = self.score_smooth.iter().filter(|&&x| x != 0).count();
            self.vox_edge = if uniformity_idx == 1 { VoxEdge::NoEdge } else { VoxEdge::MDirectEdge };
        }

        // ZICO: PPI: preferred point index
        // Update ppi. Get the first largest index
        let mut max_index: usize = 0;
        let mut max_score: u16 = 0;
        // ZICO: Can optimise into peak finding algo
        for (index, score) in self.score_smooth.iter().enumerate() {
            if *score > max_score {
                max_index = index;
                max_score = *score;
            }
        }
        self.vox_ppi = max_index as u8;
        self.update_flag = false;
    }

    pub fn set_num_of_total_points(&mut self, point_count: u8) {
        self.total_points = point_count;
        self.vox_edge = if self.total_points == 1 { VoxEdge::SDirectEdge } else { VoxEdge::MDirectEdge };
    }
}

#[cfg(test)]
mod tests {
    use kdtree::KdTree;
    use crate::common::point_set3d::{Normal3D, Point3D};
    use super::*;

    // For references
    // orientation 1 is positive x
    // orientation 2 is negative x
    // orientation 3 is positive y
    // orientation 4 is negative y
    // orientation 5 is positive z
    // orientation 6 is negative z
    #[test]
    fn test_initial_segmentation_one_in_each_direction() {
        let mut point_set = PointSet3::default();
        // There has to be a better way
        point_set.add_point(Point3D { x: 1, y: 0, z: 0 });
        point_set.add_point(Point3D { x: 3, y: 0, z: 0 });
        point_set.add_point(Point3D { x: 0, y: 1, z: 0 });
        point_set.add_point(Point3D { x: 0, y: 3, z: 0 });
        point_set.add_point(Point3D { x: 0, y: 0, z: 1 });
        point_set.add_point(Point3D { x: 0, y: 0, z: 3 });
        point_set.normals.push(Normal3D { x: 1.0, y: 0.0, z: 0.0});
        point_set.normals.push(Normal3D { x: -1.0, y: 0.0, z: 0.0});
        point_set.normals.push(Normal3D { x: 0.0, y: 1.0, z: 0.0});
        point_set.normals.push(Normal3D { x: 0.0, y: -1.0, z: 0.0});
        point_set.normals.push(Normal3D { x: 0.0, y: 0.0, z: 1.0});
        point_set.normals.push(Normal3D { x: 0.0, y: 0.0, z: -1.0});
        let axis_weights = Vector3D { x: 1.0, y: 1.0, z: 1.0 };
        let partition = PatchSegmenter::initial_segmentation(
            &point_set, &orientations6, orientations6Count, &axis_weights
        );
        let expected_partition = vec![0, 1, 2, 3, 4, 5];
        assert_eq!(partition, expected_partition)
    }
    #[test]
    fn test_initial_segmentation_all_in_same_direction() {
        let mut point_set = PointSet3::default();
        // There has to be a better way
        point_set.add_point(Point3D { x: 1, y: 0, z: 0 });
        point_set.add_point(Point3D { x: 3, y: 0, z: 0 });
        point_set.add_point(Point3D { x: 0, y: 1, z: 0 });
        point_set.add_point(Point3D { x: 0, y: 3, z: 0 });
        point_set.add_point(Point3D { x: 0, y: 0, z: 1 });
        point_set.add_point(Point3D { x: 0, y: 0, z: 3 });
        // ZICO: Ideally should be normalized
        point_set.normals.push(Normal3D { x: 1.0, y: 0.0, z: 0.0});
        point_set.normals.push(Normal3D { x: 0.9, y: 0.1, z: 0.0});
        point_set.normals.push(Normal3D { x: 0.9, y: -0.1, z: 0.0});
        point_set.normals.push(Normal3D { x: 0.6, y: 0.3, z: 0.0});
        point_set.normals.push(Normal3D { x: 0.6, y: -0.3, z: 0.0});
        point_set.normals.push(Normal3D { x: 0.1, y: 0.0, z: 0.0});
        let axis_weights = Vector3D { x: 1.0, y: 1.0, z: 1.0 };
        let partition = PatchSegmenter::initial_segmentation(
            &point_set, &orientations6, orientations6Count, &axis_weights
        );
        let expected_partition = vec![0, 0, 0, 0, 0, 0];
        assert_eq!(partition, expected_partition)
    }
    #[test]
    fn test_compute_adjacency_info() {
        let mut point_set = PointSet3::default();
        point_set.add_point(Point3D { x: 1, y: 0, z: 0 });
        point_set.add_point(Point3D { x: 2, y: 0, z: 0 });
        point_set.add_point(Point3D { x: 3, y: 0, z: 0 });
        point_set.add_point(Point3D { x: 3, y: 1, z: 0 });
        point_set.add_point(Point3D { x: 4, y: 0, z: 0 });
        point_set.add_point(Point3D { x: 5, y: 0, z: 0 });
        let mut kd_tree = PCCKdTree::new();
        kd_tree.build_from_point_set(&point_set);
        let adj_matrix = PatchSegmenter::compute_adjacency_info(
            &point_set,
            &kd_tree,
            3
        );
        println!("{:?}", adj_matrix);
        assert_eq!(adj_matrix[0], vec![0, 1, 2]);
        assert_eq!(adj_matrix[4], vec![4, 5, 2]);
    }
}
