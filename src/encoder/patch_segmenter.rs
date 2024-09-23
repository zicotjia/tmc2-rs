use std::collections::HashMap;
use std::ffi::c_double;
use cgmath::InnerSpace;
use num_traits::abs;
use num_traits::real::Real;
use crate::common::{INFINITE_DEPTH, INFINITE_NUMBER, INVALID_PATCH_INDEX};
use crate::common::math::BoundingBox;
use crate::common::point_set3d::{Color3B, Point3D, PointSet3};
use crate::decoder::Patch;
use crate::encoder::Vector3D;
use crate::encoder::constants::orientations::{orientations6, orientations6Count};
use crate::encoder::kd_tree::{NNResult, PCCKdTree};

type Voxels = HashMap<u64, Vec<usize>>;

struct PatchSegmenterParams {
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
#[derive(Clone, Default)]
pub struct PatchSegmenter {
    pub nb_thread: usize,
    pub box_min_depths: Vec<Patch>,
    pub box_max_depths: Vec<Patch>
}
impl PatchSegmenter {
    fn compute(
        geometry: PointSet3,
        // frame_index: usize,
        params: PatchSegmenterParams
    ) {
        // Determine Orientation
        // ZICO: For now assume only the base 6 projection plane
        let orientations = orientations6;
        let orientations_count = orientations6Count;
        let geometry_vox;

        // ZICO: For now assumes gridBasedSegmentation is ff
        if params.grid_based_segmentation {
            unimplemented!("Grid based segmentation (Voxelization")
        } else {
            geometry_vox = geometry;
        }

        // ZICO: For now assume PointSet always have a normal
        if !geometry_vox.with_normals {
            unimplemented!("Normal calculation")
        }
        assert!(geometry_vox.with_normals);

        let partition: Vec<usize>;
        // Initial Segmentation
        if params.additional_projection_plane_mode == 0 {
            partition = Self::initial_segmentation(&geometry_vox, &orientations, orientations_count, &params.weight_normal)
        } else {
            unimplemented!("initialSegmentation for additional projection plane")
        }

        // Build KDTree
        let mut kd_tree = PCCKdTree::new();
        kd_tree.build_from_point_set(&geometry_vox);



        // ZICO: For now lets skip all segmentation refinement
        // if params.grid_based_refine_segmentation {
        //     unimplemented!("grid-based refine segmentation")
        // } else {
        //     unimplemented!("refine segmentation")
        // }

        // ZICO: Implement Segment Patches

        // let resampled = PointSet3::default();
        // let patchPartition: Vec<usize>;
        // let resampledPatchPartition: Vec<usize>;
        // let rawPoints: Vec<usize>;


    }

    // ZICO: Come back later
    // pub fn convert_points_to_voxels(
    //     source: &PointSet3,
    //     geo_bits: usize,
    //     vox_dim: usize
    // ) -> (PointSet3, Voxels) {
    //     let geo_bits2 = geo_bits << 1;
    //     let mut vox_dim_shift = 0;
    //     let mut i = vox_dim;
    //     while i > 1  {
    //         vox_dim_shift += 1;
    //         i >>= 1;
    //     }
    //     let vox_dim_half = vox_dim >> 1;
    //     let mut voxel_point_set = PointSet3::default();
    //     voxel_point_set.reserve(source.len());
    //     let voxels = Voxels::new();
    //
    //     let sub_to_ind = |x: u64, y: u64, z: u64| -> u64 {
    //         x + (y << geo_bits) + (z << geo_bits2)
    //     };
    //
    //     for point in source.positions.iter() {
    //         let x0 = ((point.x as usize) + vox_dim_half) >> vox_dim_shift;
    //     }
    //     (voxel_point_set, voxels)
    // }

    // Segment into partitions based on the normal
    pub fn initial_segmentation(
        geometry: &PointSet3,
        orientations: &[Vector3D; 6], //Hardcoded to 6 for now
        orientations_count: usize,
        axis_weight: &Vector3D
    ) -> Vec<usize> {
        let mut partition = Vec::new();
        // ZICO(QUESTION): What is axis_weight used for?
        let point_count = geometry.point_count();
        partition.resize(point_count, 0);
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

    pub fn segment_patches(
        points: &PointSet3,
        kd_tree: &PCCKdTree,
        params: &PatchSegmenterParams,
        partition: &mut Vec<usize>,
        patch_partition: &mut Vec<usize>,
        resampled_patch_partition: &mut Vec<usize>,
        mut raw_points: Vec<usize>,
        resampled: &PointSet3,
        sub_point_cloud: &mut Vec<PointSet3>,
        distance_source_rec: &f32,
        orientations: &[Vector3D; 6],
        orientation_count: usize
    ) {
        let max_nn_count = params.max_nn_count_patch_segmentation;
        let min_point_count_per_cc = params.min_point_count_per_cc_patch_segmentation;
        let occupancy_resolution = params.occupancy_resolution;
        let quantizer_size_x = params.quantizer_size_x;
        let quantizer_size_y = params.quantizer_size_y;
        let max_allowed_dist2_raw_points_detection = params.max_allowed_dist_2_raw_points_detection;
        let max_allowed_dist2_raw_points_selection = params.max_allowed_dist_2_raw_points_selection;
        let eom_single_layer_mode = params.eom_single_layer_mode;
        // let eom_fix_bit_count = params.eom_fix_bit_count;
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
        let min_gradient = params.min_gradient;
        let min_num_high_gradient_points = params.min_num_high_gradient_points;
        let enable_point_cloud_partitioning = params.enable_point_cloud_partitioning;

        let mut roi_bounding_box_min_x = params.roi_bounding_box_min_x.clone();
        let mut roi_bounding_box_max_x = params.roi_bounding_box_max_x.clone();
        let mut roi_bounding_box_min_y = params.roi_bounding_box_min_y.clone();
        let mut roi_bounding_box_max_y = params.roi_bounding_box_max_y.clone();
        let mut roi_bounding_box_min_z = params.roi_bounding_box_min_z.clone();
        let mut roi_bounding_box_max_z = params.roi_bounding_box_max_z.clone();

        let mut num_cuts_along_1st_longest_axis = params.num_cuts_along_1st_longest_axis;
        let mut num_cuts_along_2nd_longest_axis = params.num_cuts_along_2nd_longest_axis;
        let mut num_cuts_along_3rd_longest_axis = params.num_cuts_along_3rd_longest_axis;

        let point_count = points.point_count();

        patch_partition.resize(point_count, 0);
        resampled_patch_partition.reserve(point_count);
        let mut nn_result: NNResult;
        let mut frame_pcc_color: Vec<Color3B> = Vec::new();
        frame_pcc_color.reserve(point_count);
        assert!(&points.with_colors);
        // Add colors
        points.colors.iter().for_each(|color| frame_pcc_color.push(color.clone()));

        // ZICO: Ignore enable cloud partitioning

        // ZICO: does d0 refer to original point clouds and d1 refer to after processing?
        let num_d0_points = 0;
        let num_d1_points = 0;

        // Compute adjacency info
        let mut adj: Vec<Vec<usize>> = Vec::new();                     // Adjacency list
        let mut adj_dist: Vec<Vec<f64>> = Vec::new();                  // Adjacency distance matrix
        let mut flag_exp: Vec<bool> = Vec::new();                      // Flag for expansion (Boolean flags)
        let mut num_rois: i32 = 0;                                     // Number of Regions of Interest (ROIs)
        let mut num_chunks: i32 = 0;                                   // Number of chunks (or segments)
        let mut points_chunks: Vec<PointSet3> = Vec::new();         // Vector of point cloud chunks
        let mut points_index_chunks: Vec<Vec<usize>> = Vec::new();     // Vector of point indices for each chunk
        let mut point_count_chunks: Vec<usize> = Vec::new();           // Vector of point counts for each chunk
        let mut kdtree_chunks: Vec<PCCKdTree> = Vec::new();            // Vector of KD-Trees for each chunk
        let mut bounding_box_chunks: Vec<BoundingBox> = Vec::new();       // Vector of bounding boxes for each chunk
        let mut adj_chunks: Vec<Vec<Vec<usize>>> = Vec::new();         // Adjacency list for each chunk

        // ZICO: for now patch_expansion and partitioning is disabled
        // Maybe implement partitioning cause its more memory efficient
        if patch_expansion_enabled {
            unimplemented!("Compute Adjacency Info Dist");
        } else {
            if enable_point_cloud_partitioning {
                unimplemented!("Point Cloud Partitioning");
            } else {
                adj = Self::compute_adjacency_info(
                    &points,
                    &kd_tree,
                    max_nn_count
                );
            }
        }
        // Extract Patches
        let raw_points_distance: Vec<f64> = vec![f64::MAX; point_count];
        raw_points = (0..point_count).collect();
        let raw_points_chunks: Vec<Vec<usize>>;
        let raw_points_distance_chunks: Vec<Vec<f64>>;
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
        let number_of_eom: usize = 0;

        // Must go through all raw points
        // Maybe made into function
        while !raw_points.is_empty() {
            // This algo go through all raw points. Group them into connected components
            // until not more raw points
            let mut connected_components: Vec<Vec<usize>> = Vec::new();
            if !enable_point_cloud_partitioning {
                let mut fifo: Vec<usize> = Vec::with_capacity(point_count);
                // ZICO: flags for what? UPDATE: its just check if points have been visited
                let mut flags: Vec<bool> = vec![true; point_count];
                // ZICO: Can try changing 256
                connected_components.reserve(256);

                // ZICO: Must we go through all points for each connected components?
                for index in &raw_points {
                    let index = *index;
                    if flags[index] && raw_points_distance[index] > max_allowed_dist2_raw_points_detection {
                        flags[index] = false;
                        let connected_components_index = connected_components.len();
                        // There are 6 partitions for now
                        let cluster_index = partition[index];
                        // Add an extra element
                        connected_components.resize(connected_components_index + 1, vec![]);
                        let mut connected_component = &mut connected_components[connected_components_index];
                        // Maybe copy the iterator instead of reference it
                        fifo.push(index);
                        connected_component.push(index);
                        while !fifo.is_empty() {
                            let current = fifo.pop().unwrap();
                            for neighbour in &adj[current] {
                                let neighbour = *neighbour;
                                // If the neighbour is also within the same partition
                                // and if the neighbour has not been in any connected components
                                // we consider them to be in the same connected components
                                // Current adj list implementation has point also be connected to itself
                                // For now lets just add neigbour != current check
                                // ZICO: come back to this latyer
                                if partition[neighbour] == cluster_index && flags[neighbour] && neighbour != current {
                                    flags[neighbour] = false;
                                    fifo.push(neighbour);
                                    connected_component.push(neighbour);
                                }
                            }
                        }
                        // If connected_component size is less than min_point
                        // Don't add it????
                        // Isn't this lossy
                        if connected_component.len() < min_point_count_per_cc {
                            connected_components.resize(connected_components_index, vec![]);
                        } else {
                            unimplemented!("print debug line here maybe?")
                        }
                    }
                }
                // println!("number of CC: {}", connected_components.len());
            } else {
                unimplemented!("cloud partitioning")
            }
            if connected_components.is_empty() {break;}
            if high_gradient_separation {
                unimplemented!("separate high hradient points")
            }
            if patch_expansion_enabled {
                unimplemented!("patch expansion enabled")
            }

            // ZICO: C++ version pass Patches from the frame context
            // Might have to do that here too but for now lets create then return
            let mut patches: Vec<Patch> = Vec::with_capacity(256);

            // Now we finally start creating the patch
            // Maybe make this its own function?
            for connected_component in connected_components.iter() {
                let patch_index = patches.len();

                // ZICO: Is this technique faster?
                // Why not instantiate a patch then push to patches
                patches.resize(patch_index + 1, Patch::default());
                let mut patch = patches.get_mut(patch_index).unwrap();

                // ZICO: By right i need only d0 for now right?
                let d0_count_per_patch: usize = 0;
                let d1_count_per_patch: usize = 0;
                let eom_count_per_patch: usize = 0;
                patch.set_index(patch_index);

                // ZICO: Skip EOM field, let's minimally update Patch struct
                let cluster_index = partition[connected_component[0]];
                let is_additional_projection_plane = cluster_index > 5;

                // ZICO: Skip additional projection plane logic
                patch.set_view_id(cluster_index as u8);
                patch.set_best_match_idx(INVALID_PATCH_INDEX);

                // Initialize GPAData
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

                // filter depth
                let mut peakPerBlock: Vec<i16> = Vec::new();
                let peak_number = if patch.projection_mode == 0 { INFINITE_DEPTH } else { 0 };
                peakPerBlock.resize(patch.size_uv0.0 * patch.size_uv0.1, peak_number);

                // C++ version iterate through i64 but the patch.size and others use size_t
                for v in 0..patch.size_v {
                    for u in 0..patch.size_u {
                        let p = v * patch.size_u + u;
                        let depth0 = patch.depth.0[p];

                        if depth0 == INFINITE_DEPTH {
                            continue;
                        }

                        let u0 = u / patch.occupancy_resolution;
                        let v0 = v / patch.occupancy_resolution;
                        let p0 = v0 * patch.size_uv0.0 + u0;

                        if patch.projection_mode == 0 {
                            peakPerBlock[p0] = peakPerBlock[p0].min(depth0);
                        } else {
                            peakPerBlock[p0] = peakPerBlock[p0].max(depth0);
                        }
                    }
                }

                for v in 0..patch.size_v  {
                    for u in 0..patch.size_u {
                        let p = v * patch.size_u + u;
                        let depth0 = patch.depth.0[p];

                        if depth0 == INFINITE_DEPTH {
                            continue;
                        }

                        let u0 = u / patch.occupancy_resolution;
                        let v0 = v / patch.occupancy_resolution;
                        let p0 = v0 * patch.size_uv0.0 + u0;

                        let tmp_a = (depth0 - peakPerBlock[p0]).abs();
                        let tmp_b = surface_thickness as i16 + projection_direction_type * depth0;
                        let tmp_c = projection_direction_type * patch.d1 as i16 + max_allowed_depth as i16;

                        if depth0 != INFINITE_DEPTH {
                            if tmp_a > 32 || tmp_b > tmp_c {
                                patch.depth.0[p] = INFINITE_DEPTH;
                                patch.depth_0pc_idx[p] = INFINITE_NUMBER;
                            }
                        }
                    }
                }

                if eom_single_layer_mode {
                    unimplemented!("eom single layer mode");
                } else {
                    patch.depth.1 = patch.depth.0.clone();
                    let patch_surface_thickness = surface_thickness;
                    if use_surface_separation {
                        unimplemented!("Use surface separation")
                    }

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
                            let u = point[tangent_axis] as usize - patch.uv1.0 ;
                            let v = point[bitangent_axis] as usize - patch.uv1.1;
                            assert!(u >= 0 && u < patch.size_u);
                            assert!(v >= 0 && v < patch.size_v);
                            let p = v * patch.size_u + u;
                            let depth0 = patch.depth.0[p];
                            let deltaD = projection_direction_type * (d - depth0);
                            // ZICO: C++ version do this instead of depth0 >= INFINITE DEPTH
                            if !(depth0 < INFINITE_DEPTH) {continue};
                            let is_color_similar = Self::colorSimilarity(
                                &frame_pcc_color[i],
                                &frame_pcc_color[patch.depth_0pc_idx[p]], 128);

                        }
                    }

                }

            }
        }

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
            use rayon::prelude::*;
            adj_matrix
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, adj_list)| {
                    let point = &point_cloud.positions[index];
                    let result = kd_tree.search(point, max_NN_count);
                    *adj_list = result.indices_;
                });
        }


        #[cfg(not(feature = "use_rayon"))]
        for (index, point) in point_cloud.positions.iter().enumerate() {
            let result = kd_tree.search(point, max_NN_count);
            adj_matrix[index] = result.indices_;
        }

        adj_matrix
    }

    pub fn colorSimilarity(color_d1_candidate: &Color3B, color_d0: &Color3B, threshold: u8) -> bool {
        (color_d0[0] as i16 - color_d1_candidate[0] as i16).abs() < threshold as i16 &&
            (color_d0[1] as i16 - color_d1_candidate[1] as i16).abs() < threshold as i16 &&
            (color_d0[2] as i16 - color_d1_candidate[2] as i16).abs() < threshold as i16
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
        println!("{:?}", partition);
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
        println!("{:?}", partition);
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
