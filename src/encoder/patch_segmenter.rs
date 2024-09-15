use std::collections::HashMap;
use std::ffi::c_double;
use cgmath::InnerSpace;
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
        // let eom_single_layer_mode = params.eom_single_layer_mode;
        // let eom_fix_bit_count = params.eom_fix_bit_count;
        let surface_thickness = params.surface_thickness;
        let max_allowed_depth = params.max_allowed_depth;
        let min_level = params.min_level;
        // let use_enhanced_occupancy_map_code = params.use_enhanced_occupancy_map_code;
        // let create_sub_point_cloud = params.create_sub_point_cloud;
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
        while !raw_points.is_empty() {
            // This algo go through all raw points. Group them into connected components
            // until not more raw points
            let mut connected_components: Vec<Vec<usize>>;
            if !enable_point_cloud_partitioning {
                let mut fifo: Vec<usize> = Vec::with_capacity(point_count);
                // ZICO: flags for what? UPDATE: its just check if points have been visited
                let mut flags: Vec<bool> = vec![true; point_count];
                // ZICO: Can try changing 256
                connected_components.reserve(256);

                // ZICO: Must we go through all points for each connected components?
                for index in raw_points.iter() {
                    if flags[index] && raw_points_distance[index] > max_allowed_dist2_raw_points_detection {
                        flags[index] = false;
                        let connected_components_index = connected_components.len();
                        // There are 6 partitions for now
                        let cluster_index = partition[index];
                        // Add an extra element
                        connected_components.resize(connected_components_index + 1, vec![]);
                        let mut connected_component = &connected_components[connected_components_index];
                        // Maybe copy the iterator instead of reference it
                        fifo.push(*index);
                        connected_component.push(*index);
                        while !fifo.is_empty() {
                            let current = fifo.pop().unwrap();
                            for neighbour in adj[current].iter() {
                                // If the neighbour is also within the same partition
                                // and if the neighbour has not been in any connected components
                                // we consider them to be in the same connected components
                                // Current adj list implementation has point also be connected to itself
                                // For now lets just add neigbour != current check
                                // ZICO: come back to this latyer
                                if partition[neighbour] == cluster_index && flags[neighbour] && *neighbour != current {
                                    flags[neighbour] = false;
                                    fifo.push(*neighbour);
                                    connected_component.push(*neighbour);
                                }
                            }
                        }
                        // If connected_component size is less than min_point
                        // Don't add it????
                        // Isn't this lossy
                        if connected_component.len() < min_point_count_per_cc {
                            connected_components.resize(connected_components_index);
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
            // Now we finally start creating the patch
            for connected_component in connected_components.iter() {

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
