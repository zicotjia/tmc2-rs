use std::path::PathBuf;
use cgmath::Vector3;

#[derive(Debug, Clone, Default)]
enum PCCColorTransform {
    #[default]
    COLOR_TRANSFORM_NONE,
    COLOR_TRANSFORM_RGB_TO_YCBCR
}

#[derive(Debug, Clone, Default)]
enum PCCCodecId {
    // ZICO: Let's FFMPEG for now
    // JMAPP,
    // HMAPP,
    // SHMAPP,
    // JLMIB,
    // HMLIB,
    // VTMLIB,
    #[default]
    FFMPEG
}


// ZICO: Maybe break down to multiple smaller structs
// ZICO: Will comment out unused params later
#[derive(Debug, Clone)]
pub struct Params {
    pub start_frame_number: usize,
    pub configuration_folder: String,
    pub uncompressed_data_folder: String,
    pub compressed_stream_path: String,
    pub reconstructed_data_path: String,
    pub color_transform: PCCColorTransform,
    pub color_space_conversion_path: String,
    pub video_encoder_occupancy_path: String,
    pub video_encoder_geometry_path: String,
    pub video_encoder_attribute_path: String,
    pub video_encoder_occupancy_codec_id: PCCCodecId,
    pub video_encoder_geometry_codec_id: PCCCodecId,
    pub video_encoder_attribute_codec_id: PCCCodecId,
    pub video_encoder_internal_bitdepth: usize,
    pub byte_stream_video_coder_occupancy: bool,
    pub byte_stream_video_coder_geometry: bool,
    pub byte_stream_video_coder_attribute: bool,
    pub use_3dmc: bool,
    pub use_pcc_rdo: bool,
    pub color_space_conversion_config: String,
    pub inverse_color_space_conversion_config: String,
    pub nb_thread: usize,
    pub frame_count: usize,
    pub group_of_frames_size: usize,
    pub uncompressed_data_path: String,
    pub forced_ssvh_unit_size_precision_bytes: u32,

    // packing
    pub minimum_image_width: usize,
    pub minimum_image_height: usize,

    // video encoding
    pub geometry_qp: i32,
    pub attribute_qp: i32,
    pub delta_qp_d0: i32,
    pub delta_qp_d1: i32,
    pub delta_qp_t0: i32,
    pub delta_qp_t1: i32,
    pub aux_geometry_qp: i32,
    pub aux_attribute_qp: i32,
    pub geometry_config: String,
    pub geometry0_config: String,
    pub geometry1_config: String,
    pub attribute_config: String,
    pub attribute0_config: String,
    pub attribute1_config: String,
    pub multiple_streams: bool,

    // segmentation
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
    pub log2_quantizer_size_x: usize,
    pub log2_quantizer_size_y: usize,
    pub min_point_count_per_cc_patch_segmentation: usize,
    pub max_nn_count_patch_segmentation: usize,
    pub surface_thickness: usize,
    pub min_level: usize,
    pub max_allowed_dist2_raw_points_detection: f64,
    pub max_allowed_dist2_raw_points_selection: f64,
    pub lambda_refine_segmentation: f64,
    pub map_count_minus1: usize,

    // occupancy map encoding
    pub max_candidate_count: usize,
    pub occupancy_precision: usize,
    pub occupancy_map_config: String,
    pub occupancy_map_qp: usize,
    pub eom_fix_bit_count: usize,
    pub occupancy_map_refinement: bool,

    // hash
    pub decoded_atlas_information_hash: usize,

    // smoothing
    pub neighbor_count_smoothing: usize,
    pub radius2_smoothing: f64,
    pub radius2_boundary_detection: f64,
    pub threshold_smoothing: f64,
    pub grid_smoothing: bool,
    pub grid_size: usize,
    pub flag_geometry_smoothing: bool,

    // Patch Expansion (m47772, CE2.12)
    pub patch_expansion: bool,

    // color smoothing
    pub threshold_color_smoothing: f64,
    pub threshold_color_difference: f64,
    pub threshold_color_variation: f64,
    pub cgrid_size: usize,
    pub flag_color_smoothing: bool,

    // color pre-smoothing
    pub threshold_color_pre_smoothing: f64,
    pub threshold_color_pre_smoothing_local_entropy: f64,
    pub radius2_color_pre_smoothing: f64,
    pub neighbor_count_color_pre_smoothing: usize,
    pub flag_color_pre_smoothing: bool,

    // coloring
    pub best_color_search_range: usize,

    // Improved color transfer
    pub num_neighbors_color_transfer_fwd: i32,
    pub num_neighbors_color_transfer_bwd: i32,
    pub use_dist_weighted_average_fwd: bool,
    pub use_dist_weighted_average_bwd: bool,
    pub skip_avg_if_identical_source_point_present_fwd: bool,
    pub skip_avg_if_identical_source_point_present_bwd: bool,
    pub dist_offset_fwd: f64,
    pub dist_offset_bwd: f64,
    pub max_geometry_dist2_fwd: f64,
    pub max_geometry_dist2_bwd: f64,
    pub max_color_dist2_fwd: f64,
    pub max_color_dist2_bwd: f64,

    // Exclude color outliers
    pub exclude_color_outlier: bool,
    pub threshold_color_outlier_dist: f64,

    // lossless
    pub no_attributes: bool,
    pub raw_points_patch: bool,
    pub attribute_video_444: bool,

    // raw points video
    pub use_raw_points_separate_video: bool,
    pub geometry_aux_video_config: String,
    pub attribute_aux_video_config: String,

    // scale and bias
    pub model_scale: f32,
    pub model_origin: Vector3<f32>,

    // patch sampling resolution
    pub level_of_detail_x: usize,
    pub level_of_detail_y: usize,
    pub keep_intermediate_files: bool,
    pub absolute_d1: bool,
    pub absolute_t1: bool,
    pub constrained_pack: bool,

    // dilation
    pub group_dilation: bool,

    // EOM
    pub enhanced_occupancy_map_code: bool,

    // Lossy occupancy Map coding
    pub offset_lossy_om: usize,
    pub threshold_lossy_om: usize,
    pub prefilter_lossy_om: bool,

    // reconstruction
    pub remove_duplicate_points: bool,
    pub point_local_reconstruction: bool,
    pub patch_size: usize,
    pub plrl_number_of_modes: usize,
    pub single_map_pixel_interleaving: bool,

    // visual quality
    pub patch_color_subsampling: bool,
    pub surface_separation: bool,
    pub high_gradient_separation: bool,
    pub min_gradient: f64,
    pub min_num_high_gradient_points: usize,

    // Flexible Patch Packing
    pub packing_strategy: usize,
    pub attribute_bg_fill: usize,
    pub safe_guard_distance: usize,
    pub use_eight_orientations: bool,

    // Lossy raw points Patch
    pub lossy_raw_points_patch: bool,
    pub min_norm_sum_of_inv_dist4mp_selection: f64,

    // GPA
    pub global_patch_allocation: i32,

    // GTP
    pub global_packing_strategy_gof: i32,
    pub global_packing_strategy_reset: bool,
    pub global_packing_strategy_threshold: f64,

    // low delay encoding
    pub low_delay_encoding: bool,

    // 3D geometry padding
    pub geometry_padding: usize,

    // EOM
    pub enhanced_pp: bool,
    pub min_weight_epp: f64,

    // Additional Projection Plane
    pub additional_projection_plane_mode: i32,
    pub partial_additional_projection_plane: f64,

    // 3D and 2D bit depths
    pub geometry_3d_coordinates_bitdepth: usize,
    pub geometry_nominal_2d_bitdepth: usize,

    // Partitions and tiles
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
    pub num_rois: i32,

    // Sort raw points by Morton code
    pub morton_order_sort_raw_points: bool,
    pub attribute_raw_separate_video_width: usize,

    // Patch block filtering
    pub pbf_enable_flag: bool,
    pub pbf_passes_count: i16,
    pub pbf_filter_size: i16,
    pub pbf_log2_threshold: i16,

    // re
    pub patch_precedence_order_flag: bool,
    pub max_num_ref_atlas_list: usize,
    pub max_num_ref_atlas_frame: usize,

    pub log2_max_atlas_frame_order_cnt_lsb: usize,
    pub tile_segmentation_type: usize,
    pub num_max_tile_per_frame: usize,
    pub uniform_partition_spacing: bool,
    pub tile_partition_width: usize,
    pub tile_partition_height: usize,
    pub tile_partition_width_list: Vec<i32>,
    pub tile_partition_height_list: Vec<i32>,

    // Profile tier level
    pub tier_flag: bool,
    pub profile_codec_group_idc: usize,
    pub profile_toolset_idc: usize,
    pub profile_reconstruction_idc: usize,
    pub level_idc: usize,
    pub avc_codec_id_index: usize,
    pub hevc_codec_id_index: usize,
    pub shvc_codec_id_index: usize,
    pub vvc_codec_id_index: usize,

    // Profile toolset constraints information
    pub one_v3c_frame_only_flag: bool,
    pub eom_constraint_flag: bool,
    pub max_map_count_minus1: usize,
    pub max_atlas_count_minus1: usize,
    pub multiple_map_streams_constraint_flag: bool,
    pub plr_constraint_flag: bool,
    pub attribute_max_dimension_minus1: usize,
    pub attribute_max_dimension_partitions_minus1: usize,
    pub no_eight_orientations_constraint_flag: bool,
    pub no_45_degree_projection_patch_constraint_flag: bool,

    // reconstruction options
    pub pixel_deinterleaving_type: usize,
    pub point_local_reconstruction_type: usize,
    pub reconstruct_eom_type: usize,
    pub duplicated_point_removal_type: usize,
    pub reconstruct_raw_type: usize,
    pub apply_geo_smoothing_type: usize,
    pub apply_attr_smoothing_type: usize,
    pub attr_transfer_filter_type: usize,
    pub apply_occupancy_synthesis_type: usize,

    // SHVC
    pub shvc_layer_index: usize,
    pub shvc_rate_x: usize,
    pub shvc_rate_y: usize,
}