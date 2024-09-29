use std::cmp::{Ordering, PartialEq};
use cgmath::{InnerSpace, Vector3};
use crate::encoder::Vector3D;
use std::collections::BinaryHeap;
use cgmath::Matrix3;
use num_traits::Zero;
use crate::common::math::diagonalize;
use crate::common::point_set3d::PointSet3;
use crate::encoder::kd_tree::{PCCKdTree};
use crate::encoder::normals_generator::NormalsGeneratorOrientation::PCC_NORMALS_GENERATOR_ORIENTATION_VIEW_POINT;
use rayon::prelude::*;

type Matrix3D = Matrix3<f64>;


#[derive(PartialEq, Eq)]
pub enum NormalsGeneratorOrientation {
    PCC_NORMALS_GENERATOR_ORIENTATION_NONE,
    PCC_NORMALS_GENERATOR_ORIENTATION_SPANNING_TREE,
    PCC_NORMALS_GENERATOR_ORIENTATION_VIEW_POINT,
    PCC_NORMALS_GENERATOR_ORIENTATION_CUBEMAP_PROJECTION
}

pub struct NormalsGenerator3Parameters {
    pub view_point: Vector3<f64>,
    pub radius_normal_smoothing: f64,
    pub radius_normal_estimation: f64,
    pub radius_normal_orientation: f64,
    pub weight_normal_smoothing: f64,
    pub number_of_nearest_neighbors_in_normal_smoothing: usize,
    pub number_of_nearest_neighbors_in_normal_estimation: usize,
    pub number_of_nearest_neighbors_in_normal_orientation: usize,
    pub number_of_iterations_in_normal_smoothing: usize,
    pub orientation_strategy: NormalsGeneratorOrientation,
    pub store_eigenvalues: bool,
    pub store_number_of_nearest_neighbors_in_normal_estimation: bool,
    pub store_centroids: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct WeightedEdge {
    weight: f64,
    start: u32,
    end: u32,
}

impl Eq for WeightedEdge {}

impl Ord for WeightedEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.weight == other.weight {
            if self.start == other.start {
                self.end.cmp(&other.end)
            } else {
                self.start.cmp(&other.start)
            }
        } else {
            self.weight.partial_cmp(&other.weight).unwrap()
        }
    }
}

impl PartialOrd for WeightedEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct NormalsGenerator3 {
    // pub normals: Vec<Vector3<f64>>,
    pub eigenvalues: Vec<Vector3D>,
    pub barycenters: Vec<Vector3D>,
    pub number_of_nearest_neighbors: Vec<u32>,
    pub visited: Vec<u32>,
    pub edges: BinaryHeap<WeightedEdge>,
    // // ZICO: We don't need this for now, let Rayon decide
    // // pub nb_thread: usize,
}

impl NormalsGenerator3 {
    pub fn init(point_count: usize, params: &NormalsGenerator3Parameters) -> NormalsGenerator3 {
        NormalsGenerator3 {
            // normals: vec![Vector3::new(0.0, 0.0, 0.0); point_count],
            eigenvalues:
            if params.store_eigenvalues { vec![Vector3::new(0.0, 0.0, 0.0); point_count] } else { vec![] },
            barycenters:
            if params.store_centroids { vec![Vector3::new(0.0, 0.0, 0.0); point_count] } else { vec![] },
            number_of_nearest_neighbors:
            if params.store_number_of_nearest_neighbors_in_normal_estimation { vec![0; point_count] } else { vec![] },
            visited: vec![],
            edges: Default::default()
        }
    }

    pub fn compute(
        points: &PointSet3,
        kd_tree: &PCCKdTree,
        params: &NormalsGenerator3Parameters) -> Vec<Vector3<f64>> {
        let mut generator = Self::init(points.point_count(), params);
        let mut normals = generator.compute_normals(points, kd_tree, params);
        generator.orient_normals(&mut normals, points, params);
        normals
    }

    // ZICO: C++ version update normal in place,
    // For the sake of parallelization, I make this return a normal instead
    pub fn compute_normal(
        &self,
        index: usize,
        point_cloud: &PointSet3,
        kd_tree: &PCCKdTree,
        params: &NormalsGenerator3Parameters
    )  -> Vector3<f64> {
        // ZICO: For convenience’s sake, make a copy of the current position but in f64
        let position_vector3d: Vector3D = Vector3D::new(
            point_cloud.positions[index][0] as f64,
            point_cloud.positions[index][1] as f64,
            point_cloud.positions[index][2] as f64
        );
        let mut bary: Vector3D = position_vector3d;
        let mut normal = Vector3::zero();
        let mut eigenval = Vector3::zero();
        let mut cov_mat = Matrix3::zero();

        let n_neighbors = kd_tree.search(&point_cloud.positions[index], 10);

        if n_neighbors.len() > 1 {
            // Compute barycenter
            bary = Vector3::zero();
            for &neighbor_idx in n_neighbors.indices.iter() {
                bary.x += point_cloud.positions[neighbor_idx][0] as f64;
                bary.y += point_cloud.positions[neighbor_idx][1] as f64;
                bary.z += point_cloud.positions[neighbor_idx][2] as f64;
            }
            bary /= n_neighbors.len() as f64;

            // Compute covariance matrix
            cov_mat = Matrix3::zero();
            for &neighbor_idx in n_neighbors.indices.iter() {
                let mut pt: Vector3<f64> = Vector3::zero();
                pt.x = point_cloud.positions[neighbor_idx][0] as f64 - bary.x;
                pt.y = point_cloud.positions[neighbor_idx][1] as f64 - bary.y;
                pt.z = point_cloud.positions[neighbor_idx][2] as f64 - bary.z;
                cov_mat.x.x += pt.x * pt.x;
                cov_mat.y.y += pt.y * pt.y;
                cov_mat.z.z += pt.z * pt.z;
                cov_mat.x.y += pt.x * pt.y;
                cov_mat.x.z += pt.x * pt.z;
                cov_mat.y.z += pt.y * pt.z;
            }
            cov_mat.y.x = cov_mat.x.y;
            cov_mat.z.x = cov_mat.x.z;
            cov_mat.z.y = cov_mat.y.z;
            cov_mat /= (n_neighbors.len() - 1) as f64;

            // Diagonalize covariance matrix (equivalent to PCCDiagonalize)
            let (mut q, mut d) = diagonalize(&cov_mat);

            d.x.x = d.x.x.abs();
            d.y.y = d.y.y.abs();
            d.z.z = d.z.z.abs();

            // Extract normal and eigenvalues
            if d.x.x < d.y.y && d.x.x < d.z.z {
                normal = q.x;
                eigenval.x = d.x.x;
                if d.y.y < d.z.z {
                    eigenval.y = d.y.y;
                    eigenval.z = d.z.z;
                } else {
                    eigenval.y = d.z.z;
                    eigenval.z = d.y.y;
                }
            } else if d.y.y < d.z.z {
                normal = q.y;
                eigenval.x = d.y.y;
                if d.x.x < d.z.z {
                    eigenval.y = d.x.x;
                    eigenval.z = d.z.z;
                } else {
                    eigenval.y = d.z.z;
                    eigenval.z = d.x.x;
                }
            } else {
                normal = q.z;
                eigenval.x = d.z.z;
                if d.x.x < d.y.y {
                    eigenval.y = d.x.x;
                    eigenval.z = d.y.y;
                } else {
                    eigenval.y = d.y.y;
                    eigenval.z = d.x.x;
                }
            }
        }

        // Flip normal based on viewpoint
        if normal.dot(params.view_point - position_vector3d) < 0.0 {
            normal = -normal;
        }

        // Store eigenvalues, barycenters, and number of neighbors if needed
        // if params.store_eigenvalues {
        //     self.eigenvalues[index] = eigenval;
        // }
        // if params.store_centroids {
        //     self.barycenters[index] = bary;
        // }
        // if params.store_number_of_nearest_neighbors_in_normal_estimation {
        //     self.number_of_nearest_neighbors[index] = n_neighbors.len() as u32;
        // }
        normal
    }

    pub fn compute_normals(
        &self,
        point_cloud: &PointSet3,
        kd_tree: &PCCKdTree,
        params: &NormalsGenerator3Parameters
    ) -> Vec<Vector3<f64>> {
        let point_count = point_cloud.point_count();
        let mut normals: Vec<Vector3<f64>> = vec![Vector3::zero(); point_count];

        // ZICO: Let Rayon do chunking
        // let chunk_count = 64;
        // let sub_ranges = Self::divide_range(0, point_count, chunk_count);

        // ZICO: Need major refactoring due to Rust.
        // Refactor compute_normal to return normal vector instead of modifying in place
        // Then assign the normal vector to self
        #[cfg(feature = "use_rayon")]
        {
            normals
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, vec)| {
                    *vec = self.compute_normal(index, point_cloud, kd_tree, params)
                })
        }

        // Sequential version, don't need to work chunk
        #[cfg(not(feature = "use_rayon"))]
        for i in 0..point_count { normals[i] = self.compute_normal(i, point_cloud, kd_tree, params ) }
        normals
    }


    // Given two index. Partition into a specified number of chunk
    fn divide_range(start: usize, end: usize, chunk_count: usize) -> Vec<usize> {
        let element_count = end - start;
        let mut sub_ranges: Vec<usize> = Vec::new();
        if element_count == chunk_count {
            sub_ranges.resize(element_count + 1, 0);
            for i in start..=end { sub_ranges[i - start] = i }
        } else {
            sub_ranges.resize(chunk_count + 1, 0);
            let step = element_count as f64 / (chunk_count + 1) as f64;
            let mut pos = start as f64;
            for i in 0..chunk_count {
                sub_ranges[i] = pos as usize;
                pos += step;
            }
            sub_ranges[chunk_count] = end;
        }
        sub_ranges
    }

    // ZICO: Let's use view point adjustment for now
    pub fn orient_normals(
        &self,
        normals: &mut Vec<Vector3<f64>>,
        point_cloud: &PointSet3,
        params: &NormalsGenerator3Parameters
    ) {
        if params.orientation_strategy == PCC_NORMALS_GENERATOR_ORIENTATION_VIEW_POINT {
            #[cfg(feature = "use_rayon")]
            {
                use rayon::prelude::*;

                normals.par_iter_mut().enumerate().for_each(|(index, normal)| {
                    let view_direction = Vector3::new(
                        params.view_point.x - point_cloud.positions[index].x as f64,
                        params.view_point.y - point_cloud.positions[index].y as f64,
                        params.view_point.z - point_cloud.positions[index].z as f64,
                    );
                    if normal.dot(view_direction) < 0.0 {
                        *normal = -*normal;
                    }
                })
            }

            #[cfg(not(feature = "use_rayon"))]
            {
                normals.iter_mut().enumerate().for_each(|(index, normal)| {
                    let view_direction = Vector3::new(
                        params.view_point.x - point_cloud.positions[index].x as f64,
                        params.view_point.y - point_cloud.positions[index].y as f64,
                        params.view_point.z - point_cloud.positions[index].z as f64,
                    );
                    if normal.dot(view_direction) < 0.0 {
                        *normal = -*normal;
                    }
                })
            }

        }
    }
}

mod tests {
    use std::fs;
    use vivotk::formats::PointCloud;
    use vivotk::formats::pointxyzrgba::PointXyzRgba;
    use vivotk::pcd::{read_pcd_file, read_pcd_header};
    use super::*;

    #[test]
    fn test_compute_normals() {
        let file_path = "../pointClouds/longdress/Pcd";
        let mut paths = fs::read_dir(file_path).unwrap();
        paths.next();
        let sample_pcd_file_path = "./test_files/pcd/longdress_vox10_1051.pcd";
        // let sample_pcd_file_path = paths.next().unwrap().unwrap().path();
        
        let header = read_pcd_header(sample_pcd_file_path.clone());
        let ptcl = read_pcd_file(sample_pcd_file_path.clone());

        let ptcl = PointCloud::<PointXyzRgba>::from(ptcl.unwrap());
        let point_cloud = PointSet3::from(ptcl.points);

        let mut kd_tree = PCCKdTree::new();
        kd_tree.build_from_point_set(&point_cloud);

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
        normal_generator.compute_normals(&point_cloud, &kd_tree, &param);
    }
}