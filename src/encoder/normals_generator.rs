use std::cmp::Ordering;
use cgmath::Vector3;
use crate::encoder::Vector3D;
use std::collections::BinaryHeap;
use crate::common::point_set3d::PointSet3;

// ZICO: Super headache to implement, will use the vvtk normal estimator for now

// ZICO: Dunno how this is used yet
enum NormalsGeneratorOrientation {
    PCC_NORMALS_GENERATOR_ORIENTATION_NONE,
    PCC_NORMALS_GENERATOR_ORIENTATION_SPANNING_TREE,
    PCC_NORMALS_GENERATOR_ORIENTATION_VIEW_POINT,
    PCC_NORMALS_GENERATOR_ORIENTATION_CUBEMAP_PROJECTION
}

struct NormalsGenerator3Parameters {
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

pub struct PCCNormalsGenerator3 {
    pub normals: Vec<Vector3<f64>>,
    pub eigenvalues: Vec<Vector3D>,
    pub barycenters: Vec<Vector3D>,
    pub number_of_nearest_neighbors_in_normal_estimation: Vec<u32>,
    pub visited: Vec<u32>,
    pub edges: BinaryHeap<WeightedEdge>,
    pub nb_thread: usize,
}

impl PCCNormalsGenerator3 {
    pub fn new() -> Self {
        Self {
            normals: Vec::new(),
            eigenvalues: Vec::new(),
            barycenters: Vec::new(),
            number_of_nearest_neighbors_in_normal_estimation: Vec::new(),
            visited: Vec::new(),
            edges: BinaryHeap::new(),
            nb_thread: 0,
        }
    }

    pub fn clear(&mut self) {
        self.normals.clear();
    }

    pub fn init(&mut self, point_count: usize, params: &NormalsGenerator3Parameters) {
        // Initialize the struct
    }

    // ZICO: Come back layer to implement kdtree for normal estimation
    // pub fn compute(
    //     &mut self,
    //     point_cloud: &PointSet3,
    //     kdtree: &PCCKdTree,
    //     params: &NormalsGenerator3Parameters,
    //     nb_thread: usize,
    // ) {
    //     // Compute logic
    // }

    pub fn get_normals(&self) -> &Vec<Vector3D> {
        &self.normals
    }

    pub fn get_normal(&self, pos: usize) -> Vector3D {
        assert!(pos < self.normals.len());
        self.normals[pos].clone()
    }

    pub fn get_eigenvalues(&self, pos: usize) -> Vector3D {
        assert!(pos < self.eigenvalues.len());
        self.eigenvalues[pos].clone()
    }

    pub fn get_centroid(&self, pos: usize) -> Vector3D {
        assert!(pos < self.barycenters.len());
        self.barycenters[pos].clone()
    }

    pub fn get_number_of_nearest_neighbors_in_normal_estimation(&self, index: usize) -> u32 {
        assert!(index < self.number_of_nearest_neighbors_in_normal_estimation.len());
        self.number_of_nearest_neighbors_in_normal_estimation[index]
    }

    pub fn get_normal_count(&self) -> usize {
        self.normals.len()
    }

    // ZICO: Implement the rest later
}