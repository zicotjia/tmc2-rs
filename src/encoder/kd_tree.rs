use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use crate::common::point_set3d::Point3D;

// NN: nearest_neighbour
pub struct NNQuery3 {
    pub point: Point3D,
    pub radius: f64,
    pub nearest_neighbour_count: usize
}

#[derive(Debug, Clone, Default)]
pub struct NNResult {
    pub indices_: Vec<usize>,
    pub dist_: Vec<f64>
}

impl NNResult {
    pub fn reserve(&mut self, size: usize) {
        self.indices_.reserve(size);
        self.dist_.reserve(size);
    }
    pub fn size(&self) -> usize {
        self.indices_.len()
    }
    pub fn push(&mut self, index: usize, dist: f64) {
        self.indices_.push(index);
        self.dist_.push(dist)
    }
    pub fn pop(&mut self) {
        self.indices_.pop();
        self.dist_.pop();
    }
}

// C++ version use kdtree adapter, let's hardcode with kdtree crate for now
pub struct PCCKdTree {
    pub kdtree_: KdTree<f64, usize, [f64; 3]>
}

impl PCCKdTree {
    pub fn clear(&mut self) {
        self.kdtree_ = KdTree::new(3); // Reinitialize the KdTree to clear it
    }

    pub fn new() -> Self {
        PCCKdTree {
            kdtree_: KdTree::new(3)
        }
    }

    pub fn search(&mut self, point: &Point3D, num_results: usize, result:  &NNResult) {
        let mut pointVec = [0.0; 3];
        pointVec[0] = point.x as f64; pointVec[1] = point.y as f64; pointVec[2] = point.z as f64;
        self.kdtree_.nearest(&pointVec, num_results, &squared_euclidean).unwrap();
    }

    // TBC
}