use clap::value_parser;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use crate::common::point_set3d::{Point3D, PointSet3};

// NN: nearest_neighbour
pub struct NNQuery3 {
    pub point: Point3D,
    pub radius: f64,
    pub nearest_neighbour_count: usize
}

#[derive(Debug, Clone, Default)]
pub struct NNResult {
    pub indices_: Vec<usize>,
    pub squared_dists: Vec<f64>
}

impl NNResult {
    pub fn reserve(&mut self, size: usize) {
        self.indices_.reserve(size);
        self.squared_dists.reserve(size);
    }
    pub fn size(&self) -> usize {
        self.indices_.len()
    }
    pub fn push(&mut self, index: usize, dist: f64) {
        self.indices_.push(index);
        self.squared_dists.push(dist)
    }
    pub fn pop(&mut self) {
        self.indices_.pop();
        self.squared_dists.pop();
    }
}

impl From<Vec<(f64, &usize)>> for NNResult {
    fn from(value: Vec<(f64, &usize)>) -> Self {
        let (dist, indices): (Vec<f64>, Vec<usize>) = value
            .into_iter()
            .map(|element| (element.0, element.1)
            ).unzip();
        NNResult {
            indices_: indices,
            squared_dists: dist
        }
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

    pub fn build_from_point_set(&mut self, point_set: &PointSet3) {
        point_set.positions.iter().enumerate().for_each(|(index, position)| {
            self.add(position, index)
        });
    }

    // Add a point entry into the kdTree
    pub fn add(&mut self, point: &Point3D, index: usize) {
        let mut pointVec = [0.0; 3];
        pointVec[0] = point.x as f64; pointVec[1] = point.y as f64; pointVec[2] = point.z as f64;
        self.kdtree_.add(pointVec, index).expect("Failed to add into kdtrees");
    }

    // Find the num_results nearest neighbour
    pub fn search(&self, point: &Point3D, num_results: usize) -> NNResult {
        let mut pointVec = [0.0; 3];
        pointVec[0] = point.x as f64; pointVec[1] = point.y as f64; pointVec[2] = point.z as f64;
        let nearest = self.kdtree_.nearest(&pointVec, num_results, &squared_euclidean).unwrap();
        NNResult::from(nearest)
    }

    // Find the neighbour within a certain radius
    pub fn searchRadius(&self, point: &Point3D, num_results: usize, radius: f64) -> NNResult {
        let mut pointVec = [0.0; 3];
        pointVec[0] = point.x as f64; pointVec[1] = point.y as f64; pointVec[2] = point.z as f64;
        let nearest = self.kdtree_.within(&pointVec, radius, &squared_euclidean).unwrap();
        NNResult::from(nearest)
    }

}

mod tests {
    use super::*;

    #[test]
    fn test_search() {
        // Nearest neighbour of a in order = [a, b, c, d]
        let a = Point3D { x: 1f64 as u16, y: 1f64 as u16, z: 1f64 as u16};
        let b = Point3D { x: 2f64 as u16, y: 2f64 as u16, z: 2f64 as u16};
        let c = Point3D { x: 0f64 as u16, y: 0f64 as u16, z: 0f64 as u16};
        let d = Point3D { x: 3f64 as u16, y: 3f64 as u16, z: 3f64 as u16};
        let mut kdtree = PCCKdTree::new();
        kdtree.add(&a, 0);
        kdtree.add(&b, 1);
        kdtree.add(&c, 2);
        kdtree.add(&d, 3);
        let nearestTwo = kdtree.search(&a, 2);
        assert_eq!(nearestTwo.indices_, [0, 1]);
        assert_eq!(nearestTwo.squared_dists, [0.0, 3.0]);
        println!("{:?}", kdtree.search(&a, 4));
    }
}