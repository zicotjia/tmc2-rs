use std::cmp::{max, min};
use crate::common::point_set3d::PointSet3;
use crate::encoder::encoder_parameters::EncoderParams;
use crate::encoder::Vector3D;

struct Encoder {
    pub(crate) params: EncoderParams
}

impl Encoder {
    fn calculate_weight_value(&self, source: &PointSet3, geometry_bit_depth_3d: usize) -> [f64; 3] {
        let max_value = 1_usize << geometry_bit_depth_3d;
        let mut weight_value = [1.0; 3];
        let mut pj_face = vec![false; max_value * max_value * 3];

        if self.params.enhanced_pp {
            let size1f = max_value * max_value;

            for point in &source.positions {
                let p0 = max(0, min((max_value - 1) as i32, point[0] as i32)) as usize;
                let p1 = max(0, min((max_value - 1) as i32, point[1] as i32)) as usize;
                let p2 = max(0, min((max_value - 1) as i32, point[2] as i32)) as usize;

                pj_face[p2 * max_value + p1] = true;                   // YZ
                pj_face[p0 * max_value + p2 + size1f] = true;          // ZX
                pj_face[p1 * max_value + p0 + size1f * 2] = true;      // XY
            }

            // pj_cnt: [idx, value]
            let mut pj_cnt = [(0, 0_usize); 3];

            for idx in 0..max_value * max_value {
                if pj_face[idx] { pj_cnt[0].1 += 1; }
                if pj_face[idx + size1f] { pj_cnt[1].1 += 1; }
                if pj_face[idx + size1f * 2] { pj_cnt[2].1 += 1; }
            }

            pj_cnt.sort_by(|a, b| a.1.cmp(&b.1));

            let mut axis_weight = [1.0; 6];

            if (pj_cnt[0].1 as f64 / pj_cnt[2].1 as f64) >= self.params.min_weight_epp {
                for i in 0..3 {
                    let idx_t = pj_cnt[i].0;
                    axis_weight[idx_t] = pj_cnt[i].1 as f64 / pj_cnt[2].1 as f64;
                    axis_weight[idx_t + 3] = axis_weight[idx_t];
                }
            } else {
                axis_weight[pj_cnt[0].0] = self.params.min_weight_epp;
                axis_weight[pj_cnt[0].0 + 3] = self.params.min_weight_epp;
                axis_weight[pj_cnt[2].0] = 1.0;
                axis_weight[pj_cnt[2].0 + 3] = 1.0;

                let tmp_b = pj_cnt[1].1 as f64 / pj_cnt[2].1 as f64;
                let tmp_a = pj_cnt[0].1 as f64 / pj_cnt[2].1 as f64;
                axis_weight[pj_cnt[1].0] = self.params.min_weight_epp + (tmp_b - tmp_a) / (1.0 - tmp_a) * (1.0 - self.params.min_weight_epp);
                axis_weight[pj_cnt[1].0 + 3] = axis_weight[pj_cnt[1].0];
            }
            weight_value[0] = axis_weight[0];
            weight_value[1] = axis_weight[1];
            weight_value[2] = axis_weight[2];
        }
        weight_value
    }
}