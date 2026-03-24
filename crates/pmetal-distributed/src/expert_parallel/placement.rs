//! Expert placement solver: decides which experts live on which nodes.

/// Expert placement plan across nodes.
///
/// Maps each expert ID to the rank that owns its weights and computes
/// on its tokens.
#[derive(Debug, Clone)]
pub struct ExpertPlacement {
    /// experts_per_rank[rank] = list of expert IDs owned by that rank.
    pub experts_per_rank: Vec<Vec<usize>>,
    /// Total number of experts across all ranks.
    pub total_experts: usize,
    /// Top-k experts activated per token.
    pub top_k: usize,
    /// Reverse map: expert_id → rank.
    rank_for_expert: Vec<usize>,
}

impl ExpertPlacement {
    /// Uniform division: each rank gets `total_experts / world_size` experts.
    ///
    /// For 512 experts across 4 nodes: rank 0 gets [0..128),
    /// rank 1 gets [128..256), etc. Remainder experts go to the last rank.
    pub fn uniform(total_experts: usize, world_size: usize, top_k: usize) -> Self {
        let base = total_experts / world_size;
        let remainder = total_experts % world_size;

        let mut experts_per_rank = Vec::with_capacity(world_size);
        let mut rank_for_expert = vec![0usize; total_experts];
        let mut offset = 0;

        for rank in 0..world_size {
            let count = if rank < remainder { base + 1 } else { base };
            let experts: Vec<usize> = (offset..offset + count).collect();
            for &eid in &experts {
                rank_for_expert[eid] = rank;
            }
            experts_per_rank.push(experts);
            offset += count;
        }

        Self {
            experts_per_rank,
            total_experts,
            top_k,
            rank_for_expert,
        }
    }

    /// Weighted division by available RAM on each node.
    ///
    /// Nodes with more RAM get more experts, proportional to their share
    /// of total cluster RAM.
    pub fn weighted(total_experts: usize, node_ram: &[u64], top_k: usize) -> Self {
        let world_size = node_ram.len();
        let total_ram: u64 = node_ram.iter().sum();

        if total_ram == 0 {
            return Self::uniform(total_experts, world_size, top_k);
        }

        let mut experts_per_rank = Vec::with_capacity(world_size);
        let mut rank_for_expert = vec![0usize; total_experts];
        let mut assigned = 0;

        for (rank, &ram) in node_ram.iter().enumerate() {
            let count = if rank == world_size - 1 {
                // Last rank gets all remaining to avoid rounding errors.
                total_experts - assigned
            } else {
                let proportion = ram as f64 / total_ram as f64;
                (proportion * total_experts as f64).round() as usize
            };

            let experts: Vec<usize> = (assigned..assigned + count).collect();
            for &eid in &experts {
                if eid < total_experts {
                    rank_for_expert[eid] = rank;
                }
            }
            experts_per_rank.push(experts);
            assigned += count;
        }

        Self {
            experts_per_rank,
            total_experts,
            top_k,
            rank_for_expert,
        }
    }

    /// Which rank owns a given expert.
    pub fn rank_for_expert(&self, expert_id: usize) -> usize {
        self.rank_for_expert[expert_id]
    }

    /// Expert IDs owned by a given rank.
    pub fn local_expert_ids(&self, rank: usize) -> &[usize] {
        &self.experts_per_rank[rank]
    }

    /// Number of experts on a given rank.
    pub fn num_local_experts(&self, rank: usize) -> usize {
        self.experts_per_rank[rank].len()
    }

    /// Convert a global expert ID to a local index within its owning rank.
    pub fn global_to_local(&self, expert_id: usize) -> usize {
        let rank = self.rank_for_expert[expert_id];
        let first = self.experts_per_rank[rank][0];
        expert_id - first
    }

    /// Convert a local expert index on a rank to the global expert ID.
    pub fn local_to_global(&self, rank: usize, local_idx: usize) -> usize {
        self.experts_per_rank[rank][local_idx]
    }

    /// World size (number of ranks).
    pub fn world_size(&self) -> usize {
        self.experts_per_rank.len()
    }

    /// Check if a given expert is local to this rank.
    pub fn is_local(&self, expert_id: usize, rank: usize) -> bool {
        self.rank_for_expert[expert_id] == rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_placement_512_experts_4_nodes() {
        let p = ExpertPlacement::uniform(512, 4, 8);
        assert_eq!(p.world_size(), 4);
        assert_eq!(p.num_local_experts(0), 128);
        assert_eq!(p.num_local_experts(1), 128);
        assert_eq!(p.num_local_experts(2), 128);
        assert_eq!(p.num_local_experts(3), 128);

        assert_eq!(p.rank_for_expert(0), 0);
        assert_eq!(p.rank_for_expert(127), 0);
        assert_eq!(p.rank_for_expert(128), 1);
        assert_eq!(p.rank_for_expert(511), 3);
    }

    #[test]
    fn uniform_placement_remainder() {
        let p = ExpertPlacement::uniform(10, 3, 2);
        // 10 / 3 = 3 remainder 1 → ranks get 4, 3, 3
        assert_eq!(p.num_local_experts(0), 4);
        assert_eq!(p.num_local_experts(1), 3);
        assert_eq!(p.num_local_experts(2), 3);

        // Verify all experts assigned
        let total: usize = (0..3).map(|r| p.num_local_experts(r)).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn weighted_placement_proportional() {
        // Node 0 has 2x the RAM of nodes 1 and 2.
        let p = ExpertPlacement::weighted(12, &[200, 100, 100], 2);
        // Node 0 should get ~6, nodes 1 and 2 ~3 each.
        assert_eq!(p.num_local_experts(0), 6);
        assert_eq!(p.num_local_experts(1), 3);
        assert_eq!(p.num_local_experts(2), 3);
    }

    #[test]
    fn global_to_local_mapping() {
        let p = ExpertPlacement::uniform(8, 2, 2);
        assert_eq!(p.global_to_local(0), 0);
        assert_eq!(p.global_to_local(3), 3);
        assert_eq!(p.global_to_local(4), 0);
        assert_eq!(p.global_to_local(7), 3);
    }

    #[test]
    fn local_to_global_roundtrip() {
        let p = ExpertPlacement::uniform(8, 2, 2);
        for eid in 0..8 {
            let rank = p.rank_for_expert(eid);
            let local = p.global_to_local(eid);
            assert_eq!(p.local_to_global(rank, local), eid);
        }
    }

    #[test]
    fn is_local_check() {
        let p = ExpertPlacement::uniform(8, 2, 2);
        assert!(p.is_local(0, 0));
        assert!(p.is_local(3, 0));
        assert!(!p.is_local(4, 0));
        assert!(p.is_local(4, 1));
    }
}
