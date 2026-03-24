//! Optimizer state partitioning for ZeRO Stage 1+.

use std::collections::HashMap;

/// ZeRO optimization stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeROStage {
    /// Stage 1: Partition optimizer states (Adam m/v) across ranks.
    /// Each rank stores optimizer state only for its assigned params.
    Stage1,
    /// Stage 2: Stage 1 + reduce-scatter gradients.
    /// Gradients are reduced and scattered so each rank only receives
    /// the gradient shard for its owned parameters.
    Stage2,
}

/// Manages parameter-to-rank assignment for ZeRO optimization.
///
/// Parameters are assigned round-robin to ranks by sorted name.
/// This ensures deterministic assignment across all ranks without
/// requiring communication.
#[derive(Debug, Clone)]
pub struct ZeROPartitioner {
    /// ZeRO optimization stage.
    pub stage: ZeROStage,
    /// This rank's index.
    pub rank: usize,
    /// Total number of ranks.
    pub world_size: usize,
    /// Maps parameter name → owning rank.
    pub param_to_rank: HashMap<String, usize>,
    /// Parameters owned by this rank (sorted).
    pub owned_params: Vec<String>,
    /// All parameter names (sorted, for deterministic ordering).
    pub all_params: Vec<String>,
}

impl ZeROPartitioner {
    /// Create a new partitioner with round-robin parameter assignment.
    ///
    /// Parameters are sorted by name and assigned to ranks in order:
    /// param[0] → rank 0, param[1] → rank 1, ..., param[N] → rank N%world_size.
    pub fn new(
        all_param_names: &[String],
        stage: ZeROStage,
        rank: usize,
        world_size: usize,
    ) -> Self {
        let mut sorted_params: Vec<String> = all_param_names.to_vec();
        sorted_params.sort();

        let mut param_to_rank = HashMap::with_capacity(sorted_params.len());
        let mut owned_params = Vec::new();

        for (i, name) in sorted_params.iter().enumerate() {
            let owner = i % world_size;
            param_to_rank.insert(name.clone(), owner);
            if owner == rank {
                owned_params.push(name.clone());
            }
        }

        Self {
            stage,
            rank,
            world_size,
            param_to_rank,
            owned_params,
            all_params: sorted_params,
        }
    }

    /// Create a partitioner with explicit parameter-to-rank assignment.
    ///
    /// Useful for MoE-aware partitioning where expert parameters should
    /// be co-located with expert weights.
    pub fn with_assignment(
        assignment: HashMap<String, usize>,
        stage: ZeROStage,
        rank: usize,
        world_size: usize,
    ) -> Self {
        let mut all_params: Vec<String> = assignment.keys().cloned().collect();
        all_params.sort();

        let owned_params: Vec<String> = all_params
            .iter()
            .filter(|name| assignment.get(*name) == Some(&rank))
            .cloned()
            .collect();

        Self {
            stage,
            rank,
            world_size,
            param_to_rank: assignment,
            owned_params,
            all_params,
        }
    }

    /// Whether this rank owns a parameter's optimizer state.
    pub fn owns_param(&self, param_name: &str) -> bool {
        self.param_to_rank.get(param_name) == Some(&self.rank)
    }

    /// Which rank owns a parameter's optimizer state.
    pub fn owner_of(&self, param_name: &str) -> Option<usize> {
        self.param_to_rank.get(param_name).copied()
    }

    /// Number of parameters owned by this rank.
    pub fn num_owned(&self) -> usize {
        self.owned_params.len()
    }

    /// Total number of parameters across all ranks.
    pub fn num_total(&self) -> usize {
        self.all_params.len()
    }

    /// Memory reduction factor compared to non-ZeRO training.
    ///
    /// For Stage 1 (optimizer states only): world_size reduction in optimizer memory.
    /// For Stage 2: world_size reduction in optimizer + gradient memory.
    pub fn memory_reduction_factor(&self) -> f32 {
        self.world_size as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param_names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("param_{i:03}")).collect()
    }

    #[test]
    fn round_robin_assignment() {
        let params = param_names(10);
        let p = ZeROPartitioner::new(&params, ZeROStage::Stage1, 0, 3);

        assert_eq!(p.num_total(), 10);
        // Rank 0 gets params 0, 3, 6, 9 → 4 params
        assert_eq!(p.num_owned(), 4);
        assert!(p.owns_param("param_000"));
        assert!(!p.owns_param("param_001"));
        assert!(p.owns_param("param_003"));
    }

    #[test]
    fn all_params_covered() {
        let params = param_names(7);
        let world_size = 3;

        let mut total_owned = 0;
        for rank in 0..world_size {
            let p = ZeROPartitioner::new(&params, ZeROStage::Stage1, rank, world_size);
            total_owned += p.num_owned();

            // Verify no overlap between ranks.
            for name in &p.owned_params {
                assert_eq!(p.owner_of(name), Some(rank));
            }
        }

        assert_eq!(total_owned, 7);
    }

    #[test]
    fn explicit_assignment() {
        let mut assignment = HashMap::new();
        assignment.insert("expert.0.weight".to_string(), 0);
        assignment.insert("expert.1.weight".to_string(), 0);
        assignment.insert("expert.2.weight".to_string(), 1);
        assignment.insert("expert.3.weight".to_string(), 1);

        let p = ZeROPartitioner::with_assignment(assignment, ZeROStage::Stage2, 0, 2);
        assert_eq!(p.num_owned(), 2);
        assert!(p.owns_param("expert.0.weight"));
        assert!(p.owns_param("expert.1.weight"));
        assert!(!p.owns_param("expert.2.weight"));
    }

    #[test]
    fn memory_reduction_factor() {
        let params = param_names(10);
        let p = ZeROPartitioner::new(&params, ZeROStage::Stage1, 0, 4);
        assert_eq!(p.memory_reduction_factor(), 4.0);
    }
}
