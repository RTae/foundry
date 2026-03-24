# Gather-Once (Materialize-Once) Memory Optimization for Pairwise Tensors in RFD3

## Motivation
Protein diffusion models like RFD3 require O(N²) pairwise tensors (e.g., P_LL) for each protein, which can become a major memory bottleneck, especially for large proteins. Existing solutions either keep the dense tensor in memory (wasting GPU RAM after the initial phase) or recompute the required values on demand (trading memory for compute).

## Proposed Approach: Gather-Once / Materialize-Once
- **Dense Phase:** Compute and use the full dense pairwise tensor (P_LL) during the initial model steps that require dense access.
- **Transition:** After the last dense access, gather the sparse working set (the subset of P_LL needed for subsequent steps) into a new tensor.
- **Free:** Immediately free the original dense tensor to reclaim memory.
- **Sparse Phase:** Reuse the gathered sparse tensor for all subsequent accesses, avoiding both memory waste and recomputation.

## Key Benefits
- **Memory Efficiency:** Frees up significant GPU memory after the dense phase, enabling larger proteins or batch sizes.
- **No Recomputation:** Avoids the compute overhead of on-demand recomputation (as in RFD3's low_memory mode).
- **Simplicity:** Minimal code changes; can be implemented as a mode or utility in the model pipeline.

## Closest Existing Concepts
- **Recompute-on-demand:** RFD3's low_memory mode, but this does not materialize a sparse working set for reuse.
- **Chunked computation:** (e.g., FastFold/AutoChunk) splits tensors but does not gather and reuse a sparse subset.
- **Materialized views (DB analogy):** Conceptually similar, but not implemented in protein modeling frameworks.

## Novelty
No existing protein diffusion model or system implements this gather-once/materialize-once approach for O(N²) pairwise tensors. The method is distinct from paging (PagedAttention), chunking, or recompute-on-demand.

## Implementation Plan
1. Profile memory and access patterns in RFD3 to empirically motivate the approach.
2. Integrate gather-once/materialize-once as a new mode ("Mode C") in RFD3.
3. Benchmark memory usage, speed, and accuracy across protein sizes.
4. Generalize to other models (e.g., OpenFold) if successful.