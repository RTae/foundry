# Why optimizing irregular sparse attention for AI4S is a good research target

### Question 1: Is sparse neighbor computation common in AI4S?

Yes. Sparse or sub-quadratic neighbor computation already appears across a range of AI4S and adjacent 3D modeling papers:

| Type | Category | Model / Paper | Venue / Status | AI4S Focus / Domain | Sparsity Mechanism |
|---|---|---|---|---|---|
| Deployed model | Protein structure design / diffusion | SALAD | *Nature Machine Intelligence*, 2025 | Protein diffusion | KNN sparse attention with pair features |
| Deployed model | Protein structure design / diffusion | RFD3 | 2025 repo model | Protein design | KNN sparse attention with pair bias |
| Deployed model | Molecular simulation / force fields | EquiformerV2 | *ICLR*, 2024 | Molecular simulation; SOTA on OC20 | Neighbor-list graph attention with interactions constrained by a radial cutoff graph |
| Related paper | Physics-informed modeling | Transolver: A Fast and Scalable Physics-Informed Solver for PDEs | *ICML*, 2024 | Physics-informed neural networks and large-scale physical systems | Physics-Attention uses learned slicing to project irregular 3D spatial data into 1D sequences, yielding sparse, sub-linear attention |
| Related paper | Irregular 3D data | Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems | *ICML*, 2025 | Large-scale physical systems on irregular grids | Uses ball tree partitioning and hierarchical coarsening/refinement to achieve linear-time attention over fixed-size local neighborhoods |
| Related paper | Irregular 3D data | BSA: Ball Sparse Attention for Large-scale Geometries | *ICML* workshop, 2025; arXiv preprint | Large-scale physical systems and irregular geometries | Adapts Native Sparse Attention to unordered point sets using ball-tree neighborhoods, yielding sub-quadratic sparse attention with global receptive field |
| Related paper | 3D spatial modeling | OctFormer: Octree-based Transformers for 3D Point Clouds | *CVPR*, 2023 | 3D spatial geometry | Uses octrees and Z-order sequencing, restricting attention to non-overlapping local regions |
| Related paper | Graph representation learning | DYNAMO-GAT: A Dynamical Systems-Inspired Pruning Strategy for Addressing Oversmoothing | *ICML*, 2025 | Graph representation learning | Adaptive pruning in GATs enforces sparse attention patterns during training |

---

### Question 2: Has anyone tried to optimize sparse attention?

Yes — in two directions. Each solved part of the problem but not the full pattern.

**Direction A: Block-sparse / flexible attention (LLM/general)**

| System | Venue | Contribution | Limitation |
|---|---|---|---|
| FlexAttention | PyTorch, Dong et al., 2024 | Supports arbitrary score modifications with block-sparse masks | Requires block-contiguous sparsity aligned to tile boundaries; does not naturally match irregular KNN indices scattered across memory |
| FlashInfer | *MLSys*, 2025 | Provides a customizable attention engine using BSR format | Requires block-contiguous sparsity aligned to tile boundaries; does not naturally match irregular KNN indices scattered across memory |

**Direction B: Irregular sparse GNN acceleration (systems)**

| System | Venue | Contribution | Limitation |
|---|---|---|---|
| DFSS | *PPoPP*, 2023 | Fused kernel that dynamically prunes dense attention scores to N:M structured sparse patterns | Optimizes simple SpMM-style aggregation, not softmax attention with pairwise bias and gating |
| DynaX | *ASPLOS*, 2025 | Extends DFSS with variable X:M pruning | Same limitation: focuses on SpMM-style aggregation |
| N:M Sparsity-Oriented Graph Reordering | *PPoPP*, 2025 | Transforms irregular graph sparsity into structured patterns for sparse tensor cores; reports up to 43x speedup on SpMM | Same limitation: focuses on SpMM-style aggregation |

All of these systems identify irregular memory access from neighbor gathering as a GPU performance problem. But they optimize simple SpMM aggregation rather than attention with softmax, pairwise bias, and gating.

---

### The gap

| Direction | What it solved | What remains open |
|---|---|---|
| Direction A | Flexible sparse attention | Block-aligned only; no irregular KNN indices |
| Direction B | Irregular sparse computation | Simple SpMM only; no attention + pair bias + softmax |

Nobody has built a fused kernel for irregular gather + attention + pair bias + softmax — the specific combination used by RFD3, SALAD, and structurally similar to the neighbor computation in EquiformerV2, MACE, and Allegro.

---

### Baselines for evaluation

| Baseline | Why it's fair |
|---|---|
| RFD3's current PyTorch implementation | The unfused gather + einsum + softmax path that everyone actually uses today |
| FlexAttention | Show it cannot express the KNN pattern, or show accuracy/performance gap with block-mask approximation |
| cuSPARSE / DGL SpMM | Standard library for sparse neighbor aggregation on GPUs |
| DFSS (*PPoPP* 2023) | Show their N:M pruning doesn't apply because you never have dense scores to prune |

---

### Papers to read

**Must read (kernel design and argument):**

| Priority | Paper | Why read it |
|---|---|---|
| 1 | **DFSS** (*PPoPP*, 2023) | Closest kernel-level prior work; fused SDDMM design you may adapt |
| 2 | **SALAD** (*Nat. Mach. Intell.*, 2025) | Closest architecture to RFD3; plausible second evaluation target |
| 3 | **N:M Graph Reordering** (*PPoPP*, 2025) | Irregular sparsity analysis at your target venue |
| 4 | **FlashAttention** (*NeurIPS*, 2022) | IO-complexity analysis you may need to adapt for the sparse case |
| 5 | **FlashAttention-T** (*PPoPP*, 2026) | Recent FlashAttention variant at your venue; relevant tensor core techniques |

**Must read (positioning and related work):**

| Priority | Paper | Why read it |
|---|---|---|
| 6 | **EquiformerV2** (*ICLR*, 2024) | Shows the pattern extends beyond protein design |
| 7 | **FlexAttention** (PyTorch, 2024) | Helps position why block-sparse masks do not match KNN indices cleanly |
| 8 | **FlashInfer** (*MLSys*, 2025) | Helps position BSR limitations against your irregular pattern |
| 9 | **DynaX** (*ASPLOS*, 2025) | Extends DFSS with algorithm-hardware co-design |
| 10 | **BSA** (ICML 2025 workshop / arXiv 2025) | Closest ball-tree sparse-attention follow-up to Erwin on irregular geometries |

**Skim (background and calibration):**

| Priority | Paper | Why read it |
|---|---|---|
| 11 | **Dai et al.** (*PPoPP*, 2026) | Calibrate what PPoPP expects for block-sparse LLM attention |
| 12 | **AlphaFold 3** (*Nature*, 2024) | Motivation for why dense pairwise modeling is expensive |
| 13 | **ADiT** (*ICML*, 2025) | Example of a system that avoids sparse attention entirely |
| 14 | **MACE** (*NeurIPS*, 2022) | Neighbor-list pattern in force fields |
| 15 | **Allegro** (*Nat. Commun.*, 2023) | Related neighbor-list pattern with different compute |