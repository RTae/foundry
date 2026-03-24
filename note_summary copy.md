1. Has any work proposed materializing or caching a sparse subset of pairwise interaction features in protein structure prediction or protein diffusion models to reduce GPU memory, instead of storing the full pairwise matrix or recomputing it at every step?


# **Sparse pairwise features in protein models:** existing work is related but not exactly your proposal

## Relevant ideas in current literature

**1. Sparse / reduced pairwise representations**

- Classical protein design often uses **sparse residue interaction graphs**, keeping only residue pairs within distance/energy cutoffs to reduce search space; omitted interactions are not materialized, but this is for energy evaluation rather than GPU activations  (Jain et al., 2017).  
- Distance-matrix sparsification has been studied for **protein–protein structure alignment**, where only a subset of inter-residue distances is stored while still allowing reconstruction/alignments  (Mucherino et al., 2011).  
- Coarse-grained and MD codes (UNRES, GROMACS) widely use **neighbor lists / interaction lists** so that only near-neighbor pairs are stored and updated, instead of a full O(N²) pair list  (Páll et al., 2020; Sieradzan et al., 2022). This is conceptually identical to caching a sparse set of pairwise interactions, but in physics-based simulation rather than deep learning.

### Example table

| Context | What’s stored | Sparsity use | Citations |
|--------|---------------|--------------|-----------|
| Protein design | Sparse residue interaction graphs (subset of pairwise energies) | Prunes combinatorial search; ignores long-range pairs |  (Jain et al., 2017)|
| MD / coarse-grained | Verlet / neighbor lists of close atom/residue pairs | Avoids full O(N²) pairwise list, updated periodically |  (Páll et al., 2020; Sieradzan et al., 2022)|
| Distance-matrix alignment | Sparse distance matrices | Enables tractable exact alignments |  (Mucherino et al., 2011)|

**Figure 1:** Where sparse pairwise information is explicitly materialized.

**2. Memory-efficient deep protein models**

- Recent works on structure/diffusion or AF-like models mainly reduce memory via **operator/kernel design or dimensionality reduction**, not via explicit sparse caching of pairwise tensors:
  - SALAD uses **sparse all-atom denoising**, but “sparse” refers to atom selection & model size, not a cached sparse pair-tensor  (Jendrusch & Korbel, 2025).  
  - SiDGen uses **coarse-stride folding with nearest-neighbor upsampling** to avoid quadratic pair tensors, effectively working on a reduced-resolution pair representation rather than a sparse cache  (Sanghvi et al., 2025).  
  - MegaFold optimizes EvoAttention and triangle updates with **kernel fusion and memory-efficient attention**, but still treats pair features as dense activations  (La et al., 2025).  
  - ESME leverages FlashAttention and recomputation for sequence models; they explicitly note FlashAttention is unusable for **structure prediction because full attention matrices must be stored**  (Çelik & Xie, 2024).

No retrieved work explicitly describes **materializing and reusing a dynamically selected sparse subset of pairwise interaction features during training/inference** for protein structure or protein diffusion models to save GPU memory.

## Conclusion

Existing protein design and MD communities rely heavily on sparse interaction lists/graphs, and some generative models reduce pairwise costs via coarse graining or kernel tricks. However, no cited work matches a scheme that **caches a learnable sparse subset of pairwise features instead of the full pairwise matrix** in AlphaFold-like or diffusion-style protein structure models. This appears to be an open, system/architecture-level research direction.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Jendrusch, M., & Korbel, J. (2025). Efficient protein structure generation with sparse denoising models. *Nature machine intelligence*, 7, 1429 - 1445. https://doi.org/10.1038/s42256-025-01100-z
 
Sanghvi, S., Ranjan, N., & Karmakar, T. (2025). SiDGen: Structure-informed Diffusion for Generative modeling of Ligands for Proteins. *ArXiv*, abs/2511.09529. https://doi.org/10.48550/arxiv.2511.09529
 
Çelik, M., & Xie, X. (2024). Efficient inference, training, and fine-tuning of protein language models. *iScience*, 28. https://doi.org/10.1101/2024.10.22.619563
 
Páll, S., Zhmurov, A., Bauer, P., Abraham, M., Lundborg, M., Gray, A., Hess, B., & Lindahl, E. (2020). Heterogeneous Parallelization and Acceleration of Molecular Dynamics Simulations in GROMACS. *The Journal of chemical physics*, 153 13, 134110. https://doi.org/10.1063/5.0018516
 
La, H., Gupta, A., Morehead, A., Cheng, J., & Zhang, M. (2025). MegaFold: System-Level Optimizations for Accelerating Protein Structure Prediction Models. *ArXiv*, abs/2506.20686. https://doi.org/10.48550/arxiv.2506.20686
 
Jain, S., Jou, J., Georgiev, I., & Donald, B. (2017). A critical analysis of computational protein design with sparse residue interaction graphs. *PLoS Computational Biology*, 13. https://doi.org/10.1371/journal.pcbi.1005346
 
Sieradzan, A., Sans-Duñó, J., Lubecka, E., Czaplewski, C., Lipska, A., Leszczynski, H., Ocetkiewicz, K., Proficz, J., Czarnul, P., Krawczyk, H., & Liwo, A. (2022). Optimization of parallel implementation of UNRES package for coarse‐grained simulations to treat large proteins. *Journal of Computational Chemistry*, 44, 602. https://doi.org/10.1002/jcc.27026
 
Mucherino, A., Wohlers, I., Klau, G., & Andonov, R. (2011). Sparsifying Distance Matrices for Protein-Protein Structure Alignments. *Mathematical Programming*. 
 


2. Do any deep learning systems detect when a tensor transitions from dense access to sparse access during iterative inference (such as diffusion denoising), and automatically change the tensor's storage representation?

# No, **current systems don’t automatically flip dense↔sparse formats based on per-step access patterns**



**Figure 1:** Consensus on automatic dense-to-sparse storage switching.

## What exists today

**1. Adaptive *format* selection is mostly offline or per-tensor, not per-iteration**

- Sparse tensor libraries and compilers choose formats (COO, CSR, CSF, HiCOO, ALTO, etc.) based on global tensor statistics, not on dynamic access changes during inference  (Laukemann et al., 2024; Helal et al., 2021; Sun et al., 2021).  
- SpTFS predicts the best sparse storage format for a given tensor using ML, but this happens once per tensor for MTTKRP, not as the computation pattern evolves  (Sun et al., 2021).  
- ALTO and AGCRS provide mode-agnostic, compressed formats whose parameters are tuned to tensor characteristics, again static with respect to an iterative solver or network  (Laukemann et al., 2024; Helal et al., 2021; Mehrab et al., 2025).

**2. Dynamic sparse compilation / dispatch, but driven by *sparsity layout*, not access pattern**

- STen in PyTorch dispatches to dense or various sparse kernels based on the tensor’s **declared sparsity layout**, and may on-the-fly convert between sparse formats or back to dense if that is faster  (Ivanov et al., 2023). This is a one-shot decision per operator call, not an online “detect dense→sparse transition and flip representation” within a long iterative process.  
- Frameworks like SparTA and TSTC/STCO track sparsity attributes (TeSA, TIE) through the graph and generate specialized sparse operators, but the representation is tied to the pruning/structured sparsity pattern, not monitored access locality over time  (Zheng et al., 2022; Huang et al., 2024).

### Examples of related but different adaptivity

| System / idea | Adaptivity type | Granularity | Citations |
|---------------|-----------------|------------|-----------|
| STen | Pick sparse vs dense impl, convert layouts | Per op-call |  (Ivanov et al., 2023)|
| SparTA (TeSA) | End-to-end sparse operator specialization | Model / layer |  (Zheng et al., 2022)|
| SpTFS, ALTO | Choose/storage format based on tensor stats | Per tensor |  (Laukemann et al., 2024; Sun et al., 2021; Helal et al., 2021)|

**Figure 2:** Scope and level of tensor adaptivity in systems.

## Conclusion

There is active work on choosing sparse formats and kernels, and even switching between them, but it is driven by *static sparsity patterns* or coarse profiling, not by **online detection that an iteratively-used tensor has shifted from dense to sparse access and then changing its storage representation mid-run**. That specific kind of automatic, per-tensor, per-iteration adaptation appears to be an open research area.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Laukemann, J., Helal, A., Isaac, S., Anderson, G., Checconi, F., Jesmin, Y., Tithi, J., Ranadive, T., Gravelle, B., Choi, J., & Tithi, J. (2024). Accelerating Sparse Tensor Decomposition Using Adaptive Linearized Representation. *IEEE Transactions on Parallel and Distributed Systems*, 36, 1025-1041. https://doi.org/10.1109/tpds.2025.3553092
 
Huang, S., Liu, F., Li, T., Wang, Z., Yang, N., Li, H., & Jiang, L. (2024). STCO: Enhancing Training Efficiency via Structured Sparse Tensor Compilation Optimization. *ACM Transactions on Design Automation of Electronic Systems*, 30, 1 - 22. https://doi.org/10.1145/3701033
 
Ivanov, A., Dryden, N., Ben-Nun, T., Ashkboos, S., & Hoefler, T. (2023). STen: Productive and Efficient Sparsity in PyTorch. *ArXiv*, abs/2304.07613. https://doi.org/10.48550/arxiv.2304.07613
 
Sun, Q., Liu, Y., Yang, H., Dun, M., Luan, Z., Gan, L., Yang, G., & Qian, D. (2021). Input-Aware Sparse Tensor Storage Format Selection for Optimizing MTTKRP. *IEEE Transactions on Computers*, 71, 1968-1981. https://doi.org/10.1109/tc.2021.3113028
 
Mehrab, M., Opi, H., & Hasan, K. (2025). AGCRS: An Adaptive Generalized Storage Scheme for Large Sparse Tensors. *2025 IEEE High Performance Extreme Computing Conference (HPEC)*, 1-6. https://doi.org/10.1109/hpec67600.2025.11196572
 
Zheng, N., Lin, B., Zhang, Q., , L., Yang, Y., Yang, F., Wang, Y., Yang, M., & Zhou, L. (2022). SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute. **, 213-232. 
 
Helal, A., Laukemann, J., Checconi, F., Tithi, J., Ranadive, T., Petrini, F., & Choi, J. (2021). ALTO: adaptive linearized storage of sparse tensors. *Proceedings of the 35th ACM International Conference on Supercomputing*. https://doi.org/10.1145/3447818.3461703
 
Huang, S., Liu, F., Li, T., Wang, Z., Li, H., & Jiang, L. (2024). TSTC: Enabling Efficient Training via Structured Sparse Tensor Compilation. *2024 29th Asia and South Pacific Design Automation Conference (ASP-DAC)*, 884-889. https://doi.org/10.1109/asp-dac58780.2024.10473981
 
3. How do protein diffusion generative models like RFDiffusion, FrameDiff, Chroma, or Genie handle the memory cost of pairwise atom or residue embeddings during iterative denoising steps?

# **Protein diffusion models control pairwise memory by avoiding full N×N tensors and using sparse or factorized structures**, rather than storing dense pairwise features at every step.

## RFdiffusion and RoseTTAFold-style models

RoseTTAFold/RFdiffusion maintain **per-residue (“state”) and residue–residue (“pair”) embeddings**, but these are held only for the *current* diffusion step and layer, not as a full time-history  (Michalewicz et al., 2025).  
Pair features are updated with triangle/attention blocks that are quadratic in sequence length, but the model does **not cache pairwise embeddings across denoising steps**; each step discards intermediates after backprop/inference, so peak memory is dominated by one forward pass plus activations needed for training  (Lisanza et al., 2024; Michalewicz et al., 2025).  

SCUBA-D and ProteinGenerator, which also use RoseTTAFold-derived architectures, follow the same pattern: diffusion is over sequence/structure, while the network internally recomputes pairwise features each step, yielding run-time and memory roughly linear in the number of denoising steps and quadratic in length per step  (Liu et al., 2024; Lisanza et al., 2024).  

## Chroma: sub-quadratic long-range reasoning

Chroma explicitly addresses **scaling and memory** by replacing full all-to-all pairwise attention with a **random long-range graph neural network** whose connectivity scales **O(N) or O(N log N)** instead of O(N²)  (Ingraham et al., 2022).  
Inter-residue geometry is represented in a factorized way and solved via a fast global consensus step, avoiding a persistent dense N×N pair tensor per layer and step  (Ingraham et al., 2022).  

### Example of architectural strategies

| Model family | Memory strategy for pairwise info | Citations |
|-------------|-----------------------------------|-----------|
| RFdiffusion / PG / SCUBA-D | Recompute pair embeddings per step; no time caching; quadratic per step |  (Liu et al., 2024; Lisanza et al., 2024; Michalewicz et al., 2025)|
| Chroma | Sparse/random long-range graph edges; sub‑quadratic scaling; no full dense N×N |  (Ingraham et al., 2022)|
| MapDiff, graph-based IPF | Graph edges over structure, not dense all-residue matrices |  (Bai et al., 2024; Yi et al., 2023; Bai et al., 2025)|

**Figure 1:** Architectural strategies to limit pairwise memory

## Conclusion

Current protein diffusion generators reduce pairwise-memory cost mainly by (i) **not storing full pairwise histories across steps**, and (ii) using **graph/sparse or sub-quadratic architectures** instead of dense N×N attention at every layer. Explicit step-to-step caching of large pairwise tensors is generally avoided.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Liu, Y., Wang, S., Dong, J., Chen, L., Wang, X., Wang, L., Li, F., Wang, C., Zhang, J., Wang, Y., Wei, S., Chen, Q., & Liu, H. (2024). De novo protein design with a denoising diffusion network independent of pretrained structure prediction models. *Nature Methods*, 21, 2107 - 2116. https://doi.org/10.1038/s41592-024-02437-w
 
Lisanza, S., Gershon, J., Tipps, S., Sims, J., Arnoldt, L., Hendel, S., Simma, M., Liu, G., Yase, M., Wu, H., Tharp, C., Li, X., Kang, A., Brackenbrough, E., Bera, A., Gerben, S., Wittmann, B., McShan, A., & Baker, D. (2024). Multistate and functional protein design using RoseTTAFold sequence space diffusion. *Nature Biotechnology*, 43, 1288 - 1298. https://doi.org/10.1038/s41587-024-02395-w
 
Bai, P., Miljkovi'c, F., Liu, X., De Maria, L., Croasdale-Wood, R., Rackham, O., & Lu, H. (2024). Mask-prior-guided denoising diffusion improves inverse protein folding. *Nature Machine Intelligence*, 7, 876 - 888. https://doi.org/10.1038/s42256-025-01042-6
 
Yi, K., Zhou, B., Shen, Y., Liò, P., & Wang, Y. (2023). Graph Denoising Diffusion for Inverse Protein Folding. *ArXiv*, abs/2306.16819. https://doi.org/10.48550/arxiv.2306.16819
 
Ingraham, J., Baranov, M., Costello, Z., Frappier, V., Ismail, A., Tie, S., Wang, W., Xue, V., Obermeyer, F., Beam, A., & Grigoryan, G. (2022). Illuminating protein space with a programmable generative model. *Nature*, 623, 1070 - 1078. https://doi.org/10.1038/s41586-023-06728-8
 
Michalewicz, K., Jin, C., Teare, P., Diethe, T., Barahona, M., Bravi, B., & Mullokandov, A. (2025). Protein generation with embedding learning for motif diversification. **. 
 
Bai, P., Miljković, F., Liu, X., De Maria, L., Croasdale-Wood, R., Rackham, O., & Lu, H. (2025). Mask-prior-guided denoising diffusion improves inverse protein folding. *Nature Machine Intelligence*. https://doi.org/10.1038/s42256-025-01042-6


4. Is pre-gathering or pre-indexing sparse features from a dense tensor more efficient than recomputing them on the fly in GPU-based neural network inference, particularly for attention mechanisms with fixed neighbor indices?

# **Pre-indexing sparse attention with fixed neighbors is usually more efficient than recomputing on the fly, if reuse is high**

## When pre-gathering / pre-indexing helps

For attention mechanisms with **fixed neighbor indices reused across many queries, layers, or tokens**, precomputing an index structure (or pre-gathered layouts) and then running **specially tuned sparse kernels** is generally more efficient than repeatedly computing/gathering from a dense tensor each step.

Sparse CNN and attention systems show that once a sparsity pattern is known, using a **coordinate / CSR-like index plus sparse kernels** reduces both memory traffic and FLOPs vs. dense-style gathers, especially at high sparsity  (Hackel et al., 2018; Zhu et al., 2019; Yao et al., 2018). GPU libraries such as TorchSparse++ auto-tune between gather‑GEMM‑scatter and more fused dataflows, precisely because naïve per-step gathers can be the bottleneck; optimized kernels that assume a fixed sparse pattern achieve up to ~3× end‑to‑end speedups  (Tang et al., 2023).  

In long‑context LLMs, systems like **MInference** and **FlexPrefill** explicitly **build and reuse sparse index sets for attention** (e.g., A‑shape, Vertical‑Slash, Block‑Sparse heads) and then execute attention via pattern-specific sparse kernels, demonstrating up to **10× latency reduction** for prefill compared with dense attention  (Jiang et al., 2024; Lai et al., 2025). These works show that paying an upfront cost to construct indices is amortized over many attention computations as sequence length grows  (Jiang et al., 2024; Lai et al., 2025).  

Hardware–software co-design for sparse attention (e.g., Sanger, RM‑STC) likewise assumes **structured sparse patterns plus prearranged indices** to avoid repeated irregular gathers and to keep PEs well utilized  (Lu et al., 2021; Huang et al., 2023).  

### Practical takeaway for fixed-neighbor attention

- **High reuse of a fixed neighbor pattern (e.g., local windows, k-NN graph that doesn’t change)** → pre-index neighbors and use a sparse / block-sparse attention kernel.  
- **Low reuse or changing neighbors** → per-step dense computation or lightweight on-the-fly selection may be preferable, since index construction and indirect memory accesses can dominate  (Hassani et al., 2025; Lai et al., 2025).  

## Conclusion

For GPU-based attention with **fixed, repeatedly used neighbor indices**, pre-gathering or pre-indexing sparse features and running pattern-aware sparse kernels is typically more efficient than recomputing dense features or doing ad hoc gathers each time, especially at large sequence lengths or high sparsity.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Hackel, T., Usvyatsov, M., Galliani, S., Wegner, J., & Schindler, K. (2018). Inference, Learning and Attention Mechanisms that Exploit and Preserve Sparsity in CNNs. *International Journal of Computer Vision*, 128, 1047 - 1059. https://doi.org/10.1007/s11263-020-01302-5
 
Tang, H., Yang, S., Liu, Z., Hong, K., Yu, Z., Li, X., Dai, G., Wang, Y., & Han, S. (2023). TorchSparse++: Efficient Training and Inference Framework for Sparse Convolution on GPUs. *2023 56th IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 225-239. https://doi.org/10.1145/3613424.3614303
 
Hassani, A., Zhou, F., Kane, A., Huang, J., Chen, C., Shi, M., Walton, S., Hoehnerbach, M., Thakkar, V., Isaev, M., Zhang, Q., Xu, B., Wu, H., Hwu, W., Liu, M., & Shi, H. (2025). Generalized Neighborhood Attention: Multi-dimensional Sparse Attention at the Speed of Light. *ArXiv*, abs/2504.16922. https://doi.org/10.48550/arxiv.2504.16922
 
Zhu, M., Zhang, T., Gu, Z., & Xie, Y. (2019). Sparse Tensor Core: Algorithm and Hardware Co-Design for Vector-wise Sparse Neural Networks on Modern GPUs. *Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture*. https://doi.org/10.1145/3352460.3358269
 
Jiang, H., Li, Y., Zhang, C., Wu, Q., Luo, X., Ahn, S., Han, Z., Abdi, A., Li, D., Lin, C., Yang, Y., & Qiu, L. (2024). MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention. *ArXiv*, abs/2407.02490. https://doi.org/10.48550/arxiv.2407.02490
 
Lai, X., Lu, J., Luo, Y., , Y., & Zhou, X. (2025). FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference. *ArXiv*, abs/2502.20766. https://doi.org/10.48550/arxiv.2502.20766
 
Yao, Z., Cao, S., Xiao, W., Zhang, C., & Nie, L. (2018). Balanced Sparsity for Efficient DNN Inference on GPU. **, 5676-5683. https://doi.org/10.1609/aaai.v33i01.33015676
 
Huang, G., Wang, Z., Tsai, P., Zhang, C., Ding, Y., & Xie, Y. (2023). RM-STC: Row-Merge Dataflow Inspired GPU Sparse Tensor Core for Energy-Efficient Sparse Acceleration. *2023 56th IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 338-352. https://doi.org/10.1145/3613424.3623775
 
Lu, L., Jin, Y., Bi, H., Luo, Z., Li, P., Wang, T., & Liang, Y. (2021). Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture. *MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture*. https://doi.org/10.1145/3466752.3480125
 

4. Has any work proposed pre-materializing the sparse attention bias from a dense pairwise representation and freeing the dense tensor, rather than gathering from the dense tensor at every attention call?

# No existing work directly pre-materializes a sparse bias tensor from a dense pairwise bias and then frees the dense tensor

## What current work does with attention bias

Most efficiency work around attention **with bias** focuses on *how to apply the bias without paying dense quadratic cost each call*, not on a one‑time materialization of a sparse biased tensor.

**FlashBias** targets dense pairwise biases (e.g., relative position, AlphaFold pair biases) and shows that reading a full dense bias matrix from HBM breaks the FlashAttention-style fused pipeline  (Wu et al., 2025). Instead of pre-gathering a sparse tensor, it:
- Analyzes the bias’ **low‑rank structure** and factorizes it so the bias is reconstructed on the fly using small matrices  (Wu et al., 2025).  
- Keeps attention fully dense (or block‑dense) but with cheaper bias application, achieving 1.5–2× speedups on GPUs  (Wu et al., 2025).  

This is essentially the opposite of your proposal: it **avoids storing the full dense bias** by using a compact parametric (low‑rank) form, not by precomputing and storing a sparse subset.

Sparse‑attention accelerators and methods (Sanger, CPSAA, SPRINT, static structured sparse masks, SEA, etc.) typically:  
- Build or predict **sparse masks on scores**, then do SDDMM/SpMM or dense-with-mask kernels  (Lu et al., 2021; Li et al., 2022; Yazdanbakhsh et al., 2022; Dai et al., 2023; Lee et al., 2023).  
- Sometimes encode masks in efficient formats (e.g., FlatCSR) to reduce irregular access  (Lee et al., 2023).  

However, these works either:
- Treat the bias as part of the score computation (QKᵀ + bias) without separating and sparsifying it, or  
- Study sparsity in the attention *weights* themselves, not in a separate pre‑materialized bias tensor  (Lu et al., 2021; Dai et al., 2023; Yazdanbakhsh et al., 2022; Lee et al., 2023).

## Conclusion

Within the reviewed literature, there is **no explicit proposal** that, starting from a dense pairwise bias, pre‑computes and stores a *sparse* attention-bias tensor (e.g., CSR over fixed neighbors), frees the dense tensor, and then uses only that sparse object at each attention call. Closest ideas are low‑rank compression of dense bias (FlashBias) and efficient sparse masks over scores, making your exact pattern a plausible, but currently underexplored, design point.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Lu, L., Jin, Y., Bi, H., Luo, Z., Li, P., Wang, T., & Liang, Y. (2021). Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture. *MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture*. https://doi.org/10.1145/3466752.3480125
 
Li, H., Jin, H., Zheng, L., Liao, X., Huang, Y., Liu, C., Xu, J., Duan, Z., Chen, D., & Gui, C. (2022). CPSAA: Accelerating Sparse Attention Using Crossbar-Based Processing-In-Memory Architecture. *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 43, 1741-1754. https://doi.org/10.1109/tcad.2023.3344524
 
Dai, S., Genc, H., Venkatesan, R., & Khailany, B. (2023). Efficient Transformer Inference with Statically Structured Sparse Attention. *2023 60th ACM/IEEE Design Automation Conference (DAC)*, 1-6. https://doi.org/10.1109/dac56929.2023.10247993
 
Wu, H., Guo, M., , Y., Sun, Y., Wang, J., Matusik, W., & Long, M. (2025). FlashBias: Fast Computation of Attention with Bias. *ArXiv*, abs/2505.12044. https://doi.org/10.48550/arxiv.2505.12044
 
Yazdanbakhsh, A., Moradifirouzabadi, A., Li, Z., & Kang, M. (2022). Sparse Attention Acceleration with Synergistic In-Memory Pruning and On-Chip Recomputation. *2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 744-762. https://doi.org/10.1109/micro56248.2022.00059
 
Lee, H., Kim, J., Willette, J., & Hwang, S. (2023). SEA: Sparse Linear Attention with Estimated Attention Mask. *ArXiv*, abs/2310.01777. https://doi.org/10.48550/arxiv.2310.01777
 

5. Has any work proposed pre-materializing the sparse attention bias from a dense pairwise representation and freeing the dense tensor, rather than gathering from the dense tensor at every attention call?

# No existing work directly pre-materializes a sparse bias tensor from a dense pairwise bias and then frees the dense tensor

## What current work does with attention bias

Most efficiency work around attention **with bias** focuses on *how to apply the bias without paying dense quadratic cost each call*, not on a one‑time materialization of a sparse biased tensor.

**FlashBias** targets dense pairwise biases (e.g., relative position, AlphaFold pair biases) and shows that reading a full dense bias matrix from HBM breaks the FlashAttention-style fused pipeline  (Wu et al., 2025). Instead of pre-gathering a sparse tensor, it:
- Analyzes the bias’ **low‑rank structure** and factorizes it so the bias is reconstructed on the fly using small matrices  (Wu et al., 2025).  
- Keeps attention fully dense (or block‑dense) but with cheaper bias application, achieving 1.5–2× speedups on GPUs  (Wu et al., 2025).  

This is essentially the opposite of your proposal: it **avoids storing the full dense bias** by using a compact parametric (low‑rank) form, not by precomputing and storing a sparse subset.

Sparse‑attention accelerators and methods (Sanger, CPSAA, SPRINT, static structured sparse masks, SEA, etc.) typically:  
- Build or predict **sparse masks on scores**, then do SDDMM/SpMM or dense-with-mask kernels  (Lu et al., 2021; Li et al., 2022; Yazdanbakhsh et al., 2022; Dai et al., 2023; Lee et al., 2023).  
- Sometimes encode masks in efficient formats (e.g., FlatCSR) to reduce irregular access  (Lee et al., 2023).  

However, these works either:
- Treat the bias as part of the score computation (QKᵀ + bias) without separating and sparsifying it, or  
- Study sparsity in the attention *weights* themselves, not in a separate pre‑materialized bias tensor  (Lu et al., 2021; Dai et al., 2023; Yazdanbakhsh et al., 2022; Lee et al., 2023).

## Conclusion

Within the reviewed literature, there is **no explicit proposal** that, starting from a dense pairwise bias, pre‑computes and stores a *sparse* attention-bias tensor (e.g., CSR over fixed neighbors), frees the dense tensor, and then uses only that sparse object at each attention call. Closest ideas are low‑rank compression of dense bias (FlashBias) and efficient sparse masks over scores, making your exact pattern a plausible, but currently underexplored, design point.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Lu, L., Jin, Y., Bi, H., Luo, Z., Li, P., Wang, T., & Liang, Y. (2021). Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture. *MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture*. https://doi.org/10.1145/3466752.3480125
 
Li, H., Jin, H., Zheng, L., Liao, X., Huang, Y., Liu, C., Xu, J., Duan, Z., Chen, D., & Gui, C. (2022). CPSAA: Accelerating Sparse Attention Using Crossbar-Based Processing-In-Memory Architecture. *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 43, 1741-1754. https://doi.org/10.1109/tcad.2023.3344524
 
Dai, S., Genc, H., Venkatesan, R., & Khailany, B. (2023). Efficient Transformer Inference with Statically Structured Sparse Attention. *2023 60th ACM/IEEE Design Automation Conference (DAC)*, 1-6. https://doi.org/10.1109/dac56929.2023.10247993
 
Wu, H., Guo, M., , Y., Sun, Y., Wang, J., Matusik, W., & Long, M. (2025). FlashBias: Fast Computation of Attention with Bias. *ArXiv*, abs/2505.12044. https://doi.org/10.48550/arxiv.2505.12044
 
Yazdanbakhsh, A., Moradifirouzabadi, A., Li, Z., & Kang, M. (2022). Sparse Attention Acceleration with Synergistic In-Memory Pruning and On-Chip Recomputation. *2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 744-762. https://doi.org/10.1109/micro56248.2022.00059
 
Lee, H., Kim, J., Willette, J., & Hwang, S. (2023). SEA: Sparse Linear Attention with Estimated Attention Mask. *ArXiv*, abs/2310.01777. https://doi.org/10.48550/arxiv.2310.01777
 
