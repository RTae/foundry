# RFdiffusion3 (RFD3) – Architecture & Pipeline

This document summarizes how the RFD3 model in `models/rfd3` is organized and how inference/training flow through the codebase.

The structure of this document is intentionally top-down:
1. Start with a high-level block diagram.
2. Move to the denoising/recycling control flow.
3. Drill into module internals and per-step execution details.

## High-level purpose
RFD3 is an AlphaFold3-inspired diffusion model for de novo biomolecular interaction design. It denoises atom-level coordinates while optionally predicting sequences, supporting diverse conditioning tasks (motifs, nucleic acids, small molecules, symmetry, etc.).

## Repository layout (RFD3-specific)
- `src/rfd3/engine.py` – inference engine that orchestrates parsing inputs, composing Hydra configs, running batches, and saving CIF/JSON outputs.
- `src/rfd3/cli.py` – Typer CLI (`rfd3 design ...`) that composes Hydra configs and calls the engine.
- `src/rfd3/model/` – core model: `RFD3` wrapper, diffusion module, sampler, layer building blocks, cfg utilities.
- `src/rfd3/trainer/` – Lightning-Fabric trainer (`AADesignTrainer`), recycling schedule, validation dump helpers.
- `src/rfd3/transforms/` – data transforms/conditioning pipelines for training & inference (motifs, symmetry, hbonds, NA/ppi, etc.).
- `configs/` – Hydra defaults for inference, model, trainer, datasets, callbacks, logging.
- `docs/` – user docs & examples; `.json` example inputs live in `docs/examples/`.

## Inference stack
1. **CLI entry** (`rfd3 design ...`): `cli.py` composes `configs/inference.yaml` plus overrides. If no engine provided, defaults to `inference_engine=rfdiffusion3`.
2. **Inference engine** (`engine.py`, class `RFD3InferenceEngine`):
   - Loads inputs from JSON/YAML or PDB via `rfd3.inference.input_parsing`.
   - Applies optional `specification` overrides and validates inputs.
   - Builds data loader via `assemble_distributed_inference_loader_from_json`.
   - Instantiates model + sampler using checkpoint config, applying overrides from Hydra.
   - Iterates batches, calling `AADesignTrainer.validation_step` to run forward diffusion rollout and collect metrics.
   - Dumps outputs as `.cif.gz` (plus trajectories if requested) and metadata JSON.
3. **Sampler** (`model/inference_sampler.py`):
   - Constructs EDM-style noise schedule (AF3 supplement) with optional partial diffusion.
   - Initializes noisy structure (motif atoms can be kept fixed; optional jitter).
   - Steps through schedule calling diffusion module; supports classifier-free guidance and symmetry handling.
   - Optionally realigns motif each step and collects trajectories/entropy.

Conceptually, one inference run has two nested loops:
- **Outer loop (denoising timesteps)**: advances structure from higher noise to lower noise.
- **Inner loop (recycling)**: refines predictions multiple times at a single timestep.

## Model core (`model/RFD3.py` + `model/RFD3_diffusion_module.py`)
- **TokenInitializer**: builds initial per-atom (`Q_L`, `C_L`, pairwise `P_LL`) and per-token (`S_I`, `Z_II`) features from parsed inputs (`f`). Supports chunked pairwise mode (`RFD3_LOW_MEMORY_MODE=1`).
- **DiffusionModule** (`RFD3DiffusionModule`): UNet-like, recycling architecture:
  - Time embedding via Fourier embeddings -> projection into atom/token channels.
  - **LocalAtomTransformer encoder**: atom-level self-attention using neighborhood indices (`create_attention_indices`) and optional chunked pairwise embedder.
  - Downcasts pooled atom reps to token space (`Downcast`) and sequence head prep.
  - **DiffusionTokenEncoder**: fuses token + pairwise features with current coordinates; updates `S_I`, `Z_II`.
  - **LocalTokenTransformer**: token-level attention operating on CA positions; optional full attention unless low-memory.
  - **CompactStreamingDecoder**: cross-updates atom/token streams; outputs refined atom reps (`Q_L`) and auxiliary heads.
  - **Output heads**: position update head (`to_r_update`), scaling back to coordinates (EDM / noise_pred / unconditioned), sequence logits via `LinearSequenceHead`, pairwise distogram bucketing for recycling.
  - **Recycling**: `forward_with_recycle` reuses outputs (`X_L`, `D_II_self`) for `n_recycle` iterations (train: provided; eval: config default).
- **Classifier-free guidance**: optional, strips conditioning features (`cfg_features`) for reference pass, blended inside sampler.

## Training loop (`trainer/rfd3.py`)
- Trainer (`AADesignTrainer`) extends `FabricTrainer` (Lightning Fabric):
  - Assembles network inputs (noisy coords + timestep + feature dict), handles NaNs, and enforces shape checks.
  - Randomizes recycle count per example using schedule from `trainer/recycling.py`.
  - Loss/metrics configured via Hydra (`configs/trainer/loss/` and `configs/trainer/metrics/`).
  - Validation/test phases also use full diffusion rollout; metrics include backbone, hbonds, design quality.

## Data & conditioning pipeline
- **Transforms** (`transforms/pipelines.py`, `design_transforms.py`, `training_conditions.py`, etc.): build feature dict `f` with atom/token mappings, masks for fixed motifs, symmetry constraints, NA/small-molecule conditioning, hbonds (HBPLUS), virtual atoms, and RASA.
- **Input specs**: JSON/YAML schemas in `docs/input.md`; convenience examples in `docs/examples/*.json` with matching markdown guides.
- **Hydra configs**: 
  - `configs/inference_engine/*.yaml` set engine defaults (batch size, sampler params, cleanup flags).
  - `configs/model/rfd3_base.yaml` wires channels (`c_atom`, `c_atompair`, `c_s`, `c_z`), module choices, and sampler defaults.
  - `configs/model/samplers/*.yaml`, `schedulers/*.yaml`, `optimizers/*.yaml` provide swap-in components.

## Outputs
- Structures written as `*.cif.gz` (and optionally denoised/noisy trajectories) with conditioning annotations (`SAVED_CONDITIONING_ANNOTATIONS`).
- Metadata JSON per design includes seeds, cfg parameters, metrics, and alignment info.

## Overall architecture (high-level)
```mermaid
flowchart LR
    IN[Residue and atomic features] --> FI[Feature Initializer\nTokenInitializer]

   subgraph DNS[Denoising Step]
      direction LR
      subgraph RSTEP[Recycle Step]
         direction LR
         ENC[Encoder\nLocalAtomTransformer] --> TR[Transformer\nLocalTokenTransformer]
         TR --> DEC[Decoder\nCompactStreamingDecoder]
      end

      DEC --> HDS[Heads\nposition + sequence + distogram]
      HDS -. Recycling .-> ENC
      HDS --> XNEXT[X_t -> X_t-1]
   end

   FI --> ENC
   XNEXT --> SAMP[Sampler Step\nEDM + CFG + symmetry]
    SAMP --> O[Output\nstructures + metadata]

    SAMP -. Next timestep .-> ENC
```

This top-level diagram is intentionally block-oriented. The next sections break down each module in detail.

Reading guide for this diagram:
- `Feature Initializer` prepares model state from parsed residue/atom features.
- `Recycle Step` is the inner refinement loop (`Encoder -> Transformer -> Decoder -> Heads`).
- `Denoising Step` performs one transition from `X_t` to `X_t-1`.
- `Sampler Step` advances the schedule and decides whether another timestep is needed.

## Bridge: from overview to internals
The overview above tells you **what** happens in broad blocks. The next diagram shows **how** one denoising step executes in the model runtime:
- Stage A: initialize state from `f` and timestep inputs.
- Stage B: run recycle iterations (`encoder -> token encoder -> transformer -> decoder -> heads`).
- Stage C: produce `X_t-1` and auxiliary predictions.
- Stage D: sampler decides whether to continue to the next timestep.

## Model runtime flow (detailed control flow)
```mermaid
flowchart LR
   subgraph OuterLoop[Outer loop: denoising timesteps]
      direction LR
      A0[Step input\nX_t + t + f] --> A1[Stage A\nTokenInitializer]
      A1 --> B0[Stage B entry\nDiffusionModule state]

      subgraph InnerLoop[Inner loop: recycle iterations]
         direction LR
               B0 --> B1[Atom encoder\nLocalAtomTransformer\nlocal or sparse pair-bias attention]
               B1 --> B2[Token encoder\nDiffusionTokenEncoder]
               B2 --> B3[Token transformer\nLocalTokenTransformer\nlocal attention indices]
               B3 --> B4[Decoder\nCompactStreamingDecoder\nupcast or downcast cross-attention]
         B4 --> B5[Heads\nto_r_update + sequence + distogram]
         B5 --> BR{More recycle iterations?}
         BR -- yes --> B1
         BR -- no --> C0[Stage C entry\nrecycle complete]
      end

      C0 --> C1[scale_positions_out]
      C1 --> C2[Step output\nX_t-1 + D_II_self + sequence outputs]
      C2 --> D1[Stage D\nSampler transition]
      D1 --> D2{More denoising steps?}
      D2 -- yes --> A0
      D2 -- no --> Z[Final structures + metadata]
   end

   subgraph CFG[Optional CFG reference path]
      R0[strip f by cfg_features] --> R1[TokenInitializer ref]
      R1 --> R2[Reference forward]
   end
   D1 -. blends with .- R2
```

Key signals: `f` (conditioning features), `X_t` (coordinates at current step), `t` (noise level). The inner recycle loop refines within one timestep, while the outer denoising loop advances from `X_t` to `X_t-1` until sampling completes.

This diagram is a runtime control-flow view. It emphasizes loop boundaries, stage transitions, and where CFG blending enters the denoising step.

## Module breakdown (encoder, transformer, decoder)
```mermaid
flowchart LR
   In[f, X_t, t, initializer outputs] --> Prep[Time + preprocessing\nFourierEmbedding, process_r/c, token pooling/projection]
   Prep --> Loop{{Recycle loop\nfor i in n_recycle}}

   subgraph Core[Core model path]
      direction LR
      ENC[Encoder\nLocalAtomTransformer]
      TOK[Token encoder\nDiffusionTokenEncoder + Pairformer stack]
      TR[Token transformer\nLocalTokenTransformer]
      DEC[Decoder\nCompactStreamingDecoder\nn_blocks: Upcast then AtomTransformer\nthen Downcast once]
      HD[Heads\nto_r_update + LinearSequenceHead + distogram bucketizer]
      ENC --> TOK --> TR --> DEC --> HD
   end

   Loop --> ENC
   HD --> Loop

   IDX_a[create_attention_indices (atom path)] -.-> ENC
   IDX_a -.-> DEC
   IDX_t[create_attention_indices (token path per recycle)] -.-> TR

   HD --> Xout[final X_L]
   HD --> Sout[final sequence outputs]
   HD --> Dout[final D_II_self]
```

Legend: solid arrows are main data flow; dashed arrows are attention-control paths.

This module-level diagram aligns with `RFD3_diffusion_module.py`, `layers/encoders.py`, and `layers/blocks.py`:
- `DiffusionTokenEncoder` mixes token/pairwise features, appends optional distogram + self-conditioning, and runs its internal `PairformerBlock` stack before returning `S_I, Z_II`.
- `LocalTokenTransformer` builds fresh CA-based attention indices each recycle (it does not reuse `f["attn_indices"]`) and updates `A_I` with local or full attention depending on `RFD3_LOW_MEMORY_MODE`.
- `CompactStreamingDecoder` loops `n_blocks` times over `Upcast -> StructureLocalAtomTransformerBlock`; it performs **one** `Downcast` afterward (on detached `Q_L`, `A_I`, `S_I`) to refresh token features for heads/recycling.
- Output heads live in `RFD3DiffusionModule.process_`: `to_r_update` + `scale_positions_out` for coordinates, `LinearSequenceHead` for logits/indices, and `bucketize_scaled_distogram` for `D_II_self`.
- When `RFD3_LOW_MEMORY_MODE=1`, encoder/decoder use `chunked_pairwise_embedder`; otherwise they consume full `P_LL`.

Attention mapping in code:
- `GatedCrossAttention`: [models/rfd3/src/rfd3/model/layers/attention.py](models/rfd3/src/rfd3/model/layers/attention.py#L92)
- `Decoder Upcast` cross-attention path: [models/rfd3/src/rfd3/model/layers/blocks.py](models/rfd3/src/rfd3/model/layers/blocks.py#L478)
- `Decoder Downcast` cross-attention path: [models/rfd3/src/rfd3/model/layers/blocks.py](models/rfd3/src/rfd3/model/layers/blocks.py#L532)
- `LocalAttentionPairBias` and sparse attention path: [models/rfd3/src/rfd3/model/layers/attention.py](models/rfd3/src/rfd3/model/layers/attention.py#L198)
- `create_attention_indices`: [models/rfd3/src/rfd3/model/layers/block_utils.py](models/rfd3/src/rfd3/model/layers/block_utils.py#L179)
- `PairformerBlock` implementation used by token encoding: [models/rfd3/src/rfd3/model/layers/pairformer_layers.py](models/rfd3/src/rfd3/model/layers/pairformer_layers.py#L100)
- Default cross-attention config for upcast/downcast: [models/rfd3/configs/model/components/rfd3_net.yaml](models/rfd3/configs/model/components/rfd3_net.yaml#L67) and [models/rfd3/configs/model/components/rfd3_net.yaml](models/rfd3/configs/model/components/rfd3_net.yaml#L76)

## Detailed execution order (single diffusion step)
1. Build conditioning and geometry inputs:
   - `f` carries token/atom mappings, masks, motif constraints, symmetry metadata, and optional conditioning features.
   - `X_t` and `t` define the current diffusion state.
2. Run `TokenInitializer`:
   - Produces atom stream states (`Q_L`, `C_L`), atom-pair states (`P_LL`), and token stream states (`S_I`, `Z_II`).
3. Prepare timestep-conditioned states inside the diffusion module:
   - Time embeddings are produced and injected into atom/token streams.
   - Atom-level states are projected/pool-coupled into token-level states.
4. Run atom encoder path:
   - `LocalAtomTransformer` updates atom-local context from neighborhood attention.
   - `Downcast` pools atom information into token-aligned channels.
5. Run token encoder/transformer path:
   - `DiffusionTokenEncoder` fuses token states with pairwise/context and current coordinates.
   - `LocalTokenTransformer` performs token-level attention updates.
6. Decode and project outputs:
   - `CompactStreamingDecoder` runs `n_blocks` of [Upcast -> `StructureLocalAtomTransformerBlock`], then a single `Downcast` (detached) to refresh `A_I`.
   - Heads in `RFD3DiffusionModule.process_`: `to_r_update` + `scale_positions_out` -> `X_out_L`; `LinearSequenceHead` -> logits/indices; distogram bucketization -> `D_II_self`.
7. Recycle boundary:
   - (`X_out_L`, `D_II_self`) are fed into the next recycle iteration when `n_recycle > 0`.
8. Sampler update:
   - `InferenceSampler` applies schedule logic (EDM-style), optional CFG blending, and optional symmetry constraints to produce the next step state.

## Recycle state summary
- Position state: `X_L` (updated coordinates after output scaling).
- Pairwise memory: `D_II_self` (distogram buckets used as iterative context).
- Conditioning state: `f` remains fixed unless explicitly modified by CFG feature stripping in the reference pass.

Practical interpretation:
- Recycling improves consistency at a fixed timestep before moving to the next denoising step.
- Denoising changes the noise level (schedule step) and carries forward refined structure state.

## Quick references
- Run inference: `rfd3 design out_dir=<dir> inputs=models/rfd3/docs/examples/demo.json dump_trajectories=True prevalidate_inputs=True`.
- Model toggle: set `RFD3_LOW_MEMORY_MODE=1` to enable chunked pairwise embeddings.
- CFG features controlled via `inference_sampler.cfg_features` (see default list in `configs/inference_engine/rfdiffusion3.yaml`).
- Symmetry sampler: `inference_sampler.kind=symmetry` plus symmetry spec in input JSON.
