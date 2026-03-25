
# RFD3 Model Architecture: Denoising and Recycling Loops

This document provides a detailed, code- and paper-accurate overview of the RFD3 model architecture, focusing on the interplay between the denoising (diffusion) and recycling (refinement) loops. The diagrams and descriptions below are formatted for clarity and flow, and use terminology consistent with the published paper.


## Model Flow Diagram

The following diagram shows the high-level flow of the RFD3 model, with encoder and decoder roles annotated:

```mermaid
flowchart LR
    IN[Residue features & Atomic features]
    FI[Feature Initializer<br><i>2 blocks<br><b>Encoder</b></i>]
    IN --> FI --> STEPIN[Step input<br><i>X_t, t, features</i>]
    subgraph DenoisingLoop["Denoising Loop (Diffusion)"]
        direction LR
        STEPIN --> TKINIT[Feature Initializer<br><i>2 blocks<br><b>Encoder</b></i>]
        TKINIT --> ATOM1
        subgraph RecycleLoop["Recycling (Refinement)"]
            direction LR
            ATOM1[Atom Transformer<br><i>3 blocks<br><b>Encoder</b></i>]
            TOKEN[Token Transformer<br><i>18 blocks<br><b>Encoder</b></i>]
            ATOM2[Atom Transformer<br><i>3 blocks<br><b>Decoder</b></i>]
            ATOM1 --> TOKEN --> ATOM2
            ATOM2 -- "Recycling" --> ATOM1
        end
        ATOM2 --> OUT1[Post-processing<br><i>Denoising output</i>]
        OUT1 --> OUT2[Step output<br><i>X_t-1, predictions</i>]
        OUT2 -- "More denoising steps?" --> STEPIN
    end
    OUT2 --> FINAL[Output<br><i>Final structure, metadata</i>]

    style IN fill:#e8f4fd,stroke:#2196F3,stroke-width:2px,color:#1565C0
    style FI fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style STEPIN fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,color:#6A1B9A
    style TKINIT fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style ATOM1 fill:#e8f5e9,stroke:#4CAF50,stroke-width:2px,color:#1B5E20
    style TOKEN fill:#e3f2fd,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    style ATOM2 fill:#fce4ec,stroke:#E91E63,stroke-width:2px,color:#880E4F
    style OUT1 fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,color:#6A1B9A
    style OUT2 fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,color:#6A1B9A
    style FINAL fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#004D40
    style DenoisingLoop fill:#fafafa,stroke:#616161,stroke-width:2px,stroke-dasharray:5 5
    style RecycleLoop fill:#f5f5f5,stroke:#00BCD4,stroke-width:2px,stroke-dasharray:3 3
```

**Diagram Notes:**
- <b>Encoder</b> blocks (Feature Initializer, Atom Transformer, Token Transformer) process and encode input features and intermediate representations.
- <b>Decoder</b> block (final Atom Transformer) refines and produces the output for each recycle iteration.
- The recycling loop enables iterative refinement within each denoising step.
- The denoising loop iterates over diffusion timesteps, progressively denoising the structure.

## Block-by-Block Layer Breakdown

---

### Feature Initializer (TokenInitializer)

The Feature Initializer prepares the model's internal representations from raw input features. It embeds residue and atomic features, builds pairwise representations, and runs a small Pairformer stack for initial mixing.

```mermaid
flowchart TD
    subgraph Embedders["1. Feature Embedding"]
        direction TB
        A1a[Linear: atom features to c_atom]
        A1b[ReLU activation]
        A1c[Linear: c_atom to c_atom]
        A2a[Linear: token features to c_s]
        A2b[ReLU activation]
        A2c[Linear: c_s to c_s]
        A1a --> A1b --> A1c
        A2a --> A2b --> A2c
    end

    subgraph Downcast["2. Downcast: Atom to Token"]
        direction TB
        DC1[Gather atoms per token]
        DC2[Weighted sum pooling]
        DC3[Output: S_init_I]
        DC1 --> DC2 --> DC3
    end

    subgraph PairInit["3. Pairwise Representation Init"]
        direction TB
        P1[LinearNoBias: S to Z_i]
        P2[LinearNoBias: S to Z_j]
        P3[Outer sum: Z_init_II = Z_i + Z_j]
        P4[RelativePositionEncoding x2]
        P5[LinearNoBias: bond features]
        P6[PositionPairDistEmbedder]
        P7[Sum all into Z_init_II]
        P1 --> P3
        P2 --> P3
        P3 --> P7
        P4 --> P7
        P5 --> P7
        P6 --> P7
    end

    subgraph Pairformer["4. PairformerBlock x2"]
        direction TB
        PF1[AttentionPairBias<br><i>Multi-head attention on S_I<br>with bias from Z_II</i>]
        PF2[Transition: S_I<br><i>RMSNorm, Linear, SiLU, Linear</i>]
        PF3[Transition: Z_II<br><i>RMSNorm, Linear, SiLU, Linear</i>]
        PF4[Residual connections on S_I and Z_II]
        PF1 --> PF2 --> PF3 --> PF4
    end

    subgraph AtomPairMLP["5. Atom Pair MLP: Build P_LL"]
        direction TB
        M1[Linear: project single_l, single_m]
        M2[Linear: project Z_II to atom level]
        M3[SinusoidalDistEmbed: motif positions]
        M4[PositionPairDistEmbedder: ref positions]
        M5[Sum all pair contributions]
        M6[ReLU, Linear]
        M7[ReLU, Linear]
        M8[ReLU, Linear]
        M9[Output: P_LL]
        M1 --> M5
        M2 --> M5
        M3 --> M5
        M4 --> M5
        M5 --> M6 --> M7 --> M8 --> M9
    end

    Embedders --> Downcast --> PairInit --> Pairformer --> AtomPairMLP

    style A1a fill:#e8f5e9,stroke:#4CAF50,stroke-width:2px,color:#1B5E20
    style A1b fill:#e8f5e9,stroke:#4CAF50,stroke-width:1px,color:#1B5E20
    style A1c fill:#e8f5e9,stroke:#4CAF50,stroke-width:2px,color:#1B5E20
    style A2a fill:#e3f2fd,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    style A2b fill:#e3f2fd,stroke:#1976D2,stroke-width:1px,color:#0D47A1
    style A2c fill:#e3f2fd,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    style DC1 fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style DC2 fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style DC3 fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style P1 fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,color:#6A1B9A
    style P2 fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,color:#6A1B9A
    style P3 fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,color:#6A1B9A
    style P4 fill:#ede7f6,stroke:#673AB7,stroke-width:2px,color:#311B92
    style P5 fill:#ede7f6,stroke:#673AB7,stroke-width:2px,color:#311B92
    style P6 fill:#ede7f6,stroke:#673AB7,stroke-width:2px,color:#311B92
    style P7 fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,color:#6A1B9A
    style PF1 fill:#fce4ec,stroke:#E91E63,stroke-width:2px,color:#880E4F
    style PF2 fill:#fce4ec,stroke:#E91E63,stroke-width:2px,color:#880E4F
    style PF3 fill:#fce4ec,stroke:#E91E63,stroke-width:2px,color:#880E4F
    style PF4 fill:#fce4ec,stroke:#E91E63,stroke-width:1px,color:#880E4F
    style M1 fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#004D40
    style M2 fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#004D40
    style M3 fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#004D40
    style M4 fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#004D40
    style M5 fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#004D40
    style M6 fill:#e0f2f1,stroke:#009688,stroke-width:1px,color:#004D40
    style M7 fill:#e0f2f1,stroke:#009688,stroke-width:1px,color:#004D40
    style M8 fill:#e0f2f1,stroke:#009688,stroke-width:1px,color:#004D40
    style M9 fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#004D40
    style Embedders fill:#f1f8e9,stroke:#8BC34A,stroke-width:2px
    style Downcast fill:#fff8e1,stroke:#FFC107,stroke-width:2px
    style PairInit fill:#f3e5f5,stroke:#AB47BC,stroke-width:2px
    style Pairformer fill:#fce4ec,stroke:#EC407A,stroke-width:2px
    style AtomPairMLP fill:#e0f2f1,stroke:#26A69A,stroke-width:2px
```

- **Feature Embedding:** Two parallel paths — atom features and token features — each go through Linear + ReLU + Linear to produce initial embeddings.
- **Downcast:** Gathers atoms belonging to each token and pools them via weighted sum to produce token-level representations (S_init_I).
- **Pairwise Init:** Builds Z_init_II by outer-summing projected single reps, then adding relative position encodings, bond features, and distance embeddings.
- **PairformerBlock x2:** Each block runs AttentionPairBias on S_I (with bias from Z_II), then two separate Transition layers (RMSNorm → Linear → SiLU → Linear) for S_I and Z_II, with residual connections.
- **Atom Pair MLP:** Collects projections of single reps, Z_II, motif distances, and reference distances, sums them, then passes through 3 layers of ReLU → Linear to produce P_LL.

---

### Atom Transformer (LocalAtomTransformer, 3 blocks)

Each of the 3 blocks is a `StructureLocalAtomTransformerBlock`. The attention and MLP sub-layers are broken down below.

```mermaid
flowchart TD
    INPUT[Q_L input] --> BLOCK1

    subgraph BLOCK1["Block 1 of 3"]
        direction TB
        subgraph Attn1["Local Attention with Pair Bias"]
            direction TB
            N1[AdaLN: condition on C_L<br><i>Scale and shift from single reps</i>]
            QKV1[LinearNoBias: Q, K, V projections]
            QKNORM1[RMSNorm on Q and K]
            GATHER1[Gather sparse attention indices]
            BIAS1[LinearNoBias: P_LL to per-head bias]
            SDPA1[Scaled Dot-Product Attention<br><i>with pair bias, sparse</i>]
            GATE1[Linear + Sigmoid gating]
            OUT1A[LinearNoBias: output projection]
            N1 --> QKV1 --> QKNORM1 --> SDPA1
            GATHER1 --> SDPA1
            BIAS1 --> SDPA1
            SDPA1 --> GATE1 --> OUT1A
        end
        RES1[+ Residual: Q_L = Q_L + attn_out]
        subgraph MLP1["Conditioned Transition (SwiGLU)"]
            direction TB
            AN1[AdaLN: condition on C_L]
            LIN1A[Linear: c_atom to 4*c_atom]
            LIN1B[Linear: c_atom to 4*c_atom]
            SILU1[SiLU activation on branch A]
            MULT1[Element-wise multiply: SiLU_A * B]
            LIN1C[Linear: 4*c_atom to c_atom]
            AN1 --> LIN1A --> SILU1 --> MULT1
            AN1 --> LIN1B --> MULT1
            MULT1 --> LIN1C
        end
        RES1B[+ Residual: Q_L = Q_L + mlp_out]
        Attn1 --> RES1 --> MLP1 --> RES1B
    end

    BLOCK1 --> REPEAT["Repeat for Block 2 and Block 3<br><i>Same architecture, shared weights within each block</i>"]
    REPEAT --> OUTQ[Q_L output]

    style INPUT fill:#e8f5e9,stroke:#4CAF50,stroke-width:2px,color:#1B5E20
    style N1 fill:#e8f5e9,stroke:#4CAF50,stroke-width:2px,color:#1B5E20
    style QKV1 fill:#e3f2fd,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    style QKNORM1 fill:#e3f2fd,stroke:#1976D2,stroke-width:1px,color:#0D47A1
    style GATHER1 fill:#e8eaf6,stroke:#3F51B5,stroke-width:2px,color:#1A237E
    style BIAS1 fill:#e8eaf6,stroke:#3F51B5,stroke-width:2px,color:#1A237E
    style SDPA1 fill:#e3f2fd,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    style GATE1 fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style OUT1A fill:#e3f2fd,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    style RES1 fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style AN1 fill:#e8f5e9,stroke:#4CAF50,stroke-width:2px,color:#1B5E20
    style LIN1A fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,color:#6A1B9A
    style LIN1B fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,color:#6A1B9A
    style SILU1 fill:#f3e5f5,stroke:#9C27B0,stroke-width:1px,color:#6A1B9A
    style MULT1 fill:#ede7f6,stroke:#673AB7,stroke-width:2px,color:#311B92
    style LIN1C fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,color:#6A1B9A
    style RES1B fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style REPEAT fill:#fafafa,stroke:#9E9E9E,stroke-width:2px,stroke-dasharray:5 5,color:#616161
    style OUTQ fill:#e8f5e9,stroke:#4CAF50,stroke-width:2px,color:#1B5E20
    style Attn1 fill:#e3f2fd,stroke:#1976D2,stroke-width:2px
    style MLP1 fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px
    style BLOCK1 fill:#e8f5e9,stroke:#388E3C,stroke-width:2px
```

**Attention sub-layer:**
- **AdaLN:** Adaptive Layer Normalization conditioned on single representations (C_L). Produces scale/shift parameters.
- **Q, K, V projections:** Three separate LinearNoBias layers projecting input to query, key, and value.
- **Q/K RMSNorm:** Normalizes query and key vectors for stable attention.
- **Sparse index gathering:** Collects attention indices (nearest neighbors in 3D space).
- **Pair bias:** Projects P_LL to per-head bias via LinearNoBias, added to attention logits.
- **Scaled Dot-Product Attention:** Computes sparse attention with pair bias.
- **Sigmoid gating:** Linear → Sigmoid produces a gate that modulates the attention output.
- **Output projection:** LinearNoBias maps back to model dimension.

**MLP sub-layer (SwiGLU):**
- **AdaLN:** Conditions on C_L.
- **Two parallel Linear projections:** Both map c_atom → 4*c_atom.
- **SiLU on branch A:** Applies SiLU activation to one branch.
- **Element-wise multiply:** Multiplies the SiLU branch with the gate branch (SwiGLU pattern).
- **Linear projection:** Maps 4*c_atom back to c_atom.

---

### Token Transformer (LocalTokenTransformer, 18 blocks)

Uses the same `StructureLocalAtomTransformerBlock` as the Atom Transformer, but operates on token-level features (A_I) with pair bias from Z_II. Sparse attention indices are built from 3D coordinates (128 keys, 2-4 neighbors).

```mermaid
flowchart TD
    TINPUT[A_I input] --> IDX[Build sparse attention indices<br><i>from 3D coords, 128 keys, 2-4 neighbors</i>]
    IDX --> TBLOCK1

    subgraph TBLOCK1["Block 1 of 18"]
        direction TB
        subgraph TAttn["Local Attention with Pair Bias"]
            direction TB
            TN1[AdaLN: condition on S_I<br><i>Scale and shift from single reps</i>]
            TQKV[LinearNoBias: Q, K, V projections]
            TQKNORM[RMSNorm on Q and K]
            TGATHER[Gather sparse attention indices]
            TBIAS[LinearNoBias: Z_II to per-head bias]
            TSDPA[Scaled Dot-Product Attention<br><i>with pair bias, sparse or full</i>]
            TGATE[Linear + Sigmoid gating]
            TOUT[LinearNoBias: output projection]
            TN1 --> TQKV --> TQKNORM --> TSDPA
            TGATHER --> TSDPA
            TBIAS --> TSDPA
            TSDPA --> TGATE --> TOUT
        end
        TRES1[+ Residual: A_I = A_I + attn_out]
        subgraph TMLP["Conditioned Transition (SwiGLU)"]
            direction TB
            TAN[AdaLN: condition on S_I]
            TLIN1[Linear: c_s to 4*c_s]
            TLIN2[Linear: c_s to 4*c_s]
            TSILU[SiLU activation on branch A]
            TMULT[Element-wise multiply: SiLU_A * B]
            TLIN3[Linear: 4*c_s to c_s]
            TAN --> TLIN1 --> TSILU --> TMULT
            TAN --> TLIN2 --> TMULT
            TMULT --> TLIN3
        end
        TRES2[+ Residual: A_I = A_I + mlp_out]
        TAttn --> TRES1 --> TMLP --> TRES2
    end

    TBLOCK1 --> TREPEAT["Repeat for Blocks 2-18<br><i>Same architecture per block</i>"]
    TREPEAT --> TOUTQ[A_I output]

    style TINPUT fill:#e3f2fd,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    style IDX fill:#e8eaf6,stroke:#3F51B5,stroke-width:2px,color:#1A237E
    style TN1 fill:#e3f2fd,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    style TQKV fill:#e3f2fd,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    style TQKNORM fill:#e3f2fd,stroke:#1976D2,stroke-width:1px,color:#0D47A1
    style TGATHER fill:#e8eaf6,stroke:#3F51B5,stroke-width:2px,color:#1A237E
    style TBIAS fill:#e8eaf6,stroke:#3F51B5,stroke-width:2px,color:#1A237E
    style TSDPA fill:#e3f2fd,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    style TGATE fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style TOUT fill:#e3f2fd,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    style TRES1 fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style TAN fill:#e3f2fd,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    style TLIN1 fill:#ede7f6,stroke:#673AB7,stroke-width:2px,color:#311B92
    style TLIN2 fill:#ede7f6,stroke:#673AB7,stroke-width:2px,color:#311B92
    style TSILU fill:#ede7f6,stroke:#673AB7,stroke-width:1px,color:#311B92
    style TMULT fill:#ede7f6,stroke:#673AB7,stroke-width:2px,color:#311B92
    style TLIN3 fill:#ede7f6,stroke:#673AB7,stroke-width:2px,color:#311B92
    style TRES2 fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style TREPEAT fill:#fafafa,stroke:#9E9E9E,stroke-width:2px,stroke-dasharray:5 5,color:#616161
    style TOUTQ fill:#e3f2fd,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    style TAttn fill:#e3f2fd,stroke:#1976D2,stroke-width:2px
    style TMLP fill:#ede7f6,stroke:#673AB7,stroke-width:2px
    style TBLOCK1 fill:#e3f2fd,stroke:#1565C0,stroke-width:2px
```

**Key differences from Atom Transformer:**
- Operates on token-level reps (A_I) instead of atom-level (Q_L).
- Pair bias comes from Z_II (token-pair) instead of P_LL (atom-pair).
- Conditioned on S_I (token single reps) instead of C_L (atom conditioning).
- Uses 128 attention keys with 2-4 spatial neighbors for sparse attention.
- 18 blocks (vs. 3 for atom transformer) — this is the largest computation in each recycle.

---

### DiffusionTokenEncoder (Self-Conditioning)

Sits between the Atom Encoder and Token Transformer. Conditions token and pair representations using noise-level and distogram information.

```mermaid
flowchart TD
    subgraph SinglePath["Single Representation Path (S_I)"]
        direction TB
        S1[RMSNorm]
        S2[Linear: c_s to 2*c_s]
        S3[SiLU activation]
        S4[Linear: 2*c_s to c_s]
        S5[Residual connection]
        S6["Repeat Transition x2"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6
    end

    subgraph PairPath["Pair Representation Path (Z_II)"]
        direction TB
        D1[Compute pairwise distances from R_L]
        D2a[SinusoidalDistEmbed: distances to c_z]
        D2b[Bucketize: 65-bin Gaussian PDF]
        D3[Concatenate: Z_init_II + distogram]
        D4[RMSNorm]
        D5[Linear: cat_c_z to c_z]
        D6[RMSNorm, Linear, SiLU, Linear]
        D7[Residual connection]
        D8["Repeat Pair Transition x2"]
        D1 --> D2a --> D3
        D1 --> D2b --> D3
        D3 --> D4 --> D5 --> D6 --> D7 --> D8
    end

    subgraph PairformerMix["PairformerBlock x2: S_I and Z_II Mixing"]
        direction TB
        PFa[AttentionPairBias on S_I<br><i>with bias from Z_II</i>]
        PFb[Transition: S_I<br><i>RMSNorm, Linear, SiLU, Linear</i>]
        PFc[Transition: Z_II<br><i>RMSNorm, Linear, SiLU, Linear</i>]
        PFd[Residual connections]
        PFe["Repeat PairformerBlock x2"]
        PFa --> PFb --> PFc --> PFd --> PFe
    end

    SinglePath --> PairformerMix
    PairPath --> PairformerMix
    PairformerMix --> ENCOUT[Output: S_I, Z_II<br><i>Conditioned on noise and distogram</i>]

    style S1 fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#004D40
    style S2 fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#004D40
    style S3 fill:#e0f2f1,stroke:#009688,stroke-width:1px,color:#004D40
    style S4 fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#004D40
    style S5 fill:#e0f2f1,stroke:#009688,stroke-width:1px,color:#004D40
    style S6 fill:#e0f2f1,stroke:#009688,stroke-width:2px,stroke-dasharray:3 3,color:#004D40
    style D1 fill:#fce4ec,stroke:#E91E63,stroke-width:2px,color:#880E4F
    style D2a fill:#fce4ec,stroke:#E91E63,stroke-width:2px,color:#880E4F
    style D2b fill:#fce4ec,stroke:#E91E63,stroke-width:2px,color:#880E4F
    style D3 fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style D4 fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style D5 fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#E65100
    style D6 fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,color:#6A1B9A
    style D7 fill:#f3e5f5,stroke:#9C27B0,stroke-width:1px,color:#6A1B9A
    style D8 fill:#f3e5f5,stroke:#9C27B0,stroke-width:2px,stroke-dasharray:3 3,color:#6A1B9A
    style PFa fill:#e8eaf6,stroke:#3F51B5,stroke-width:2px,color:#1A237E
    style PFb fill:#e8eaf6,stroke:#3F51B5,stroke-width:2px,color:#1A237E
    style PFc fill:#e8eaf6,stroke:#3F51B5,stroke-width:2px,color:#1A237E
    style PFd fill:#e8eaf6,stroke:#3F51B5,stroke-width:1px,color:#1A237E
    style PFe fill:#e8eaf6,stroke:#3F51B5,stroke-width:2px,stroke-dasharray:3 3,color:#1A237E
    style ENCOUT fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#004D40
    style SinglePath fill:#e0f7fa,stroke:#00ACC1,stroke-width:2px
    style PairPath fill:#fce4ec,stroke:#EC407A,stroke-width:2px
    style PairformerMix fill:#e8eaf6,stroke:#5C6BC0,stroke-width:2px
```

**Single Representation Path (S_I):**
- Two Transition layers, each: RMSNorm → Linear (expand) → SiLU → Linear (contract) → Residual.

**Pair Representation Path (Z_II):**
- Compute pairwise distances from noisy coordinates (R_L).
- Embed distances via SinusoidalDistEmbed (continuous) or 65-bin Gaussian PDF bucketing (discrete).
- Concatenate Z_init_II with distogram embedding.
- Project via RMSNorm → Linear to c_z.
- Two Pair Transition layers: RMSNorm → Linear → SiLU → Linear → Residual.

**PairformerBlock x2:**
- Each block: AttentionPairBias on S_I (bias from Z_II), then separate Transition layers for S_I and Z_II, with residual connections.
- Repeated twice for deeper mixing.


### Summary Table

| Component               | Class                    | Blocks | Key Layers                                                                 |
|-------------------------|--------------------------|--------|---------------------------------------------------------------------------|
| Feature Initializer     | TokenInitializer         | —      | 1D Embedders, Downcast, RelPos, 2 PairformerBlocks, Atom Pair MLP        |
| Atom Transformer        | LocalAtomTransformer     | 3      | AdaLN, Sparse Local Attention + Pair Bias, SwiGLU MLP, Residual          |
| DiffusionTokenEncoder   | DiffusionTokenEncoder    | —      | 2 Transitions, Distogram Embed, 2 Pair Transitions, 2 PairformerBlocks   |
| Token Transformer       | LocalTokenTransformer    | 18     | AdaLN, Sparse Local Attention + Pair Bias (128 keys), SwiGLU MLP, Residual |

All Atom Transformer and Token Transformer blocks share the same internal architecture (`StructureLocalAtomTransformerBlock`): **Local Attention with Pair Bias + Conditioned Transition (SwiGLU)**. The difference is in the number of blocks, the input level (atom vs. token), and the attention indices used.


## Step-by-Step Model Flow

This section describes the RFD3 model flow in detail, matching the diagram and paper terminology:

1. **Residue features & Atomic features:**
    - The model receives residue-level and atomic-level features as input, as shown in the paper's diagram.

2. **Feature Initializer (2 blocks, Encoder):**
    - Prepares the initial model state from the input features and coordinates.
    - Consists of 2 encoder blocks that embed and project the input data into the internal representations required for downstream processing.

3. **Step Input:**
    - Represents the current state at diffusion step $t$ (i.e., $X_t$, $t$, and features $f$).
    - This node is the entry point for each denoising (diffusion) step.

4. **Feature Initializer (2 blocks, Encoder):**
    - Initializes the state for the current denoising step.
    - Sets up the token-level and pairwise representations that will be refined in the recycling loop.

5. **Recycling (Refinement):**
    - For each denoising step, the model performs several recycling iterations to iteratively refine the representations.
    - **Atom Transformer (3 blocks, Encoder):** Processes atomic-level features, applying local self-attention and updating atom representations.
    - **Token Transformer (18 blocks, Encoder):** Processes token-level (residue-level) features, applying transformer layers to capture long-range dependencies and context.
    - **Atom Transformer (3 blocks, Decoder):** Further refines atomic features after token-level processing and produces the output for each recycle iteration.
    - The recycling loop repeats for a set number of iterations, with outputs from the final Atom Transformer feeding back to the first Atom Transformer for further refinement.

6. **Post-processing:**
    - After the final recycling iteration, the model post-processes the refined representations.
    - This typically involves scaling or transforming coordinates and preparing outputs for the next denoising step.

7. **Step Output:**
    - Produces the denoised coordinates $X_{t-1}$ and other predictions for the current step.
    - If more denoising steps remain, the output is fed back as the next step's input.

8. **Denoising Loop:**
    - The outer loop iterates over diffusion timesteps, each time running the recycling loop and post-processing.
    - The process continues until the final timestep is reached.

9. **Output:**
    - After all denoising steps are complete, the model outputs the final structure and any associated metadata or predictions.

**Key Points:**
- The recycling loop is nested inside the denoising loop, enabling multiple refinement steps per diffusion timestep.
- The architecture and terminology now match the published paper's diagram, and the diagram accurately reflects the data/control flow in the RFD3 model codebase.