# Architecture — Fine-Tuned Donut for Scanned Document Extraction

**Project:** Vision-Language model (Donut + LoRA) that extracts 10 header fields from
degraded scanned Indian bank statements.
**Status:** Training complete; inference-loader fix under validation.

---
         
## 1. System context (C4 — Level 1)

```mermaid
flowchart LR
    user([ML Engineer / Aashish])
    subgraph System[Document Extraction System]
        nb[Kaggle Notebook<br/>train + evaluate]                                                                
        model[(Fine-tuned Donut<br/>+ LoRA adapter)]
    end
    hf[(HuggingFace Hub<br/>donut-base weights)]
    ds[(AgamiAI Bank<br/>Statements dataset)]

    user -->|runs cells| nb
    ds -->|source docs| nb
    hf -->|pretrained weights| nb
    nb -->|saves| model
    model -->|predictions| nb
    nb -->|F1 / metrics| user
```

**One-line:** pixels in → structured tagged string out, no OCR step.

---

## 2. Container / component view

```mermaid
flowchart TB
    subgraph Data[Data Pipeline]
        D1[download_all<br/>fetch 250 docs]
        D2[batch_degrade<br/>clean/light/medium/heavy]
        D3[prepare_dataset<br/>train/val/test split]
        D4[prepare_donut_dataset<br/>image + target_sequence jsonl]
        D1 --> D2 --> D3 --> D4
    end

    subgraph Train[Training]
        T1[Load donut-base]
        T2[Add 22 tags<br/>resize embeddings]
        T3[Wrap with LoRA<br/>Q,V + embed/lm_head]
        T4[Seq2SeqTrainer<br/>early stopping]
        T5[Save best model]
        T1 --> T2 --> T3 --> T4 --> T5
    end

    subgraph Eval[Inference and Scoring]
        I1[predict_donut<br/>generate on test set]
        I2[scorer<br/>precision/recall/F1]
        I1 --> I2
    end

    D4 --> T1
    T5 --> I1
```

---

## 3. Model architecture (the two halves)

```mermaid
flowchart LR
    img[Image<br/>1280x960x3]
    subgraph Enc[Swin Encoder - the eye]
        p[Patchify 4x4]
        w[Windowed attention<br/>+ shifted windows]
        m[Patch merging<br/>hierarchy]
        p --> w --> m
    end
    mem[(Image memory<br/>~1200 x 1024)]
    subgraph Dec[BART Decoder - the hand]
        se[Self-attention]
        ca[Cross-attention<br/>into memory]
        lm[lm_head to vocab]
        se --> ca --> lm
    end
    out[Tagged string<br/>s_extraction ...]

    img --> Enc --> mem --> Dec --> out
```

- **Encoder (Swin):** turns 3.6M pixels into ~1,200 semantic vectors.
- **Decoder (BART):** autoregressively writes tokens, cross-attending into the image memory.
- **Custom tags:** 22 new vocabulary tokens (`<s_account_number>` … `</s_extraction>`).

---

## 4. Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Base model | `donut-base` (~200M) | OCR-free, fits free T4, strong doc pretraining |
| Fine-tuning | LoRA on Q/V | Cheap, high-impact steering; avoids overfitting small data |
| New tokens | full-train `embed_tokens`+`lm_head` | Random-init tags can't be learned by low-rank adapters |
| Memory | grad checkpointing + fp16 + batch1/accum8 | Fits 15 GB T4 |
| Image size | 1280×960 | Preserves tiny digits on dense statements |
| Stopping | early stop, best on `eval_loss` | Avoids overfitting past epoch 5 |

---

## 5. Known risk / the inference bug

The training process sets `pad_token_id` / `decoder_start_token_id`, but the **separate
inference process** reloads base Donut and does **not** inherit them — causing empty
generation (0.000 F1). Fix: re-apply this config in `load_finetuned_model()` and mirror it
onto `generation_config`. See `LLD.md` §4.
