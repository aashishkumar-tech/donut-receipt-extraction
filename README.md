# Fine-Tuned Vision-Language Model for Scanned Document Extraction

# Fine-Tuned Vision-Language Model for Scanned Document Extraction

**Status:** Validated — **F1 0.901** on receipts (CORD-v2), vs 0.026 on bank statements
**Owner:** Aashish
**Last updated:** 2026-09-27

## Result

| Use case | Data | Precision | Recall | **F1** |
|---|---|---|---|---|
| Bank statements | 580 synthetic, heavily degraded | 0.031 | 0.022 | 0.026 |
| **Receipts (CORD-v2)** | 800 real photos | 0.908 | 0.894 | **0.901** |
| Reference — `donut-base-finetuned-cord-v2` | full fine-tune | 0.985 | 0.967 | 0.976 |

Same model, same LoRA recipe, same scorer. The 35x difference came from **legible inputs
plus a checkpoint that was actually saved correctly** — see
[FINDINGS.md](FINDINGS.md) for the save bug that made two runs score 0.070 instead of 0.901.

- Notebook: [`notebooks/donut_cord_receipts.ipynb`](notebooks/donut_cord_receipts.ipynb) — self-contained, Kaggle T4, ~7 h
- Model: [`aashishkumar-tech/donut-cord-summary`](https://huggingface.co/aashishkumar-tech/donut-cord-summary)
- Serving: [`serve.py`](serve.py) — FastAPI, `POST /extract`

## Repository contents

| Path | What it is |
|---|---|
| [FINDINGS.md](FINDINGS.md) | **Start here** — results, what broke, what the numbers actually mean |
| [notebooks/donut_cord_receipts.ipynb](notebooks/donut_cord_receipts.ipynb) | The working run — train, verify, infer, score, demo UI |
| [serve.py](serve.py) | FastAPI inference service, `POST /extract` |
| [results/](results/) | `report.json` + `test_preds.jsonl` — the evidence behind the numbers above |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System overview and data flow |
| [HLD.md](HLD.md) | High-level design — components and responsibilities |
| [LLD.md](LLD.md) | Low-level design — modules, configs, interfaces |
| [notebooks/archive_bank_statements_0.026.ipynb](notebooks/archive_bank_statements_0.026.ipynb) | Prior attempt (F1 0.026). Reference only — will not run from this repo |

## One-line summary

Testing whether a small vision-language model, fine-tuned on domain-specific
scanned documents, can match or beat both a traditional OCR+rules pipeline
and a zero-shot large VLM — at a fraction of the cost — for structured field
extraction from degraded scanned documents.

**Answer: yes, on legible inputs.** A 200M-parameter Donut with LoRA on 800 images
reaches 92% of a full fine-tune's F1 for ~7 GPU-hours. On unreadable inputs no model
helps, and neither did a 7B VLM.

## Three transferable lessons

1. **Verify checkpoints by loss, not by shape.** A 90%-shape-match guard passed on a
   checkpoint containing zero trained weights. Compare in-memory vs on-disk loss on the
   same batches, and `assert`.
2. **Benchmark against a published model with your own scorer.** "0.901" only means
   something next to the 0.976 that a known-good model scores through the same function.
3. **A heuristic validated on synthetic corruption may not survive real data.** The
   Laplacian quality gate gave +lift on artificially blurred scans and **−0.026** on real
   photos.

## Quickstart

**Reproduce the result.** Upload `notebooks/donut_cord_receipts.ipynb` to Kaggle, select a
**T4 GPU**, turn **Internet ON**, run top to bottom. Cell 1.1 uninstalls `torchao` (Kaggle
ships 0.10, which current `peft` rejects) — **restart the kernel after it**, then continue
from 1.2. Training is ~7 h.

Two cells matter more than they look:

- **5.1b — save parity.** Compares in-memory vs on-disk loss on the same batches and
  raises if they differ. This caught a silent bug where the saved checkpoint contained no
  trained weights; fixing it moved F1 from 0.070 to 0.901 **without retraining**.
- **8.1 — reference baseline.** Scores a published CORD model through the identical
  scorer. Reference high + ours low ⇒ our bug. Both low ⇒ the scorer is at fault.

**Try the model.** Run cell 11 for a Gradio demo (public link, ~72 h), or serve it
locally — no GPU and no Kaggle needed, the weights come from the Hub:

```bash
python -m venv .venv && .venv\Scripts\activate    # source .venv/bin/activate on unix
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only, much smaller
pip install -r requirements.txt
uvicorn serve:app --port 8000
```

```bash
curl -F "file=@receipt.jpg" http://localhost:8000/extract
# {"filename":"receipt.jpg","fields":{"total_price":"60000", ...}}
```

First start downloads ~800 MB and caches it. `/docs` gives a browser upload form;
`/health` is the readiness probe. Set `MODEL_PATH` to use a local folder instead of the
Hub. Expect ~4 s per receipt on CPU — the guided decoder makes six sequential passes.

## Known limitations

- **Overfits after epoch 6** — `eval_loss` 0.1706 @ e6 → 0.2036 @ e20, and
  `save_total_limit=1` discarded the better checkpoint.
- **No abstain signal.** ~10% of values are wrong and nothing flags them. Decoder
  sequence log-probability is the right fix; the Laplacian gate is not.
- **`discount_price` F1 0.667** on 6 test samples — not a meaningful number.
- **Receipt-domain only.** Trained on CORD-v2; expect a drop on other layouts.

## Quick links

- Receipt dataset: [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- Bank-statement dataset: [AgamiAI/Indian-Bank-Statements](https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements) (Apache 2.0)
- Calibration dataset: FUNSD (real scanned forms)
- Degradation engine: Augraphy / custom OpenCV pipeline
- Models: Donut-base (~200M) + LoRA r=64; Qwen2-VL-7B (QLoRA) evaluated and rejected
- Compute: free Kaggle/Colab T4
