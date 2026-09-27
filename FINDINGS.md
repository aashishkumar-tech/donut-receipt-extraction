# Donut LoRA Fine-Tuning — Findings & Next Steps

_Last updated: 2026-09-27_

## UPDATE 2026-09-27 — Same method, readable inputs: **F1 0.026 → 0.901**

The bank-statement conclusion was "the pipeline is sound, the images are unreadable."
That claim was untestable on unreadable data, so it was re-run on **CORD-v2 receipts**
(real photos, legible) with the **same** recipe. Notebook:
[`notebooks/donut_cord_receipts.ipynb`](notebooks/donut_cord_receipts.ipynb).
Model: [`aashishkumar-tech/donut-cord-summary`](https://huggingface.co/aashishkumar-tech/donut-cord-summary).

| Model | Precision | Recall | **F1** |
|---|---|---|---|
| Bank statements (this repo, 580 docs) | 0.031 | 0.022 | 0.026 |
| **Receipts — LoRA r=64, 800 imgs, 20 epochs, 7 h on one T4** | 0.908 | 0.894 | **0.901** |
| Reference `donut-base-finetuned-cord-v2` (full fine-tune) | 0.985 | 0.967 | 0.976 |

Per field: `changeprice` 0.946, `cashprice` 0.917, `total_price` 0.916,
`subtotal_price` 0.868, `tax_price` 0.854, `discount_price` 0.667 (only 6 test samples).
LoRA on 800 images lands **0.075 behind a full fine-tune** — the method works.

### Finding 1 — the save bug was never actually fixed, and it cost two runs

The 2026-09-22 note above says the corrupt-checkpoint problem was solved by
"reconstructing a clean plain model". **It was not.** The hand-rolled state-dict fold
(`removeprefix('base_model.model.')`, `.base_layer`, `.modules_to_save.default`) silently
kept **base** weights everywhere once LoRA covered ~10 module types plus two
`ModulesToSaveWrapper`s. The guard passed because `matched: 484` counted **shape**
matches — which untouched base weights satisfy trivially.

Detected by a new check that compares **loss on identical batches**, in-memory vs on-disk:

```
    trained   on_disk     delta
     0.0000    9.1680    9.1680      -> max |delta| = 11.5135  SAVE BUG
```

Fix: delete the manual fold, use PEFT's `merge_and_unload()`. Re-scoring the **same**
weights took F1 from **0.070 → 0.901 with no retraining**.

> **A shape/key-count check proves nothing about a checkpoint. Compare losses.**
> Every run in this repo scored before 2026-09-27 should be treated as suspect — the
> 0.026 bank-statement number was very likely measured on an untrained checkpoint too.

### Finding 2 — the quality gate did **not** transfer to real photos

The 2026-09-23 recommendation was a Laplacian legibility gate. On receipts it **hurts**:

| pct | coverage | auto F1 |
|---|---|---|
| p0 (ungated) | 100% | **0.901** |
| p90 | 10% | 0.875 |

Lift @p90 = **−0.026**. On bank statements the blur was applied *artificially*, so the
Laplacian score tracked the exact corruption hurting accuracy — the gate was measuring
the augmentation, not legibility. On natural photos a low score usually means a sparse,
plain receipt, not an illegible one. **Ungated at 100% coverage beats every gated point.**
The right abstain signal is decoder sequence log-probability, not an image statistic.

### What held from the earlier findings
- **fp32 on T4** — loss fell 2.088 → 0.007; no repeat of the fp16 `loss=47` collapse.
- **Guided per-field decoding** — zero tag corruption across 100 test receipts.
- **Honest scorer over trained fields only** — plus a *published reference model* scored
  by the same function, which is what makes 0.901 meaningful rather than self-reported.
- **Flat scalar fields only** — every field ≥ 0.85 except the 6-sample `discount_price`.

### Open item
`eval_loss` bottomed at **0.1706 (epoch 6)** and rose to **0.2036 (epoch 20)** — overfit
after epoch 6, and `save_total_limit=1` discarded the better checkpoint. An 8-epoch rerun
with `load_best_model_at_end=True` should close part of the 0.075 gap in ~3 h.

---

## UPDATE 2026-09-23 — Decoding fixed; real bottleneck is input legibility

### What changed
- **Free-generation collapse is solved via guided (constrained) decoding.** Free-running
  `generate()` looped `<s_extraction>` / emitted doubled, out-of-order tags even with
  `repetition_penalty` + beams. Fix: **supply the tag skeleton and fill one field value
  at a time**, banning every other structural tag, stopping on `</s_field>`. Output is
  now clean and parseable:
  ```
  Parsed2 -> {'account_number': '60273451072', 'micr_code': '911778412',
              'opening_balance': '45012', ...}
  ```
  Implemented in `predict_donut.py` `generate_prediction()` (cell 4.2) and mirrored in
  the 5.1b debug cell.

### Honest full-test result (run `20260922_143157`, 580 docs, header-only F1)
```
MICRO  precision=0.031  recall=0.022  F1=0.026   (was 0.009)
```
| Signal              | Value                                             |
|---------------------|---------------------------------------------------|
| Best fields         | account_number 0.072, end_date 0.064, ifsc 0.058  |
| Worst fields        | statement_date 0.000, account_holder 0.002        |
| Tier gradient       | clean 0.047 → light 0.040 → medium 0.019 → heavy 0.002 |
| likely_hallucination| 575 / 580                                         |

The clean→heavy gradient is monotonic: **accuracy tracks image quality, not model tricks.**

### Root cause (confirmed by two independent methods)
- Teacher-forcing shows the model *reads* clean values correctly, but accuracy collapses
  as degradation rises.
- **A vision LLM (Option C) was independently tried and ALSO failed on medium/heavy
  degraded images.** Two unrelated approaches hitting the same floor ⇒ the degraded
  images **do not contain recoverable text**. This is an **input-legibility problem, not
  a model-capability problem.** Fine-tuning vs LLM is a wash on unreadable input.

### Recommended path forward (not "scrap the use case")
1. **Quality-gated pipeline** — score each image (Laplacian variance / OCR confidence);
   auto-extract clean+light, **route medium+heavy to restoration or human review**
   (abstain instead of hallucinate).
2. **Image restoration** on the deferred bucket — deskew, denoise, Sauvola binarization,
   Real-ESRGAN super-resolution — then re-extract.
3. **Re-capture / higher-quality source** where possible.
4. Report **coverage + accuracy on the auto bucket** (high F1 on the readable subset)
   rather than a single blended number dragged down by unreadable inputs.

---

_(Prior 2026-09-22 findings below.)_

## TL;DR (UPDATED — it was NOT a data problem)
The model **can read values off the image** — teacher-forced PRED matches GOLD almost
exactly (ifsc_code, micr_code, bank_name, account_holder all EXACT on the first sample).
The earlier **0.000 F1 was caused by uploading a corrupt PEFT checkpoint** to HF: the
backup missed `modules_to_save` (embed_tokens / lm_head), so on restore those weights
were randomly re-initialized → out-of-range tag ids → CUDA index assert / hallucinated
values. **Fix = save the reconstructed clean plain model and guard the upload**, not more
data. A hard pre-upload guard now refuses to push any model that still has PEFT keys or
the wrong vocab (cell 3.6b).

### Evidence (2026-09-22, 400 docs / 1,600 imgs, 12 epochs, val_loss ~0.93)
```
PRED: <s_account_number> 788439336112 <s_ifsc_code> PROG0920445 <s_micr_code> 456316854
      <s_bank_name> Progressive National Bank <s_account_holder> TRADE LINKS INDIA ...
GOLD: <s_account_number> 78439336112  <s_ifsc_code> PROG0920445 <s_micr_code> 456316854
      <s_bank_name> Progressive National Bank <s_account_holder> TRADE LINKS INDIA ...
```
Near-perfect teacher-forced reading. Remaining gap is exposure bias in free-generation,
addressable via decoding (bad_words/no_repeat already in place) — NOT data volume.

---

## (Superseded) earlier hypothesis
The note below concluded the fix was "more and more diverse training data". That was
based on the corrupt-checkpoint run and is now known to be wrong — kept for history.

---

## Honest baseline (run `20260920_013232_donut_finetuned`, 20 docs)

Scored on **only the 10 trained `HEADER_FIELDS`** (cell 5.4):

| FIELD             | TP | FP | FN | F1    |
|-------------------|----|----|----|-------|
| account_number    | 0  | 20 | 20 | 0.000 |
| ifsc_code         | 0  | 0  | 20 | 0.000 |
| micr_code         | 0  | 2  | 20 | 0.000 |
| bank_name         | 0  | 1  | 20 | 0.000 |
| account_holder    | 0  | 0  | 20 | 0.000 |
| opening_balance   | 0  | 2  | 20 | 0.000 |
| closing_balance   | 0  | 8  | 20 | 0.000 |
| start_date        | 0  | 3  | 20 | 0.000 |
| end_date          | 0  | 0  | 20 | 0.000 |
| statement_date    | 0  | 4  | 20 | 0.000 |

**MICRO precision=0.000  recall=0.000  F1=0.000**

This is the official baseline to beat after the data run.

---

## What we ruled out (already fixed / not the cause)
- **Save/load corruption** — reconstructed clean plain `VisionEncoderDecoderModel`;
  reload check passes (`matched: 484`, vocab 57547, 0 peft keys, single
  `model.safetensors`).
- **Generation config** — `pad`/`decoder_start`/`eos` re-applied in the fresh process;
  `decoder_start` prepended to match training.
- **Generation collapse** — model over-emitted `<s_extraction>`; suppressed via
  `bad_words_ids` + `no_repeat_ngram_size=3`. Predictions are now non-empty (20/20).
- **Scorer / formatting** — 5.3 confirmed gold is a clean flat dict with correct keys;
  5.4 normalizes tags/whitespace/number formats. Not a scoring artifact.

## Root cause (confirmed)
- **Teacher-forcing ≈ 0.53 token match, free-gen = 0.000.** That large gap = severe
  **exposure bias on top of undertraining**. The decoder learned the *tag grammar* but
  never learned reliable **image→text reading**.
- Driver: **too little and too uniform data** — ~210 docs across only **3 templates**.
- Secondary: **field-scope mismatch** — gold has ~20 fields (incl. `transactions`,
  `branch_*`, `account_type`, `currency`, `customer_id`, `interest_rate`), but only 10
  header tags were trained. Even a perfect model caps at ~10/20 on the repo scorer.

---

## Next steps (priority order)

### 1. More templates (dominant lever)
- Go from 3 templates → **10–20+** distinct layouts.
- Vary bank names, column order, fonts, header/logo position, date formats.
- Find the template generator in the repo:
  ```bash
  ls src/data/
  grep -rl "template" src/data/ | head
  ```

### 2. Then scale volume
- In the data cell (2.3): change `--n 250` → **`--n 2000+`**.
- Keep the 4 degradation tiers (clean/light/medium/heavy) — good augmentation that
  matches the test tiers.
- **Target: ~2,000–5,000 training images across 10+ templates.**

### 3. Rebuild + back up
- Rerun cells 2.3 → 2.4 to regenerate images and manifests.
- Confirm the dataset cell (3.1) prints a much larger `Train:` count.
- Re-run the **backup-data** cell so the ~2h rebuild is never repeated.
- Watch disk/session limits: `df -h /kaggle/working` before generating; batch if tight.

### 4. Align scored fields to trained fields
- Either train **all** fields the scorer checks (including `transactions` — needs
  `MAX_LENGTH` 1024+ and longer training), **or** keep scoring restricted to the 10
  header fields (cell 5.4) for an honest measure.

### 5. Retrain & re-measure
- Retrain (cells 3.3 → 3.6), then rerun **cell 5.4**. Beat the 0.000 baseline above.

### 6. Only after data is bigger
- Consider more epochs / higher LoRA rank, and/or `predict_with_generate=True` during
  eval to expose the model to its own outputs (reduces exposure bias). Not worth it
  before the data is scaled — data diversity is the real fix.

---

## Key facts / config
- Model: `naver-clova-ix/donut-base` (Swin + BART), PEFT LoRA r=16 / alpha=32,
  `modules_to_save=[embed_tokens, lm_head]`.
- Tokenizer: 57547 (base 57525 + 22 custom tags). `TASK_START=<s_extraction>` (57525),
  `TASK_END=</s_extraction>` (57526), `decoder_start=0`, `pad=1`.
- Env: Kaggle 2×T4, torch 2.10.0+cu128, fp32 (fp16 caused loss=47 / 0.000 F1 on T4).
- Paths: `SAVE_PATH=/kaggle/working/donut_finetuned_v2`, `MODEL_REPO=<user>/donut-bank-model`.
