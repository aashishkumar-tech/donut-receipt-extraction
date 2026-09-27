# Results

Artifacts from the run reported in [`../README.md`](../README.md). Download these from
the Kaggle session (`/kaggle/working/cord/`) before it expires — `/kaggle/working` is
wiped when the session ends.

| File | Source cell | What it is |
|---|---|---|
| `report.json` | 10.1 | Micro/per-field P/R/F1 for ours and the reference model, plus the full quality-gate sweep |
| `test_preds.jsonl` | 6.1 | 100 rows of `gold` vs `pred` vs `ref_pred`, one per test receipt |

Both are committed as evidence for the F1 0.901 claim. Model weights are **not** stored
here — they live at
[`aashishkumar-tech/donut-cord-summary`](https://huggingface.co/aashishkumar-tech/donut-cord-summary).

## Inspecting the errors

```python
import json
rows = [json.loads(l) for l in open('results/test_preds.jsonl')]
norm = lambda v: ''.join(c for c in str(v or '') if c.isdigit())

for r in rows:
    for f, g in r['gold'].items():
        p = r['pred'].get(f)
        if norm(g) and norm(p) != norm(g):
            print(f"{r['doc_id']:<12} {f:<16} gold={g!r:<12} pred={p!r}")
```

`subtotal_price` (F1 0.868) and `tax_price` (0.854) account for most of the ~40 misses.
