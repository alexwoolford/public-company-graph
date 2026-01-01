# Entity Resolution Evaluation

This document describes the process for evaluating entity resolution quality using AI-assisted labeling.

## Ground Truth Dataset

**Location:** `data/er_ground_truth_labeled.csv`

### Baseline Results (December 2025)

| Metric | Count | % |
|--------|-------|---|
| ✓ Correct | 70 | 70.0% |
| ✗ Incorrect | 25 | 25.0% |
| ? Ambiguous | 5 | 5.0% |
| **Estimated Precision** | | **73.7%** |

### Common Error Patterns Identified

1. **Generic words matching tickers/names**
   - "Target" (goal) → Target Corp (TGT)
   - "MA" (Marketing Authorization) → Mastercard (MA)

2. **Exchange/venue mentions**
   - "Nasdaq" (exchange listing) → Nasdaq Inc (NDAQ)

3. **Wrong relationship type**
   - Partner labeled as competitor
   - Supplier labeled as customer

4. **Legal/figurative references**
   - "Wayfair" (Supreme Court case) → Wayfair Inc
   - "Starbucks of Marijuana" (analogy) → Starbucks Corp

## Reproducible Process

### Step 1: Create Ground Truth Sample

```bash
# Extract 100 random business relationships from the graph
python scripts/create_er_ground_truth.py --num-samples 100 --output data/er_ground_truth.csv
```

### Step 2: AI-Assisted Labeling

```bash
# Label with gpt-5.2-pro (most accurate, ~30-60 min for 100 samples)
python scripts/label_er_with_ai.py \
    --model gpt-5.2-pro \
    --reasoning-effort high \
    --input data/er_ground_truth.csv \
    --output data/er_ground_truth_labeled.csv

# If interrupted, resume without re-processing completed records:
python scripts/label_er_with_ai.py --resume
```

### Step 3: Calculate Metrics

```python
import csv
from collections import Counter

with open('data/er_ground_truth_labeled.csv') as f:
    rows = list(csv.DictReader(f))

labels = Counter(r['ai_label'] for r in rows)
precision = labels['correct'] / (labels['correct'] + labels['incorrect'])
print(f"Precision: {precision:.1%}")
```

## Model Options

| Model | Accuracy | Speed | Cost | Notes |
|-------|----------|-------|------|-------|
| `gpt-5.2-pro` | Highest | ~30s/record | $0.10-0.20/record | Recommended |
| `gpt-5.2` | High | ~5s/record | $0.02-0.05/record | Good balance |
| `o3` | High | ~10s/record | $0.05-0.10/record | Reasoning model |

## Files

- `data/er_ground_truth.csv` - Original 100 samples (unlabeled)
- `data/er_ground_truth_labeled.csv` - Samples with AI labels
- `scripts/create_er_ground_truth.py` - Sample extraction script
- `scripts/label_er_with_ai.py` - AI labeling script

## Expanding the Dataset

To create additional labeled samples:

```bash
# Generate new samples (different random seed)
python scripts/create_er_ground_truth.py --num-samples 100 --output data/er_ground_truth_batch2.csv

# Label them
python scripts/label_er_with_ai.py \
    --input data/er_ground_truth_batch2.csv \
    --output data/er_ground_truth_batch2_labeled.csv
```

## Using Ground Truth for Evaluation

After implementing ER improvements, compare against the labeled data:

```bash
# Run evaluation script (to be implemented)
python scripts/evaluate_er.py \
    --ground-truth data/er_ground_truth_labeled.csv \
    --compare-new-algorithm
```

This will show precision/recall changes before and after improvements.
