# CS2 Major Swiss System Predictor

[中文](README_CN.md) | English

This project predicts CS2 Major Swiss stage results using ELO + Monte Carlo simulation + Valve's official Buchholz pairing rules.

Basically runs 100k Swiss round simulations, then brute-force searches through 10 million Pick'Em combinations to find the optimal prediction. Calculates team ratings based on historical match data.

**v2.0 supports GPU acceleration**: Dozens of times faster than v1.0's 16-core CPU (thanks to **[Tenzray](https://github.com/Tenzray)**'s PR).

## Data Format (Must follow this format when modifying match and team data)

### `data/cs2_cleaned_matches.csv`

CSV file without header, 7 columns:

```
date,team1,score1,score2,team2,tournament,format
2025-11-21,Team A,2,1,Team B,Example Tournament,bo3
2025-11-20,Team C,16,14,Team D,Example League,bo1
```

- `date`: YYYY-MM-DD format
- `team1`, `team2`: Team names (must match your TEAMS list in code)
- `score1`, `score2`: Match scores
- `tournament`: Tournament name (just for reference, write whatever)
- `format`: `bo1`, `bo3`, or `bo5`

### `data/hltv_ratings.txt` (optional)

CSV with header, HLTV ratings:

```csv
team,Maps,KD_diff,KD,Rating
Team A,120,+250,1.08,1.09
Team B,95,+180,1.06,1.07
```

Only `team` and `Rating` columns matter. Grab latest ratings from HLTV.org if you want more accurate initial values.

**Note:** The actual filename in the repository is `hltv_ratings.txt` (not `战队属性.txt`).

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** For GPU acceleration, ensure you have CUDA installed. PyTorch will automatically use GPU if available.

### 2. Configure simulation settings

Edit `batchsize.yaml` to adjust performance:

```yaml
simulation:
  num_simulations: 500  # Number of Monte Carlo simulations

performance:
  eval_batch_size: 5000  # Adjust based on GPU memory
  save_every: 1000000    # Checkpoint frequency

device:
  use_gpu: true          # true for GPU, false for CPU
  gpu_id: 0              # GPU device ID (usually 0)
```

### 3. Edit team data

Edit `cs2_gen_preresult.py` (current team names are just examples, change based on actual situation):

```python
TEAMS = [
    "FURIA", "Natus Vincere", "Vitality", "FaZe", 
    "Falcons", "B8", "The MongolZ", "Imperial",
    # ... 16 teams total
]

ROUND1_MATCHUPS = [
    ("FURIA", "Natus Vincere"),
    ("Vitality", "FaZe"),
    # ... 8 matchups total
]
```

Manually input stage X participating teams and round 1 matchups. Order matters! ROUND1_MATCHUPS defines initial seeds for Buchholz pairing.

### 4. Run two-step workflow

**Step 1: Generate simulation data**

```bash
python cs2_gen_preresult.py
```

This creates `output/intermediate_sim_data.json` with 100k Swiss round simulations.

**Step 2: GPU-accelerated Pick'Em optimization**

```bash
python cs2_gen_final.py
```

## Core Logic

### ELO Rating System

Standard ELO with some improvements:

- Time decay (50-day half-life) - old matches weighted less
- Format weights: BO1=1.0, BO3=1.2, BO5=1.5 (theoretically should be fine)
- Adaptive K-factor: starts at K=50 for quick adjustment, drops to K=30 after 30 matches
- Blends HLTV ratings with historical data (more history = less HLTV weight)

Win probability: `P = 1 / (1 + 10^((rating2-rating1)/400))`

### Buchholz Swiss Pairing

Based on [Valve's official rules](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md)

### Pick'Em Search

Brute-force enumerates all valid combinations:

- 2 teams with 3-0
- 6 teams with 3-1 or 3-2
- 2 teams with 0-3

Total: C(16,6) × C(10,2) × C(8,2) = 10,090,080 combinations

Selects the combo with highest success rate (at least 5 hits) from 100k simulations.

**v2.0 GPU Optimization:** Uses PyTorch tensors and matrix multiplication for batch evaluation, achieving significant speedup on NVIDIA GPUs.

### Final Output

**Generated files:**

- `output/final_prediction.json` - Full results with best Pick'Em combination
- `output/optimized_report.txt` - Human-readable recommendations
- `output/intermediate_sim_data.json` - Cached simulation data (from step 1)
- `gpu_checkpoint.json` - Auto-saved progress (can resume if interrupted)

**Notes:**

- Each team needs at least 10 historical matches for reliable predictions
- Delete `gpu_checkpoint.json` if you want to restart optimization from scratch
- The two-step design allows you to rerun step 2 with different settings without regenerating simulations
- Old v1.0 single-file version is preserved as `cs2_swiss_predictor_old.py`

Finally, this project is inspired by [claabs/cs-buchholz-simulator](https://github.com/claabs/cs-buchholz-simulator)
