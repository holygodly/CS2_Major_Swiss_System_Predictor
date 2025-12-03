# CS2 Major Swiss System Predictor

[中文文档](README_CN.md) | English

A CS2 Swiss system prediction tool based on historical match data and ELO algorithm.

## Features

- **Adaptive ELO System**: Considers opponent strength, time decay, and adaptive K-factor
- **Monte Carlo Simulation**: 100,000 Swiss rounds simulation using Buchholz pairing algorithm
- **Pick'Em Optimizer**: Brute-force search through 10M+ combinations to find optimal predictions
- **Checkpoint Resume**: Supports resuming from interruptions during long runs
- **Universal Design**: Works with any CS2 Swiss format event (8/11/16/24 teams)

## Quick Start

### Prerequisites

```bash
# Python 3.11+
pip install pandas
```

### Usage

This system uses pure Python implementation of the Buchholz pairing algorithm with no additional dependencies.

#### 1. Prepare Data

**`data/cs2_cleaned_matches.csv`** - Historical match records

Format specification (CSV without header):

- Column 1: date - Match date (YYYY-MM-DD)
- Column 2: team1 - Team 1 name
- Column 3: score1 - Team 1 score
- Column 4: score2 - Team 2 score
- Column 5: team2 - Team 2 name
- Column 6: tournament - Tournament name
- Column 7: format - Match format (bo1/bo3/bo5)

Example:

```csv
2025-11-21,Team A,2,1,Team B,Example Tournament,bo3
2025-11-20,Team C,16,14,Team D,Example League,bo1
2025-11-19,Team A,3,0,Team C,Example Cup,bo5
```

**`战队属性.txt`** - HLTV ratings (Optional)

Format: CSV with header containing:

- team - Team name (must match TEAMS list in code)
- Maps - Number of maps played (for sample size confidence adjustment)
- KD_diff - K/D difference (optional)
- KD - K/D ratio (optional)
- Rating - HLTV rating (core data, typically 0.9-1.1)

Example:

```csv
team,Maps,KD_diff,KD,Rating
Team A,120,+250,1.08,1.09
Team B,95,+180,1.06,1.07
Team C,110,-50,1.02,1.05
```

Tip: Get latest ratings from HLTV.org

#### 2. Configure Code

Edit the configuration section at the top of **`cs2_swiss_predictor.py`**:

```python
# Participating teams (16 teams)
TEAMS = [
    "FURIA", "Natus Vincere", "Vitality", "FaZe", "Falcons", "B8",
    "The MongolZ", "Imperial", "MOUZ", "PARIVISION", "Spirit", "Liquid",
    "G2", "Passion UA", "paiN", "3DMAX"
]

# Round 1 matchups (8 BO1 matches)
ROUND1_MATCHUPS = [
    ("FURIA", "Natus Vincere"),
    ("Vitality", "FaZe"),
    ("Falcons", "B8"),
    ("The MongolZ", "Imperial"),
    ("MOUZ", "PARIVISION"),
    ("Spirit", "Liquid"),
    ("G2", "Passion UA"),
    ("paiN", "3DMAX")
]

# External data file paths
MATCHES_FILE = 'cs2_cleaned_matches.csv'
TEAM_RATINGS_FILE = '战队属性.txt'
```

**Important**: The order of ROUND1_MATCHUPS determines seed positions for the Buchholz pairing system!

#### 3. Run Prediction

```bash
python cs2_swiss_predictor.py
```

The program will automatically:

1. Load historical data and calculate ELO ratings
2. Run 100,000 Swiss round Monte Carlo simulations
3. Brute-force search for optimal Pick'Em combinations (~20 hours)
4. Generate result reports

Output files:

- `prediction_results.json` - Complete prediction results (JSON format)
- `optimized_report.txt` - Readable recommendation report
- `checkpoint_progress.json` - Progress checkpoint (for resume)
- `checkpoint_best.json` - Best solution backup

## Project Structure

```
cs2_major_prediction_system/
├── cs2_swiss_predictor.py           # Main program (only file you need to run)
├── cs2_cleaned_matches.csv          # Historical match data
├── 战队属性.txt                      # HLTV ratings (optional)
├── prediction_results.json          # Output: Complete predictions
├── optimized_report.txt             # Output: Readable report
├── checkpoint_progress.json         # Output: Progress checkpoint
└── checkpoint_best.json             # Output: Best solution backup
```

## Core Algorithms

### 1. Adaptive ELO System

The system dynamically adjusts initial rating weights based on historical data coverage:

**Initialization Strategy**:

- CSV matches < 10: External rating 70% weight (insufficient data, rely on external ratings)
- CSV matches 10-20: Weight transitions linearly (70% → 35%)
- CSV matches 20-30: External rating 35% weight
- CSV matches > 30: External rating 20% weight (sufficient data, historical matches dominate)

**Adaptive K-factor**:

- Matches < 15: K=50 (fast adjustment)
- Matches 15-30: K=40 (balanced)
- Matches > 30: K=30 (stable convergence)

**Other Factors**:

- Time decay: 50-day half-life (older matches have reduced weight)
- Format weight: BO1=1.0, BO3=1.2, BO5=1.5
- Sample size confidence: Maps data for Bayesian adjustment of external ratings

**Calculation Formula**:

```python
expected = 1 / (1 + 10^((elo2 - elo1) / 400))
new_elo = old_elo + K * format_weight * time_weight * (actual - expected)
```

### 2. Buchholz Swiss System (Valve Official Rules)

**Implementation Standard**: Strictly follows [Valve Official Rules](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md)

**Pairing Rules**:

- **Round 1**: Fixed matchup table (1v9, 2v10, 3v11...)
- **Round 2-3**: Highest seed vs lowest seed (avoid rematches)
- **Round 4-5**: Use 15-priority pairing table (avoid rematches)

**Seed Calculation** (Mid-stage Seed):

1. Current record (W-L)
2. Difficulty Score (sum of opponent win-loss differences)
3. Initial seed

**Difficulty Score (Buchholz)**:

```python
difficulty = sum(opponent_wins - opponent_losses)
```

**15-Priority Pairing Table** (Round 4-5):

```
Priority 1:  1v6, 2v5, 3v4
Priority 2:  1v6, 2v4, 3v5
Priority 3:  1v5, 2v6, 3v4
...
Priority 15: 1v2, 3v4, 5v6
```

The system selects the first priority that **doesn't create a rematch**.

### 3. Pick'Em Optimization

**Rules**:

| Category | Count | Description                      |
| -------- | ----- | -------------------------------- |
| 3-0      | 2     | Predict exactly 3-0 advancement  |
| 3-1/3-2  | 6     | Predict 3-1 or 3-2 advancement   |
| 0-3      | 2     | Predict 0-3 elimination          |

**Constraint**: Each team can only be selected once, total 10 teams

**Search Space**:

- Advances combinations (6 teams): C(16,6) = 8,008
- Sub-combinations for each advances:
  * 3-0 combinations (2 teams): C(10,2) = 45
  * 0-3 combinations (2 teams): C(8,2) = 28
  * Sub-combinations: 45 × 28 = 1,260
- Total: 8,008 × 1,260 = 10,090,080 combinations

## Usage Scenarios

### Scenario 1: 16-team Swiss (Standard Major)

```python
TEAMS = [
    "Team A", "Team B", "Team C", "Team D",
    "Team E", "Team F", "Team G", "Team H",
    "Team I", "Team J", "Team K", "Team L",
    "Team M", "Team N", "Team O", "Team P"
]

ROUND1_MATCHUPS = [
    ("Team A", "Team I"),  # Seed 1 vs Seed 9
    ("Team B", "Team J"),  # Seed 2 vs Seed 10
    ("Team C", "Team K"),
    ("Team D", "Team L"),
    ("Team E", "Team M"),
    ("Team F", "Team N"),
    ("Team G", "Team O"),
    ("Team H", "Team P")
]
```

### Scenario 2: 11-team Swiss (Stage 2 Advancement)

```python
TEAMS = [
    "Seed 1", "Seed 2", "Seed 3", "Seed 4", "Seed 5",
    "Seed 6", "Seed 7", "Seed 8", "Seed 9", "Seed 10", "Seed 11"
]

ROUND1_MATCHUPS = [
    ("Seed 1", "Seed 6"),
    ("Seed 2", "Seed 7"),
    ("Seed 3", "Seed 8"),
    ("Seed 4", "Seed 9"),
    ("Seed 5", "Seed 10")
    # Seed 11 gets a bye
]
```

### Scenario 3: 8-team Small Event

```python
TEAMS = [
    "Team A", "Team B", "Team C", "Team D",
    "Team E", "Team F", "Team G", "Team H"
]

ROUND1_MATCHUPS = [
    ("Team A", "Team E"),
    ("Team B", "Team F"),
    ("Team C", "Team G"),
    ("Team D", "Team H")
]
```

## Notes

### Data Requirements

**Recommended**:
- Historical matches covering 3-6 months
- At least 10 match samples per team
- Use latest HLTV ratings (optional)

**Note**:
- Teams without HLTV ratings will be initialized with default ELO=1000
- Teams not found in CSV will trigger warnings

### Configuration Tips

**Key Points**:
- Order of ROUND1_MATCHUPS determines Buchholz pairing rules
- Ensure all participating teams are in the TEAMS list
- Verify round 1 matchups match actual schedule

### Performance Tips

**Best Practices**:
- Use 16-core CPU for optimal performance
- Allow ~20 hours for complete search
- Can interrupt at any time, program auto-saves progress

## Multi-Stage Event Prediction

For events with multiple stages (e.g., Stage 1 → Stage 2 → Stage 3):

1. **Update match data**:
   Add previous stage's actual match results to cs2_cleaned_matches.csv

2. **Update code configuration**:
   ```python
   TEAMS = ["List of teams advancing to Stage 2"]
   ROUND1_MATCHUPS = [("Round 1 matchups for new stage")]
   ```

3. **Re-run**:
   ```bash
   python cs2_swiss_predictor.py
   ```

**Key**: Use latest historical data for each stage to keep ELO ratings reflecting current team states

## FAQ

**Q: Why is ROUND1_MATCHUPS order important?**

A: The Buchholz pairing system relies on initial seed positions. The order of round 1 matchups determines each team's initial seed, affecting subsequent pairing logic (high difficulty vs low difficulty, same difficulty sorted by seed).

**Q: What if a team has no historical data?**

A: The system will initialize according to this priority:
1. Try reading from HLTV ratings file
2. If no external rating, use default ELO (1000)
3. Display warning message

**Q: Why does optimization take 20 hours?**

A: Total search space is ~10.09M combinations. Even with boolean array optimization and 16-core parallelization, complete search requires significant time. You can interrupt and resume via checkpoint at any time.

**Q: What does Pick'Em success rate mean?**

A: It represents the proportion of 100,000 simulations where this prediction combination hits 5 or more picks. This is the minimum standard for Pick'Em rewards.

**Q: How does checkpoint resume work?**

A: The program saves a checkpoint every 200 combinations, recording current progress. If interrupted, re-running will automatically resume from the checkpoint.

## Future Improvements

- [ ] **Pruning Optimization**: Implement L1/L2 pruning logic, expected to reduce search space by 50-70%
- [ ] **Adaptive Parallelism**: Auto-adjust worker count based on CPU cores
- [ ] **Incremental Updates**: Support recalculating only changed portions of ratings
- [ ] **Visualization Interface**: Add web interface to display prediction results and simulation process
- [ ] **Multi-Event Support**: Extend to other esports using Swiss system

Contributions and suggestions welcome!

## Dependencies

- Python 3.11+
- pandas

## Credits

- Buchholz pairing algorithm inspired by [claabs/cs-buchholz-simulator](https://github.com/claabs/cs-buchholz-simulator)
- Team rating data from [HLTV.org](https://www.hltv.org/)
- Swiss system rules reference from Valve CS2 Major official rules

## License

MIT License
