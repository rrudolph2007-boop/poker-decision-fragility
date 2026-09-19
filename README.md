# Poker Decision Fragility

Poker Decision Fragility is a probabilistic Texas Hold'em research project designed to measure how robust a poker decision remains when an opponent's strategy is uncertain.

Rather than assuming that an opponent has one perfectly known range, the program generates multiple plausible versions of that opponent, estimates Hero's equity against each version, calculates the expected value of calling, and measures how often the correct decision changes.

The project currently combines:

- Monte Carlo simulation
- Texas Hold'em hand evaluation
- 169 starting-hand classes
- Card-blocker handling
- Opponent statistics such as VPIP, PFR, and 3-bet frequency
- Position-adjusted opponent modeling
- Bayesian-style range inference
- Beta-distributed uncertainty
- Multiway equity simulation
- Expected value calculations
- A custom Decision Fragility metric
- Automated testing
- Reproducible random seeds

The long-term goal is to replace the current heuristic opponent model with a model trained on real poker hand-history data.

---

## Core Research Question

Traditional poker equity calculations usually assume that an opponent's range is known.

In reality, the range itself is uncertain.

A player might have:

- 30% VPIP
- 22% PFR
- 8% 3-bet
- only 40 observed hands

Those statistics provide evidence about how the player behaves, but they do not determine one exact range.

Poker Decision Fragility asks:

> How stable is a poker decision when reasonable uncertainty about the opponent's strategy is taken into account?

Instead of calculating equity against one assumed range, the program generates many plausible opponent models and observes whether the decision remains profitable across them.

---

# Decision Fragility

Let:

```text
q = proportion of plausible opponent models
    where calling has positive expected value
```

Decision Fragility is defined as:

```text
F = 1 - |2q - 1|
```

This produces a value between 0 and 1.

### Example

If all 50 opponent models say calling is profitable:

```text
q = 1.00
F = 0.00
```

The decision is robust.

If none of the models say calling is profitable:

```text
q = 0.00
F = 0.00
```

The models again strongly agree.

If exactly half say calling is profitable:

```text
q = 0.50
F = 1.00
```

The decision is maximally fragile.

Therefore:

```text
0% Decision Fragility
Strong model agreement

100% Decision Fragility
Maximum model disagreement
```

Decision Fragility is a project-defined metric. It is not an established poker statistic.

---

# Current Simulation Structure

The current version simulates a six-player active pot:

```text
Hero

Modeled Opponent
    +
Random Opponent
    +
Random Opponent
    +
Random Opponent
    +
Random Opponent
```

One opponent receives a probabilistically inferred range.

The other four opponents currently receive uniformly random legal starting hands.

All six players are evaluated on the same five-card board.

The simulation correctly handles multiway split pots.

For example:

```text
Hero wins outright
Equity contribution = 1.00

Hero ties one player
Equity contribution = 0.50

Hero ties two players
Equity contribution = 0.333...

Hero loses
Equity contribution = 0.00
```

---

# Opponent Range Inference

The modeled opponent is described using:

```text
VPIP
PFR
3-bet frequency
Number of observed hands
Position
Observed action
Hero card blockers
```

The current system estimates:

```text
P(action | hand)
```

for every starting-hand class.

It then uses a Bayesian-style update:

```text
P(hand | action)
    proportional to
P(action | hand) × P(hand)
```

The prior probability of a hand class is based on the number of physical combinations still available after Hero's cards are removed.

For example:

```text
AA
normally has 6 combinations

AKs
normally has 4 combinations

AKo
normally has 12 combinations
```

Hero's cards can remove some of these combinations through blocker effects.

---

# Opponent Behavior Model

The current opponent model is heuristic.

Observed player statistics are first adjusted according to position.

For example, the current model assumes that players generally:

```text
play tighter from early position
play wider from late position
```

The program then uses logistic functions to convert starting-hand strength into probabilities such as:

```text
P(raise | hand)

P(call | hand)

P(3-bet | hand)
```

The logistic thresholds are calibrated so that the weighted average action frequency approximately matches the opponent's position-adjusted statistics.

For example, if adjusted PFR is:

```text
20%
```

the model finds a threshold that causes the weighted average:

```text
P(raise | hand)
```

across all available starting-hand combinations to be approximately 20%.

---

# Modeling Uncertainty

The project models uncertainty in several ways.

## Player Statistic Uncertainty

Observed VPIP, PFR, and 3-bet percentages are treated as uncertain estimates.

The program samples plausible versions of these statistics using Beta distributions.

A player observed for:

```text
20 hands
```

receives much more uncertainty than a player observed for:

```text
5,000 hands
```

This allows sample size to affect how confident the program is about the opponent.

## Range Shape Uncertainty

Each plausible opponent model also receives a slightly different logistic slope.

This represents uncertainty about the exact relationship between hand strength and the player's likelihood of taking an action.

---

# Monte Carlo Equity

The program uses Monte Carlo simulation extensively.

For every plausible opponent model:

1. A physical hand is sampled from the modeled opponent's inferred range.
2. Four additional opponents receive random legal hands.
3. A five-card board is dealt.
4. Every player's seven-card poker hand is evaluated.
5. Hero's share of the pot is recorded.
6. The process repeats thousands of times.

Hero's estimated equity is:

```text
total pot share
-----------------
number of trials
```

Increasing the number of Monte Carlo trials generally reduces simulation error.

Typical Monte Carlo error decreases approximately according to:

```text
O(n^-1/2)
```

while runtime grows approximately according to:

```text
O(n)
```

---

# Expected Value

The program calculates the expected value of calling using:

```text
EV = equity × (current pot + call amount) - call amount
```

The break-even equity is:

```text
call amount
--------------------------
current pot + call amount
```

A model considers the call profitable when:

```text
EV > 0
```

Decision Fragility measures how consistently that condition holds across plausible opponent models.

---

# Starting-Hand Strength Model

The project contains all 169 canonical Texas Hold'em starting-hand classes.

Examples include:

```text
AA
KK
AKs
AKo
76s
T9o
```

Baseline strength for each class is estimated using Monte Carlo simulation against a random opponent.

These values are cached in a JSON file so they do not need to be regenerated every time the program runs.

The cache filename automatically reflects the number of baseline simulations used.

Example:

```text
hand_strengths_1000.json
```

---

# Project Architecture

```text
poker-decision-fragility/
│
├── main.py
├── config.py
│
├── poker/
│   ├── __init__.py
│   ├── cards.py
│   ├── evaluator.py
│   ├── hands.py
│   └── equity.py
│
├── models/
│   ├── __init__.py
│   ├── opponent_model.py
│   └── fragility.py
│
├── tests/
│   ├── test_evaluator.py
│   ├── test_hands.py
│   ├── test_opponent_model.py
│   └── test_fragility.py
│
├── README.md
└── .gitignore
```

### `main.py`

Handles the main application flow:

```text
User input
Opponent-range inference
Simulation execution
Results
CSV output
Graphs
```

### `config.py`

Contains simulation settings and modeling assumptions.

Examples include:

```text
Random seed
Number of Monte Carlo trials
Number of plausible opponent models
Position multipliers
Logistic slopes
Beta prior settings
Range-shape uncertainty
```

### `poker/cards.py`

Contains:

```text
Ranks
Suits
Rank values
52-card deck
```

### `poker/evaluator.py`

Evaluates seven-card Texas Hold'em hands.

Supported categories include:

```text
High card
Pair
Two pair
Three of a kind
Straight
Flush
Full house
Four of a kind
Straight flush
```

### `poker/hands.py`

Generates:

```text
169 starting-hand classes
Physical card combinations
Blocker-adjusted combinations
```

### `poker/equity.py`

Handles baseline Monte Carlo hand-strength estimation and cache loading.

### `models/opponent_model.py`

Contains the current heuristic behavioral model:

```text
Position adjustment
Logistic action probabilities
Threshold calibration
```

### `models/fragility.py`

Contains:

```text
Beta uncertainty sampling
Plausible opponent generation
Weighted range sampling
Multiway Monte Carlo simulation
Expected value calculations
Decision Fragility
Summary statistics
```

---

# Configuration

Important settings are centralized in:

```text
config.py
```

Current settings include:

```python
RANDOM_SEED = 42

BASELINE_TRIALS_PER_CLASS = 1000

NUMBER_OF_MODELS = 50

TRIALS_PER_MODEL = 6000

RANDOM_OPPONENTS = 4
```

Centralizing these settings makes experiments easier to understand and reproduce.

It also clearly separates:

```text
Model assumptions
```

from:

```text
Model implementation
```

This will become particularly important as heuristic assumptions are replaced with learned parameters.

---

# Reproducibility

The development version currently uses a fixed random seed.

This means that, given the same:

```text
Inputs
Model settings
Cached hand-strength model
Random seed
```

the Monte Carlo process should reproduce the same pseudo-random sequence.

This is useful for:

```text
Debugging
Testing
Comparing model changes
Research reproducibility
```

---

# Automated Testing

The project uses `pytest`.

Current tests cover areas including:

```text
Poker hand rankings
Wheel straights
Split board hands
Starting-hand generation
Physical combination counts
Card blockers
Sigmoid behavior
Position adjustments
Action probability calibration
Beta uncertainty
Weighted range construction
Decision-model helper functions
```

Run the complete test suite with:

```bash
python -m pytest -v
```

---

# Running the Program

Install required packages:

```bash
python -m pip install matplotlib pytest
```

Run the application:

```bash
python main.py
```

The program will request:

```text
Hero's two cards

Modeled opponent VPIP
Modeled opponent PFR
Modeled opponent 3-bet percentage
Number of observed hands
Opponent position
Observed opponent action

Current pot
Amount required to call
```

Example:

```text
Hero hand:
AH JD

Opponent VPIP:
35

Opponent PFR:
28

Opponent 3-bet:
9

Observed hands:
25

Position:
UTG

Observed action:
RAISE

Current pot:
100

Call:
129
```

---

# Output

The program reports information including:

```text
Position-adjusted opponent statistics
Most likely opponent starting hands
Posterior range mass
Break-even equity
Average Hero equity
Minimum and maximum Hero equity
Approximate model equity interval
Average call EV
Minimum and maximum call EV
Number of profitable models
Decision Fragility score
Simulation runtime
```

The program also produces CSV output containing the inferred range and model-level simulation results.

---

# Visualizations

The current program generates two graphs.

## Decision Fragility Graph

Plots Hero's equity across plausible opponent models.

A horizontal line represents the break-even equity required to call.

Models above that line consider the call profitable.

Models below it consider the call unprofitable.

## Inferred Opponent Range

Displays the starting-hand classes with the largest posterior range mass after observing the opponent's action.

---

# Current Limitations

The current opponent-range model is intentionally a research prototype.

### Heuristic opponent model

Position multipliers and logistic action curves are manually specified assumptions.

They are not currently:

```text
learned from real poker hand histories
derived from solver GTO ranges
personalized from complete player histories
```

### Random additional opponents

Only one opponent receives an inferred behavioral range.

The four additional opponents currently receive uniformly random legal hands.

Future versions could infer a separate range for every active opponent.

### Baseline hand-strength noise

Starting-hand strength is estimated using Monte Carlo simulation.

Therefore, the baseline strength estimates contain sampling error.

### Hidden-card information

Real poker datasets often do not reveal the hole cards of players who fold.

This creates a major challenge for future machine-learning development because training only on showdown hands can introduce selection bias.

### Decision Fragility measures decision disagreement

The current metric considers whether:

```text
EV > 0
```

under each plausible model.

A model with:

```text
EV = +0.01
```

is treated as profitable just like a model with:

```text
EV = +100
```

Therefore, the current metric measures disagreement in the sign of EV rather than the magnitude of EV differences.

---

# Machine Learning Roadmap

The next major phase of the project is replacing heuristic opponent-range inference with probabilities learned from actual poker decisions.

The target behavioral model will initially estimate:

```text
P(action | hand, position, game context)
```

Possible input features include:

```text
Starting hand
Position
Table size
Effective stack size
VPIP
PFR
3-bet frequency
Number of observed hands
Previous action
Previous raise size
Pot size
Players remaining
Betting history
```

The model can then be combined with Bayesian inference to estimate:

```text
P(hand | action, context)
```

---

# Planned Machine Learning Pipeline

```text
Poker hand histories
        ↓
Parse individual player decisions
        ↓
Structured training dataset
        ↓
Train behavioral model
        ↓
P(action | hand, context)
        ↓
Probabilistic opponent range
        ↓
Monte Carlo equity
        ↓
Expected value
        ↓
Decision Fragility
```

The first machine-learning models will likely favor interpretable methods such as:

```text
Multiclass logistic regression
Gradient-boosted decision trees
```

before exploring more complex neural or sequence models.

---

# Long-Term Player Modeling

Eventually, the project could combine a population model with player-specific evidence.

A new opponent might begin with:

```text
Population poker model
```

As more hands are observed:

```text
0 hands
Mostly population prior

50 hands
Population model + limited individual evidence

500 hands
Increasingly personalized model

5,000 hands
Strong player-specific behavioral model
```

This would allow uncertainty to decrease naturally as the amount of observed information increases.

---

# Future Development

Current:

```text
✓ Texas Hold'em hand evaluator
✓ 169 starting-hand classes
✓ Physical card combinations
✓ Hero blocker handling
✓ Monte Carlo starting-hand strength
✓ Position-adjusted opponent modeling
✓ Bayesian-style range inference
✓ Beta uncertainty modeling
✓ Six-player multiway simulation
✓ Split-pot equity handling
✓ Expected value calculation
✓ Decision Fragility metric
✓ Fixed random seed
✓ Automated testing
✓ Modular architecture
✓ Centralized configuration
```

Next:

```text
→ Acquire real poker hand-history data
→ Design the machine-learning dataset schema
→ Parse hands into individual player decisions
→ Train P(action | hand, position, context)
→ Measure out-of-sample prediction performance
→ Evaluate probability calibration
→ Replace heuristic action likelihoods with learned probabilities
```

Later:

```text
→ Player-specific learned models
→ Action-conditioned ranges for every opponent
→ Effective stack depth
→ Bet-size modeling
→ Full preflop action histories
→ Postflop range inference
→ Flop, turn, and river decision modeling
→ Sequence models for complete betting histories
→ EV sensitivity metrics
→ Decision Fragility zones
```

---

# Research Direction

The current version should be viewed as the mathematical and software framework for a more data-driven system.

The progression is:

```text
Hand-designed heuristic model
            ↓
Data-driven population model
            ↓
Player-specific probabilistic model
            ↓
Range uncertainty
            ↓
Equity uncertainty
            ↓
Decision Fragility
```

The ultimate goal is not to predict an opponent's exact cards.

Instead, the goal is to produce a calibrated probability distribution over possible holdings based on observed behavior and game context, and then determine how sensitive a poker decision is to uncertainty in that distribution.

---

# Status

Active research and development project.

The current opponent model is heuristic.

The next major development phase is real-data collection and machine-learning-based behavioral modeling.
