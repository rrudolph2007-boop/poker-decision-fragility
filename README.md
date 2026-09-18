# Poker Decision Fragility

Poker Decision Fragility is a probabilistic Texas Hold'em modeling project that studies how reliable a poker decision remains when the opponent's strategy is uncertain.

Traditional equity calculators usually assume that an opponent's range is already known. In real poker, that assumption is rarely true. This project instead models uncertainty in the opponent's range and asks a different question:

> Does a decision remain profitable when our assumptions about the opponent change?

## Current Features

The project currently includes:

* A custom Texas Hold'em hand evaluator
* Monte Carlo preflop equity simulation
* Support for all 169 starting hand classes and 1,326 physical starting combinations
* Automatic baseline hand strength estimation
* Opponent modeling using VPIP, PFR, 3 bet frequency, position, and observed action
* Weighted opponent range inference
* Card removal and blocker effects
* Statistical uncertainty based on the number of observed hands
* Expected value calculations
* Multiple plausible opponent models
* A custom Decision Fragility metric
* Equity and range visualizations
* CSV exports for further analysis

## Decision Fragility

Instead of calculating equity against one fixed opponent range, the program generates many plausible versions of the opponent model.

For each model, it calculates:

1. The opponent's weighted range
2. Hero equity
3. Call expected value
4. Whether the decision is profitable

Let

q = proportion of plausible models in which the decision is profitable

The Decision Fragility Score is defined as:

F = 1 - |2q - 1|

The score ranges from 0 to 1.

* F = 0 means the decision is completely robust across the tested models
* F = 1 means the decision is maximally fragile, with the models split approximately 50/50 on whether the decision is profitable

Decision Fragility is a project defined metric and is not an established poker statistic.

## Opponent Range Inference

The program does not assume that an opponent has one exact range.

Instead, it assigns probability weight to each of the 169 starting hand classes based on:

* Baseline hand strength
* Opponent VPIP
* Opponent PFR
* Opponent 3 bet frequency
* Table position
* Observed preflop action
* Available card combinations after blockers

The model estimates an action likelihood for each hand class and then normalizes those values into a weighted opponent range.

Conceptually:

P(Hand | Action) ∝ P(Action | Hand) × P(Hand)

This produces a probability distribution over possible opponent holdings rather than a simple list of hands that are either "in" or "out" of the range.

## Monte Carlo Simulation

The project uses Monte Carlo simulation to estimate poker equity.

For a fixed opponent model:

Equity = (Wins + 0.5 × Ties) / Simulations

Monte Carlo estimation error decreases approximately according to:

O(n^(-1/2))

while simulation runtime grows approximately according to:

O(n)

This creates a tradeoff between computational cost and statistical precision.

## Model Uncertainty

Opponent statistics are themselves uncertain, especially when only a small number of hands have been observed.

The project uses probability distributions around observed player statistics to generate multiple plausible versions of the same opponent.

A player observed for 50 hands therefore produces more model uncertainty than a player observed for 5,000 hands.

Decision Fragility measures whether that uncertainty is large enough to change the recommended mathematical decision.

## Important Limitations

This project is currently a research prototype.

The opponent action model uses parameterized assumptions to translate statistics and hand strength into action probabilities. The position adjustments and action curves are not claimed to represent solved GTO strategy.

The project therefore does not claim to perfectly predict an opponent's range.

Its purpose is to study how poker decisions behave when the opponent model itself is uncertain.

A major future goal is to replace heuristic action models with probabilities learned from real poker hand history data.

## Example Research Question

Suppose Hero is considering a call.

A traditional calculator might estimate:

Hero Equity: 47.8%

Break Even Equity: 46.5%

That result suggests a profitable call.

Poker Decision Fragility asks an additional question:

What happens if our opponent model is slightly wrong?

For example:

Average Equity: 47.8%

90% Model Equity Interval: 43.9% to 51.4%

Profitable Models: 29 / 50

Decision Fragility: 84%

The expected decision may be profitable, but the result is highly sensitive to assumptions about the opponent.

## Project Goal

The long term goal is to build a poker decision engine that models three different sources of uncertainty:

1. Card uncertainty
2. Opponent hand uncertainty
3. Opponent strategy uncertainty

Rather than outputting only a single equity estimate, the system attempts to quantify how much confidence should be placed in the resulting decision.

## Roadmap

Planned improvements include:

* Automated testing for hand evaluation
* Modular project structure
* Reproducible experiments using random seeds
* Confidence interval analysis
* Real poker hand history parsing
* Data driven opponent action models
* Player specific range learning
* Postflop decision modeling
* Improved EV analysis
* Sensitivity analysis for individual hand classes

## Status

Active development.
