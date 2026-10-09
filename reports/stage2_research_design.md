# Stage 2: Research Design

## 1. Working title
Explainable Demand Forecasting and Simulation-Based Inventory Decision Support
Using Tabular Foundation Models (title to be finalised in Stage 12).

## 2. Research gap (to be verified against literature in Stage 12)
The base paper, as described in the project brief, compares transformer-based
tabular models and classical ML for inventory prediction and cost optimisation.
This project differs in the following planned ways. These are design choices,
NOT claims of novelty or superiority:
1. Uses a pretrained tabular foundation model (TabPFN) and checks honestly
   whether it beats simple baselines.
2. Separates model tuning from decision optimisation: the optimiser works on an
   explicit cost function with explicit constraints.
3. Uses leakage-aware, chronological evaluation with a test set used once.
4. Labels every simulated quantity and reports negative results.

## 3. Research questions and hypotheses
RQ1: Does TabPFN improve weekly product-level demand forecasts over baselines?
  H0: no difference in MAE versus the best baseline on the test weeks.
  Evidence: paired bootstrap confidence interval of MAE difference (over products).
  TabPFN counts as better only if the interval excludes zero.
RQ2: Can forecasts and predicted cost be turned into lower-cost inventory
  decisions under stated assumptions and constraints?
  Evidence: simulated total cost of optimised policy vs baseline policies on the
  test weeks, across a range of assumed cost parameters. SIMULATION ONLY.
RQ3: Do SHAP, permutation importance and what-if analysis help users understand
  predictions and compare decisions?
  Evidence available: technical validity checks (explanation stability, sanity
  checks). A user study is out of scope, so usefulness for people CANNOT be
  claimed.

## 4. Data and targets
- Modelling window: 2015-01-01 to 2017-09-30 (structure changes from Oct 2017).
- Unit of analysis: product x week (Monday-start, full weeks only).
- Demand target: weekly units sold per product.
- Eligible products: [after checks] (zero-sales weeks <= 20%, sold in first 8 and
  last 8 weeks of the window).
- Cost target (PROVISIONAL): implied cost ratio = 1 - Order Item Profit Ratio.
  Derived from profit data, NOT an observed supply-chain cost. Final choice
  [after checks].

## 5. Split (chronological, never random)
- Train: first 70% of weeks. Validation: next 15%. Test: last 15%.
- Exact dates: reports/stage2_split.json.
- All hyperparameter and model choices use validation only.
- Test set is evaluated once, after all choices are frozen.
- Lag and rolling features use only information from before the target week.

## 6. Models
Demand: previous week, 4-week moving average, seasonal naive (if justified),
  Random Forest, XGBoost, TabPFN (TabPFN-TS only if the setup supports it).
Cost: product-mean baseline, Random Forest, XGBoost, CatBoost.

## 7. Metrics
MAE, RMSE, R2. WAPE (sum of absolute errors / sum of actuals).
MASE (error scaled by the in-sample naive error).
MAPE is NOT used: 63% of product-weeks have zero sales.

## 8. Uncertainty
- Paired bootstrap over products for metric differences (95% interval).
- Repeat stochastic models over 5 fixed seeds; report mean and spread.
- Report negative results.

## 9. Ablations (feasible with this data)
- Without lag/rolling features.
- Without discount rate.
- Without shipping-mode features.
- TabPFN vs the same features in XGBoost.

## 10. Decision optimisation (design preview)
- Decision variables per product: reorder point s, order-up-to level S (or order
  quantity Q), and shipping mode.
- Objective: ordering cost + holding cost + shipping cost + stockout penalty +
  late-delivery penalty.
- Inputs from data: forecast demand, forecast error spread (validation
  residuals), delivery days and late rates by shipping mode.
- ASSUMED (not in data, always labelled): initial stock, ordering cost, holding
  rate, stockout penalty, shipping-cost multipliers, budget, supplier capacity.
- Constraints: target service level, budget, capacity, lead-time limit.
- Method: SciPy-based constrained search. Optuna is used only for model tuning.
- Baseline policies: fixed order quantity; moving-average order-up-to rule.
- Evaluation: simulate each policy on the test weeks with actual demand; verify
  that every constraint holds; sensitivity analysis over cost assumptions.

## 11. Explainability
SHAP global and local explanations, permutation importance on validation/test
data. Explanations describe the model, NOT causes in the real world.

## 12. Threats to validity
- Demand patterns look weakly structured (see stage1 findings); models may not
  beat naive baselines.
- Data may be partly synthetic or rule-generated; origin unknown.
- No stock, supplier, or true cost data; optimisation is a simulation.
- Lead time is delivery time to customers, not supplier replenishment time.
- Single company, 2015 to 2017; regime change after Sept 2017 excluded.
- Cost target is derived from profit, not observed.

## 13. Findings from feasibility checks (Stage 2)

Evidence (all from train/validation weeks only, produced by src/stage2_checks.py):
- 96 products sold in the 142 full weeks; 14 had <= 20% zero-sales weeks.
- Median lag-1 autocorrelation -0.033 and lag-52 autocorrelation 0.003 over 9
  dense products (99 training weeks).
- Validation one-week-ahead MAE: training mean 22.85, 4-week average 24.56,
  previous week 30.25. These ratios are close to what noise around a constant
  level would produce. This is evidence, not proof (9 products, 21 weeks,
  lags 1 and 52 only).
- Unit cost equals price*(1-discount rate)*(1-profit ratio) in 100% of rows.
- Cost ratio (1 - profit ratio) is not explained by product, category,
  shipping mode, market, segment, payment type or region (R2 = 0.000 each).

Design updates:
1. Product eligibility is defined from TRAINING weeks only (low zero-week share
   and a sale in the last 8 training weeks). The earlier rule used the full
   window and leaked test-period information. Final thresholds: [after
   stage2_eligibility.py].
2. Baselines: training mean, 26-week moving average, 4-week moving average,
   previous week. The training mean is the main baseline to beat.
3. Paired bootstrap resamples weeks (not products); the product count is small.
4. Cost model target is the cost ratio, compared with a product-mean baseline.
   Dollar unit cost = net price x predicted ratio. Unit-cost accuracy driven by
   price and discount arithmetic is NOT reported as model skill.
5. Late-delivery risk uses empirical late rates by shipping mode (no ML model).
6. The test set is evaluated once, after all choices are frozen. Products that
   stop selling during the test period are reported in the error analysis.

Expected outcome (stated before running experiments): TabPFN may not beat the
training-mean baseline, and the cost ratio may not be predictable. These would
be reported as negative results.

Added threats to validity:
- The product catalog changes between April and September 2017, inside the
  test window.
- Demand and cost-ratio patterns resemble randomly assigned values; the data
  origin is unknown.