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

## 14. Final product selection (Stage 2, frozen)

- Eligibility (training weeks only): zero-sales share <= 0.20 and at least one
  sale in the last 8 training weeks. Result: 54 products (the original catalog).
- No eligible product was silent in the last 8 validation weeks.
- Tiers by mean training units per week: Tier A (>= 50) and Tier B (< 50).
  Counts and statistics: reports/eligible_products.csv.
- Metrics are reported per tier AND pooled, because a handful of high-volume
  products dominate pooled absolute errors. Scaled metrics (WAPE, MASE against
  the training mean) are used to compare tiers.
- Thresholds are frozen. Aggregate full-window statistics were seen before the
  thresholds were set, so no further tuning of selection rules is allowed.
- Rows for TabPFN: 54 products x training weeks, before lag features
  (about 5,300 rows), within current TabPFN size guidance.

  ## 15. Stage 5 baseline results and pre-specified decision rules

Observed on VALIDATION weeks (reports/baseline_validation.csv): the training mean
has the lowest MAE in every group. Relative to it, MAE is +1.6% (MA-26), +8.3%
(MA-4) and +35.1% (last week) pooled. RMSE ratios to the training mean are close
to 1.41, 1.12 and 1.02 (last week, MA-4, MA-26), which is what random scatter
around a fixed level would produce. This is evidence from 21 weeks, not proof.

R2 is not used to judge skill: pooled and Tier A R2 are inflated by volume
differences between products, and Tier B R2 is about zero.

Decision rules, fixed BEFORE any machine-learning model is run:
1. A model "improves on the baseline" only if the 95% paired week-bootstrap
   interval of (model MAE - training-mean MAE) on VALIDATION weeks lies entirely
   below zero for the pooled data. Tiers are reported separately.
2. If no model meets rule 1, the RQ1 conclusion is "no improvement detected on
   this data". The training mean is then the demand forecast used by the
   optimizer, with TabPFN as a sensitivity comparison.
3. Hyperparameters: library defaults first. Any tuning (small fixed grid or
   Optuna) uses train -> validation only, and the number of trials is logged.
4. The test set is evaluated once, after rules 1 to 3 have been applied.

## 16. Stage 6 outcome on VALIDATION weeks (library defaults, no tuning)

Software: tabpfn 9.1.0, weights tabpfn-v3.5-20260909 (non-commercial license
accepted by the author), local CPU inference. Models trained on 3,942 rows with
10 features, evaluated on 1,134 validation rows. Predictions clipped at zero.

Pooled MAE relative to the training mean (95% paired week-bootstrap interval of
the MAE difference in brackets):
- Random Forest (5 seeds): +2.8% [-0.06, +0.49] (no detectable difference)
- XGBoost (defaults; seed has no effect, so effectively 1 run): +12.7% [+0.48, +1.33]
- TabPFN (5 seeds): +13.1% [+0.58, +1.32]
Tier A (9 high-volume products): TabPFN +25.1% [+3.45, +8.03]; XGBoost +16.1%;
Random Forest +3.6% (not distinguishable). Tier B (45 products): TabPFN -0.3%
[-0.05, +0.03] (not distinguishable).

Decision rule 1 (95% interval entirely below zero) is NOT met by any model.
Rule 2 therefore applies: the RQ1 conclusion is "no improvement over a
per-product training mean detected on this data", and the training mean is the
demand forecast used by the optimizer, with TabPFN as a sensitivity comparison.

Limits of this conclusion: one dataset; 21 validation weeks; default settings;
a single feature set. These results say nothing about TabPFN in general.
The feature ablation (Section 17) tests whether the history features cause the
extra error.

## 17. Stage 6b ablation (VALIDATION weeks, library defaults)

Variants: full (10 features), history_only (no week of year), id_season (product
code + week of year), id_only (product code only).
Findings:
1. Random Forest and XGBoost with id_only reproduce the training mean (pooled MAE
   7.294 / 7.297 vs 7.297). Trees can learn each product's level; any extra error
   comes from additional features.
2. Adding week of year to the product code raises Random Forest's pooled MAE by
   12.7% (interval above zero), consistent with no detectable seasonality.
3. TabPFN with numeric product code does not reproduce the training mean even with
   product code only (+14.6% pooled, interval above zero), and is indistinguishable
   from TabPFN with all features. History features are therefore not the cause of
   its extra error. No variant matches the training mean pooled.
4. Possible explanation (UNTESTED when written): the integer product code is read as
   an ordered number, which may blur unrelated products. Tested in Section 18.
Caveats: many intervals are reported from 21 validation weeks; differences of a few
hundredths between variants are within noise.

Pre-registration note: variants in this section and in Section 18 were chosen after
seeing the validation results in Section 16. They are exploratory. The
pre-specified comparison remains the one in Section 16.