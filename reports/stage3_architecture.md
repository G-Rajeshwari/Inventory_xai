# Stage 3: Architecture and model choices

## Pipeline
raw CSV -> data -> features -> split -> baselines / demand models / cost models
-> metrics -> optimizer -> simulate -> explain / what-if -> reports -> Streamlit.
The dashboard reads saved artifacts only; it computes no hardcoded results.

## Frozen inputs (Stages 1-2)
- Window 2015-01 to 2017-09; weekly product table; split in stage2_split.json.
- 54 eligible products (selected on training weeks); Tier A (9, >= 50 units/week)
  and Tier B (45). Natural gap between 94.5 and 8.2 units/week.
- Frozen constants live in src/config.py.

## Demand models (1 week ahead)
Baselines: training mean (main), MA-26, MA-4, last week.
Learned: Random Forest, XGBoost, TabPFN; same features, same rows, global model
with product id as a feature. TabPFN-TS only if install, licence and fit are
verified in Stage 6. Features use past weeks only.

## Cost model
Target: cost ratio (1 - profit ratio) at order-line level. Baselines: global
mean, product mean. Models: RF, XGBoost, CatBoost. Formula columns excluded.
Dollar cost = net price x predicted ratio (derived, not observed).

## Optimizer (to be built in Stage 7)
Weekly review, order-up-to level and shipping mode per product. Cost = ordering
+ holding + shipping + stockout + late delivery. Data-derived inputs: forecast,
residual spread, late rate by shipping mode, cost ratio. Assumed inputs
(labelled simulation): replenishment lead time, ordering cost, holding rate,
stockout penalty, shipping multipliers, budget, capacity.
Constraints: service level, budget, capacity, scheduled-days limit.
Solver: SciPy. Comparison: optimizer fed with training-mean forecast vs TabPFN
forecast, to measure whether better forecasts change decisions.

## Explainability
SHAP (tree models), permutation importance (all models, held-out data), random
noise control feature. Explanations describe models, not causes.

## Reproducibility
Seeds in config, saved run configs, test set evaluated once.