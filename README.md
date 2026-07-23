# Kaggle — Spaceship Titanic (write-up)

Documented approach to the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition: predict which passengers were transported to another dimension (binary classification, ~8700 train rows, mixed numeric/categorical features).

**Public leaderboard: 0.80243** (top scores on this permanent competition hover around 0.81–0.82).

## Approach

1. **EDA** — missing-value patterns (CryoSleep ↔ spending columns), group structure in `PassengerId`, cabin deck/side extraction
2. **Feature engineering** — total spending, spending ratios, group size, cabin deck/num/side split, `CryoSleep` imputation from spending
3. **Models** — regularized gradient boosting (LightGBM / XGBoost) vs. HistGradientBoosting baseline, stratified 5-fold CV
4. **Validation discipline** — all preprocessing inside the CV loop (no leakage), single held-out seed for final comparison

## Reproduce

```bash
pip install -r requirements.txt
kaggle competitions download -c spaceship-titanic -p data/
python -m src.train
```

## Results

| Model | CV accuracy (5-fold) | Public LB |
|-------|----------------------|-----------|
| HistGradientBoosting | **0.8115 ± 0.0063** | **0.80243** |
| LightGBM (600 trees, lr 0.03) | 0.8103 ± 0.0053 | — |

The ~0.9pt gap between CV and LB is expected sampling noise on ~4300 test rows — no sign of leakage or overfitting.

## Ideas to push further

- Group-level target encoding (careful: must stay inside CV folds)
- Imputation of `HomePlanet`/`Destination` from group members
- Model blending (HistGB + LightGBM + CatBoost)
