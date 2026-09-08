# Spaceship Titanic

My notes on the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition. About 8700 passengers, and you have to predict which ones were transported to another dimension. Tabular data, lots of missing values.

Public leaderboard: **0.80243**.

## What I tried

I started by looking at where the missing values were, and two things stood out.

First, `PassengerId` is not just an id. It has the form `gggg_pp`, where the first part is a travel group. So passengers travelling together share a prefix, and group size turns out to be informative. Same idea for `Cabin`, which packs deck, number and side into one string.

Second, passengers in cryosleep spend nothing, by definition. That gives you a way to fill in missing `CryoSleep` values from the spending columns instead of just using the mode, and it works in both directions.

From there the features were: group size and a "travelling alone" flag, deck / cabin number / side, total spending, a "spent nothing" flag, and spending split into luxury (spa, VR deck, room service) versus basic (food court, shopping mall).

I compared HistGradientBoosting against LightGBM with 5-fold stratified CV. They ended up within a hair of each other:

| | CV accuracy | Leaderboard |
|---|---|---|
| HistGradientBoosting | 0.8115 ± 0.0063 | 0.80243 |
| LightGBM | 0.8103 ± 0.0053 | not submitted |

One thing I paid attention to: all the imputation and encoding sits inside the sklearn `Pipeline` that gets passed to `cross_val_score`. That way the imputation statistics are recomputed on each training fold. Doing it before the split would leak information from the validation fold and inflate the score.

The categorical encoding differs per model, one-hot for HistGradientBoosting and ordinal for LightGBM, since LightGBM handles categories natively.

## About the CV to leaderboard gap

CV says 0.8115, the leaderboard says 0.8024, so roughly 0.9 points apart. With ~4300 test rows and a fold-to-fold standard deviation of 0.0063 already, that is normal sampling noise. A gap of several points would have been the warning sign.

## Running it

```bash
pip install -r requirements.txt
kaggle competitions download -c spaceship-titanic -p data/ --unzip
python -m src.train
```

## Things I did not get to

- Target encoding at the group level, being careful to keep it inside the CV folds
- Filling `HomePlanet` and `Destination` from other members of the same group
- Blending several models, though I doubt it buys much here
- SHAP values, to check whether the features I think matter actually do
