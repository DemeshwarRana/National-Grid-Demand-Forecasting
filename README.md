# National Grid Demand Forecasting

Forecasting next-day electricity demand for Great Britain's national grid using recurrent neural networks (SimpleRNN, LSTM, GRU), with a strong emphasis on validating the data and the evaluation itself before trusting the results.

## Business Problem

Great Britain's electricity grid runs on a constraint that gets little attention until it fails: supply and demand have to match almost exactly, in real time, every second. Electricity can't be stored at scale, so if the National Grid Electricity System Operator (ESO) gets its demand forecast wrong, it either runs excess generation (wasted cost, wasted carbon) or too little (risking blackouts, or an expensive scramble to bring extra generation online at short notice through the balancing mechanism).

This problem has gotten harder over the past decade. Wind and solar generation are weather-dependent rather than following the predictable rhythm of coal and gas plants, heating and transport are gradually electrifying and shifting demand patterns, and one-off shocks like COVID lockdowns temporarily rewrote demand entirely. A forecasting approach trained on outdated patterns risks learning relationships that no longer hold.

**Objective:** build a next-day demand forecasting model that meaningfully beats a naive "tomorrow repeats today" baseline, using real historical grid data, and measure that improvement honestly rather than assuming it exists.

## Dataset

- **Source:** [National Grid Energy Consumption 2009-2025](https://www.kaggle.com/datasets/tomfarnell/national-grid-energy-consumption-2009-2025) (Kaggle, sourced from National Grid ESO)
- **Granularity:** half-hourly demand, generation mix, and calendar data
- **Used range:** filtered to 2022 onwards to avoid pandemic-era distortion, with a further correction described below

## Data Validation

A meaningful part of this project was validating the data itself before trusting any model result:

- **Missing values & duplicate timestamps:** checked and handled (interconnector columns with 62-93% missingness were dropped rather than imputed). An initial check found 32 duplicate timestamps; tracing the root cause showed these came from the UK's October "fall-back" daylight saving day (which has 50 half-hourly settlement periods instead of 48) rolling over into the next calendar day. Dropping periods 49/50 fixed the underlying issue, confirmed by re-checking the index (0 duplicates remaining).
- **Timestamp continuity:** checked the time index itself for gaps, not just missing values in existing rows. Found 6 missing timestamps, all falling exactly on the UK's daylight saving "clocks forward" dates, an expected characteristic of the settlement period system (that day only has 46 half-hour periods instead of 48), not a data error.
- **A real data quality issue, caught and fixed:** a year-by-year comparison showed 2025's mean demand was ~25% higher than every other year. Investigating further showed the 2025 data only contained January and two days of February, an incomplete, winter-only partial year that was skewing comparisons and sitting inside the original test split. The incomplete tail was dropped and the pipeline rerun to confirm the fix didn't change results for the worse.
- **Outlier check:** no extreme outliers found in the demand column (|z| > 4).

## Methodology

1. Convert settlement date + period into a proper datetime index
2. Engineer calendar features (day of week, weekend flag)
3. Create the forecasting target: demand shifted 48 half-hour steps (one full day) ahead
4. Split chronologically 80/10/10 (train/validation/test), no shuffling, to avoid leaking future information
5. Standardize features and target
6. Build 48-step (one day) input sequences
7. Compare three recurrent architectures (SimpleRNN, LSTM, GRU) under identical conditions: same seeds, dropout (0.3), and early stopping
8. Train the winning architecture (GRU) with more patience for a proper final run
9. Evaluate against a naive persistence baseline, not just in isolation

## Results

| Model | Validation Loss |
|---|---|
| SimpleRNN | 0.089 |
| LSTM | 0.089 |
| **GRU (selected)** | **0.082** |

After selecting GRU, further tuning (an additional GRU layer, learning rate) brought validation loss down to 0.080 for the final model.

**Final model (GRU) on the test set:**
- **R² = 0.869** (vs. naive persistence baseline of **0.790**)
- **MAE = 1,708.0 MW** (~7% of typical demand)
- **RMSE = 2,200.2 MW**

The gap between RMSE and MAE indicates the model's largest errors are concentrated on a subset of days, visually confirmed to be the coldest, highest-demand periods in the test window, the model tracks typical daily demand well but underestimates the sharpest peaks.

A higher learning rate (0.01) and a larger unit count (64) were tested against the defaults (0.001 and 32); the higher learning rate performed worse, confirming 0.001 as the better choice, while more units gave a small improvement.

## Strengths

- Model accuracy was checked against a naive baseline before being trusted, beating a repetitive "tomorrow repeats today" guess is what actually proves the model learned real patterns.
- Three architectures were compared under identical, controlled conditions rather than picking one and assuming it was right.
- Data validation went beyond a basic missing-value check, including timestamp continuity and cross-year comparison.
- A real data quality issue was caught, traced to its root cause, fixed, and the fix was verified rather than assumed.
- Limitations are stated specifically (which season the test set covers, where predictions are weakest) rather than generically.

## Limitations

- The chronological 80/10/10 split means the test set spans only autumn into early winter; performance on spring/summer demand was not directly evaluated. A 70/15/15 split was tested as an alternative (it would extend test coverage into summer) but scored lower on accuracy, so 80/10/10 was kept.
- The model underestimates peak demand during the coldest part of the test period, the costliest kind of error for a grid operator.
- Outlier and timestamp checks were run on the demand column and time index only; other columns were not checked to the same depth.
- The naive persistence baseline is a useful sanity check but a simple one; a stronger baseline might narrow the model's apparent advantage further.

## Future Work

- Adding temperature data would likely help address the peak-demand underestimation, since the model currently has no way to anticipate a cold snap.
- A multi-year evaluation (e.g., train on 2022-2023, test on 2024) would give a fuller picture of accuracy across all seasons. This is feasible with the data already collected but wasn't run here due to the added training time per fold.
- Further hyperparameter tuning (sequence length, a lower learning rate, dropout rate) remains open.

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Plotly

## Repository Structure

```
├── National_Grid_Demand_Forecasting_.ipynb  # Full pipeline: data validation, modeling, evaluation
├── data/                                     # Raw dataset (National Grid ESO, via Kaggle)
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook National_Grid_Demand_Forecasting_.ipynb
```

Run cells in order from top to bottom. The dataset is expected at `data/National Grid Data 2009-2025.csv`.

## References

National Grid ESO (2025) *Historic demand data*. Kaggle. Available at: https://www.kaggle.com/datasets/tomfarnell/national-grid-energy-consumption-2009-2025

Hochreiter, S. and Schmidhuber, J. (1997) 'Long short-term memory', *Neural Computation*, 9(8), pp. 1735-1780. Available at: https://doi.org/10.1162/neco.1997.9.8.1735

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. and Bengio, Y. (2014) 'Learning phrase representations using RNN encoder-decoder for statistical machine translation', *arXiv preprint*. Available at: https://arxiv.org/abs/1406.1078

Abadi, M. et al. (2015) *TensorFlow: large-scale machine learning on heterogeneous systems*. Available at: https://www.tensorflow.org/

Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830. Available at: https://jmlr.org/papers/v12/pedregosa11a.html

McKinney, W. (2010) 'Data structures for statistical computing in Python', *Proceedings of the 9th Python in Science Conference*, pp. 56-61. Available at: https://conference.scipy.org/proceedings/scipy2010/mckinney.html

## License

MIT, see [LICENSE](LICENSE).
