# Predicting Plant-Level Crude Steel Production

This project predicts the annual crude steel production of individual iron and steel plants from their characteristics (capacity, production route, location, age, workforce). It uses the **Global Iron and Steel Tracker (GIST)** from Global Energy Monitor and covers the full machine learning workflow: data loading and validation, cleaning, feature engineering, baseline and linear models, cross-validation, hyperparameter tuning, experiment tracking and model storage.

## Data

Source: [Global Energy Monitor – Global Iron and Steel Tracker](https://globalenergymonitor.org/projects/global-iron-steel-tracker), June 2026 release (V1), plant-level file. The three sheets are used as CSV files:

| File | Content | Rows | Granularity |
|---|---|---|---|
| `plant_data.csv` | Location, owner, equipment, age, workforce, upstream capacities | 1,293 | one row per plant |
| `plant_capacities.csv` | Nominal steel and iron capacities by status | 1,845 | several rows per plant (by status) |
| `plant_production.csv` | Production by type, 2019–2025 | 1,393 | one row per plant and production type |

The three tables are joined on `GEM plant ID` to obtain one row per plant. The unit-level files (steel units, iron units) and the met coal / iron ore file are not used.

**Target:** `crude_steel_production_ttpa`, the most recent available crude steel production (thousand tonnes per annum). The model is trained on `log1p(production)`.

## Requirements

Python 3.11 or newer.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pandera optuna "optuna-integration[mlflow]" mlflow joblib scipy
```

Optional: `skrub` (alternative baseline pipeline), `polars`, and TabFM (bonus in Task 3.2, installed from its GitHub repository).

## How to run

1. Place the three CSV files in the same folder as the notebook.
2. Run the notebook cells in order. Each task depends on variables created in earlier tasks.
3. Outputs are written to the working directory (see [Outputs](#outputs)).

To browse the MLflow runs locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Workflow

### 1. Data setup and exploration

**1.1 Load and inspect.** The capacity table is filtered to operating units (`operating`, `operating pre-retirement`) and summed per plant. For production, the crude steel rows are selected and the latest year with a value is kept, along with that year (`production_year`). The result has 1,293 plants and 55 columns, of which 424 plants have a production value.

**1.2 Schema check with Pandera.** A `DataFrameSchema` built from `df.columns` checks dtypes, required columns (including the target), non-negative capacities, known regions and equipment codes, and a row-level check that production does not exceed 1.1 × capacity. On the raw data, the numeric columns fail because of text placeholders such as `"unknown"` and `">0"`, 869 targets are missing, and 17 plants produce more than 1.1 × capacity.

**1.3 Cleaning.** Decisions based on the schema results:

- **Coerce** text placeholders to NaN.
- **Drop** 869 plants without a target and 30 plants without operating capacity (their production dates from before they closed).
- **Fill** missing route capacities with 0 (no such equipment), and drop the sparse upstream capacity columns (78–99% missing).
- **Fix** an implausible workforce value (3 employees for 2.4 Mt) by setting it to missing.
- **Relax** the production ≤ capacity check to a warning.
- **Log-transform** production, capacity and workforce (skewness goes from about 2.5 to about 0).

The final table has 394 plants.

**1.4 Feature engineering.**

| Feature | Meaning |
|---|---|
| `log_capacity_per_worker` | Labour productivity / automation |
| `eaf_share`, `bof_share` | Production route (mini-mill vs. integrated) |
| `iron_self_sufficiency`, `is_integrated`, `has_dri` | Vertical integration |
| `equip_*`, `n_equipment_types` | Technology mix and plant complexity |
| `data_age` | Age of the production figure (2019–2025) |

Utilisation (production / capacity) is deliberately **not** used as a feature, because it contains the target.

**1.5 Correlations.** Capacity dominates (r = 0.88 with log production), followed by the production route. Several features are redundant (`is_integrated` / `equip_BF` r = 0.98, `bof_share` / `equip_BOF` r = 0.95, `eaf_share` / `bof_share` r = −0.93), so one feature per group is kept for modelling.

### 2. Baseline and linear models

- **2.1** `DummyRegressor` (mean, median) and a scikit-learn `Pipeline` (median imputation with missing indicators, one-hot encoding, standard scaling, Ridge). All preprocessing is fitted on the training split only.
- **2.2** Multiple linear regression with a reduced, non-collinear feature set for interpretation. The capacity elasticity is about 1 (1% more capacity means about 1% more production).

Train/test split: 80/20, `random_state=42` (315 / 79 plants).

### 3. Model evaluation and selection

- **3.1** 5-fold cross-validation (`KFold(5, shuffle=True, random_state=42)`) on the training set.
- **3.2** Comparison of Linear Regression, Ridge and Random Forest on the same folds.
- **3.3** Tuning with `RandomizedSearchCV` (Random Forest, 40 iterations) and `GridSearchCV` (Ridge alpha).

### 4. Model lifecycle

- **4.1** MLflow tracking (`sqlite:///mlflow.db`, experiment `gist_steel_production`): parameters, CV and test metrics, and artifacts (prediction plot, feature list, Pandera schema).
- **4.2** Optuna study (TPE sampler, 30 trials) on the Random Forest, with trials logged to the same MLflow store (experiment `rf_optuna`) via `MLflowCallback`.
- **4.3** The best pipeline (preprocessing and model together) is saved with `joblib`, reloaded, and re-evaluated on the test set with identical results.

## Results

Cross-validation on the training set (5 folds, metrics on the log scale):

| Model | CV RMSE | CV MAE | CV R² |
|---|---|---|---|
| Dummy (median) | 1.190 | 0.941 | −0.05 |
| Linear Regression | 0.574 | 0.340 | 0.75 |
| Ridge (alpha = 1) | 0.564 | 0.330 | 0.76 |
| Random Forest (500 trees) | 0.576 | 0.341 | 0.76 |
| Random Forest (RandomizedSearchCV) | 0.558 | 0.340 | 0.77 |
| Random Forest (Optuna, 30 trials) | 0.553 | – | – |
| **Ridge (tuned, alpha ≈ 17.8)** | **0.548** | **0.329** | **0.78** |

**Selected model: tuned Ridge.** It has the best cross-validation score, trains in milliseconds and is interpretable. The differences between the models (about 0.01–0.03 RMSE) are smaller than the fold-to-fold standard deviation (about 0.12–0.14), which suggests that the remaining error comes from utilisation being hard to predict from plant characteristics, not from the choice of model.

On the held-out test set, the tuned Ridge reaches RMSE ≈ 696 ttpa, MAE ≈ 463 ttpa and R² ≈ 0.80 (log scale).

## Outputs

| File | Description |
|---|---|
| `best_pipeline_ridge.joblib` | Full fitted pipeline (imputer, encoder, scaler, Ridge) |
| `best_pipeline_ridge_metadata.json` | Hyperparameters, features, target transform, CV score, library versions |
| `mlflow.db`, `mlruns/` | MLflow tracking store and artifacts |

Using the saved model:

```python
import joblib
import numpy as np

pipeline = joblib.load("best_pipeline_ridge.joblib")
log_pred = pipeline.predict(X_new)       # X_new: same raw feature columns as in training
production_ttpa = np.expm1(log_pred)     # back-transform from log1p
```

`X_new` must contain the columns listed in the metadata file. Missing values and text categories are handled inside the pipeline.

## Limitations

- **Small, selected sample.** Only about a third of plants report production, probably favouring large plants and countries with good disclosure.
- **Mixed years.** The target comes from different years (2019–2025) depending on the plant. `data_age` partly controls for this.
- **Capacity mismatch.** Only current operating capacity is used, while some production figures predate retirements or expansions (17 plants produce more than 1.1 × capacity).
- **Noisy test estimate.** With 79 test plants, a single split is noisy. Cross-validation is the more reliable comparison.
- **Leakage in exploration.** The median imputation in Task 1.3 used all rows. The models from Task 2 onwards redo imputation inside the pipeline, so reported scores are not affected.

## Licences

- The GIST data belong to Global Energy Monitor; see their website for terms of use and the recommended citation.
- The TabFM pretrained weights (bonus in Task 3.2) are licensed for non-commercial, non-production use only, which is one reason TabFM is not the deployed model.
