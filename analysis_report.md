# Insurance Conversion EDA (from `data_(2).csv`)

## 1) Dataset overview
- Rows: **10,000**
- Columns: **12**
- Target: `has_sale` (binary conversion label)
- Positive class rate: **21.94%** (2,194 conversions)

## 2) Data quality findings
- Missing values:
  - `membership_level`: **10.38%** missing
  - `platform`: **10.01%** missing
- `commission_rate` has only one unique value, so it is non-informative and should be removed in modeling.
- `id` behaves like an identifier and should not be used as a feature.
- `dt` is a date column and should be transformed into useful behavioral fields (e.g., weekday/weekend/day-of-month).

## 3) Conversion patterns discovered
### Categorical patterns
- `platform`
  - desktop: **39.77%** conversion
  - mobile: **14.15%** conversion
  - missing platform: **14.99%** conversion
  - **Insight**: platform is a very strong signal.
- `gender`
  - M: **21.91%**
  - F: **21.98%**
  - **Insight**: almost no separation by gender.
- `membership_level`
  - Bronze: **21.61%**
  - Silver: **21.95%**
  - Gold: **22.18%**
  - Platinum: **22.27%**
  - Missing: **22.74%**
  - **Insight**: weak-to-moderate signal; missingness itself might carry signal.

### Numeric patterns (quintile check)
- `age`: strongest in middle-age ranges (roughly 26–43), much lower for youngest and oldest users.
- `monthly_cost`: higher conversion in lower monthly-cost range, then decreasing trend.
- `session_time`: weak non-monotonic relationship; still potentially useful in interaction with other features.
- `household_income`: slight uplift for higher-income segments.

## 4) Preprocessing plan
1. Drop leakage/non-informative columns: `id`, constant features (e.g., `commission_rate`).
2. Parse `dt` and create: `day_of_week`, `is_weekend`, `day_of_month`.
3. Missing value handling:
   - numeric: median imputation
   - categorical: most-frequent imputation
4. One-hot encode categorical variables (`platform`, `gender`, `membership_level`, engineered date categories if needed).
5. Scale numeric variables for linear models.
6. Add interaction-style ratios:
   - `cost_income_ratio = monthly_cost / household_income`
   - `session_per_age = session_time / age`

## 5) Modeling strategy
Because class balance is ~22% positives, pure accuracy can be misleading. We optimize for ranking users likely to convert.

Recommended optimization metric:
- **Primary**: PR-AUC (Average Precision)
- **Secondary**: ROC-AUC
- Optional operating threshold tuning: F2-score (if recall is more valuable for business)

Algorithms to run:
1. Logistic Regression (interpretable baseline)
2. Balanced Random Forest (`imblearn`)
3. LightGBM baseline
4. LightGBM + Optuna hyperparameter tuning (optimize CV PR-AUC)

Implementation is provided in `modeling_pipeline.py`.
