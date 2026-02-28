
---

# 📊 Advanced Statistical & Spatial Data Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Statistics](https://img.shields.io/badge/Focus-Statistical%20Modeling-green)
![Spatial Analysis](https://img.shields.io/badge/Domain-Spatial%20%26%20Time%20Series-orange)
![Status](https://img.shields.io/badge/Project-Academic%20Research-critical)

---

## 📌 Project Overview

This repository presents a structured implementation of advanced statistical methodologies across temporal, spatial, multivariate, and distribution-sensitive modeling frameworks.

The objective is to rigorously evaluate structural dependencies in data and demonstrate appropriate model selection under violations of classical assumptions such as independence, homoscedasticity, and spatial randomness.

The project integrates:

* Time Series Modeling
* Spatial Autocorrelation Analysis
* Principal Component Analysis
* Quantile Regression
* Environmental–Health Statistical Modeling
* Urban Surface Spatial Metrics

---

## 📁 Repository Structure

```
├── Auto_Correlation.ipynb
├── Principle Component Analysis.ipynb
├── Quantile Regression.ipynb
├── Spatial Auto Correlation.ipynb
├── Tempreture_Health.ipynb
├── Times_series_modal.ipynb
├── UrbanSurface.ipynb
└── README.md
```

---

# 🔍 Notebook Summaries & Statistical Interpretation

---

## 1️⃣ Auto_Correlation.ipynb

### Objective

To assess temporal dependence and evaluate violation of independence assumptions.

### Statistical Interpretation

Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) reveal statistically significant serial dependence across multiple lags. The null hypothesis of independence is rejected.

This implies:

* Classical regression assumptions are invalid.
* ARIMA-class models are required.
* Ignoring serial dependence would yield inefficient and biased estimators.

---

## 2️⃣ Principle Component Analysis.ipynb

### Objective

To reduce dimensionality while preserving variance structure.

### Statistical Interpretation

Eigenvalue decomposition of the covariance matrix demonstrates strong multicollinearity among predictors. A limited number of orthogonal principal components explain the majority of total variance.

Implications:

* Dimensionality reduction improves numerical stability.
* Variance inflation is mitigated.
* Information loss is statistically minimal.

---

## 3️⃣ Quantile Regression.ipynb

### Objective

To estimate conditional relationships across distributional quantiles.

### Statistical Interpretation

Estimated regression coefficients vary across conditional quantiles, indicating heterogeneity in predictor effects.

Key inference:

* OLS estimates are insufficient when variance is non-constant.
* Tail behavior differs significantly from mean behavior.
* Quantile regression provides robust inference under heteroscedasticity.

---

## 4️⃣ Spatial Auto Correlation.ipynb

### Objective

To detect geographic clustering and spatial dependence.

### Statistical Interpretation

Moran’s I statistic indicates statistically significant positive spatial autocorrelation. The null hypothesis of spatial randomness is rejected.

Consequences:

* Spatial independence assumption is violated.
* Classical regression models are misspecified.
* Spatial econometric models are required.

---

## 5️⃣ Tempreture_Health.ipynb

### Objective

To model environmental effects on health indicators.

### Statistical Interpretation

Temperature variability exhibits statistically significant association with health outcomes. Evidence suggests potential non-linear and seasonal components.

Policy implication:

Environmental factors materially influence public health metrics and must be incorporated into predictive models.

---

## 6️⃣ Times_series_modal.ipynb

### Objective

To construct and validate forecasting models.

### Statistical Interpretation

Stationarity tests necessitated differencing. The fitted ARIMA model satisfies residual diagnostics and captures stochastic structure adequately.

Inference:

* Model assumptions are statistically validated.
* Forecasts are structurally defensible.
* Residual independence confirms adequate model specification.

---

## 7️⃣ UrbanSurface.ipynb

### Objective

To analyze spatial heterogeneity in urban surface characteristics.

### Statistical Interpretation

Spatial metrics reveal clustering and heterogeneity, rejecting the assumption of spatial uniformity.

Conclusion:

Urban structure is governed by systematic geographic processes rather than random dispersion.

---

# 📈 Global Statistical Conclusions

Across all analyses:

1. Independence assumptions are routinely violated in real-world datasets.
2. Distributional heterogeneity invalidates mean-based inference alone.
3. Spatial and temporal dependence must be explicitly modeled.
4. Dimensionality reduction enhances stability without sacrificing explanatory power.
5. Diagnostic validation is mandatory for inferential credibility.

This project demonstrates rigorous applied statistical reasoning across multiple advanced frameworks.

---

# 🛠 Technical Stack

* Python 3.x
* numpy
* pandas
* statsmodels
* scikit-learn
* matplotlib
* geopandas

---

# 📚 Methodological Domains Covered

* Time Series Econometrics
* Spatial Statistics
* Multivariate Analysis
* Robust Regression
* Environmental Statistical Modeling

---

# 🎯 Academic Value

This repository demonstrates:

* Correct identification of assumption violations
* Appropriate model selection
* Diagnostic validation
* Formal statistical interpretation

It is structured for academic submission, research presentation, or advanced coursework evaluation.

---
