[English](README.md) | [简体中文](README.zh-CN.md)

# U.S. Inflation and Macroeconomic Factors

This individual course project examines monthly U.S. inflation dynamics with Federal Reserve Economic Data. The reproducible workflow combines an autoregressive inflation model with monetary, labor-market, energy-price, and real-activity indicators and tests whether the main conclusions survive alternative specifications.

## Research design

- **Outcome:** year-over-year CPI inflation.
- **Baseline:** AR(3) time-series regression.
- **Covariates:** monetary conditions, labor-market measures, money supply, WTI oil-price growth, and real-activity indicators.
- **Inference:** Newey-West HAC standard errors with 12 lags.
- **Diagnostics:** descriptive statistics, correlations, ADF tests, VIF, Durbin-Watson, Ljung-Box, and heteroskedasticity checks.
- **Robustness:** alternative inflation measures, lag orders, sample definitions, HAC bandwidths, inflation-change models, and pre/post-pandemic subsamples.

## Main findings

| Quantity | Result |
|---|---:|
| Analysis period after lag construction | April 2001-May 2026 |
| Baseline observations | 298 |
| Selected autoregressive order | 3 |
| Adjusted R-squared | 0.953 |
| Sum of lagged-inflation coefficients | approximately 0.871 |

Inflation is highly persistent in the sample. WTI oil-price growth has a stable positive conditional association with year-over-year CPI inflation, while money-supply growth is less stable across specifications. The post-pandemic subsample has a higher inflation intercept. These are observational associations, not identified causal effects.

## Run the workflow

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The command rebuilds the processed data, analysis sample, source audit, regression output, diagnostics, and robustness tables.

## Repository guide

```text
code/            data cleaning and empirical analysis
data/raw/        downloaded FRED series
data/processed/  harmonized monthly data and source audit
data/final/      analysis sample
results/tables/  descriptive, diagnostic, baseline, and robustness results
results/logs/    machine-readable analysis summary
```

Series identifiers and source provenance are recorded in `data/processed/data_source_audit.csv`.

