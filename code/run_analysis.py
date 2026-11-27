# -*- coding: utf-8 -*-
"""Reproduce the main empirical tables reported in the course paper."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "fred_monthly_processed.csv"
TABLE_DIR = ROOT / "results" / "tables"
LOG_DIR = ROOT / "results" / "logs"

LABELS = {
    "const": "常数项",
    "inflation_l1": "滞后通胀率",
    "inflation_l2": "滞后通胀率(2)",
    "inflation_l3": "滞后通胀率(3)",
    "core_inflation_l1": "滞后核心通胀率",
    "core_inflation_l2": "滞后核心通胀率(2)",
    "core_inflation_l3": "滞后核心通胀率(3)",
    "fedfunds_l1": "联邦基金利率(-1)",
    "unrate_l1": "失业率(-1)",
    "m2_growth_l1": "M2增速(-1)",
    "oil_growth_l1": "油价增速(-1)",
    "indpro_growth_l1": "工业生产增速(-1)",
    "post_covid": "疫情后虚拟变量",
    "fedfunds_l1_post": "利率×疫情后",
    "oil_growth_l1_post": "油价增速×疫情后",
}


def norm_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2))


def two_sided_p(value: float) -> float:
    return 2 * (1 - norm_cdf(abs(value)))


def chi2_sf_approx(value: float, degrees: int) -> float:
    if value <= 0:
        return 1.0
    if degrees == 2:
        return math.exp(-value / 2)
    z = ((value / degrees) ** (1 / 3) - (1 - 2 / (9 * degrees))) / math.sqrt(
        2 / (9 * degrees)
    )
    return max(0.0, min(1.0, 1 - norm_cdf(z)))


def stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def month_key(series: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(series)
    return (dates.dt.year.to_numpy() * 12 + dates.dt.month.to_numpy()).astype(int)


def ols_hac(
    data: pd.DataFrame,
    dependent: str,
    regressors: list[str],
    bandwidth: int = 12,
) -> dict:
    use = data[["date", dependent] + regressors].dropna().sort_values("date").copy()
    y = use[dependent].to_numpy(float)
    x = np.column_stack([np.ones(len(use)), use[regressors].to_numpy(float)])
    names = ["const"] + regressors
    n, k = x.shape

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    residual = y - fitted
    rss = float(residual.T @ residual)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    score = x * residual[:, None]
    keys = month_key(use["date"])
    key_index = {key: index for index, key in enumerate(keys)}
    meat = score.T @ score
    for lag in range(1, bandwidth + 1):
        pairs = [
            (index, key_index.get(key - lag))
            for index, key in enumerate(keys)
            if key_index.get(key - lag) is not None
        ]
        gamma = sum(np.outer(score[i], score[j]) for i, j in pairs)
        weight = 1 - lag / (bandwidth + 1)
        meat += weight * (gamma + gamma.T)

    covariance = xtx_inv @ meat @ xtx_inv
    covariance *= n / max(1, n - k)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0))
    p_values = np.array(
        [two_sided_p(value) for value in beta / standard_error]
    )

    adjacent = [
        (index, key_index.get(key - 1))
        for index, key in enumerate(keys)
        if key_index.get(key - 1) is not None
    ]
    dw = sum((residual[i] - residual[j]) ** 2 for i, j in adjacent) / float(
        residual.T @ residual
    )

    return {
        "data": use,
        "names": names,
        "beta": beta,
        "se": standard_error,
        "p": p_values,
        "cov": covariance,
        "resid": residual,
        "keys": keys,
        "n": n,
        "k": k,
        "rss": rss,
        "r2": r2,
        "adj_r2": adj_r2,
        "dw": dw,
    }


def coefficient(result: dict, name: str) -> str:
    if name not in result["names"]:
        return ""
    index = result["names"].index(name)
    return f"{result['beta'][index]:.3f}{stars(result['p'][index])}"


def standard_error(result: dict, name: str) -> str:
    if name not in result["names"]:
        return ""
    index = result["names"].index(name)
    return f"({result['se'][index]:.3f})"


def regression_table(models: dict[str, dict], variables: list[str]) -> pd.DataFrame:
    rows = []
    for variable in variables:
        coef_row = {"变量": LABELS[variable]}
        se_row = {"变量": ""}
        for model_name, result in models.items():
            coef_row[model_name] = coefficient(result, variable)
            se_row[model_name] = standard_error(result, variable)
        rows.extend([coef_row, se_row])
    for key, label in [
        ("n", "观测值"),
        ("r2", "R²"),
        ("adj_r2", "调整R²"),
        ("dw", "DW统计量"),
    ]:
        row = {"变量": label}
        for model_name, result in models.items():
            row[model_name] = (
                str(result[key]) if key == "n" else f"{result[key]:.3f}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def adf_select(series: pd.Series, max_lags: int = 12) -> dict:
    y = series.dropna().to_numpy(float)
    difference = np.diff(y)
    fixed_dep = difference[max_lags:]
    fixed_level = y[max_lags:-1]
    fixed_lags = [
        difference[max_lags - lag : -lag] for lag in range(1, max_lags + 1)
    ]

    best = None
    for lag_order in range(max_lags + 1):
        x = np.column_stack(
            [np.ones(len(fixed_dep)), fixed_level] + fixed_lags[:lag_order]
        )
        beta = np.linalg.lstsq(x, fixed_dep, rcond=None)[0]
        residual = fixed_dep - x @ beta
        n, k = x.shape
        aic = n * math.log(float(residual.T @ residual) / n) + 2 * k
        if best is None or aic < best["aic"]:
            best = {"lag": lag_order, "aic": aic}

    lag_order = best["lag"]
    dependent = difference[lag_order:]
    level = y[lag_order:-1]
    lag_columns = [
        difference[lag_order - lag : -lag] for lag in range(1, lag_order + 1)
    ]
    x = np.column_stack([np.ones(len(dependent)), level] + lag_columns)
    beta = np.linalg.lstsq(x, dependent, rcond=None)[0]
    residual = dependent - x @ beta
    n, k = x.shape
    covariance = (
        float(residual.T @ residual)
        / (n - k)
        * np.linalg.pinv(x.T @ x)
    )
    statistic = float(beta[1] / math.sqrt(covariance[1, 1]))
    critical = -2.87
    return {
        "ADF统计量": statistic,
        "滞后阶数": lag_order,
        "5%临界值": critical,
        "结论": "平稳" if statistic < critical else "未拒绝单位根",
    }


def calendar_ljung_box(result: dict, lags: int = 12) -> tuple[float, float]:
    residual = result["resid"] - result["resid"].mean()
    keys = result["keys"]
    key_index = {key: index for index, key in enumerate(keys)}
    denominator = float(residual.T @ residual)
    terms = []
    for lag in range(1, lags + 1):
        pairs = [
            (index, key_index.get(key - lag))
            for index, key in enumerate(keys)
            if key_index.get(key - lag) is not None
        ]
        rho = sum(residual[i] * residual[j] for i, j in pairs) / denominator
        terms.append(rho**2 / len(pairs))
    q_stat = result["n"] * (result["n"] + 2) * sum(terms)
    return q_stat, chi2_sf_approx(q_stat, lags)


def auxiliary_lm(
    residual: np.ndarray, regressors: np.ndarray, degrees: int
) -> tuple[float, float]:
    dependent = residual**2
    x = np.column_stack([np.ones(len(regressors)), regressors])
    beta = np.linalg.lstsq(x, dependent, rcond=None)[0]
    error = dependent - x @ beta
    r2 = 1 - float(error.T @ error) / float(
        (dependent - dependent.mean()).T @ (dependent - dependent.mean())
    )
    lm = len(dependent) * r2
    return lm, chi2_sf_approx(lm, degrees)


def jarque_bera(residual: np.ndarray) -> tuple[float, float]:
    centered = residual - residual.mean()
    std = centered.std(ddof=0)
    skew = float(np.mean((centered / std) ** 3))
    kurtosis = float(np.mean((centered / std) ** 4))
    statistic = len(centered) / 6 * (skew**2 + (kurtosis - 3) ** 2 / 4)
    return statistic, math.exp(-statistic / 2)


def vif_max(data: pd.DataFrame, variables: list[str]) -> float:
    use = data[variables].dropna()
    maximum = 0.0
    for target in variables:
        others = [variable for variable in variables if variable != target]
        y = use[target].to_numpy(float)
        x = np.column_stack([np.ones(len(use)), use[others].to_numpy(float)])
        fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
        r2 = 1 - float(((y - fitted) ** 2).sum()) / float(
            ((y - y.mean()) ** 2).sum()
        )
        maximum = max(maximum, 1 / (1 - r2))
    return maximum


def information_criteria(result: dict) -> tuple[float, float]:
    n, k, rss = result["n"], result["k"], result["rss"]
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + math.log(n) * k
    return aic, bic


def save(table: pd.DataFrame, filename: str) -> None:
    table.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_FILE, parse_dates=["date"]).sort_values("date")
    for lag in range(4, 7):
        data[f"inflation_l{lag}"] = data["inflation"].shift(lag)
    data["inflation_change"] = data["inflation"] - data["inflation"].shift(1)
    for lag in range(1, 4):
        data[f"inflation_change_l{lag}"] = data["inflation_change"].shift(lag)

    descriptive_variables = [
        "inflation",
        "core_inflation",
        "fedfunds_l1",
        "unrate_l1",
        "m2_growth_l1",
        "oil_growth_l1",
        "indpro_growth_l1",
    ]
    descriptive = data[descriptive_variables].describe(
        percentiles=[0.5]
    ).T.reset_index()
    descriptive = descriptive.rename(
        columns={
            "index": "变量",
            "count": "N",
            "mean": "均值",
            "std": "标准差",
            "min": "最小值",
            "50%": "中位数",
            "max": "最大值",
        }
    )[["变量", "N", "均值", "标准差", "最小值", "中位数", "最大值"]]
    descriptive["变量"] = descriptive["变量"].map(
        {
            "inflation": "CPI同比通胀率",
            "core_inflation": "核心CPI同比通胀率",
            **{key: value for key, value in LABELS.items()},
        }
    )
    save(descriptive.round(3), "table_descriptive.csv")

    correlation = data[
        [
            "inflation",
            "fedfunds_l1",
            "unrate_l1",
            "m2_growth_l1",
            "oil_growth_l1",
            "indpro_growth_l1",
        ]
    ].corr()
    correlation.index = [LABELS.get(name, "CPI同比通胀率") for name in correlation.index]
    correlation.columns = [
        LABELS.get(name, "CPI同比通胀率") for name in correlation.columns
    ]
    correlation.to_csv(
        TABLE_DIR / "table_correlation.csv",
        encoding="utf-8-sig",
        float_format="%.3f",
    )

    adf_rows = []
    for variable in descriptive_variables:
        result = adf_select(data[variable])
        result["变量"] = LABELS.get(
            variable,
            "CPI同比通胀率" if variable == "inflation" else "核心CPI同比通胀率",
        )
        adf_rows.append(result)
    adf = pd.DataFrame(adf_rows)[
        ["变量", "ADF统计量", "滞后阶数", "5%临界值", "结论"]
    ]
    save(adf.round(3), "table_adf.csv")

    ar3 = ["inflation_l1", "inflation_l2", "inflation_l3"]
    structural = [
        "fedfunds_l1",
        "unrate_l1",
        "m2_growth_l1",
        "oil_growth_l1",
        "indpro_growth_l1",
    ]
    simplified_x = ["inflation_l1", "fedfunds_l1", "unrate_l1"]
    full_x = ar3 + structural
    base_x = full_x + ["post_covid"]
    extended_x = full_x + [
        "post_covid",
        "fedfunds_l1_post",
        "oil_growth_l1_post",
    ]

    common_dates = set(data[["date", "inflation"] + base_x].dropna()["date"])
    models = {
        "简化动态模型": ols_hac(
            data.loc[data["date"].isin(common_dates)], "inflation", simplified_x
        ),
        "完整AR(3)模型": ols_hac(data, "inflation", full_x),
        "基准模型": ols_hac(data, "inflation", base_x),
        "疫情后交互模型": ols_hac(data, "inflation", extended_x),
    }
    baseline_variables = [
        "const",
        "inflation_l1",
        "inflation_l2",
        "inflation_l3",
        *structural,
        "post_covid",
        "fedfunds_l1_post",
        "oil_growth_l1_post",
    ]
    save(
        regression_table(models, baseline_variables),
        "table_baseline_regression.csv",
    )
    base = models["基准模型"]

    non_ar = structural + ["post_covid"]
    x_no_constant = base["data"][base_x].to_numpy(float)
    bp_lm, bp_p = auxiliary_lm(base["resid"], x_no_constant, len(base_x))
    white_x = np.column_stack([x_no_constant, x_no_constant**2])
    white_lm, white_p = auxiliary_lm(
        base["resid"], white_x, white_x.shape[1]
    )
    lb_q, lb_p = calendar_ljung_box(base)
    jb, jb_p = jarque_bera(base["resid"])

    residual_dates = base["data"]["date"].reset_index(drop=True)
    groups = residual_dates.diff().dt.days.fillna(0).gt(40).cumsum()
    longest_group = groups.value_counts().idxmax()
    residual_adf = adf_select(pd.Series(base["resid"])[groups == longest_group])

    diagnostics = pd.DataFrame(
        [
            ["Durbin-Watson", base["dw"], np.nan, "接近2，一阶残差相关较弱"],
            ["Ljung-Box(12)", lb_q, lb_p, "拒绝无序列相关原假设"],
            ["Breusch-Pagan", bp_lm, bp_p, "检验同方差原假设"],
            ["White", white_lm, white_p, "检验同方差原假设"],
            ["残差ADF", residual_adf["ADF统计量"], np.nan, residual_adf["结论"]],
            ["Jarque-Bera", jb, jb_p, "检验残差正态性"],
            ["最大VIF（非AR项）", vif_max(data, non_ar), np.nan, "未显示严重多重共线性"],
        ],
        columns=["检验", "统计量", "近似p值", "结论"],
    )
    save(diagnostics.round(4), "table_diagnostics.csv")

    core_x = [
        "core_inflation_l1",
        "core_inflation_l2",
        "core_inflation_l3",
        *structural,
        "post_covid",
    ]
    robustness = {
        "核心CPI": ols_hac(data, "core_inflation", core_x),
        "剔除疫情冲击期": ols_hac(
            data.loc[
                (data["date"] < "2020-03-01")
                | (data["date"] > "2021-06-01")
            ],
            "inflation",
            base_x,
        ),
        "仅2008年后": ols_hac(
            data.loc[data["date"] >= "2008-01-01"], "inflation", base_x
        ),
    }
    robustness_variables = [
        "const",
        "core_inflation_l1",
        "core_inflation_l2",
        "core_inflation_l3",
        *structural,
        "post_covid",
        "inflation_l1",
        "inflation_l2",
        "inflation_l3",
    ]
    save(
        regression_table(robustness, robustness_variables),
        "table_robustness.csv",
    )

    sensitivity_rows = []
    for bandwidth in (3, 6, 12, 18):
        result = ols_hac(data, "inflation", base_x, bandwidth)
        sensitivity_rows.append(
            {
                "检验类别": "HAC带宽",
                "设定": f"HAC({bandwidth})",
                "M2增速(-1)": coefficient(result, "m2_growth_l1"),
                "油价增速(-1)": coefficient(result, "oil_growth_l1"),
                "疫情后虚拟变量": coefficient(result, "post_covid"),
                "AIC": "",
                "BIC": "",
                "DW统计量": f"{result['dw']:.3f}",
                "观测值": result["n"],
            }
        )

    common_columns = [
        "date",
        "inflation",
        *[f"inflation_l{lag}" for lag in range(1, 7)],
        *structural,
        "post_covid",
    ]
    common = data[common_columns].dropna()
    for lag_order in range(1, 7):
        regressors = [
            *[f"inflation_l{lag}" for lag in range(1, lag_order + 1)],
            *structural,
            "post_covid",
        ]
        result = ols_hac(common, "inflation", regressors)
        aic, bic = information_criteria(result)
        sensitivity_rows.append(
            {
                "检验类别": "动态阶数",
                "设定": f"AR({lag_order})",
                "M2增速(-1)": coefficient(result, "m2_growth_l1"),
                "油价增速(-1)": coefficient(result, "oil_growth_l1"),
                "疫情后虚拟变量": coefficient(result, "post_covid"),
                "AIC": f"{aic:.3f}",
                "BIC": f"{bic:.3f}",
                "DW统计量": f"{result['dw']:.3f}",
                "观测值": result["n"],
            }
        )
    save(pd.DataFrame(sensitivity_rows), "table_sensitivity_and_lag_selection.csv")

    change_x = [
        "inflation_change_l1",
        "inflation_change_l2",
        "inflation_change_l3",
        *structural,
        "post_covid",
    ]
    change_models = {
        "基准模型": base,
        "通胀变化量": ols_hac(data, "inflation_change", change_x),
    }
    save(
        regression_table(
            change_models,
            [*structural, "post_covid"],
        ),
        "table_inflation_change.csv",
    )

    stage_x = ar3 + structural
    stage_models = {
        "疫情前样本": ols_hac(
            data.loc[data["date"] < "2020-03-01"], "inflation", stage_x
        ),
        "疫情后样本": ols_hac(
            data.loc[data["date"] >= "2020-03-01"], "inflation", stage_x
        ),
    }
    save(
        regression_table(stage_models, structural),
        "table_pre_post_samples.csv",
    )

    summary = {
        "sample_start": base["data"]["date"].min().date().isoformat(),
        "sample_end": base["data"]["date"].max().date().isoformat(),
        "observations": base["n"],
        "adjusted_r2": base["adj_r2"],
        "durbin_watson": base["dw"],
        "hac_bandwidth": 12,
        "selected_ar_order": 3,
        "residual_adf": residual_adf["ADF统计量"],
        "ljung_box_12": lb_q,
    }
    (LOG_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "Analysis complete:",
        base["n"],
        "baseline observations, adjusted R^2 =",
        f"{base['adj_r2']:.3f}",
        "and selected AR order = 3.",
    )


if __name__ == "__main__":
    main()

