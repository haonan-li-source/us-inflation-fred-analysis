# -*- coding: utf-8 -*-
"""Clean FRED monthly data and construct the paper's analysis sample."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "fred_macro_monthly.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FINAL_DIR = ROOT / "data" / "final"

SERIES_META = {
    "CPIAUCSL": ("居民消费价格指数", "U.S. Bureau of Labor Statistics"),
    "CPILFESL": ("核心居民消费价格指数", "U.S. Bureau of Labor Statistics"),
    "FEDFUNDS": ("联邦基金有效利率", "Board of Governors of the Federal Reserve System"),
    "UNRATE": ("失业率", "U.S. Bureau of Labor Statistics"),
    "M2SL": ("M2货币存量", "Board of Governors of the Federal Reserve System"),
    "MCOILWTICO": ("WTI原油价格", "U.S. Energy Information Administration"),
    "INDPRO": ("工业生产指数", "Board of Governors of the Federal Reserve System"),
}


def yoy_log(series: pd.Series) -> pd.Series:
    """Return 100 times the 12-month log difference."""
    return 100 * (np.log(series) - np.log(series.shift(12)))


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_FILE)
    raw["date"] = pd.to_datetime(raw["observation_date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    work = raw.loc[raw["date"] >= "2000-01-01"].copy()
    work = work.rename(
        columns={
            "CPIAUCSL": "cpi",
            "CPILFESL": "core_cpi",
            "FEDFUNDS": "fedfunds",
            "UNRATE": "unrate",
            "M2SL": "m2",
            "MCOILWTICO": "oil_wti",
            "INDPRO": "indpro",
        }
    )

    work["inflation"] = yoy_log(work["cpi"])
    work["core_inflation"] = yoy_log(work["core_cpi"])
    work["m2_growth"] = yoy_log(work["m2"])
    work["oil_growth"] = yoy_log(work["oil_wti"])
    work["indpro_growth"] = yoy_log(work["indpro"])

    lag_vars = [
        "inflation",
        "core_inflation",
        "fedfunds",
        "unrate",
        "m2_growth",
        "oil_growth",
        "indpro_growth",
    ]
    for variable in lag_vars:
        for lag in (1, 2, 3):
            work[f"{variable}_l{lag}"] = work[variable].shift(lag)

    work["post_covid"] = (work["date"] >= "2020-03-01").astype(int)
    work["fedfunds_l1_post"] = work["fedfunds_l1"] * work["post_covid"]
    work["oil_growth_l1_post"] = work["oil_growth_l1"] * work["post_covid"]

    processed_columns = [
        "date",
        "cpi",
        "core_cpi",
        "fedfunds",
        "unrate",
        "m2",
        "oil_wti",
        "indpro",
        "inflation",
        "core_inflation",
        "m2_growth",
        "oil_growth",
        "indpro_growth",
        "inflation_l1",
        "inflation_l2",
        "inflation_l3",
        "core_inflation_l1",
        "core_inflation_l2",
        "core_inflation_l3",
        "fedfunds_l1",
        "unrate_l1",
        "m2_growth_l1",
        "oil_growth_l1",
        "indpro_growth_l1",
        "post_covid",
        "fedfunds_l1_post",
        "oil_growth_l1_post",
    ]
    processed = work[processed_columns].copy()
    processed.to_csv(
        PROCESSED_DIR / "fred_monthly_processed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    required = [
        "date",
        "inflation",
        "core_inflation",
        "inflation_l1",
        "inflation_l2",
        "inflation_l3",
        "core_inflation_l1",
        "core_inflation_l2",
        "core_inflation_l3",
        "fedfunds_l1",
        "unrate_l1",
        "m2_growth_l1",
        "oil_growth_l1",
        "indpro_growth_l1",
        "post_covid",
        "fedfunds_l1_post",
        "oil_growth_l1_post",
    ]
    sample = processed[required].dropna().reset_index(drop=True)
    sample.to_csv(FINAL_DIR / "analysis_sample.csv", index=False, encoding="utf-8-sig")

    audit_rows = []
    for code, (label, source) in SERIES_META.items():
        valid = raw[["date", code]].dropna()
        audit_rows.append(
            {
                "FRED代码": code,
                "变量": label,
                "起始日期": valid["date"].min().date().isoformat(),
                "结束日期": valid["date"].max().date().isoformat(),
                "观测数": len(valid),
                "原始机构": source,
                "网址": f"https://fred.stlouisfed.org/series/{code}",
                "访问日期": "2026-06-16",
            }
        )
    pd.DataFrame(audit_rows).to_csv(
        PROCESSED_DIR / "data_source_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Data preparation complete:",
        sample["date"].min().date(),
        "to",
        sample["date"].max().date(),
        "with",
        len(sample),
        "valid observations.",
    )


if __name__ == "__main__":
    main()

