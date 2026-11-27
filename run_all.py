# -*- coding: utf-8 -*-
"""Run data cleaning and empirical analysis in sequence."""

from code.clean_data import main as clean_data
from code.run_analysis import main as run_analysis


if __name__ == "__main__":
    clean_data()
    run_analysis()
    print("Workflow complete. See data/processed, data/final, and results/tables.")

