# Sports Quant Research Engine

A Python-based quantitative research framework for analysing sports betting markets, testing betting hypotheses, and evaluating strategy robustness using historical market data.

## Project Purpose

This project is designed to turn raw sports betting market data into structured research outputs.

The focus is not simply predicting winners. The aim is to evaluate whether a betting idea has evidence, realistic risk characteristics, and repeatable long-term behaviour.

## Current Research Focus

- Horse racing Betfair Starting Price analysis
- Greyhound racing results and tips analysis
- Price-band performance
- Strike rate and ROI analysis
- Drawdown and losing-run evaluation
- Strategy robustness across time periods

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- PyArrow
- Jupyter Notebooks
- PyCharm Professional
- Git / GitHub
- CSV / Parquet

## Repository Structure

```text
sports-quant-research/
│
├── config/          Configuration files
├── data/sample/     Small sample datasets only
├── docs/            Methodology and research notes
├── reports/examples/ Example outputs and diagnostics
├── scripts/         Runnable research pipeline scripts
├── src/             Reusable Python modules
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Research Workflow

```
Raw Data
   ↓
Cleaning and Normalisation
   ↓
Feature Engineering
   ↓
Strategy Testing
   ↓
Performance Evaluation
   ↓
Reports and Diagnostics
```

---

## Example Metrics

The project is designed to evaluate strategies using metrics such as:
* Profit and loss
* Return on investment
* Strike rate
* Average odds / BSP
* Maximum drawdown
* Longest losing run
* Price-band performance
* Year-by-year robustness
* Commission-adjusted returns

---

## Example Output

Example files are included to demonstrate the intended research workflow:

- `data/sample/sample_betting_results.csv`
-  `reports/examples/price_band_analysis.csv`

These files show how raw betting results can be converted into summary analysis by 
price band, including bets, wins, strike rate, profit/loss, and ROI.

---

## Why This Project Matters

Sports betting markets can be analysed in a similar way to financial markets. A strategy should not only be 
judged by whether it made money historically, but also by whether the returns appear robust, explainable, 
and realistic after costs.

This project reflects my wider interest in quantitative research, market behaviour, risk management, 
and systematic strategy development.

---

## Project Status
Active research project.

Current development priorities:
1. Standardise the project structure
2. Add small sample datasets
3. Produce reproducible example reports
4. Improve strategy diagnostics
5. Build clearer links between research outputs and written analysis

---

## Author

Ben Cole
Aspiring quantitative researcher focused on Python, data analysis, systematic trading, 
sports market research, and risk-aware strategy development.
