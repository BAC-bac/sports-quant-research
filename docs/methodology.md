# Methodology

## Research Philosophy

This project is built around the idea that betting strategies should be tested like quantitative trading strategies.

A profitable backtest is not enough on its own. A strategy also needs to be examined for robustness, drawdown behaviour, sample size, market logic, and sensitivity to costs.

## Key Questions

The project aims to answer questions such as:

- Does a strategy perform consistently across time?
- Is the profit driven by a small number of outliers?
- How severe are the losing runs?
- Does the edge survive commission?
- Does performance vary by odds range, market type, or year?
- Is the strategy realistic to execute?

## Core Analysis Areas

### 1. Data Cleaning

Raw sports market data often needs standardising before analysis. This includes cleaning dates, market names, runner names, prices, results, and missing values.

### 2. Feature Engineering

Features may include price bands, implied probabilities, market type, runner counts, volume measures, and historical performance categories.

### 3. Strategy Testing

Strategies are tested by applying clear rules to historical data and measuring the resulting profit and loss.

### 4. Risk Evaluation

Risk is evaluated using drawdown, losing streaks, volatility of returns, and year-by-year performance.

### 5. Robustness Review

A strategy is more interesting when it performs across multiple periods, markets, or conditions rather than relying on one narrow historical window.

## Current Limitations

This project is research-focused and should not be treated as financial or betting advice. Historical performance does not guarantee future results.