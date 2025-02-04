# Menchero Method Performance Attribution

**Menchero Method Performance Attribution** is a Python package for performing single- and multi-period performance attribution using the [Menchero multiperiod smoothing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=574762) method, with support for both **sector-level** and **stock-level** breakdowns. Stock level attributions are calculated using our own internally developed adjustments descibed [here](https://www.differential.co.za/stock-level-extensions-to-mencheros-method-for-portfolio-attributions/)

This library provides:
- **Menchero smoothing** for accurate multi-period attribution of returns.
- **Quality assurance checks** to compare calculated attributions vs. actual active returns.
- **PnL (Profit & Loss) handling** at both sector and stock levels.
- Flexible, **pandas DataFrame**-based interface for integration with Python analytics workflows.

---

## Table of Contents

- [Features](#features)  
- [Installation](#installation)  
- [Quickstart Example](#quickstart-example)  
- [Usage](#usage)  
  - [Sector-Level Attributions](#sector-level-attributions)  
  - [Stock-Level Attributions](#stock-level-attributions)  
- [How It Works](#how-it-works)  
- [Validation and Testing](#validation-and-testing)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Features

- **Menchero multiperiod** smoothing approach to handle multi-day or multi-month performance attribution.  
- **Sector-level** vs. **stock-level** breakdowns:  
  - *Sector-level*: See which sectors drove active performance over time.  
  - *Stock-level*: Drill down to individual stock attributions, with optional re-allocation logic.  
- **PnL inclusion**: Decide whether to include profit-and-loss percentages in your allocation effect.  
- **Flexible QA checks**: Compare the summed selection/allocation to the final cumulative active return.  
- **Verbose** logging mode for debugging and demonstration.  

---

## Installation

You can install this package via pip:

```bash
pip install menchero-method-attributions
```

---

## Quickstart Example

```python
import pandas as pd
from menchero_method import sectorAttributions, stockAttributions

# Example: create or load your data as a pandas DataFrame
data = pd.DataFrame({
    'date': [...],         # Python 'date' objects or Timestamps
    'sector': [...],       # Sector names (str)
    'stock': [...],        # Stock names (str), optional for sector-level
    'portfolio_weight': [...],
    'benchmark_weight': [...],
    'stock_return': [...],
    'pnl_pct': [...]
})

# Sector-level attributions
sector_output = sectorAttributions(
    raw=data, 
    pnl=True,      # include PnL in calculations
    check=False,   # skip QA checks for brevity
    verbose=False  # no debug printing
)

# Stock-level attributions
stock_output = stockAttributions(
    raw=data, 
    pnl=True,
    check=True,    # perform QA checks 
    verbose=True   # print debug info
)

print(sector_output.head())
print(stock_output[0].head())  # if check=True, returns (stock_output, qa_checks)
```

---

## Usage

### Sector-Level Attributions

```python
from menchero_method import sectorAttributions

# Example DataFrame 'df' with columns: 
#   date, sector, portfolio_weight, benchmark_weight, stock_return, pnl_pct
sector_result = sectorAttributions(
    raw=df,
    pnl=True,       # include or exclude PnL in your calculation
    check=False,    # set True to return QA checks
    verbose=True    # optional logging
)
```

- If ```check=True```, returns a tuple ```(sector_result, qa_checks)```.
- The returned DataFrame (or first element of the tuple) is in a “long” format, with columns like ```[date, sector, variable, variable_value]```.

### Stock-Level Attributions

```python
from menchero_method import stockAttributions

# Example DataFrame 'df' with columns: 
#   date, sector, stock, portfolio_weight, benchmark_weight, stock_return, pnl_pct
stock_result = stockAttributions(
    raw=df,
    pnl=False,     # do not include PnL
    check=True,
    verbose=False
)
```

- If ```check=True```, returns ```(stock_output, qa_checks)```. Otherwise, returns only ```stock_output```.
- The DataFrame(s) have columns like ```[date, sector, stock, variable, variable_value]```.
- Internally, ```stockAttributions``` calculates sector-level attributions first, then modifies them at the stock level via ```_attributionModify```.

---

## How It Works

1) Input Validation – The library checks that your input DataFrame has the required columns, correct data types, and that weights sum to approximately 1.
2) Menchero Method – The core _menchero function calculates:
    - Allocation, Selection, Interaction effects.
    - Cumulative returns and active return.
    - Menchero smoothing via multiplicative and corrective factors to properly handle compounding.
3) QA Checks (optional) – Sums allocation/selection across dates and compares to the final cumulative active return to ensure correctness.
4) Stock-Level Adjustments – For stock-level attributions, we apply _attributionModify to adjust the final allocation/selection by sector-level results.

---

## Validation and Testing

We maintain a comprehensive test suite that includes:

- **Unit tests** for internal functions (```_menchero```, ```_attributionCheck```, ```_attributionModify```, ```_validateInputData```).
- **Integration tests** for ```sectorAttributions``` and ```stockAttributions.
- **Golden tests** with small, hand-verified datasets to confirm exact numeric results match expected Menchero smoothing outcomes.
To run the tests locally, clone this repository and run:

```python
pytest
```

*(You may need to install development dependencies like ```pytest```.)*

---

## Contributing

Contributions, bug reports, and feature requests are welcome! Please:

1) [Open an issue](ADD LINK HERE) describing the problem or suggestion.
2) Fork the repo and create a new branch for your contribution.
3) Submit a pull request with your changes and relevant tests.
We’ll review your pull request as soon as we can.

---

## License

This project is released under the [MIT License](https://opensource.org/license/MIT). Feel free to use it in commercial and private projects. See the [LICENSE](https://github.com/differentialcapital/dev-menchero-smoothing/blob/main/LICENSE) file for details.

---

**Thank you for using the Menchero Method Performance Attribution Package!** If you have any questions or feedback, please open an issue or submit a pull request. Contributions are always welcome.
