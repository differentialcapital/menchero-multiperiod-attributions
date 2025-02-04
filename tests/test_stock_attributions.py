# import libraries

# testing
import pytest
# data handling
import pandas as pd
from datetime import date
# functions
from menchero_multiperiod_attribution import stockAttributions

#### FIXTURES ####

@pytest.fixture
def stock_input_data():
    """
    Returns a small, valid DataFrame to be used as input to stockAttributions.
    It must satisfy _validateInputData requirements:
      - date (date)
      - sector (str)
      - stock (str)
      - portfolio_weight (float)
      - benchmark_weight (float)
      - stock_return (float)
      - pnl_pct (float)
    We also ensure that the weights by date sum to ~1 for both portfolio and benchmark.
    """
    data = pd.DataFrame({
        "date": [
            date(2023, 1, 1), date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 2)
        ],
        "sector": ["Tech", "Finance", "Tech", "Finance"],
        "stock": ["AAPL", "JPM", "AAPL", "JPM"],
        "portfolio_weight": [0.6, 0.4, 0.6, 0.4],
        "benchmark_weight": [0.5, 0.5, 0.5, 0.5],
        "stock_return": [0.01, 0.02, -0.005, 0.015],
        "pnl_pct": [0.0, 0.0, 0.0, 0.0],
    })
    return data

@pytest.fixture
def small_stock_data():
    """
    A tiny dataset for hand-verified Menchero calculations.
    Single sector (Tech), two stocks (AAPL, MSFT), two dates.
    """
    data = {
            "date": [date(2023,1,1), date(2023,1,1),
                     date(2023,1,2), date(2023,1,2)],
            "sector": ["Tech", "Tech", "Tech", "Tech"],
            "stock":  ["AAPL","MSFT","AAPL","MSFT"],
            "portfolio_weight": [0.7, 0.3, 0.6, 0.4],
            "benchmark_weight": [0.5, 0.5, 0.5, 0.5],
            "stock_return": [0.02, 0.04, -0.01, 0.02],
            "pnl_pct": [0.001, 0.0, 0.0, 0.0]
        }
    return pd.DataFrame(data)

#### TESTS ####

def test_stock_attributions_basic(stock_input_data):
    """
    Test that stockAttributions runs without error when check=False, pnl=True,
    returning a single DataFrame.
    """
    result = stockAttributions(
        raw=stock_input_data,
        pnl=True,
        check=False,
        verbose=False
    )
    # Because check=False, we expect a single DataFrame.
    assert isinstance(result, pd.DataFrame), (
        "stockAttributions should return a single DataFrame when check=False."
    )
    assert not result.empty, "Returned DataFrame from stockAttributions should not be empty."

    # check basic columns.
    expected_cols = {"date", "sector", "stock", "variable", "variable_value"}
    actual_cols = set(result.columns)
    missing = expected_cols - actual_cols
    assert not missing, f"Missing expected columns in final output: {missing}"


def test_stock_attributions_with_check(stock_input_data):
    """
    Test that stockAttributions returns a tuple (stock_output, qa_checks)
    when check=True.
    """
    result = stockAttributions(
        raw=stock_input_data,
        pnl=True,
        check=True,
        verbose=False
    )
    assert isinstance(result, tuple) and len(result) == 2, (
        "When check=True, stockAttributions should return a 2-tuple: (stock_output, qa_checks)."
    )
    stock_output, qa_checks = result

    # both should be non-empty DataFrames
    assert isinstance(stock_output, pd.DataFrame), (
        "First element of the tuple should be a DataFrame (stock_output)."
    )
    assert isinstance(qa_checks, pd.DataFrame), (
        "Second element of the tuple should be a DataFrame (qa_checks)."
    )
    assert not stock_output.empty, "stock_output DataFrame should not be empty."
    assert not qa_checks.empty, "qa_checks DataFrame should not be empty."

    # Optional structural checks
    for df, label in [(stock_output, "stock_output"), (qa_checks, "qa_checks")]:
        assert "date" in df.columns, f"'date' column missing in {label}."


def test_stock_attributions_pnl_true_vs_false(stock_input_data):
    """
    Verify that including PnL (pnl=True) changes the final output
    vs. excluding PnL (pnl=False).
    """
    # introduce some nonzero PnL in the data
    stock_input_data.loc[0, "pnl_pct"] = 0.005  # e.g., 0.5%

    # run with pnl=True
    stock_output_pnl_true = stockAttributions(
        raw=stock_input_data,
        pnl=True,
        check=False,
        verbose=False
    )

    # reset row to 0.0 for separate run with pnl=False
    stock_input_data.loc[0, "pnl_pct"] = 0.0
    stock_output_pnl_false = stockAttributions(
        raw=stock_input_data,
        pnl=False,
        check=False,
        verbose=False
    )

    # focus on allocation/selection rows for each result
    eff_vars = ["allocation", "selection"]
    df_true = stock_output_pnl_true[stock_output_pnl_true["variable"].isin(eff_vars)]
    df_false = stock_output_pnl_false[stock_output_pnl_false["variable"].isin(eff_vars)]

    # group by date/sector/stock/variable, sum the variable_value
    sum_true = df_true.groupby(["date", "sector", "stock", "variable"])["variable_value"].sum().reset_index()
    sum_false = df_false.groupby(["date", "sector", "stock", "variable"])["variable_value"].sum().reset_index()

    # merge on same keys
    merged = pd.merge(
        sum_true, sum_false,
        on=["date", "sector", "stock", "variable"],
        how="inner",
        suffixes=("_pnl_true", "_pnl_false")
    )

    # expect at least one difference due to the PnL when pnl=True
    diffs = merged["variable_value_pnl_true"] - merged["variable_value_pnl_false"]
    assert any(diffs != 0), (
        "Expected a difference in allocation/selection values when PnL is included vs. excluded."
    )


def test_stock_attributions_pnl_false(stock_input_data):
    """
    Test the function with pnl=False. The output should ignore PnL
    when computing attributions.
    """
    # add random PnL
    stock_input_data["pnl_pct"] = [0.001, 0.002, 0.001, 0.000]

    result = stockAttributions(
        raw=stock_input_data,
        pnl=False,
        check=False,
        verbose=False
    )
    assert isinstance(result, pd.DataFrame), (
        "stockAttributions should return a single DataFrame when check=False."
    )
    assert not result.empty, "The returned DataFrame should not be empty with pnl=False."

    # ensure 'allocation' appears in the final data
    all_vars = result["variable"].unique()
    assert "allocation" in all_vars, (
        "Expected 'allocation' in the final 'variable' column."
    )


def test_stock_attributions_verbose(capfd, stock_input_data):
    """
    Test that setting verbose=True prints debug messages.
    We'll capture stdout via capfd (pytest fixture).
    """
    _ = stockAttributions(raw=stock_input_data, pnl=False, check=False, verbose=True)
    captured = capfd.readouterr()
    # check that some debug messages appear
    assert "Starting attributions calculation at a stock level..." in captured.out
    assert "Calculated total daily returns." in captured.out


def test_stock_attributions_small_dataset(small_stock_data):
    """
    Verify the Menchero calculations on a tiny dataset by comparing
    with pre-computed 'expected' values (hand or spreadsheet).
    """
    # 1) call stockAttributions with check=False, pnl=True to ensure PnL is included
    result_df = stockAttributions(
        raw=small_stock_data,
        pnl=True,
        check=False,
        verbose=False
    )
    
    # function returns a single DataFrame in melted format.
    # filter on 'allocation' and 'selection' rows to compare numeric values.
    assert isinstance(result_df, pd.DataFrame), "Expected a single DataFrame for check=False."
    assert not result_df.empty, "Resulting DataFrame should not be empty."

    # fetch final 'allocation' and 'selection' values per date/stock.
    pivoted = (
        result_df[result_df["variable"].isin(["allocation","selection"])]
        .pivot_table(index=["date","stock"], columns="variable", values="variable_value")
        .reset_index()
    )

    # compare with expected results. 
    expected_results = {
        (date(2023,1,1),"AAPL"): {"allocation": 0, "selection": -0.001},
        (date(2023,1,1),"MSFT"): {"allocation": 0, "selection": -0.002},
        (date(2023,1,2),"AAPL"): {"allocation": 0, "selection": -0.00254},
        (date(2023,1,2),"MSFT"): {"allocation": 0, "selection": -0.003556},
    }

    # loop through pivoted results, check each row vs expected
    tolerance = 1e-6  # or whatever you prefer
    for idx, row in pivoted.iterrows():
        d, stk = row["date"], row["stock"]
        alloc_actual = row["allocation"]
        sel_actual = row["selection"]
        
        exp_alloc = expected_results[(d, stk)]["allocation"]
        exp_sel = expected_results[(d, stk)]["selection"]

        # assert they're within tolerance
        assert abs(alloc_actual - exp_alloc) < tolerance, (
            f"Allocation mismatch on {d} - {stk}. "
            f"Expected {exp_alloc}, got {alloc_actual}."
        )
        assert abs(sel_actual - exp_sel) < tolerance, (
            f"Selection mismatch on {d} - {stk}. "
            f"Expected {exp_sel}, got {sel_actual}."
        )
