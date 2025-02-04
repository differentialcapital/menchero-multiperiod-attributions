# import libraries

# testing
import pytest
# data handling
import pandas as pd
from datetime import date
#functions
from menchero_multiperiod_attribution import _menchero

#### FIXTURES ####

@pytest.fixture
def menchero_input_data():
    """
    Returns a small, valid DataFrame that can be used as input to _menchero.
    Because _validateInputData is called before this function in real usage,
    we assume this data passes all validation checks (i.e., columns present,
    no NaNs, weights sum to ~1, etc.).
    """
    # make sure the weights sum to approximately 1.0 per date
    return pd.DataFrame({
        "date": [
            date(2023, 1, 1), date(2023, 1, 1), 
            date(2023, 1, 2), date(2023, 1, 2)
        ],
        "sector": ["Tech", "Finance", "Tech", "Finance"],
        # some simple numeric columns
        "portfolio_weight": [0.6, 0.4, 0.6, 0.4],
        "benchmark_weight": [0.5, 0.5, 0.5, 0.5],
        # arbitrary returns
        "portfolio_return": [0.01, 0.02, -0.005, 0.0],
        "benchmark_return": [0.008, 0.01, -0.004, -0.001],
        # weighted returns 
        "weighted_portfolio_return": [0.006, 0.008, -0.003, 0.0],
        "weighted_benchmark_return": [0.004, 0.005, -0.002, -0.0004],
        # total daily returns
        "total_portfolio_return": [0.014, 0.014, -0.003, -0.003],
        "total_benchmark_return": [0.0096, 0.0096, -0.0024, -0.0024],
        # pnl percentage for these rows
        "pnl_pct": [0.0, 0.0, 0.0, 0.0],
    })

#### TESTS ####

def test_menchero_basic(menchero_input_data):
    """
    Test that _menchero runs without error and returns a single DataFrame
    when check=False and stock_level=False.
    """
    group_vars = ["sector"]
    result = _menchero(
        input_data=menchero_input_data, 
        group_vars=group_vars,
        check=False, 
        stock_level=False, 
        verbose=False
    )
    # Because check=False and stock_level=False, we expect a single DataFrame back
    assert isinstance(result, pd.DataFrame), (
        "Expected a single DataFrame when check=False, stock_level=False."
    )
    # Basic shape or column checks
    assert not result.empty, "The returned DataFrame should not be empty."
    expected_cols = {"date", "sector", "variable", "variable_value"}
    actual_cols = set(result.columns)
    missing = expected_cols - actual_cols
    assert not missing, f"Expected columns missing from result: {missing}"


def test_menchero_with_check(menchero_input_data):
    """
    Test that _menchero returns a tuple of two items (data, qa_checks)
    when check=True and stock_level=False.
    """
    group_vars = ["sector"]
    result = _menchero(
        input_data=menchero_input_data,
        group_vars=group_vars,
        check=True,
        stock_level=False,
        verbose=False
    )
    # Expect (data, qa_checks)
    assert isinstance(result, tuple) and len(result) == 2, (
        "Expected a 2-tuple (data, qa_checks) when check=True, stock_level=False."
    )
    data, qa_checks = result
    assert isinstance(data, pd.DataFrame), "First item of the tuple must be a DataFrame."
    assert isinstance(qa_checks, pd.DataFrame), "Second item of the tuple must be a DataFrame."
    # Check the data frames are not empty
    assert not data.empty, "Returned 'data' should not be empty."
    assert not qa_checks.empty, "Returned 'qa_checks' should not be empty."
    # Optional column checks
    for df, name in [(data, "data"), (qa_checks, "qa_checks")]:
        assert "date" in df.columns, f"'date' column missing from {name}."


def test_menchero_stock_level(menchero_input_data):
    """
    Test behavior when stock_level=True. We expect PnL to be included in 'allocation',
    and the return value to still be a single DataFrame if check=False.
    """
    group_vars = ["sector"]
    # Add some nonzero PnL to verify effect
    menchero_input_data["pnl_pct"] = [0.001, 0.002, 0.0, 0.0]

    result = _menchero(
        input_data=menchero_input_data,
        group_vars=group_vars,
        check=False,
        stock_level=True,
        verbose=False
    )
    # Because check=False and stock_level=True, we expect a single DataFrame
    assert isinstance(result, pd.DataFrame), (
        "Expected a single DataFrame when check=False, stock_level=True."
    )
    assert not result.empty, "The returned DataFrame should not be empty for stock-level calculations."

    # The final DataFrame is in long format: we have a 'variable' column containing 'allocation'
    all_vars = result["variable"].unique()
    assert "allocation" in all_vars, (
        "Expected 'allocation' in the final 'variable' column for stock-level computations."
    )

    # Optional numeric check:
    # For rows that had nonzero PnL, we might expect the resulting 'allocation' effect to differ
    allocation_rows = result[result["variable"] == "allocation"]
    assert not allocation_rows.empty, (
        "There should be some rows labeled 'allocation' if PnL was nonzero."
    )
    # You can check the sum is positive or some other logic
    sum_allocation = allocation_rows["variable_value"].sum()
    assert sum_allocation > 0, (
        "Expected a positive sum for the 'allocation' effect with nonzero PnL."
    )


def test_menchero_check_with_stock_level(menchero_input_data):
    """
    Test that _menchero returns (data, active_return) when check=True and stock_level=True.
    """
    group_vars = ["sector"]
    # This time, let's keep some PnL as well
    menchero_input_data["pnl_pct"] = [0.001, 0.002, 0.001, 0.0]

    result = _menchero(
        input_data=menchero_input_data,
        group_vars=group_vars,
        check=True,
        stock_level=True,
        verbose=False
    )
    # Expect a 2-tuple (data, active_return)
    assert isinstance(result, tuple) and len(result) == 2, (
        "Expected a 2-tuple (data, active_return) when check=True, stock_level=True."
    )
    data, active_return = result
    # Basic checks
    assert isinstance(data, pd.DataFrame), "First item must be a DataFrame."
    assert isinstance(active_return, pd.DataFrame), "Second item must be a DataFrame."
    assert not data.empty, "'data' DataFrame should not be empty."
    assert not active_return.empty, "'active_return' DataFrame should not be empty."
    # We typically expect 'date' and 'value' columns in active_return
    assert {"date", "value"}.issubset(active_return.columns), (
        "Expected 'active_return' to have 'date' and 'value' columns."
    )