# import libraries

# testing
import pytest
# data handling
import pandas as pd
from datetime import date

# functions
from menchero_multiperiod_attribution import sectorAttributions

#### FIXTURES ####

@pytest.fixture
def sector_input_data():
    """
    Returns a small, valid DataFrame to be used as input to sectorAttributions.
    It must satisfy _validateInputData requirements:
      - date (date)
      - sector (str)
      - portfolio_weight (float)
      - benchmark_weight (float)
      - stock_return (float)
      - pnl_pct (float)
    We also ensure weights per date sum ~1 for both portfolio and benchmark.
    """
    # create two dates, two sectors, minimal but valid data
    data = pd.DataFrame({
        "date": [
            date(2023, 1, 1), date(2023, 1, 1),
            date(2023, 1, 2), date(2023, 1, 2)
        ],
        "sector": ["Tech", "Finance", "Tech", "Finance"],
        "portfolio_weight": [0.6, 0.4, 0.5, 0.5],
        "benchmark_weight": [0.5, 0.5, 0.5, 0.5],
        "stock_return": [0.01, 0.02, -0.005, 0.01],
        "pnl_pct": [0.0, 0.0, 0.0, 0.0]  # start with zeros, can modify in specific tests
    })
    return data

@pytest.fixture
def small_sector_data():
    """
    A tiny dataset for verifying sectorAttributions.
    Two dates, two sectors (Tech, Finance), minimal 
    weights summing to 1, small returns, optional PnL.
    """
    data = {
        "date": [date(2023,1,1), date(2023,1,1),
                 date(2023,1,2), date(2023,1,2)],
        "sector": ["Tech", "Finance", "Tech", "Finance"],
        "portfolio_weight": [0.60, 0.40, 0.60, 0.40],
        "benchmark_weight": [0.50, 0.50, 0.50, 0.50],
        "stock_return": [0.01, 0.02, -0.005, 0.01],
        "pnl_pct": [0.001, 0.0, 0.0, 0.0],
    }
    return pd.DataFrame(data)

#### TESTS ####

def test_sector_attributions_basic(sector_input_data):
    """
    Test that sectorAttributions runs without error when check=False, pnl=True
    and returns a single DataFrame.
    """
    # call function
    result = sectorAttributions(raw=sector_input_data, pnl=True, check=False, verbose=False)

    # because check=False, expect a single DataFrame (not a tuple)
    assert isinstance(result, pd.DataFrame), (
        "sectorAttributions should return a single DataFrame when check=False."
    )
    assert not result.empty, "Returned DataFrame from sectorAttributions should not be empty."

    # check basic structure
    expected_cols = {"date", "sector", "variable", "variable_value"}
    actual_cols = set(result.columns)
    assert expected_cols.issubset(actual_cols), (
        f"Missing expected columns in the final DataFrame: {expected_cols - actual_cols}"
    )


def test_sector_attributions_with_check(sector_input_data):
    """
    Test that sectorAttributions returns a tuple (data, qa_checks) when check=True.
    """
    result = sectorAttributions(raw=sector_input_data, pnl=True, check=True, verbose=False)
    assert isinstance(result, tuple) and len(result) == 2, (
        "When check=True, sectorAttributions should return a 2-tuple: (data, qa_checks)."
    )

    data, qa_checks = result
    # both should be DataFrames
    assert isinstance(data, pd.DataFrame), "First tuple element should be a DataFrame (the main data)."
    assert isinstance(qa_checks, pd.DataFrame), "Second tuple element should be a DataFrame (the QA checks)."
    assert not data.empty, "Data DataFrame should not be empty."
    assert not qa_checks.empty, "QA checks DataFrame should not be empty."
    # optional structure checks
    for df, label in [(data, "data"), (qa_checks, "qa_checks")]:
        assert "date" in df.columns, f"'date' column missing in {label}."
        # 'qa_checks' might have columns like ['date', 'menchero_active_return', 'cumulative_active_return', 'diff', 'perc_diff']


def test_sector_attributions_pnl_true(sector_input_data):
    """
    Test that setting pnl=True includes pnl_pct in portfolio_return.
    We can do a numeric check to see if final DataFrame differs from pnl=False.
    """
    # make one row have a nonzero pnl_pct
    sector_input_data.loc[0, "pnl_pct"] = 0.003  # e.g., 0.3% PnL
    # with pnl=True, sectorAttributions should incorporate this in portfolio_return
    result_pnl_true = sectorAttributions(
        raw=sector_input_data, pnl=True, check=False, verbose=False
    )
    # because check=False, we get a single DataFrame
    assert isinstance(result_pnl_true, pd.DataFrame)

    # now compare with pnl=False
    sector_input_data.loc[0, "pnl_pct"] = 0.0
    result_pnl_false = sectorAttributions(
        raw=sector_input_data, pnl=False, check=False, verbose=False
    )

    # expect differences in the 'allocation' or 'selection' values for the date/sector with nonzero PnL.
    var_true = result_pnl_true[result_pnl_true["variable"].isin(["allocation","selection"])]
    var_false = result_pnl_false[result_pnl_false["variable"].isin(["allocation","selection"])]

    # group by date/sector/variable to sum variable_value
    sum_true = var_true.groupby(["date", "sector", "variable"])["variable_value"].sum().reset_index()
    sum_false = var_false.groupby(["date", "sector", "variable"])["variable_value"].sum().reset_index()

    # merge to compare
    merged = pd.merge(
        sum_true, sum_false,
        on=["date", "sector", "variable"],
        how="inner",
        suffixes=("_pnl_true", "_pnl_false")
    )

    # for row with 0.003 PnL (the first row), we expect a difference in either 'allocation' or 'selection' for that date & sector.
    differences = merged["variable_value_pnl_true"] - merged["variable_value_pnl_false"]
    assert any(differences != 0), (
        "Expected a difference in at least one 'allocation'/'selection' value when pnl=True vs pnl=False."
    )


def test_sector_attributions_pnl_false(sector_input_data):
    """
    Test the function with pnl=False. The portfolio_return should NOT include pnl_pct.
    We'll do a small numeric check to confirm it's ignoring pnl_pct.
    """
    # set small PnL in the data
    sector_input_data["pnl_pct"] = [0.001, 0.002, 0.001, 0.0]
    
    # run with pnl=False
    result = sectorAttributions(
        raw=sector_input_data, pnl=False, check=False, verbose=False
    )
    assert isinstance(result, pd.DataFrame), (
        "Expected a single DataFrame when check=False."
    )

    # pnl=False, final numbers in 'allocation' or 'selection' should not be affected by the nonzero PnL
    all_vars = result["variable"].unique()
    assert "allocation" in all_vars, "Expected 'allocation' in the final DataFrame's 'variable' column."
    # If you want, you can do a more advanced numeric check here.


def test_sector_attributions_verbose(capfd, sector_input_data):
    """
    Test that setting verbose=True prints some debug messages.
    We'll capture stdout via capfd (pytest fixture).
    """
    _ = sectorAttributions(raw=sector_input_data, pnl=True, check=False, verbose=True)
    captured = capfd.readouterr()
    # check that some debug messages appear
    assert "Starting attributions calculation at a sector level..." in captured.out
    assert "Calculated total daily returns." in captured.out

def test_sector_attributions_small_dataset(small_sector_data):
    """
    Verify sector-level Menchero calculations on a small dataset
    by comparing the final 'allocation' and 'selection' values
    with hand-verified or spreadsheet-derived expected results.
    """
    # call function with pnl=True, check=False
    result_df = sectorAttributions(
        raw=small_sector_data,
        pnl=True,
        check=False,
        verbose=False
    )
    
    # sectorAttributions should return a single DataFrame in melted format.
    assert isinstance(result_df, pd.DataFrame), "Expected a single DataFrame for check=False."
    assert not result_df.empty, "Resulting DataFrame should not be empty."

    # filter to final 'allocation' and 'selection' rows.
    pivoted = (
        result_df[result_df["variable"].isin(["allocation", "selection"])]
        .pivot_table(index=["date","sector"], columns="variable", values="variable_value")
        .reset_index()
    )

    # compare with known/expected results 
    expected_results = {
        (date(2023,1,1),"Tech"):    {"allocation": 0, "selection": 0},
        (date(2023,1,1),"Finance"): {"allocation": 0, "selection": 0},
        (date(2023,1,2),"Tech"):    {"allocation": -0.00126543, "selection": 0.00100835},
        (date(2023,1,2),"Finance"): {"allocation": -0.00126543, "selection": 0},
    }

    # check each row's actual vs. expected within a small tolerance.
    tolerance = 1e-5
    for idx, row in pivoted.iterrows():
        d, sec = row["date"], row["sector"]
        alloc_actual = row["allocation"]
        sel_actual = row["selection"]

        exp_alloc = expected_results[(d, sec)]["allocation"]
        exp_sel   = expected_results[(d, sec)]["selection"]

        assert abs(alloc_actual - exp_alloc) < tolerance, (
            f"Allocation mismatch on {d} - {sec}. "
            f"Expected {exp_alloc}, got {alloc_actual}."
        )
        assert abs(sel_actual - exp_sel) < tolerance, (
            f"Selection mismatch on {d} - {sec}. "
            f"Expected {exp_sel}, got {sel_actual}."

)