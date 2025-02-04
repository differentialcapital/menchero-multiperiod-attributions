# import libraries

# data handling
import pandas as pd
from datetime import date
# testing
import pytest
# functions
from menchero_multiperiod_attribution import _validateInputData

#### FIXTURES ####

@pytest.fixture
def valid_data():
    """Fixture for valid sample data."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02']).date,
        'sector': ['Tech', 'Finance', 'Tech', 'Finance'],
        'portfolio_weight': [0.5, 0.5, 0.6, 0.4],  # Sums to 1 per date
        'benchmark_weight': [0.6, 0.4, 0.5, 0.5],  # Sums to 1 per date
        'stock_return': [0.02, 0.01, 0.01, 0.0],
        'pnl_pct': [0.01, 0.0, 0.0, 0.0]
    })

@pytest.fixture
def valid_weight_data():
    """Fixture for valid weight data with tolerance."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02']).date,
        'sector': ['Tech', 'Finance', 'Tech', 'Finance'],
        'portfolio_weight': [0.495, 0.505, 0.495, 0.505],  # Sums to 1 per date
        'benchmark_weight': [0.495, 0.505, 0.495, 0.505],  # Sums to 1 per date
        'stock_return': [0.02, 0.01, 0.01, 0.0],
        'pnl_pct': [0.01, 0.0, 0.0, 0.0]
    })

@pytest.fixture
def invalid_weight_data():
    """Fixture for invalid weight data."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02']).date,
        'sector': ['Tech', 'Finance', 'Tech', 'Finance'],
        'portfolio_weight': [0.5, 0.5, 0.6, 0.3],  # Sums to 0.9 on 2023-01-02
        'benchmark_weight': [0.6, 0.4, 0.5, 0.5],  # Sums to 1 per date
        'stock_return': [0.02, 0.01, 0.01, 0.0],
        'pnl_pct': [0.01, 0.0, 0.0, 0.0]
    })

#### TESTS ####

def test_validate_input_data_valid(valid_data):
    """Test with valid input data."""
    required_columns = ["date", "sector", "portfolio_weight", "benchmark_weight", "stock_return", "pnl_pct"]
    column_types = {
        "date": date,
        "sector": str,
        "portfolio_weight": float,
        "benchmark_weight": float,
        "stock_return": float,
        "pnl_pct": float,
    }
    # This should pass without raising an exception.
    _validateInputData(valid_data, required_columns, column_types)

def test_validate_input_data_missing_column(valid_data):
    """Test with missing columns."""
    required_columns = ["date", "sector", "portfolio_weight", "benchmark_weight", "stock_return", "pnl_pct"]
    column_types = {
        "date": date,
        "sector": str,
        "portfolio_weight": float,
        "benchmark_weight": float,
        "stock_return": float,
        "pnl_pct": float,
    }
    data_missing_column = valid_data.drop(columns=["sector"])
    with pytest.raises(ValueError, match="Input data is missing the following required columns:"):
        _validateInputData(data_missing_column, required_columns, column_types)

def test_validate_input_data_empty(valid_data):
    """Test with empty data."""
    required_columns = ["date", "sector", "portfolio_weight", "benchmark_weight", "stock_return", "pnl_pct"]
    column_types = {
        "date": date,
        "sector": str,
        "portfolio_weight": float,
        "benchmark_weight": float,
        "stock_return": float,
        "pnl_pct": float,
    }
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError, match="Input data cannot be None or empty."):
        _validateInputData(empty_data, required_columns, column_types)

def test_validate_input_data_invalid_type(valid_data):
    """Test with incorrect data types."""
    required_columns = ["date", "sector", "portfolio_weight", "benchmark_weight", "stock_return", "pnl_pct"]
    column_types = {
        "date": date,
        "sector": str,
        "portfolio_weight": float,
        "benchmark_weight": float,
        "stock_return": float,
        "pnl_pct": float,
    }
    invalid_data = valid_data.copy()
    invalid_data["portfolio_weight"] = invalid_data["portfolio_weight"].astype(str)
    with pytest.raises(ValueError, match="Column 'portfolio_weight' must have datatype float."):
        _validateInputData(invalid_data, required_columns, column_types)

def test_validate_input_data_invalid_weights(invalid_weight_data):
    """Test with invalid weights that do not sum to approximately 1."""
    required_columns = ["date", "sector", "portfolio_weight", "benchmark_weight", "stock_return", "pnl_pct"]
    column_types = {
        "date": date,
        "sector": str,
        "portfolio_weight": float,
        "benchmark_weight": float,
        "stock_return": float,
        "pnl_pct": float,
    }
    with pytest.raises(ValueError, match="does not sum to approximately 1"):
        _validateInputData(invalid_weight_data, required_columns, column_types)

def test_validate_input_data_valid_weights(valid_weight_data):
    """Test with valid weights that sum to approximately 1."""
    required_columns = ["date", "sector", "portfolio_weight", "benchmark_weight", "stock_return", "pnl_pct"]
    column_types = {
        "date": date,
        "sector": str,
        "portfolio_weight": float,
        "benchmark_weight": float,
        "stock_return": float,
        "pnl_pct": float,
    }
    # This should pass without raising an exception.
    _validateInputData(valid_weight_data, required_columns, column_types)
