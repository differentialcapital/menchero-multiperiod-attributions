# import libraries

# data handling
import pandas as pd
# testing 
import pytest
# functions
from menchero_multiperiod_attribution import _attributionCheck  # Replace 'your_module' with the module name.

#### FIXTURES ####

@pytest.fixture
def attribution_data():
    """Fixture for valid attribution data."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']).date,
        'variable': ['selection', 'allocation'],
        'sector': ['Tech', 'Finance'],
        'stock': ['AAPL', 'JPM'],
        'variable_value': [0.05, 0.03]
    })

@pytest.fixture
def active_return_data():
    """Fixture for valid active return data."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']).date,
        'value': [0.08, 0.05]
    })

#### TESTS ####

def test_attribution_check_valid(attribution_data, active_return_data):
    """Test with valid inputs."""
    result = _attributionCheck(attribution_data, active_return_data, stock_level=True)
    assert not result.empty
    assert 'diff' in result.columns
    assert 'perc_diff' in result.columns

def test_attribution_check_sector_level(attribution_data, active_return_data):
    """Test for sector-level input with stock_level=False."""
    result = _attributionCheck(attribution_data, active_return_data, stock_level=False)
    assert not result.empty
    assert 'diff' in result.columns
    assert 'perc_diff' in result.columns