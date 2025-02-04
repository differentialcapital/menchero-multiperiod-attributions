# import libraries

# data handling
import pandas as pd
# testing
import pytest
# functions
from menchero_multiperiod_attribution import _attributionModify

#### FIXTURES ####

@pytest.fixture
def sector_output():
    """Fixture for sector-level attribution data."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01']).date,
        'sector': ['Tech', 'Finance'],
        'variable': ['allocation', 'allocation'],
        'variable_value': [0.05, 0.03]
    })

@pytest.fixture
def stock_output():
    """Fixture for stock-level attribution data with both selection and allocation rows."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01']).date,
        'sector': ['Tech', 'Finance', 'Tech', 'Finance'],
        'stock': ['AAPL', 'JPM', 'GOOGL', 'MS'],
        'variable': ['selection', 'selection', 'allocation', 'portfolio_weight'],
        'variable_value': [0.1, 0.2, 0.3, 0.4]
    })

#### TESTS ####

def test_attribution_modify_basic(sector_output, stock_output):
    """Test basic functionality of _attributionModify."""
    result = _attributionModify(stock_output, sector_output)
    assert not result.empty, "The result should not be empty."
    assert 'variable_value' in result.columns, "Output must include the 'variable_value' column."
    assert 'variable' in result.columns, "Output must include the 'variable' column."

def test_attribution_modify_structure(sector_output, stock_output):
    """Test the structure of the output."""
    result = _attributionModify(stock_output, sector_output)
    expected_columns = ['date', 'sector', 'stock', 'variable', 'variable_value']
    assert list(result.columns) == expected_columns, "The output structure does not match the expected format."

def test_attribution_modify_flips_selection_and_allocation(sector_output, stock_output):
    """
    Ensure _attributionModify flips:
      - original 'selection' to 'allocation'
      - original 'allocation' to 'selection'
    """
    result = _attributionModify(stock_output, sector_output)
    
    # check rows originally labeled 'selection' became 'allocation'
    original_selection_stocks = stock_output.loc[
        stock_output['variable'] == 'selection', 'stock'
    ]
    # in final result, same stocks should appear with variable == 'allocation'
    flipped_to_allocation = result.loc[
        (result['stock'].isin(original_selection_stocks)) & 
        (result['variable'] == 'allocation')
    ]
    assert not flipped_to_allocation.empty, (
        "Expected to find originally 'selection' rows flipped to 'allocation', "
        "but got an empty DataFrame."
    )
    
    # check that rows originally labeled 'allocation' became 'selection'
    original_allocation_stocks = stock_output.loc[
        stock_output['variable'] == 'allocation', 'stock'
    ]
    # in final result, same stocks should appear with variable == 'selection'
    flipped_to_selection = result.loc[
        (result['stock'].isin(original_allocation_stocks)) & 
        (result['variable'] == 'selection')
    ]
    assert not flipped_to_selection.empty, (
        "Expected to find originally 'allocation' rows flipped to 'selection', "
        "but got an empty DataFrame."
    )
