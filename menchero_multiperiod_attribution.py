# import libraries

# data handling
import pandas as pd
# type hints
from typing import Union, Tuple

#### FUNCTIONS ####

## INTERNAL FUNCTIONS ##

# function to perform qa checks on attributions
def _attributionCheck(attribution, active_return, stock_level) -> pd.DataFrame:
    
    """
    Performs quality assurance checks to compare calculated attribution values with cumulative active returns.

    Parameters:
       attribution (pd.DataFrame): DataFrame containing calculated attribution values.
       active_return (pd.DataFrame): DataFrame containing cumulative active return values.
       stock_level (bool): Whether the data corresponds to stock-level attributions.

    Returns:
       pd.DataFrame: One or more DataFrames containing results of the quality assurance checks, including differences and percentage differences.
    """
    
    # Assumes inputs are validated and non-empty (validated by _validateInputData).
    
    # detemine which columns to remove 
    if stock_level:
        cols = ['variable', 'sector', 'stock']
    else:
        cols = ['variable', 'sector']
    
    # get last data point
    last_date = attribution['date'].max()
    
    # calculate total daily active return from selection and allocation 
    qa_checks = attribution[attribution['variable'].isin(['selection', 'allocation'])].drop(columns=cols).groupby('date').sum().reset_index().rename(columns={'variable_value' : 'menchero_active_return'})
    
    # filter to final day
    qa_checks = qa_checks[qa_checks['date'] == last_date]
    active_return = active_return[active_return['date'] == last_date]

    # join, format and compare
    active_return = active_return[['date', 'value']].rename(columns={'value':'cumulative_active_return'})
    qa_checks = qa_checks.merge(active_return, on=['date'], how='left')
    qa_checks['diff'] = qa_checks['cumulative_active_return'] - qa_checks['menchero_active_return']
    qa_checks['perc_diff'] = (qa_checks['diff']) / (qa_checks['cumulative_active_return']) * 100
    
    return qa_checks

# function to modify attribution output for stock level calculations
def _attributionModify(stock_output, sector_output) -> pd.DataFrame:
    
    """
    Modifies stock-level attribution values by adjusting based on sector-level attributions.

    Parameters:
        stock_output (pd.DataFrame): DataFrame containing stock-level attribution data.
        sector_output (pd.DataFrame): DataFrame containing sector-level attribution data.

    Returns:
        pd.DataFrame: Modified stock-level attribution data with adjusted effects.
    """
    
    # extract selection effect
    a = (
        sector_output[sector_output["variable"] == "allocation"][["date", "sector", "variable_value"]]
        .rename({"variable_value": "selection"}, axis=1)
        .assign(variable="selection")
    )
    
    # extract poirtfolio weights
    b = stock_output[stock_output["variable"] == "portfolio_weight"][["date", "sector", "stock", "variable_value"]].rename({"variable_value": "portfolio_weight"}, axis=1)
    
    # combine
    c = pd.merge(a, b, on=["date", "sector"])
    
    # normalise portfolio weight by total sector weight
    c["norm_portfolio_weight"] = c["portfolio_weight"] / c.groupby(["date", "sector"])["portfolio_weight"].transform("sum")
    
    # adjust selection effect by normalised portfolio weight
    c["selection_"] = c["selection"] * c["norm_portfolio_weight"]
    
    # add adjusted selection effect to stock attributions
    stock_output = pd.merge(stock_output, c[["date", "sector", "stock", "variable", "selection_"]], on=["date", "sector", "stock", "variable"], how="left")
    stock_output["variable_value"] = stock_output.apply(lambda x: x["selection_"] if x["variable"] == "selection" else x["variable_value"], axis=1)
    stock_output = stock_output.drop("selection_", axis=1)
    stock_output["variable"] = stock_output.apply(
        lambda x: "allocation" if x["variable"] == "selection" else ("selection" if x["variable"] == "allocation" else x["variable"]), axis=1
    )
    
    return stock_output

# function to ensure input data is correct
def _validateInputData(data, required_columns, column_types, verbose=False) -> None:
    """
    Validates the input data for required columns and datatypes.
    
    Parameters:
        data (pd.DataFrame): Input dataframe to validate.
        required_columns (list): List of required column names.
        column_types (dict): Dictionary with column names as keys and expected datatypes as values.
        verbose (bool): Print details if True.
    
    Raises:
        ValueError: If validation fails.
    """
    
    # import required libraries
    import numpy as np
    from datetime import date
    
    # check input data is not empty
    if data is None or data.empty:
        raise ValueError("Input data cannot be None or empty.")
    
    # check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing the following required columns: {missing_columns}")
    
    # Check for column datatypes
    for column, dtype in column_types.items():
        if column in data.columns:
            # Special handling for date column
            if dtype == date:
                if not data[column].map(type).eq(dtype).all():
                    raise ValueError(f"Column '{column}' must have datatype {dtype.__name__}.")
            elif not data[column].map(type).eq(dtype).all():
                raise ValueError(f"Column '{column}' must have datatype {dtype.__name__}.")

    # Check weights sum to approximately 1 per date
    for weight_col in ['benchmark_weight', 'portfolio_weight']:
        if weight_col in data.columns:
            daily_sums = data.groupby('date')[weight_col].sum()
            invalid_dates = daily_sums.loc[~np.isclose(daily_sums, 1.0, atol=0.01)]  # Tolerance of 0.01
            if not invalid_dates.empty:
                raise ValueError(
                    f"Column '{weight_col}' does not sum to approximately 1 (within tolerance) "
                    f"for the following dates: {invalid_dates.index.tolist()}"
                )
    
    if verbose:
        print("Input data validation passed.")


# function to apply the menchero multiperiod smoothing method to performance attribution calculations
def _menchero(input_data, group_vars, check, stock_level=False, verbose=False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    
    """
    Applies the Menchero multiperiod amoothing method to calculate performance attributions.

    Parameters:
        input_data (pd.DataFrame): Input data containing portfolio and benchmark returns and weights.
        group_vars (list): List of variables to group by, such as 'sector' or 'stock'.
        check (bool): Whether to perform quality assurance checks on the output.
        stock_level (bool): If True, includes PnL in calculations for stock-level attributions.
        verbose (bool): If True, prints detailed logs for debugging and progress tracking.

    Returns:
        Tuple[pd.DataFrame, ...]: One or more DataFrames with calculated performance attributions, adjusted for multiperiod smoothing.
    """
    
    if verbose:
        print("Starting Menchero multiperiod smoothing...")
    
    # make a copy
    data = pd.DataFrame(input_data)
    
    if verbose:
       print(f"Input data has {data.shape[0]} rows and {data.shape[1]} columns.")
       print(f"Group variables: {group_vars}")

    
    # calculate performance attributions 
    data["allocation"] = (data["portfolio_weight"] - data["benchmark_weight"]) * (data["benchmark_return"] - data["total_benchmark_return"]) + data["pnl_pct"] * (stock_level == True)
    data["selection"] = data["benchmark_weight"] * (data["portfolio_return"] - data["benchmark_return"])
    data["interaction"] = (data["portfolio_weight"] - data["benchmark_weight"]) * (data["portfolio_return"] - data["benchmark_return"])
    
    if verbose:
        print("Calculated initial attributions.")
        print(data[["allocation", "selection", "interaction"]].head())
    
    # calculate portfolio active return 
    active_return = data.groupby("date")[["weighted_portfolio_return", "weighted_benchmark_return"]].sum()
    active_return["portfolio_return"] = (1 + active_return["weighted_portfolio_return"]).cumprod() - 1
    active_return["benchmark_return"] = (1 + active_return["weighted_benchmark_return"]).cumprod() - 1
    active_return["value"] = active_return["portfolio_return"] - active_return["benchmark_return"]
    
    if verbose:
        print("Calculated cumulative active return.")
        print(active_return.reset_index()[["date", "portfolio_return", "benchmark_return", "value"]].head())
    
    ## APPLY MENCHERO MULTIPERIOD SMOOTHING ##
    
    # calculate cumulative weighted daily returns
    # cumprod
    agg = data.groupby(["date"])[["weighted_portfolio_return", "weighted_benchmark_return"]].sum().reset_index().sort_values(["date"])
    agg["cumprod_weighted_portfolio_return"] = 1 + agg["weighted_portfolio_return"]
    agg["cumprod_weighted_benchmark_return"] = 1 + agg["weighted_benchmark_return"]
    agg["cumprod_weighted_portfolio_return"] = agg["cumprod_weighted_portfolio_return"].cumprod() - 1
    agg["cumprod_weighted_benchmark_return"] = agg["cumprod_weighted_benchmark_return"].cumprod() - 1
    # cumsum
    agg["cumsum_weighted_portfolio_return"] = agg["weighted_portfolio_return"].cumsum()
    agg["cumsum_weighted_benchmark_return"] = agg["weighted_benchmark_return"].cumsum()
    
    # calculate cumulative squared difference between weighted returns
    agg["cum_squared_diff"] = (agg["weighted_portfolio_return"] - agg["weighted_benchmark_return"]) ** 2
    agg["cum_squared_diff"] = agg["cum_squared_diff"].cumsum()

    # calculate time periods
    agg["T"] = agg.assign(T=1)["T"].cumsum()
    
    # calculate multiplicative factor for return contributions
    agg["multiplicative_factor"] = (1 + agg["cumprod_weighted_benchmark_return"]) ** ((agg["T"] - 1) / agg["T"]) * (agg["cumprod_weighted_portfolio_return"] == agg["cumprod_weighted_benchmark_return"]) + ((agg["cumprod_weighted_portfolio_return"] - agg["cumprod_weighted_benchmark_return"]) / agg["T"]) / (
        (1 + agg["cumprod_weighted_portfolio_return"]) ** (1 / agg["T"]) - (1 + agg["cumprod_weighted_benchmark_return"]) ** (1 / agg["T"])
    ) * (agg["cumprod_weighted_portfolio_return"] != agg["cumprod_weighted_benchmark_return"])
    
    if verbose:
        print("Calculated multiplicative factor.")
        print(agg[["date", "multiplicative_factor"]].head())
    
    # calculate corrective factor for return contributions
    agg["corrective_factor"] = 0 * (agg["cumprod_weighted_portfolio_return"] == agg["cumprod_weighted_benchmark_return"]) + ((agg["cumprod_weighted_portfolio_return"] - agg["cumprod_weighted_benchmark_return"]) - agg["multiplicative_factor"] * (agg["cumsum_weighted_portfolio_return"] - agg["cumsum_weighted_benchmark_return"])) / agg["cum_squared_diff"] * (
        agg["cumprod_weighted_portfolio_return"] != agg["cumprod_weighted_benchmark_return"]
    )
    
    if verbose:
        print("Calculated corrective factor.")
        print(agg[["date", "corrective_factor"]].head())
    
    # calculate adjusted return
    agg["adjusted_return"] = agg["multiplicative_factor"] + agg["corrective_factor"] * (agg["weighted_portfolio_return"] - agg["weighted_benchmark_return"])

    if verbose:
        print("Calculated return adjustment.")
        print(agg[["date", "adjusted_return"]].head())

    # function to dynamically calculate adjusted_return values for given date using calculated parameters
    def _aggs(date):
        out = agg[agg.index <= date].copy()
        out["adjusted_return"] = out["multiplicative_factor"][date] + out["corrective_factor"][date] * (out["weighted_portfolio_return"] - out["weighted_benchmark_return"])
        return out["adjusted_return"]

    # dynamically calculate adjusted returns
    agg = agg.set_index("date")
    betas = {date: _aggs(date) for date in agg.index}
    
    # function to apply adjusted returns to performance attribution values
    def _adjust(grp, g, date, c):
        df = data[(data[grp] == g) & (data.date <= date)].copy()
        df[c] = df["date"].map(betas[date]) * df[c]
        return df[c].sum()
    
    # apply adjusted returns
    data = data.sort_values(group_vars + ["date"])
    data["allocation_"] = data.apply(lambda x: _adjust(group_vars[-1], x[group_vars[-1]], x["date"], "allocation"), axis=1)
    data["selection_"] = data.apply(lambda x: _adjust(group_vars[-1], x[group_vars[-1]], x["date"], "selection"), axis=1)
    data["interaction_"] = data.apply(lambda x: _adjust(group_vars[-1], x[group_vars[-1]], x["date"], "interaction"), axis=1)
    
    if verbose:
        print("Applied adjustments to performance attributions.")
        print(data[["date", "allocation_", "selection_", "interaction_"]].head())
    
    # prepare output
    if stock_level == False:
        
        # format fund level active returns 
        active_return = active_return.reset_index()[["date", "value"]].assign(variable="active_return").assign(group="Fund") 
        
        # add results to output dataframe
        data["active_return"] = data["portfolio_return"] - data["benchmark_return"] # MAYBE I CAN REMOVE?
        data["allocation"] = data["allocation_"]
        data["selection"] = data["selection_"] + data["interaction_"]

        # convert from wide to long format
        data = pd.melt(
            data,
            id_vars=["date"] + group_vars,
            value_vars=["allocation", "selection", "portfolio_weight", "benchmark_weight", "pnl_pct"],
            value_name="value",
            var_name="effect",
            )
        
        # rename for output
        data = data.rename({"effect" : "variable", "value" : "variable_value"}, axis=1)
        
        if verbose:
            print("Formatted data for output.")
            print(data.head())
        
        # perform and return checks with output
        if check:
            
            if verbose:
                print("Performing quality assurance checks.")
            
            # perform quality assurance checks
            qa_checks = _attributionCheck(data, active_return, stock_level=False)
            
            if verbose:
                print("Quality assurance checks complete.")
                print(qa_checks)
            
            return data, qa_checks
            
        
        # return only the output    
        else:
        
            return data
    
    else:
        
        # format fund level active returns 
        active_return = active_return.reset_index()[["date", "value"]].assign(variable="active_return").assign(group="Fund") 
        
        # add results to output dataframe
        data["allocation"] = data["allocation_"]
        data["selection"] = data["selection_"] + data["interaction_"]
        
        # convert from wide to long format
        data = pd.melt(
            data,
            id_vars=["date"] + group_vars,
            value_vars=["allocation", "selection", "portfolio_weight", "benchmark_weight", "pnl_pct"],
            value_name="variable_value",
            var_name="variable",
            )
        
        # perform and return checks with output
        if check:
            
            if verbose:
                print("Formatted data for output.")
                print(data.head())
            
            return data, active_return
        
        # return only the output    
        else:
            
            if verbose:
                print("Formatted data for output.")
                print(data.head())
        
            return data

## USER FUNCTIONS ##

# function to calculate sector level attributions using Menchero multiperiod smoothing
def sectorAttributions(raw, pnl=True, check=False, verbose=False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    
    """
    Calculates performance attributions at the sector level.

    Parameters:
       raw (pd.DataFrame): Input data containing sector-level portfolio and benchmark information.
       pnl (bool): If True, includes PnL in the calculations. Defaults to True.
       check (bool): If True, performs quality assurance checks on the output. Defaults to False.
       verbose (bool): If True, prints progress and intermediate results. Defaults to False.

    Returns:
       Tuple[pd.DataFrame, ...]: One or more DataFrames with calculated sector-level attributions.
    """
    
    if verbose:
        print("Starting attributions calculation at a sector level...")
        
    # validate input data
    from datetime import date
    required_columns = ["date", "sector", "portfolio_weight", "benchmark_weight", "stock_return", "pnl_pct"]
    column_types = {"date": date, "sector": str, "portfolio_weight": float, "benchmark_weight": float, "stock_return": float, "pnl_pct": float}
    _validateInputData(raw, required_columns, column_types, verbose=verbose)    
    
    # replace missing values with 0
    sector = raw.fillna(0)
    
    # calculate returns taking pnl into account
    sector["portfolio_return"] = sector["portfolio_weight"] * sector["stock_return"] + sector["pnl_pct"] * (pnl == True)
    sector["benchmark_return"] = sector["benchmark_weight"] * sector["stock_return"]
    
    # aggregate to sector level 
    sector = sector.groupby(["date", "sector"])[["portfolio_weight", "benchmark_weight", "benchmark_return", "portfolio_return", "pnl_pct"]].sum().reset_index()

    # normalise aggregated returns by weights
    sector["portfolio_return"] = sector.apply(lambda x: x["portfolio_return"] / x["portfolio_weight"] if x["portfolio_weight"] != 0 else 0, axis=1)
    sector["benchmark_return"] = sector.apply(lambda x: x["benchmark_return"] / x["benchmark_weight"] if x["benchmark_weight"] != 0 else 0, axis=1)

    # calculate weighted returns
    sector["weighted_portfolio_return"] = sector["portfolio_return"] * sector["portfolio_weight"]
    sector["weighted_benchmark_return"] = sector["benchmark_return"] * sector["benchmark_weight"]
    
    # calculate total daily returns
    sector["total_portfolio_return"] = sector.groupby(["date"])["weighted_portfolio_return"].transform("sum")
    sector["total_benchmark_return"] = sector.groupby(["date"])["weighted_benchmark_return"].transform("sum")
    
    if verbose:
        print("Calculated total daily returns.")
        print(sector[["date", "total_portfolio_return", "total_benchmark_return"]].head())
    
    # apply menchero method
    return _menchero(input_data=sector, group_vars=['sector'], check=check, stock_level=False, verbose=verbose)

# function to calculate stock level attributions using Menchero multiperiod smoothing and stock level modifications    
def stockAttributions(raw, pnl=True, check=False, verbose=False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    
    """
    Calculates performance attributions at the stock level.

    Parameters:
        raw (pd.DataFrame): Input data containing stock-level portfolio and benchmark information.
        pnl (bool): If True, includes PnL in the calculations. Defaults to True.
        check (bool): If True, performs quality assurance checks on the output. Defaults to False.
        verbose (bool): If True, prints progress and intermediate results. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, ...]: One or more DataFrames with calculated stock-level attributions.
    """
    
    # validate input data
    from datetime import date
    required_columns = ["date", "sector", "stock", "portfolio_weight", "benchmark_weight", "stock_return", "pnl_pct"]
    column_types = {"date": date, "sector": str, "stock": str, "portfolio_weight": float, "benchmark_weight": float, "stock_return": float, "pnl_pct": float}
    _validateInputData(raw, required_columns, column_types, verbose=verbose)
    
    if verbose:
        print("Starting attributions calculation at a stock level...")
    
    # calculate sector level attributions
    sector_output = sectorAttributions(raw, pnl=pnl, check=False)
    
    if verbose:
        print("Calculated sector level attributions.")
        print(sector_output.head())
    
    # replace missing values with 0
    stock = raw.fillna(0)
    
    # rename
    stock = stock.rename({"stock_return": "portfolio_return"}, axis=1)

    # calculate returns
    stock["benchmark_return"] = stock["portfolio_return"]

    # calculate weighted returns
    stock["weighted_portfolio_return"] = stock["portfolio_return"] * stock["portfolio_weight"] + stock["pnl_pct"] * (pnl == True)
    stock["weighted_benchmark_return"] = stock["benchmark_return"] * stock["benchmark_weight"]

    # calculate total daily returns
    stock["total_portfolio_return"] = (stock.groupby(["date", "sector"])["weighted_portfolio_return"].transform("sum") / stock.groupby(["date", "sector"])["portfolio_weight"].transform("sum")).fillna(0)
    stock["total_benchmark_return"] = (stock.groupby(["date", "sector"])["weighted_benchmark_return"].transform("sum") / stock.groupby(["date", "sector"])["benchmark_weight"].transform("sum")).fillna(0)

    if verbose:
        print("Calculated total daily returns.")
        print(stock[["date", "total_portfolio_return", "total_benchmark_return"]].head())

    # perform qa checks
    if check:
        
        # apply menchero method
        stock_output, active_return = _menchero(input_data=stock, group_vars=["sector", "stock"], check=check, stock_level=True)
        
        # calculate allocation effect at a stock level
        stock_output = _attributionModify(stock_output, sector_output)
        
        if verbose:
            print("Applied stock level attribution modification.")
            print(stock_output.head())

        if verbose:
            print("Performing quality assurance checks.")

        # perform quality assurance checks
        qa_checks = _attributionCheck(stock_output, active_return, stock_level=True)
        
        if verbose:
            print("Quality assurance checks complete.")
            print(qa_checks)
            print("Formatted data for output.")
            print(stock_output.head())
        
        return stock_output, qa_checks
    
    else:
        
        stock_output = _menchero(input_data=stock, group_vars=["sector", "stock"], check=check, stock_level=True)
        
        # calculate allocation effect at a stock level
        stock_output = _attributionModify(stock_output, sector_output)
        
        if verbose:
            print("Applied stock level attribution modification.")
            print("Formatted data for output.")
            print(stock_output.head())
        
        return stock_output