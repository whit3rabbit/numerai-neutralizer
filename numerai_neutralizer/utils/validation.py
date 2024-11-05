import pandas as pd
from typing import Union, Tuple

def validate_data(
    data: Union[pd.DataFrame, pd.Series],
    name: str = "data"
) -> None:
    """Validate common data requirements."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if data.empty:
            raise ValueError(f"{name} is empty")
        if data.isna().any().any():
            raise ValueError(f"{name} contains NaN values")
    else:
        raise TypeError(f"{name} must be DataFrame or Series")

def validate_indices_match(
    df1: Union[pd.DataFrame, pd.Series],
    df2: Union[pd.DataFrame, pd.Series],
    sort: bool = True
) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
    """Validate and align indices of two dataframes/series."""
    if not isinstance(df1.index, type(df2.index)):
        raise TypeError("Index types must match")
        
    common_idx = df1.index.intersection(df2.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping indices")
        
    df1 = df1.loc[common_idx]
    df2 = df2.loc[common_idx]
    
    if sort:
        df1 = df1.sort_index()
        df2 = df2.sort_index()
        
    return df1, df2