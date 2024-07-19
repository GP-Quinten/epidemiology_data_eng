import datetime

import numpy as np
import pandas as pd


def upload_data(df):
    return df


def data_shaping(data: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape and clean the input DataFrame.

    Parameters:
    - data: Pandas DataFrame
      Input DataFrame to be reshaped and cleaned.

    Returns:
    - final_data: Pandas DataFrame
      Resulting DataFrame after reshaping and cleaning.
    """
    data = data.iloc[:-1].fillna(0)
    stacked_data = pd.melt(data, ignore_index=False).dropna()
    stacked_data.variable = stacked_data.variable.str.strip()
    stacked_data.value = stacked_data.value.replace({r"\r": ""}, regex=True)
    stacked_data["value"] = stacked_data["value"].replace(r'^""*$', np.nan, regex=True)
    stacked_data = stacked_data.dropna()
    dates = []
    for year, week in zip(stacked_data.variable, stacked_data.index):
        dates.append(
            datetime.date.fromisocalendar(int(year.strip('"')), int(week.strip('"')), 1)
        )
    final_data = pd.DataFrame({"date": dates, "total": stacked_data.value}, index=None)
    return final_data


def preprocessing_rsv(RSV_cases_SN: pd.DataFrame) -> pd.DataFrame:
    """
    Perform preprocessing on RSV cases DataFrame.

    Parameters:
    - RSV_cases_SN: Pandas DataFrame
      DataFrame containing RSV cases for the SN region.

    Returns:
    - df_complete_reg: Pandas DataFrame
      RSV cases in Saxony, Germany on a weekly basis
    """
    # List of DataFrames
    dataframes = {"SN": RSV_cases_SN}

    # Initialization of the resulting DataFrame
    df_complete_reg = pd.DataFrame()
    # Iterating over the DataFrames
    for geo, df in dataframes.items():
        geo = "DE." + geo
        # Calling the data_shaping function with the corresponding DataFrame
        df_reg = data_shaping(df)

        # Special treatment for the "DE.DE" case
        if geo == "DE.DE":
            geo = "DE"

        # Adding the "geography" column
        df_reg.insert(2, "geography", geo)

        # Concatenation with the resulting DataFrame
        df_complete_reg = pd.concat([df_complete_reg, df_reg], axis=0)

    df_complete_reg["total"] = (
        df_complete_reg["total"].str.strip('"').astype(float).astype("Int64")
    )

    return df_complete_reg
