import datetime

import numpy as np
import pandas as pd


def upload_data(df):
    print(df.columns)
    return df


def data_shaping(data):
    """
    Reshape and clean the input DataFrame for further analysis.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Reshaped and cleaned DataFrame with 'date' and 'total' columns.
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


def preprocessing_influenza(
    Influenza_cases_BB: pd.DataFrame,
    Influenza_cases_BE: pd.DataFrame,
    Influenza_cases_BW: pd.DataFrame,
    Influenza_cases_BY: pd.DataFrame,
    Influenza_cases_DE: pd.DataFrame,
    Influenza_cases_HB: pd.DataFrame,
    Influenza_cases_HE: pd.DataFrame,
    Influenza_cases_HH: pd.DataFrame,
    Influenza_cases_MV: pd.DataFrame,
    Influenza_cases_NI: pd.DataFrame,
    Influenza_cases_NW: pd.DataFrame,
    Influenza_cases_RP: pd.DataFrame,
    Influenza_cases_SH: pd.DataFrame,
    Influenza_cases_SL: pd.DataFrame,
    Influenza_cases_SN: pd.DataFrame,
    Influenza_cases_ST: pd.DataFrame,
    Influenza_cases_TH: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform preprocessing on a list of Influenza cases DataFrames.

    Parameters:
    - Influenza_cases_BB, Influenza_cases_BE, ..., Influenza_cases_TH: Pandas DataFrames
      DataFrames containing Influenza cases for different regions.

    Returns:
    - df_complete_reg: Pandas DataFrame
      Influenza cases on a weekly basis
    """

    # List of DataFrames
    dataframes = {
        "BB": Influenza_cases_BB,
        "BE": Influenza_cases_BE,
        "BW": Influenza_cases_BW,
        "BY": Influenza_cases_BY,
        "DE": Influenza_cases_DE,
        "HB": Influenza_cases_HB,
        "HE": Influenza_cases_HE,
        "HH": Influenza_cases_HH,
        "MV": Influenza_cases_MV,
        "NI": Influenza_cases_NI,
        "NW": Influenza_cases_NW,
        "RP": Influenza_cases_RP,
        "SH": Influenza_cases_SH,
        "SL": Influenza_cases_SL,
        "SN": Influenza_cases_SN,
        "ST": Influenza_cases_ST,
        "TH": Influenza_cases_TH,
    }

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

    # df_complete_reg['total'] = (
    #     df_complete_reg['total']
    #     .apply(lambda x: 0 if x.startswith('"""') and x.endswith('"""') else x)
    #     .str.strip('"""')
    #     .replace('', '0')
    #     .astype('Int64')
    # )

    df_complete_reg = df_complete_reg.sort_values(by=["date", "geography"], ascending=True).reset_index(drop=True)

    return df_complete_reg
