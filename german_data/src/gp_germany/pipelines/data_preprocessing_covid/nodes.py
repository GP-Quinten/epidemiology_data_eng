from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.signal.windows import gaussian


def upload_data(df):
    return df


def download_data(df):
    return df


def preprocessing_covid(
    RKI_full_data: pd.DataFrame, Hospitalization_ger: pd.DataFrame
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Preprocess COVID-19 data including cases, deaths, and hospitalization.

    Args:
        RKI_full_data (pd.DataFrame): Full COVID-19 data from RKI.
        Hospitalization_ger (pd.DataFrame): Hospitalization data for Germany.

    Returns:
    RKI_cases_sliced, RKI_deaths_sliced, RKI_hosp_sliced, RKI_cases_weekly, RKI_deaths_weekly, RKI_hosp_weekly

        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - RKI_cases_sliced : COVID cases on a daily basis
        - RKI_deaths_sliced : COVID deaths on a daily basis
        - RKI_hosp_sliced : COVID hospitalization on a daily basis
        - RKI_cases_weekly : COVID cases aggrated to a weekly basis (mean over a calendar week)
        - RKI_deaths_weekly : COVID deaths aggrated to a weekly basis (mean over a calendar week)
        - RKI_hosp_weekly : COVID hospitalization aggrated to a weekly basis (mean over a calendar week)
    """
    RKI_sub_data = RKI_full_data.iloc[:, [0, 1, 3, 9, 10]]

    # Group by 'IdLandkreis', 'Meldedatum', and 'Altersgruppe', then summarize using sum
    grouped_data = (
        RKI_sub_data.groupby(["IdLandkreis", "Meldedatum", "Altersgruppe"])
        .agg({"AnzahlFall": "sum", "AnzahlTodesfall": "sum"})
        .reset_index()
    )

    # Create a list of all unique 'IdLandkreis', 'Meldedatum', and 'Altersgruppe'
    all_ids = grouped_data["IdLandkreis"].unique()
    all_dates = grouped_data["Meldedatum"].unique()
    all_ages = grouped_data["Altersgruppe"].unique()

    # Create all possible combinations
    all_combinations = list(product(all_ids, all_dates, all_ages))

    # Convert to DataFrame
    all_combinations_df = pd.DataFrame(
        all_combinations, columns=["IdLandkreis", "Meldedatum", "Altersgruppe"]
    )

    # Merge with grouped_data to fill missing values with zeros
    full_data = pd.merge(
        all_combinations_df,
        grouped_data,
        how="left",
        on=["IdLandkreis", "Meldedatum", "Altersgruppe"],
    ).fillna(0)

    # Sort DataFrame by 'IdLandkreis' and 'Meldedatum'
    full_data.sort_values(by=["IdLandkreis", "Meldedatum"], inplace=True)

    # Format 'IdLandkreis' column
    full_data["IdLandkreis"] = full_data["IdLandkreis"].apply(
        lambda x: format(x, "05d")
    )

    # Reset index
    full_data.reset_index(drop=True, inplace=True)
    RKI_sub_data = full_data

    # remove unknown agegroup counts (less than 0.1% of cases)
    RKI_sub_data = RKI_sub_data[RKI_sub_data["Altersgruppe"] != "unbekannt"]
    RKI_sub_data = RKI_sub_data.pivot_table(
        index=["IdLandkreis", "Meldedatum", "Altersgruppe"],
        values=["AnzahlFall", "AnzahlTodesfall"],
        aggfunc="sum",
    ).reset_index()

    # some dates are never recorded so are removed
    dates_to_remove = pd.date_range("2020-01-01", "2020-02-24", freq="D").strftime(
        "%Y-%m-%d"
    )
    RKI_sub_data = RKI_sub_data[~RKI_sub_data["Meldedatum"].isin(dates_to_remove)]

    # Dividing dataframe into cases and deaths (below)
    RKI_LK_cases = RKI_sub_data.iloc[:, 0:4].pivot_table(
        index=["IdLandkreis", "Meldedatum"],
        columns="Altersgruppe",
        values="AnzahlFall",
        aggfunc="sum",
    )
    # creating a column total summing over the cases of the agegroups
    total_column = RKI_LK_cases.iloc[:, :].sum(axis=1)
    RKI_LK_cases["0-200"] = total_column
    RKI_LK_cases.reset_index(inplace=True)
    # renaming to english
    RKI_LK_cases.rename(columns={"Meldedatum": "date"}, inplace=True)
    RKI_LK_cases.fillna(0, inplace=True)

    # see above
    RKI_LK_deaths = RKI_sub_data.iloc[:, [0, 1, 2, 4]].pivot_table(
        index=["IdLandkreis", "Meldedatum"],
        columns="Altersgruppe",
        values="AnzahlTodesfall",
        aggfunc="sum",
    )
    total_column_deaths = RKI_LK_deaths.iloc[:, :].sum(axis=1)
    RKI_LK_deaths["0-200"] = total_column_deaths
    RKI_LK_deaths.reset_index(inplace=True)
    RKI_LK_deaths.rename(columns={"Meldedatum": "date"}, inplace=True)
    RKI_LK_deaths.fillna(0, inplace=True)

    # renaming and reordering columns
    RKI_LK_cases.columns = [
        "geography",
        "date",
        "0-4",
        "5-14",
        "15-34",
        "35-59",
        "60-79",
        "80plus",
        "0-200",
    ]
    RKI_LK_deaths.columns = [
        "geography",
        "date",
        "0-4",
        "5-14",
        "15-34",
        "35-59",
        "60-79",
        "80plus",
        "0-200",
    ]
    RKI_LK_cases = RKI_LK_cases.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 0]]
    RKI_LK_deaths = RKI_LK_deaths.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 0]]

    # Removing last three digits of ID to get to Bundesländer
    RKI_LK_cases.geography = RKI_LK_cases.geography.str[:2]
    RKI_LK_deaths.geography = RKI_LK_deaths.geography.str[:2]

    # summing over regions in the same BL
    RKI_BL_cases = (
        RKI_LK_cases.groupby(["geography", "date"])
        .agg(
            {
                "0-4": "sum",
                "5-14": "sum",
                "15-34": "sum",
                "35-59": "sum",
                "60-79": "sum",
                "80plus": "sum",
                "0-200": "sum",
            }
        )
        .reset_index()
    )

    RKI_BL_deaths = (
        RKI_LK_deaths.groupby(["geography", "date"])
        .agg(
            {
                "0-4": "sum",
                "5-14": "sum",
                "15-34": "sum",
                "35-59": "sum",
                "60-79": "sum",
                "80plus": "sum",
                "0-200": "sum",
            }
        )
        .reset_index()
    )

    # Defining dict for BL HASC Codes and applying it below
    HASC_codes = {
        "01": "DE.SH",
        "02": "DE.HH",
        "03": "DE.NI",
        "04": "DE.HB",
        "05": "DE.NW",
        "06": "DE.HE",
        "07": "DE.RP",
        "08": "DE.BW",
        "09": "DE.BY",
        "10": "DE.SL",
        "11": "DE.BE",
        "12": "DE.BB",
        "13": "DE.MV",
        "14": "DE.SN",
        "15": "DE.ST",
        "16": "DE.TH",
    }

    RKI_BL_cases.geography = RKI_BL_cases.geography.apply(lambda x: HASC_codes[str(x)])
    RKI_BL_deaths.geography = RKI_BL_deaths.geography.apply(
        lambda x: HASC_codes[str(x)]
    )

    # reordering again
    RKI_BL_cases = RKI_BL_cases.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 0]]
    RKI_BL_deaths = RKI_BL_deaths.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 0]]

    # summing over BL to get country level data
    RKI_DE_cases = (
        RKI_BL_cases.groupby(["date"])
        .agg(
            {
                "0-4": "sum",
                "5-14": "sum",
                "15-34": "sum",
                "35-59": "sum",
                "60-79": "sum",
                "80plus": "sum",
                "0-200": "sum",
            }
        )
        .reset_index()
    )

    RKI_DE_deaths = (
        RKI_BL_deaths.groupby(["date"])
        .agg(
            {
                "0-4": "sum",
                "5-14": "sum",
                "15-34": "sum",
                "35-59": "sum",
                "60-79": "sum",
                "80plus": "sum",
                "0-200": "sum",
            }
        )
        .reset_index()
    )

    # adding HASC code for Germany
    RKI_DE_cases["geography"] = pd.Series(["DE"] * len(RKI_DE_cases))
    RKI_DE_deaths["geography"] = pd.Series(["DE"] * len(RKI_DE_deaths))

    # concatination to one df containing country level on top and BL level on bottom
    RKI_cases = pd.concat([RKI_DE_cases, RKI_BL_cases], ignore_index=True)
    RKI_deaths = pd.concat([RKI_DE_deaths, RKI_BL_deaths], ignore_index=True)

    # ### Now I want to make this data weekly using calendar weeks (Monday-Sunday)
    # and add the possibility to also slice daily data to start on Mondays and end on Sundays

    def calendar_weekly(rel_data, weekly=True):
        rel_data_cp = rel_data.copy()
        rel_data_cp["date"] = pd.to_datetime(rel_data_cp["date"])
        wdays = rel_data_cp["date"].dt.day_name()
        while wdays.iloc[0] != "Monday":
            wdays = wdays.iloc[1:]
            rel_data_cp = rel_data_cp.iloc[1:, :]

        while wdays.iloc[-1] != "Sunday":
            wdays = wdays.iloc[:-1]
            rel_data_cp = rel_data_cp.iloc[:-1, :]
        if weekly == False:
            return rel_data_cp
        else:
            starting_dates = np.array(rel_data_cp.iloc[::7, 0])
            starting_dates = pd.DataFrame(starting_dates)
            numeric_columns = [
                "0-4",
                "5-14",
                "15-34",
                "35-59",
                "60-79",
                "80plus",
                "0-200",
            ]

            weekly_data = round(
                rel_data_cp[numeric_columns]
                .groupby(np.arange(len(rel_data_cp)) // 7)
                .mean(0),
                2,
            )
            rel_data_weekly = pd.DataFrame({**weekly_data}).reset_index(drop=True)
            rel_data_weekly.insert(0, "date", starting_dates)
            rel_data_weekly["geographygeo"] = rel_data.iloc[0, 8]

            return rel_data_weekly

    RKI_cases_sliced = pd.DataFrame()

    for geo in RKI_cases["geography"].unique():
        cases_sliced_tmp = calendar_weekly(
            RKI_cases[RKI_cases["geography"] == geo], weekly=False
        )
        RKI_cases_sliced = pd.concat(
            [RKI_cases_sliced, cases_sliced_tmp], ignore_index=True
        )

    RKI_deaths_sliced = pd.DataFrame()

    for geo in RKI_deaths["geography"].unique():
        deaths_sliced_tmp = calendar_weekly(
            RKI_deaths[RKI_deaths["geography"] == geo], weekly=False
        )
        RKI_deaths_sliced = pd.concat(
            [RKI_deaths_sliced, deaths_sliced_tmp], ignore_index=True
        )

    # Initialize an empty DataFrame for RKI_cases_weekly
    RKI_cases_weekly = pd.DataFrame()

    # Loop through unique geographies
    for geo in RKI_cases["geography"].unique():
        cases_weekly_tmp = calendar_weekly(RKI_cases[RKI_cases["geography"] == geo])
        RKI_cases_weekly = pd.concat(
            [RKI_cases_weekly, cases_weekly_tmp], ignore_index=True
        )

    # Initialize an empty DataFrame for RKI_cases_weekly
    RKI_deaths_weekly = pd.DataFrame()

    # Loop through unique geographies
    for geo in RKI_deaths["geography"].unique():
        deaths_weekly_tmp = calendar_weekly(RKI_deaths[RKI_deaths["geography"] == geo])
        RKI_deaths_weekly = pd.concat(
            [RKI_deaths_weekly, deaths_weekly_tmp], ignore_index=True
        )

    # ### Now Hospitalization (different file)

    RKI_hosp_data = Hospitalization_ger

    HASC_codes_hosp = {
        "0": "DE",
        "1": "DE.SH",
        "2": "DE.HH",
        "3": "DE.NI",
        "4": "DE.HB",
        "5": "DE.NW",
        "6": "DE.HE",
        "7": "DE.RP",
        "8": "DE.BW",
        "9": "DE.BY",
        "10": "DE.SL",
        "11": "DE.BE",
        "12": "DE.BB",
        "13": "DE.MV",
        "14": "DE.SN",
        "15": "DE.ST",
        "16": "DE.TH",
    }

    RKI_hosp_sub_data = RKI_hosp_data.iloc[:, [0, 2, 3, 4]]
    RKI_hosp = RKI_hosp_sub_data.pivot_table(
        index=["Bundesland_Id", "Datum"],
        columns="Altersgruppe",
        values="7T_Hospitalisierung_Faelle",
        aggfunc="sum",
        fill_value=0,
    )
    RKI_hosp.reset_index(inplace=True)
    RKI_hosp = RKI_hosp[
        [
            "Datum",
            "00-04",
            "05-14",
            "15-34",
            "35-59",
            "60-79",
            "80+",
            "00+",
            "Bundesland_Id",
        ]
    ]
    RKI_hosp["Bundesland_Id"] = RKI_hosp["Bundesland_Id"].apply(
        lambda x: HASC_codes_hosp[str(x)]
    )
    RKI_hosp.columns = [
        "date",
        "0-4",
        "5-14",
        "15-34",
        "35-59",
        "60-79",
        "80plus",
        "0-200",
        "geography",
    ]

    RKI_hosp_sliced = pd.DataFrame()

    for geo in RKI_hosp["geography"].unique():
        hosp_sliced_tmp = calendar_weekly(
            RKI_hosp[RKI_hosp["geography"] == geo], weekly=False
        )
        RKI_hosp_sliced = pd.concat(
            [RKI_hosp_sliced, hosp_sliced_tmp], ignore_index=True
        )

    RKI_hosp_weekly = pd.DataFrame()

    for geo in RKI_hosp["geography"].unique():
        hosp_weekly_tmp = calendar_weekly(RKI_hosp[RKI_hosp["geography"] == geo])
        RKI_hosp_weekly = pd.concat(
            [RKI_hosp_weekly, hosp_weekly_tmp], ignore_index=True
        )

    return (
        RKI_cases_sliced,
        RKI_deaths_sliced,
        RKI_hosp_sliced,
        RKI_cases_weekly,
        RKI_deaths_weekly,
        RKI_hosp_weekly,
    )

def smooth_n_cases(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join.

    Args:
        df_cas_pos (pd.DataFrame): Processed DataFrame for 'Cas_pos' sheets.
        df_nb_prelev (pd.DataFrame): Processed DataFrame for 'Nb_prélèvement' sheets.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data (number of pos tests + number of test).
    """
    # Assuming merged_df is your original DataFrame
    unique_combinations = merged_df["COMBI_GEO_AGE"].unique()

    # List to store historical dataframes for each combination
    df_smoothed_dataframes = []

    # Set up the number of rows and columns for subplots
    num_rows = len(unique_combinations) // 2 + len(unique_combinations) % 2
    num_cols = 2

    for i, combo in enumerate(unique_combinations):

        filtered_df = merged_df.loc[merged_df["COMBI_GEO_AGE"] == combo]
        filtered_df = filtered_df.sort_values(by="date")
        filtered_df["cases"] = filtered_df["cases"].astype(int)
        filtered_df["date"] = pd.to_datetime(filtered_df["date"])

        # Copy the filtered dataframe to avoid modifying the original
        smoothed_dataframes = filtered_df.copy()

        # Ensure 'date' is of type datetime
        smoothed_dataframes["date"] = pd.to_datetime(
            smoothed_dataframes["date"]
        )

        # Sort the DataFrame by date
        smoothed_dataframes = smoothed_dataframes.sort_values(by="date")

        # Define the Gaussian window
        window_size = 5
        STD = 4
        gaussian_window = gaussian(window_size, std=STD)
        gaussian_window /= gaussian_window.sum()

        # Apply convolution with 'same' option
        smoothed = convolve(
            smoothed_dataframes["cases"].values, gaussian_window, mode="same"
        )

        # Add the 'SMOOTHED_PASSAGE_DIAG_NB' column to the DataFrame
        smoothed_dataframes["SMOOTHED_cases"] = smoothed

        # Calculate variance for confidence interval
        variance = convolve(
            (smoothed_dataframes["cases"].values - smoothed) ** 2,
            gaussian_window,
            mode="same",
        )

        # Calculate standard deviation
        std_dev = variance**0.5

        # Set the confidence level (e.g., 95%)
        confidence_level = 0.95

        # Calculate the standard error
        margin_of_error = std_dev * 1.96  # For a 95% confidence interval

        # Add confidence interval columns to the DataFrame
        smoothed_dataframes["LOWER_CI_SMOOTHED_cases"] = (
            smoothed_dataframes["SMOOTHED_cases"] - margin_of_error
        )
        smoothed_dataframes["UPPER_CI_SMOOTHED_cases"] = (
            smoothed_dataframes["SMOOTHED_cases"] + margin_of_error
        )

        # Sort the DataFrame by date
        smoothed_dataframes.sort_values(by="date", inplace=True)

        # Check and replace NaN values with 0 in the columns used
        smoothed_dataframes["LOWER_CI_SMOOTHED_cases"] = smoothed_dataframes[
            "LOWER_CI_SMOOTHED_cases"
        ].clip(lower=0)

        # Check for NaN values in the confidence interval columns
        lower_ci_smoothed = np.where(
            np.isnan(smoothed_dataframes["LOWER_CI_SMOOTHED_cases"]),
            0,
            smoothed_dataframes["LOWER_CI_SMOOTHED_cases"],
        )
        upper_ci_smoothed = np.where(
            np.isnan(smoothed_dataframes["UPPER_CI_SMOOTHED_cases"]),
            0,
            smoothed_dataframes["UPPER_CI_SMOOTHED_cases"],
        )

        # Append the smoothed dataframe to the list
        df_smoothed_dataframes.append(smoothed_dataframes)

    # Concatenate all historical dataframes in the list
    final_smoothed_dataframe = pd.concat(df_smoothed_dataframes, ignore_index=True)

    return final_smoothed_dataframe


def baseline_n_cases(final_smoothed_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join.

    Args:
        df_cas_pos (pd.DataFrame): Processed DataFrame for 'Cas_pos' sheets.
        df_nb_prelev (pd.DataFrame): Processed DataFrame for 'Nb_prélèvement' sheets.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data (number of pos tests + number of test).
    """
    # Liste unique des combinaisons de diagnostic et d'âge
    unique_combinations = final_smoothed_dataframe["COMBI_GEO_AGE"].unique()

    # Liste pour stocker les DataFrames résultants pour chaque combinaison
    result_dataframes = []

    # Définir le décalage d'un an
    one_year_shift = pd.DateOffset(weeks=52)

    # Boucle sur chaque combinaison
    for combo in unique_combinations:
        # Filtrer le DataFrame pour la combinaison actuelle
        df = final_smoothed_dataframe[
            final_smoothed_dataframe["COMBI_GEO_AGE"] == combo
        ].copy()

        # make sure 'date' is of type datetime
        df["date"] = pd.to_datetime(df["date"])
        # Ajouter une colonne 'date_1YR_AGO' pour stocker la date correspondante il y a un an
        df["date_1YR_AGO"] = df["date"] - one_year_shift

        # Fusionner avec lui-même pour obtenir les valeurs de l'année précédente
        merged_df = pd.merge(
            df,
            df[
                [
                    "date",
                    "SMOOTHED_cases",
                    "LOWER_CI_SMOOTHED_cases",
                    "UPPER_CI_SMOOTHED_cases",
                ]
            ],
            left_on="date_1YR_AGO",
            right_on="date",
            suffixes=("", "_1YR_AGO"),
            how="left",
        )

        # Remplacer les valeurs manquantes par celles de l'année précédente
        merged_df["SMOOTHED_cases_1YR_AGO"].fillna(
            merged_df["SMOOTHED_cases"], inplace=True
        )
        merged_df["LOWER_CI_SMOOTHED_cases_1YR_AGO"].fillna(
            merged_df["LOWER_CI_SMOOTHED_cases"], inplace=True
        )
        merged_df["UPPER_CI_SMOOTHED_cases_1YR_AGO"].fillna(
            merged_df["UPPER_CI_SMOOTHED_cases"], inplace=True
        )

        # Supprimer la colonne 'date_1YR_AGO'
        merged_df.drop(["date_1YR_AGO"], axis=1, inplace=True)

        # Ajouter le DataFrame résultant à la liste
        result_dataframes.append(merged_df)

    # Concaténer tous les DataFrames résultants
    final_result_dataframe = pd.concat(result_dataframes, ignore_index=True)

    # Afficher les 50 dernières lignes du DataFrame final résultant
    final_result_dataframe.rename(
        columns={"SMOOTHED_cases_1YR_AGO": "BASELINE_cases"},
        inplace=True,
    )
    final_result_dataframe.rename(
        columns={
            "LOWER_CI_SMOOTHED_cases_1YR_AGO": "LOWER_CI_BASELINE_cases"
        },
        inplace=True,
    )
    final_result_dataframe.rename(
        columns={
            "UPPER_CI_SMOOTHED_cases_1YR_AGO": "UPPER_CI_BASELINE_cases"
        },
        inplace=True,
    )

    age_class_order = [
        "0-4",
        "5-14",
        "15-34",
        "35-59",
        "80plus",
        "0-200",
    ]

    final_result_dataframe = final_result_dataframe.sort_values(
        by=["date", "geography", "AGE_CLASS"],
        key=lambda x: pd.Categorical(x, categories=age_class_order, ordered=True),
    )

    final_result_dataframe = final_result_dataframe.drop(
        ["COMBI_GEO_AGE", "YEAR"], axis=1
    )

    final_result_dataframe["ALERT_N_CASES"] = np.where(
        (
            final_result_dataframe["cases"]
            > final_result_dataframe["UPPER_CI_BASELINE_cases"]
        ),
        1,
        0,
    )

    return final_result_dataframe


def create_alerts(df):
    '''
    This function creates alerts for the number of cases. It first smooths the number of cases and then creates a baseline for the number of cases. It then compares the number of cases to the baseline and creates an alert if the number of cases is higher than the baseline.

    Args:
        df (pd.DataFrame): DataFrame containing the number of cases for each age group (as columns) for each geography (as rows) for each date (format YYYY-MM-dd).

    Returns:
        pd.DataFrame: DataFrame containing the number of cases for each age group (as columns) for each geography (as rows) for each date (format YYYY-MM-dd). It also contains the smoothed number of cases, the baseline number of cases, and the alert (True if the number of cases is higher than the baseline, False otherwise).
    '''
    # pivot table to get the data in the right format. 0-4,5-14,15-34,35-59,60-79,80plus,0-200 are columns name. melt them in a column called AGE_CLASS. extract YEAR
    df = pd.melt(df, id_vars=['geography', 'date'], var_name='AGE_CLASS', value_name='cases')
    df["YEAR"] = pd.DatetimeIndex(df["date"]).year
    # Définissez les conditions et les valeurs correspondantes
    df["COMBI_GEO_AGE"] = (
        df["geography"] + " | " + df["AGE_CLASS"]
    )

    # create smoothed data
    df = smooth_n_cases(df)

    # create baseline data
    df = baseline_n_cases(df)


    return df