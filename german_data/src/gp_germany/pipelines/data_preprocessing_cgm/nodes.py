import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.signal.windows import gaussian


def download_data(df):
    # keep only small part of the data for testing
    # df = df.head(100)
    return df

def concatenate_data(df1, df2):
    return pd.concat([df1, df2], axis=0)

def update_data(df):
    # print("Noms des dataframes contenus dans le dictionnaire :")
    # for key in df.keys():
    #     print(key)
    return df


def upload_data(df):
    return df


def normalize_region(df: pd.DataFrame) -> List[str]:
    """
    Merge patient data and general practitioner data regions into a unified region list.

    Args:
        df (pd.DataFrame): DataFrame containing patient or general practitioner data.

    Returns:
        List[str]: Unified region list.
    """
    region = []

    for item in df["kvregion"]:

        if item == "Westfalen-Lippe" or item == "Nordrhein":
            region.append("Nordrhein-Westfalen")
        else:
            region.append(item)

    return region


def week_to_monday(week_str):
    year, week = map(int, week_str.split("-"))
    # Calculate the date of the first day (Monday) of the given calendar week
    monday = datetime.datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w").date()
    return monday.strftime("%Y-%m-%d")


def preprocessing(
    patient_data: pd.DataFrame,
    prescriptions_data: pd.DataFrame,
    pharma_data: pd.DataFrame,
    pharma_data_all_sales: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Perform preprocessing on patient and pharmacy data.

    Args:
        patient_data (pd.DataFrame): Respiratory disease related diagnoses and prescriptions aggregated to ICD-10 and ATC categories.
        prescriptions_data (pd.DataFrame): Respiratory disease related prescriptions aggregated to ATC categories.
        pharma_data (pd.DataFrame): Respiratory disease related pharmacy sells aggregated to ATC categories.

    Returns:
        - diagnosis_data_wide (pd.DataFrame) : Diagnoses of respiratory disease grouped into ICD-10 categories on a weekly basis
        - diagnosis_data_wide_rel (pd.DataFrame) :  Normalized diagnoses in percent (relative to all respiratory diagnoses)
        - pharma_data_wide (pd.DataFrame) : Pharmacy sells of respiratory disease related drugs grouped into ATC categories on a weekly basis
        - pharma_data_wide_rel (pd.DataFrame) : Normalized pharmacy sells in percent (relative to all respiratory disease related drugs)
        - prescription_data_wide (pd.DataFrame) : Prescriped drugs for respiratory diseases group into ATC categories on a weekly basis
        - prescription_data_wide_rel (pd.DataFrame) : Normalized prescriped drugs in percent (relative to all prescriped drugs for respiratory diseases)
    """

    regions = {
        "Deutschland": "DE",
        "Baden-Württemberg": "DE.BW",
        "Bayern": "DE.BY",
        "Berlin": "DE.BE",
        "Brandenburg": "DE.BB",
        "Bremen": "DE.HB",
        "Hamburg": "DE.HH",
        "Hessen": "DE.HE",
        "Mecklenburg-Vorpommern": "DE.MV",
        "Niedersachsen": "DE.NI",
        "Nordrhein-Westfalen": "DE.NW",
        "Rheinland-Pfalz": "DE.RP",
        "Saarland": "DE.SL",
        "Sachsen": "DE.SN",
        "Sachsen-Anhalt": "DE.ST",
        "Schleswig-Holstein": "DE.SH",
        "Thüringen": "DE.TH",
    }


    ### DIAG DATA PREPROCESSING ###
    # drop columns total, ratio, count_all
    patient_data.drop(columns=["total", "ratio"], inplace=True)
    patient_data["ratio_all"] = round(patient_data["ratio_all"]*100, 2)
    # drop duplicates ['week', 'category', 'kvregion', 'age_group'] -> keep the one that has the highest count_all
    patient_data = patient_data.sort_values("count_all", ascending=False).drop_duplicates(['week', 'category', 'kvregion', 'age_group'])
    diagnosis_data_wide, diagnosis_data_wide_rel, diagnosis_data_wide_ratio_all, diagnosis_data_wide_count_all, diagnosis_counts_for_trends = transforming_gp_data(
        patient_data
    )
    
    ### PRESCRIPTION DATA PREPROCESSING ###
    # drop duplicates ['week', 'category', 'kvregion', 'age_group'] -> keep the one that has the highest count_all
    prescriptions_data = prescriptions_data.sort_values("count_all", ascending=False).drop_duplicates(['week', 'category', 'kvregion', 'age_group'])
    prescriptions_data["ratio_all"] = round(
            prescriptions_data["distinct_patient_count"]
            / prescriptions_data["count_all"]
            * 100,
            2,
        )
    prescription_data_wide, prescription_data_wide_rel, prescription_data_wide_ratio_all, prescription_data_wide_count_all, prescription_counts_for_trends = transforming_gp_data(
        prescriptions_data
    )


    ### PHARMACY DATA PREPROCESSING ###
    pharma_data["atc_classification"] = (
        pharma_data["atc3_code"] + " " + pharma_data["atc3_name"]
    )
    # if column type Abverkäufe is object, convert it to float
    if pharma_data["Abverkäufe"].dtype == "object":
        # convert column Abverkäufe (type object, format: 98225,0) to int
        pharma_data["Abverkäufe"] = pharma_data["Abverkäufe"].str.replace(",", ".").astype(float).astype("Int64")   

    pharma_data = (
        pharma_data.groupby(["Woche", "Region (Bundesland)", "atc_classification", "Abgabe"])
        .agg({"Abverkäufe": "sum"})
        .astype({"Abverkäufe": int})
        .reset_index()
    )

    pharma_data.rename(columns={"Region (Bundesland)": "region", "Abgabe": "Dispensing"}, inplace=True)

    pharma_data_wide = pd.DataFrame()
    for geo in pharma_data.region.unique():
        for dispensing_mode in pharma_data.Dispensing.unique():
            pharma_wide = (
                pharma_data[(pharma_data.region == geo) & (pharma_data.Dispensing == dispensing_mode)]
                .pivot(index="Woche", columns=["atc_classification"], values="Abverkäufe")
                .reset_index()
            )
            pharma_wide.insert(1, "geography", geo)
            pharma_wide.insert(2, "dispensing_mode", dispensing_mode)
            pharma_data_wide = pd.concat([pharma_data_wide, pharma_wide])

    totalcols = [col for col in pharma_data_wide.columns if "0" in col]
    pharma_data_wide["sum_all_respiratory"] = pharma_data_wide[
        totalcols
    ].sum(axis=1)

    pharma_data_wide = data_final_transfos_pharma_data(pharma_data_wide, regions=regions)

    ### PHARMACY DATA ALL SALES PREPROCESSING ###
    if pharma_data_all_sales["Abverkäufe"].dtype == "object":
        # convert column Abverkäufe (type object, format: 98225,0) to int
        pharma_data_all_sales["Abverkäufe"] = pharma_data_all_sales["Abverkäufe"].str.replace(",", ".").astype(float).astype("Int64")
    pharma_data_all_sales.rename(columns={"Region (Bundesland)": "geography", "Abgabe": "dispensing_mode", "Abverkäufe": "count_all"}, inplace=True)
    pharma_data_wide_count_all = data_final_transfos_pharma_data(pharma_data_all_sales, regions=regions)

    # initialize new dataframe pharma_data_wide_rel copying th first 3 columns of pharma_data_wide
    pharma_data_wide_rel = pharma_data_wide.iloc[:, :3]
    numeric_cols = pharma_data_wide.select_dtypes("number").columns
    for col in numeric_cols:
        new_col_name = col + " relative"
        pharma_data_wide_rel[new_col_name] = round(
            pharma_data_wide[col]
            / pharma_data_wide["sum_all_respiratory"]
            * 100,
            2,
        )
    
    # merge pharma_data_wide with pharma_data_all_sales on date, geography, and dispensing_mode, left join, don't keep dduplicate columns
    pharma_data_wide = pd.merge(
        pharma_data_wide,
        pharma_data_wide_count_all,
        on=["date", "geography", "dispensing_mode"],
        how="left",
        # suffixes=("", "_count_all"),
    )
    # compute ratio_all 
    pharma_data_wide_ratio_all = pharma_data_wide.iloc[:, :3]
    for col in numeric_cols:
        new_col_name = col + " relative"
        pharma_data_wide_ratio_all[new_col_name] = round(
            pharma_data_wide[col]
            / pharma_data_wide["count_all"]
            * 100,
            2,
        )
    # pharma_data_wide.drop(columns=["sum_all_ICDs"], inplace=True)
    # rename column Abverkäufe to count_all
    # pharma_data_wide_ratio_all.rename(columns={"Abverkäufe": "count_all"}, inplace=True)
    
    # drop last column for diagnosis_data_wide_rel, pharma_data_wide_rel, and prescription_data_wide_rel
    diagnosis_data_wide_rel.drop(columns=diagnosis_data_wide_rel.columns[-1], inplace=True)
    pharma_data_wide_rel.drop(columns=pharma_data_wide_rel.columns[-1], inplace=True)
    prescription_data_wide_rel.drop(columns=prescription_data_wide_rel.columns[-1], inplace=True)

    # drop column 'count_all' if it exists in the dataframe pharma_data_wide
    if "count_all" in pharma_data_wide.columns:
        pharma_data_wide.drop(columns=["count_all"], inplace=True)


    return (
        diagnosis_data_wide,
        diagnosis_data_wide_rel,
        diagnosis_data_wide_ratio_all,
        diagnosis_data_wide_count_all,
        diagnosis_counts_for_trends,
        pharma_data_wide,
        pharma_data_wide_rel,
        pharma_data_wide_ratio_all,
        pharma_data_wide_count_all,
        prescription_data_wide,
        prescription_data_wide_rel,
        prescription_data_wide_ratio_all,
        prescription_data_wide_count_all,
        prescription_counts_for_trends,
    )



def pivot_transform_docmetric(data_frame, column_name, age_group=False):
    '''make a pivot transfo around the column_name column'''
    regions = {
        "Deutschland": "DE",
        "Baden-Württemberg": "DE.BW",
        "Bayern": "DE.BY",
        "Berlin": "DE.BE",
        "Brandenburg": "DE.BB",
        "Bremen": "DE.HB",
        "Hamburg": "DE.HH",
        "Hessen": "DE.HE",
        "Mecklenburg-Vorpommern": "DE.MV",
        "Niedersachsen": "DE.NI",
        "Nordrhein-Westfalen": "DE.NW",
        "Rheinland-Pfalz": "DE.RP",
        "Saarland": "DE.SL",
        "Sachsen": "DE.SN",
        "Sachsen-Anhalt": "DE.ST",
        "Schleswig-Holstein": "DE.SH",
        "Thüringen": "DE.TH",
    }
    data_frame_cp = data_frame
    data_frame_cp["region"] = normalize_region(data_frame_cp)

    if column_name == "count_all":
        # keep only the columns week, region, count_all, and remove duplicates
        df = data_frame_cp[["week", "region", "age_group", "count_all"]].drop_duplicates()
        # remove duplicates [week, region, age_group] -> keep the one that has the highest count_all
        df = df.sort_values("count_all", ascending=False).drop_duplicates(["week", "region", "age_group"])
        df.insert(0, "date", df.iloc[:, 0].apply(lambda x: week_to_monday(x)))
        df.insert(1, "geography", df.iloc[:, 2].apply(lambda x: regions[x]))
        df.fillna(0, inplace=True)
        df.columns = df.columns.str.strip()
        df = df.drop(columns=["week", "region"])
        df.columns = df.columns.str.replace(column_name + " ", "")

    else :
        if age_group:
            df = data_frame_cp.pivot_table(
                index=["week", "region"],
                columns=["category", "age_group"],
                values=[column_name],
                aggfunc="sum",
            )
        else:
            df = data_frame_cp.pivot_table(
                index=["week", "region", "age_group"],
                columns=["category"],
                values=[column_name],
                aggfunc="sum",
            )
        
        df.reset_index(inplace=True)
        df.columns = df.columns.map(" ".join)
        df.insert(0, "date", df.iloc[:, 0].apply(lambda x: week_to_monday(x)))
        df.insert(1, "geography", df.iloc[:, 2].apply(lambda x: regions[x]))
        df.fillna(0, inplace=True)
        df.columns = df.columns.str.strip()
        df = df.drop(columns=["week", "region"])
        df.columns = df.columns.str.replace(column_name + " ", "")

        totalcols = [col for col in df.columns if "0" in col]
        df["sum_all_respiratory"] = df[
                totalcols
            ].sum(axis=1)

    return df

def transforming_gp_data(data_frame):
    """
    Brings the gp data to the correct format by pivoting, and adding the correct date and region
    """

    df_distinct_counts = pivot_transform_docmetric(data_frame, "distinct_patient_count")

    # make sure all numeric columns are integers
    numeric_cols = df_distinct_counts.select_dtypes("number").columns
    df_distinct_counts[numeric_cols] = df_distinct_counts[numeric_cols].astype(int)

    # initialize new dataframe pharma_data_wide_rel copying th first 3 columns of pharma_data_wide
    df_distinct_counts_rel = df_distinct_counts.iloc[:, :4]
    for col in numeric_cols:
        new_col_name = col + " relative"
        df_distinct_counts_rel[new_col_name] = round(
            df_distinct_counts[col]
            / df_distinct_counts["sum_all_respiratory"]
            * 100,
            2,
        )
    
    df_ratio_all = pivot_transform_docmetric(data_frame, "ratio_all")
    df_count_all = pivot_transform_docmetric(data_frame, "count_all")
    # make sure all numeric columns are integers
    numeric_cols = df_distinct_counts.select_dtypes("number").columns
    df_distinct_counts[numeric_cols] = df_distinct_counts[numeric_cols].astype(int)

    # create a table just for the trends
    df_counts_for_trends = pivot_transform_docmetric(data_frame, "distinct_patient_count", age_group=True)


    return df_distinct_counts, df_distinct_counts_rel, df_ratio_all, df_count_all, df_counts_for_trends


def data_final_transfos_pharma_data(df, regions):
    df.Woche = df.Woche.str[2:12]
    df.insert(
        0, "date", pd.to_datetime(df["Woche"], format="%d.%m.%Y")
    )

    df["geography"] = df["geography"].apply(
        lambda x: regions[x]
    )


    dispensing_modes = {
        "Nichtarzneimittel": "non-medicinal",
        "nicht apothekenpflichtig": "also available outside of pharmacies",
        "apothekenpflichtig": "only available within pharmacies",
        "rezeptpflichtig": "only on prescription",
        "Betäubungsmittel": "drugs and chemicals",
        "Drogen + Chemikalien": "drugs and chemicals",
        "Gesamt": "all",
    }

    df["dispensing_mode"] = df["dispensing_mode"].apply(
        lambda x: dispensing_modes[x]
    )

    df.drop(columns=["Woche"], inplace=True)
    df.fillna(0, inplace=True)
    df.sort_values(["date", "geography", "dispensing_mode"], inplace=True)
    numeric_cols = df.select_dtypes("number").columns
    df[numeric_cols] = df[numeric_cols].astype(int)
    return df

def smooth_n_cases(df, cols, col_for_alerts, window_size = 5, upper_clip=False):

    # Assuming df is your original DataFrame
    unique_combinations = df["COMBINATIONS"].unique()

    # List to store historical dataframes for each combination
    df_smoothed_dataframes = []

    # Set up the number of rows and columns for subplots
    num_rows = len(unique_combinations) // 2 + len(unique_combinations) % 2
    num_cols = 2

    for i, combo in enumerate(unique_combinations):

        filtered_df = df.loc[df["COMBINATIONS"] == combo]
        filtered_df = filtered_df.sort_values(by=cols[0])
        # convert infinite values to 1000000000
        filtered_df.replace([np.inf, -np.inf], 1000000000, inplace=True)
        # convert NaN values to 0
        filtered_df.fillna(0, inplace=True)
        filtered_df[col_for_alerts] = filtered_df[col_for_alerts].astype(int)
        filtered_df[cols[0]] = pd.to_datetime(filtered_df[cols[0]])

        # Copy the filtered dataframe to avoid modifying the original
        smoothed_dataframes = filtered_df.copy()

        # Ensure 'date' is of type datetime
        smoothed_dataframes[cols[0]] = pd.to_datetime(
            smoothed_dataframes[cols[0]]
        )

        # Sort the DataFrame by date
        smoothed_dataframes = smoothed_dataframes.sort_values(by=cols[0])

        # Define the Gaussian window
        STD = 4
        gaussian_window = gaussian(window_size, std=STD)
        gaussian_window /= gaussian_window.sum()

        # Apply convolution with 'same' option
        smoothed = convolve(
            smoothed_dataframes[col_for_alerts].values, gaussian_window, mode="same"
        )

        # Add the 'SMOOTHED_PASSAGE_DIAG_NB' column to the DataFrame
        smoothed_dataframes["SMOOTHED_cases"] = smoothed

        # Calculate variance for confidence interval
        variance = convolve(
            (smoothed_dataframes[col_for_alerts].values - smoothed) ** 2,
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

        if upper_clip:
            smoothed_dataframes["UPPER_CI_SMOOTHED_cases"] = smoothed_dataframes["UPPER_CI_SMOOTHED_cases"].clip(upper=100)

        # Sort the DataFrame by date
        smoothed_dataframes.sort_values(by=cols[0], inplace=True)

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

def baseline_n_cases(df_smoothed, df_baseline, cols, col_for_alerts):


    # Liste unique des combinaisons de diagnostic et d'âge
    unique_combinations = df_smoothed["COMBINATIONS"].unique()

    # Liste pour stocker les DataFrames résultants pour chaque combinaison
    result_dataframes = []

    # Définir le décalage de 0 semaine
    one_year_shift = pd.DateOffset(weeks=0)

    # Boucle sur chaque combinaison
    for combo in unique_combinations:
        # Filtrer le DataFrame pour la combinaison actuelle
        df_combo_smoothed = df_smoothed[
            df_smoothed["COMBINATIONS"] == combo
        ].copy()

        df_combo_baseline = df_baseline[
            df_baseline["COMBINATIONS"] == combo
        ].copy()




        # make sure 'date' is of type datetime
        df_combo_smoothed[cols[0]] = pd.to_datetime(df_combo_smoothed[cols[0]])
        df_combo_baseline[cols[0]] = pd.to_datetime(df_combo_baseline[cols[0]])
        # # Ajouter une colonne 'date_current_year' pour stocker la date correspondante actuelle
        # df_combo_smoothed["date_current_year"] = df[cols[0]] - one_year_shift

        # Fusionner avec lui-même pour obtenir les valeurs de l'année précédente
        merged_df = pd.merge(
            df_combo_smoothed,
            df_combo_baseline[
                [
                    cols[0],
                    "SMOOTHED_cases",
                    "LOWER_CI_SMOOTHED_cases",
                    "UPPER_CI_SMOOTHED_cases",
                ]
            ],
            left_on="date",
            right_on=cols[0],
            suffixes=("", "_baseline"),
            how="left",
        )

        # Remplacer les valeurs manquantes par celles de l'année précédente
        merged_df["SMOOTHED_cases_baseline"].fillna(
            merged_df["SMOOTHED_cases"], inplace=True
        )
        merged_df["LOWER_CI_SMOOTHED_cases_baseline"].fillna(
            merged_df["LOWER_CI_SMOOTHED_cases"], inplace=True
        )
        merged_df["UPPER_CI_SMOOTHED_cases_baseline"].fillna(
            merged_df["UPPER_CI_SMOOTHED_cases"], inplace=True
        )

        # # Supprimer la colonne 'date_current_year'
        # merged_df.drop(["date_current_year"], axis=1, inplace=True)

        # Ajouter le DataFrame résultant à la liste
        result_dataframes.append(merged_df)

    # Concaténer tous les DataFrames résultants
    final_result_dataframe = pd.concat(result_dataframes, ignore_index=True)

    final_result_dataframe.rename(
        columns={"SMOOTHED_cases_baseline": "BASELINE_cases"},
        inplace=True,
    )
    final_result_dataframe.rename(
        columns={
            "LOWER_CI_SMOOTHED_cases_baseline": "LOWER_CI_BASELINE_cases"
        },
        inplace=True,
    )
    final_result_dataframe.rename(
        columns={
            "UPPER_CI_SMOOTHED_cases_baseline": "UPPER_CI_BASELINE_cases"
        },
        inplace=True,
    )

    final_result_dataframe = final_result_dataframe.sort_values(
        by=cols,
    )

    final_result_dataframe = final_result_dataframe.drop(
        ["COMBINATIONS", "YEAR"], axis=1
    )

    final_result_dataframe["ALERT_CASES"] = np.where(
        (
            final_result_dataframe[col_for_alerts]
            > final_result_dataframe["UPPER_CI_BASELINE_cases"]
        ),
        1,
        0,
    )

    return final_result_dataframe


def create_alerts(df, df_rel, df_ratio_all, cols_for_alerts):
    """
    Create alerts for the given dataframes. The alerts are based on the smoothed data and the baseline data. 
    The alerts are created based on the following conditions:
    - If the smoothed data is above the upper confidence interval of the baseline data, an alert is triggered.
    - If the smoothed data is below the lower confidence interval of the baseline data, an alert is triggered.
    - If the smoothed data is within the confidence interval of the baseline data, no alert is triggered.

    Args:
    - df (pd.DataFrame): Dataframe containing the count data.
    - df_rel (pd.DataFrame): Dataframe containing the relative data.
    - df_ratio_all (pd.DataFrame): Dataframe containing the ratio data.
    - cols_for_alerts (list): List of columns to be used for creating the alerts.

    Returns:
    - df (pd.DataFrame): Dataframe containing the alerts for the count data.
    - df_rel (pd.DataFrame): Dataframe containing the alerts for the relative data.
    - df_ratio_all (pd.DataFrame): Dataframe containing the alerts for the ratio data.
    """

    # make sure column date is of type datetime
    df[cols_for_alerts[0]] = pd.to_datetime(df[cols_for_alerts[0]])
    df_rel[cols_for_alerts[0]] = pd.to_datetime(df_rel[cols_for_alerts[0]])
    df_ratio_all[cols_for_alerts[0]] = pd.to_datetime(df_ratio_all[cols_for_alerts[0]])

    # keep only the dates after 2023-07-01
    df = df[df[cols_for_alerts[0]] >= pd.to_datetime("2023-07-01")]
    df_rel = df_rel[df_rel[cols_for_alerts[0]] >= pd.to_datetime("2023-07-01")]
    df_ratio_all = df_ratio_all[df_ratio_all[cols_for_alerts[0]] >= pd.to_datetime("2023-07-01")]


    ##############################
    ### Treat count data first ###
    df = pd.melt(df, id_vars=cols_for_alerts, var_name='MEDICAL_GROUPS', value_name='N_CASES')
    
    df["YEAR"] = pd.DatetimeIndex(df[cols_for_alerts[0]]).year
    # Définissez les conditions et les valeurs correspondantes
    df["COMBINATIONS"] = (
        df[cols_for_alerts[1]] + " | " + df[cols_for_alerts[2]] + " | " + df["MEDICAL_GROUPS"]
    )

    # create smoothed data
    df_smoothed = smooth_n_cases(df, cols_for_alerts+["MEDICAL_GROUPS"], "N_CASES", window_size=2)
    # cols_to_keep = cols_for_alerts+ ["MEDICAL_GROUPS", "COMBINATIONS", "N_CASES", "SMOOTHED_cases"]
    # # keep only the columns to keep : cols_to_keep is a list of columns to keep. Avoid error "TypeError: unhashable type: 'list'"
    # df_smoothed = df_smoothed[cols_to_keep]
    df_baseline = smooth_n_cases(df, cols_for_alerts+["MEDICAL_GROUPS"], "N_CASES", window_size=6)

    # create baseline data
    df_final = baseline_n_cases(df_smoothed, df_baseline, cols_for_alerts+["MEDICAL_GROUPS"], "N_CASES")


    #####################
    ### Treat relative data ###
    df_rel = pd.melt(df_rel, id_vars=cols_for_alerts, var_name='MEDICAL_GROUPS', value_name='CASES_INCIDENCE')
    df_rel["YEAR"] = pd.DatetimeIndex(df_rel[cols_for_alerts[0]]).year
    # Définissez les conditions et les valeurs correspondantes
    df_rel["COMBINATIONS"] = (
        df_rel[cols_for_alerts[1]] + " | " + df_rel[cols_for_alerts[2]] + " | " + df_rel["MEDICAL_GROUPS"]
    )

    # create smoothed data
    df_rel_smoothed = smooth_n_cases(df_rel, cols_for_alerts+["MEDICAL_GROUPS"], "CASES_INCIDENCE", window_size=2, upper_clip=True)
    # cols_to_keep = cols_for_alerts+ ["MEDICAL_GROUPS", "COMBINATIONS", "CASES_INCIDENCE", "SMOOTHED_cases"]
    # df_rel_smoothed = df_rel_smoothed[cols_to_keep]
    df_rel_baseline = smooth_n_cases(df_rel, cols_for_alerts+["MEDICAL_GROUPS"], "CASES_INCIDENCE", window_size=6, upper_clip=True)

   # create baseline data
    df_rel_final = baseline_n_cases(df_rel_smoothed, df_rel_baseline, cols_for_alerts+["MEDICAL_GROUPS"], "CASES_INCIDENCE")



    ########################
    ### Treat ratio data ###

    df_ratio_all = pd.melt(df_ratio_all, id_vars=cols_for_alerts, var_name='MEDICAL_GROUPS', value_name='RATIO_ALL')
    df_ratio_all["YEAR"] = pd.DatetimeIndex(df_ratio_all[cols_for_alerts[0]]).year
    # Définissez les conditions et les valeurs correspondantes
    df_ratio_all["COMBINATIONS"] = (
        df_ratio_all[cols_for_alerts[1]] + " | " + df_ratio_all[cols_for_alerts[2]] + " | " + df_ratio_all["MEDICAL_GROUPS"]
    )

    # # create smoothed data
    # df_ratio_all = smooth_n_cases(df_ratio_all, cols_for_alerts+["MEDICAL_GROUPS"], "RATIO_ALL", upper_clip=True)

    # # create baseline data
    # df_ratio_all = baseline_n_cases(df_ratio_all, cols_for_alerts+["MEDICAL_GROUPS"], "RATIO_ALL")

    # create smoothed data
    df_ratio_all_smoothed = smooth_n_cases(df_ratio_all, cols_for_alerts+["MEDICAL_GROUPS"], "RATIO_ALL", window_size=2, upper_clip=True)
    # cols_to_keep = cols_for_alerts+ ["MEDICAL_GROUPS", "COMBINATIONS", "RATIO_ALL", "SMOOTHED_cases"]
    # df_ratio_all_smoothed = df_ratio_all_smoothed[cols_to_keep]
    df_ratio_all_baseline = smooth_n_cases(df_ratio_all, cols_for_alerts+["MEDICAL_GROUPS"], "RATIO_ALL", window_size=6, upper_clip=True)

   # create baseline data
    df_ratio_all_final = baseline_n_cases(df_ratio_all_smoothed, df_ratio_all_baseline, cols_for_alerts+["MEDICAL_GROUPS"], "RATIO_ALL")

    
    return df_final, df_rel_final, df_ratio_all_final