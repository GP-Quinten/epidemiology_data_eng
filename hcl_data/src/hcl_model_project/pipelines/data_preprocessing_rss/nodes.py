from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import convolve
from scipy.signal.windows import gaussian


def upload_data(df):
    return df


def update_dataframes_SEMX_A_X(
    HCL_update: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform data preprocessing and concatenation on multiple DataFrames.
    """
    for key, df in HCL_update.items():
        print(f"Nom du DataFrame : {key}")
    print("Concatenate all dataframes: Loading...")

    dataframes_to_concat = []  # List to store DataFrames
    dataframes_to_concat_death = []  # List to store DataFrames with _II
    dataframes_to_concat_critical_health = []  # List to store DataFrames with _III
    df = []

    for key, df in HCL_update.items():
        # Find the position of "SEM" in the DataFrame name
        if "- I_" in key or "- I " in key:
            # print(key)
            dataframes_to_concat.append(df)
            # Check if the DataFrame name ends with '_II'
        elif "- II_" in key or "- II " in key:
            dataframes_to_concat_death.append(df)
            # Check if the DataFrame name ends with '_III'
        elif "- III_" in key or "- III " in key:
            dataframes_to_concat_critical_health.append(df)
    df_number_of_id = pd.concat(dataframes_to_concat, ignore_index=True)
    df_death = pd.concat(dataframes_to_concat_death, ignore_index=True)
    df_critical_health = pd.concat(
        dataframes_to_concat_critical_health, ignore_index=True
    )

    # Delete useless columns
    df_number_of_id = df_number_of_id.drop(
        ["ANNEE_SEMAINE_ENTREE_RSS", "LIB_SEM"], axis=1
    )
    df_death = df_death.drop(["ANNEE_SEMAINE_ENTREE_RSS", "LIB_SEM"], axis=1)
    df_critical_health = df_critical_health.drop(
        ["ANNEE_SEMAINE_ENTREE_RSS", "LIB_SEM"], axis=1
    )

    # Delete rows with NaN values in LIB_ANNEE_SEMAINE_ENTREE_RSS
    df_number_of_id = df_number_of_id.dropna(subset=["LIB_ANNEE_SEMAINE_ENTREE_RSS"])
    df_death = df_death.dropna(subset=["LIB_ANNEE_SEMAINE_ENTREE_RSS"])
    df_critical_health = df_critical_health.dropna(
        subset=["LIB_ANNEE_SEMAINE_ENTREE_RSS"]
    )

    df_number_of_id.rename(
        columns={"LIB_ANNEE_SEMAINE_ENTREE_RSS": "LIB_SEM"}, inplace=True
    )
    df_death.rename(columns={"LIB_ANNEE_SEMAINE_ENTREE_RSS": "LIB_SEM"}, inplace=True)
    df_critical_health.rename(
        columns={"LIB_ANNEE_SEMAINE_ENTREE_RSS": "LIB_SEM"}, inplace=True
    )

    # # Extract the date from the LIB_ANNEE_SEMAINE_ENTREE_RSS column using a regular expression
    df_number_of_id["LIB_SEM"] = df_number_of_id["LIB_SEM"].str.extract(
        r"Du (\d{2}/\d{2}/\d{2})"
    )
    df_death["LIB_SEM"] = df_death["LIB_SEM"].str.extract(r"Du (\d{2}/\d{2}/\d{2})")
    df_critical_health["LIB_SEM"] = df_critical_health["LIB_SEM"].str.extract(
        r"Du (\d{2}/\d{2}/\d{2})"
    )

    # Drop duplicate rows, keeping only the occurrence where DATE_EXTRACT is the most recent (thus accurate) for each unique combination of values
    df_number_of_id["DATE_EXTRACT"] = pd.to_datetime(df_number_of_id["DATE_EXTRACT"])
    df_number_of_id = df_number_of_id.sort_values("DATE_EXTRACT", ascending=False)
    df_number_of_id = df_number_of_id.drop_duplicates(
        ["CATEG_DIAG", "LIB_SEM", "CLAS_AGE", "CLAS_DUREERSS"]
    )

    df_death["DATE_EXTRACT"] = pd.to_datetime(df_death["DATE_EXTRACT"])
    df_death = df_death.sort_values("DATE_EXTRACT", ascending=False)
    df_death = df_death.drop_duplicates(
        ["CATEG_DIAG", "LIB_SEM", "DECES OU PAS (oui/non)"]
    )

    df_critical_health["DATE_EXTRACT"] = pd.to_datetime(
        df_critical_health["DATE_EXTRACT"]
    )
    df_critical_health = df_critical_health.sort_values("DATE_EXTRACT", ascending=False)
    df_critical_health = df_critical_health.drop_duplicates(
        ["CATEG_DIAG", "LIB_SEM", "ORIG_GEO", "SOINS CRITIQUES OU PAS (Oui/Non)"]
    )

    return df_number_of_id, df_death, df_critical_health


def translate_dataframes(df_number_of_id, df_death, df_critical_health):
    # Renommer les colonnes pour chaque DataFrame
    df_number_of_id_translate = df_number_of_id.rename(
        columns={
            "DATE_EXTRACT": "EXTRACTION_DATE",
            "CATEG_DIAG": "DIAGNOSIS_CATEGORY",
            "ANNEE_ENTREE_RSS": "ADMISSION_YEAR_RSS",
            "NUM_SEMAINE": "WEEK_NUMBER",
            "LIB_SEM": "WEEK_LABEL",
            "CLAS_AGE": "AGE_CLASS",
            "CLAS_DUREERSS": "RSS_DURATION_CLASS",
            "NRSS": "TOTAL_RSS",
            "NIPP": "NIPP",
        }
    )

    df_death_translate = df_death.rename(
        columns={
            "DATE_EXTRACT": "EXTRACTION_DATE",
            "CATEG_DIAG": "DIAGNOSIS_CATEGORY",
            "ANNEE_ENTREE_RSS": "ADMISSION_YEAR_RSS",
            "NUM_SEMAINE": "WEEK_NUMBER",
            "LIB_SEM": "WEEK_LABEL",
            "DECES OU PAS (oui/non)": "DEATH_OR_NOT",
            "NRSS": "TOTAL_RSS",
            "NIPP": "NIPP",
        }
    )

    df_critical_health_translate = df_critical_health.rename(
        columns={
            "DATE_EXTRACT": "EXTRACTION_DATE",
            "CATEG_DIAG": "DIAGNOSIS_CATEGORY",
            "ANNEE_ENTREE_RSS": "ADMISSION_YEAR_RSS",
            "NUM_SEMAINE": "WEEK_NUMBER",
            "LIB_SEM": "WEEK_LABEL",
            "ORIG_GEO": "GEOGRAPHICAL_ORIGIN",
            "SOINS CRITIQUES OU PAS (Oui/Non)": "CRITICAL_CARE_OR_NOT",
            "NRSS": "TOTAL_RSS",
            "NIPP": "NIPP",
        }
    )

    # Translation for CRITICAL_CARE_OR_NOT column
    yes_no_translation = {"Oui": "Yes", "Non": "No"}
    df_critical_health_translate["CRITICAL_CARE_OR_NOT"] = df_critical_health_translate[
        "CRITICAL_CARE_OR_NOT"
    ].replace(yes_no_translation)
    df_death_translate["DEATH_OR_NOT"] = df_death_translate["DEATH_OR_NOT"].replace(
        yes_no_translation
    )

    # Translation for GEOGRAPHICAL_ORIGIN column
    geographical_translation = {
        "LYON": "LYON",
        "RHONE (HORS LYON)": "RHONE (OUTSIDE LYON)",
        "REGION ARA (HORS RHONE)": "AUVERGNE-RHONE-ALPES (OUTSIDE RHONE)",
        "FRANCE (HORS ARA)": "FRANCE (OUTSIDE AUVERGNE-RHONE-ALPES)",
        "AUTRES": "OTHER",
    }
    df_critical_health_translate["GEOGRAPHICAL_ORIGIN"] = df_critical_health_translate[
        "GEOGRAPHICAL_ORIGIN"
    ].replace(geographical_translation)

    # Translation for DIAGNOSIS_CATEGORY column
    diagnosis_translation = {
        "COVID_19": "COVID_19",
        "GRIPPE": "FLU",
        "IR_AUTVIRUS": "RI_OTHER_VIRUS",
        "IR_GENERALE": "GENERAL_RI",
        "RSV": "RSV",
    }
    df_number_of_id_translate["DIAGNOSIS_CATEGORY"] = df_number_of_id_translate[
        "DIAGNOSIS_CATEGORY"
    ].replace(diagnosis_translation)
    df_death_translate["DIAGNOSIS_CATEGORY"] = df_death_translate[
        "DIAGNOSIS_CATEGORY"
    ].replace(diagnosis_translation)
    df_critical_health_translate["DIAGNOSIS_CATEGORY"] = df_critical_health_translate[
        "DIAGNOSIS_CATEGORY"
    ].replace(diagnosis_translation)

    # Translation for AGE_CLASS column
    age_class_translation = {
        "Moins de 1 an": "Less than 1 year",
        "[1 - 5[ an(s)": "[1 - 5[ year(s)",
        "[5 - 20[ ans": "[5 - 20[ years",
        "[20 - 50[ ans": "[20 - 50[ years",
        "[50 - 65[ ans": "[50 - 65[ years",
        "65 ans et plus": "65 years and older",
    }
    df_number_of_id_translate["AGE_CLASS"] = df_number_of_id_translate[
        "AGE_CLASS"
    ].replace(age_class_translation)

    # Translation for RSS_DURATION_CLASS column
    rss_duration_translation = {
        "< 2 jours": "Less than 2 days",
        "[2 - 5[ jours": "[2 - 5[ days",
        "5 jours et plus": "5 days and more",
    }
    df_number_of_id_translate["RSS_DURATION_CLASS"] = df_number_of_id_translate[
        "RSS_DURATION_CLASS"
    ].replace(rss_duration_translation)

    return df_number_of_id_translate, df_death_translate, df_critical_health_translate


def create_non_viral_respiratory_category(df):
    """
    Create a new category 'RI_NO_VIRAL' by subtracting TOTAL_RSS values of 'GENERAL_RI' from all other categories.

    Args:
        df (DataFrame): Input DataFrame containing diagnosis categories and TOTAL_RSS values.

    Returns:
        DataFrame: Modified DataFrame with the new category and adjusted TOTAL_RSS values.
    """
    # Drop unnecessary columns
    df = df.drop(["NIPP"], axis=1)

    # Select columns for grouping, excluding specified columns
    groupby_columns = [
        col for col in df.columns if col not in ["DIAGNOSIS_CATEGORY", "TOTAL_RSS"]
    ]

    # Create a DataFrame with TOTAL_RSS values for 'GENERAL_RI' for each combination
    TOTAL_RSS_general_ri = (
        df[df["DIAGNOSIS_CATEGORY"] == "GENERAL_RI"]
        .groupby(groupby_columns)["TOTAL_RSS"]
        .sum()
        .reset_index()
    )

    # Create a DataFrame with TOTAL_RSS values for non-'GENERAL_RI' categories for each combination
    TOTAL_RSS_non_general_ri = (
        df[df["DIAGNOSIS_CATEGORY"] != "GENERAL_RI"]
        .groupby(groupby_columns)["TOTAL_RSS"]
        .sum()
        .reset_index()
    )

    # Define new column names
    new_diag_column = "DIAGNOSIS_CATEGORY"
    new_TOTAL_RSS_column = "TOTAL_RSS"

    # Rename columns
    TOTAL_RSS_general_ri.columns = groupby_columns + [new_TOTAL_RSS_column]
    TOTAL_RSS_non_general_ri.columns = groupby_columns + [
        new_TOTAL_RSS_column + "_NON_VIRAL"
    ]

    # Merge DataFrames
    result = TOTAL_RSS_general_ri.merge(
        TOTAL_RSS_non_general_ri, on=groupby_columns, how="left"
    ).fillna(0)

    # Calculate the new TOTAL_RSS column
    result["TOTAL_RSS"] = (
        result[new_TOTAL_RSS_column] - result[new_TOTAL_RSS_column + "_NON_VIRAL"]
    )

    # Add the new diagnosis category column
    result[new_diag_column] = "RI_NO_VIRAL"

    # Concatenate with the original DataFrame
    df = pd.concat(
        [df, result.drop([new_TOTAL_RSS_column + "_NON_VIRAL"], axis=1)],
        ignore_index=True,
    )

    column_order = ["WEEK_NUMBER", "WEEK_LABEL"] + [
        col for col in df.columns if col not in ["WEEK_NUMBER", "WEEK_LABEL"]
    ]
    df = df[column_order]

    df = df.sort_values(by=["WEEK_NUMBER", "WEEK_LABEL"])

    df = df.reset_index(drop=True)

    return df


def add_new_category_to_dataframes(
    df_number_of_id_translate: pd.DataFrame,
    df_death_translate: pd.DataFrame,
    df_critical_health_translate: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add a new category 'IR_NON_VIRAL' to the input dataframes and return the modified dataframes.

    Args:
        df_number_of_id_translate (pd.DataFrame): DataFrame for 'df_number_of_id'.
        df_death_translate (pd.DataFrame): DataFrame for 'df_death'.
        df_critical_health_translate (pd.DataFrame): DataFrame for 'df_critical_health'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the modified dataframes
        for 'df_number_of_id', 'df_death', and 'df_critical_health'.
    """

    df_table_I_newcat = create_non_viral_respiratory_category(df_number_of_id_translate)
    df_table_II_newcat = create_non_viral_respiratory_category(df_death_translate)
    df_table_III_newcat = create_non_viral_respiratory_category(
        df_critical_health_translate
    )

    return df_table_I_newcat, df_table_II_newcat, df_table_III_newcat


def preprocess_table_I(
    df_table_I_newcat: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess 'table_I' from 'df_table_I_newcat'.

    Args:
        df_table_I_newcat (pd.DataFrame): DataFrame for 'df_table_I_newcat'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the original 'table_I' and the
        preprocessed 'table_I' with renamed columns and 'N_CASES' rounded and casted to int.
    """

    table_I = df_table_I_newcat[
        [
            "WEEK_LABEL",
            "DIAGNOSIS_CATEGORY",
            "AGE_CLASS",
            "RSS_DURATION_CLASS",
            "TOTAL_RSS",
        ]
    ]
    table_I = table_I.rename(columns={"TOTAL_RSS": "N_CASES"})
    table_I["N_CASES"] = table_I["N_CASES"].round().astype(int)

    # create a new dataframe that groups the data by WEEK_LABEL, DIAGNOSIS_CATEGORY, AGE_CLASS. Sums the TOTAL_RSS values and resets the index
    sum_RSS_duration = (
        table_I.groupby(["WEEK_LABEL", "DIAGNOSIS_CATEGORY", "AGE_CLASS"])["N_CASES"]
        .sum()
        .reset_index()
    )
    # add column RSS_DURATION_CLASS to the sum_RSS_duration dataframe with value "all"
    sum_RSS_duration.insert(3, "RSS_DURATION_CLASS", "all_durations")
    # concatenate the sum_RSS_duration dataframe with the table_I dataframe, on axis 0
    table_I = pd.concat([table_I, sum_RSS_duration], axis=0)

    # create a new dataframe that groups the data by WEEK_LABEL, DIAGNOSIS_CATEGORY, RSS_DURATION_CLASS. Sums the TOTAL_RSS values and resets the index
    sum_AGE_class = (
        table_I.groupby(["WEEK_LABEL", "DIAGNOSIS_CATEGORY", "RSS_DURATION_CLASS"])[
            "N_CASES"
        ]
        .sum()
        .reset_index()
    )
    # add column AGE_CLASS to the sum_AGE_class dataframe with value "all"
    sum_AGE_class.insert(2, "AGE_CLASS", "0-200")
    # concatenate the sum_AGE_class dataframe with the table_I dataframe, on axis 0
    table_I = pd.concat([table_I, sum_AGE_class], axis=0)

    # create a new dataframe that groups the data by WEEK_LABEL, RSS_DURATION_CLASS, AGE_CLASS. Sums the TOTAL_RSS values and resets the index
    sum_DIAGNOSIS_CATEGORY = (
        table_I.groupby(["WEEK_LABEL", "AGE_CLASS", "RSS_DURATION_CLASS"])["N_CASES"]
        .sum()
        .reset_index()
    )
    # add column DIAGNOSIS_CATEGORY to the sum_DIAGNOSIS_CATEGORY dataframe with value "all"
    sum_DIAGNOSIS_CATEGORY.insert(1, "DIAGNOSIS_CATEGORY", "all_resp_diags")
    # concatenate the sum_DIAGNOSIS_CATEGORY dataframe with the table_I dataframe, on axis 0
    table_I = pd.concat([table_I, sum_DIAGNOSIS_CATEGORY], axis=0)



    return table_I


def preprocess_table_II(df_table_II_newcat: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Preprocess 'table_II' from 'df_table_II_newcat'.

    Args:
        df_table_II_newcat (pd.DataFrame): DataFrame for 'df_table_II_newcat'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the original 'df_table_II_newcat' and the
        preprocessed 'table_II' with the specified operations.
    """

    df_death_renamed = df_table_II_newcat[
        ["WEEK_LABEL", "DIAGNOSIS_CATEGORY", "DEATH_OR_NOT", "TOTAL_RSS"]
    ]

    sum_TOTAL_RSS = (
        df_death_renamed.groupby(["WEEK_LABEL", "DIAGNOSIS_CATEGORY"])["TOTAL_RSS"]
        .sum()
        .reset_index()
    )
    sum_TOTAL_RSS = sum_TOTAL_RSS.sort_values(by=["WEEK_LABEL", "DIAGNOSIS_CATEGORY"])

    df_death_filtered = df_death_renamed[
        df_death_renamed["DEATH_OR_NOT"] == "Yes"
    ].drop("DEATH_OR_NOT", axis=1)

    table_II = df_death_filtered.merge(
        sum_TOTAL_RSS, on=["WEEK_LABEL", "DIAGNOSIS_CATEGORY"], suffixes=("", "_SUM")
    )
    table_II = table_II.rename(columns={"TOTAL_RSS": "N_DEATH"})
    table_II = table_II.rename(columns={"TOTAL_RSS_SUM": "N_CASES"})
    table_II["N_DEATH"] = table_II["N_DEATH"].round().astype(int)
    table_II["N_CASES"] = table_II["N_CASES"].round().astype(int)

    # create a new dataframe that groups the data by WEEK_LABEL. Sums the N_DEATH, and N_CASES and  values and resets the index
    sum_DIAGNOSIS_CATEGORY = (
        table_II.groupby(["WEEK_LABEL"])[["N_DEATH", "N_CASES"]]
        .sum()
        .reset_index()
    )
    # add column DIAGNOSIS_CATEGORY to the sum_DIAGNOSIS_CATEGORY dataframe with value "all"
    sum_DIAGNOSIS_CATEGORY.insert(1, "DIAGNOSIS_CATEGORY", "all_resp_diags")
    # concatenate the sum_DIAGNOSIS_CATEGORY dataframe with the table_II dataframe, on axis 0
    table_II = pd.concat([table_II, sum_DIAGNOSIS_CATEGORY], axis=0)

    return table_II


def preprocess_table_III(df_table_III_newcat: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Preprocess 'table_III' from 'df_table_III_newcat'.

    Args:
        df_table_III_newcat (pd.DataFrame): DataFrame for 'df_table_III_newcat'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the original 'df_table_III_newcat' and the
        preprocessed 'table_III' with the specified operations.
    """

    df_critical_health_renamed = df_table_III_newcat[
        [
            "WEEK_LABEL",
            "DIAGNOSIS_CATEGORY",
            "CRITICAL_CARE_OR_NOT",
            "GEOGRAPHICAL_ORIGIN",
            "TOTAL_RSS",
        ]
    ]

    sum_TOTAL_RSS = (
        df_critical_health_renamed.groupby(["WEEK_LABEL", "DIAGNOSIS_CATEGORY", "GEOGRAPHICAL_ORIGIN"])[
            "TOTAL_RSS"
        ]
        .sum()
        .reset_index()
    )
    sum_TOTAL_RSS = sum_TOTAL_RSS.sort_values(by=["WEEK_LABEL", "DIAGNOSIS_CATEGORY", "GEOGRAPHICAL_ORIGIN"])

    df_critical_health_filtered = df_critical_health_renamed[
        df_critical_health_renamed["CRITICAL_CARE_OR_NOT"] == "Yes"
    ].drop("CRITICAL_CARE_OR_NOT", axis=1)

    sum_critical_health_TOTAL_RSS = (
        df_critical_health_filtered.groupby(["WEEK_LABEL", "DIAGNOSIS_CATEGORY", "GEOGRAPHICAL_ORIGIN"])[
            "TOTAL_RSS"
        ]
        .sum()
        .reset_index()
    )

    table_III = sum_critical_health_TOTAL_RSS.merge(
        sum_TOTAL_RSS,
        on=["WEEK_LABEL", "DIAGNOSIS_CATEGORY", "GEOGRAPHICAL_ORIGIN"],
        suffixes=("", "_SUM"),
    )
    table_III = table_III.rename(columns={"TOTAL_RSS": "N_CRITICAL"})
    table_III = table_III.rename(columns={"TOTAL_RSS_SUM": "N_CASES"})
    table_III["N_CRITICAL"] = table_III["N_CRITICAL"].round().astype(int)
    table_III["N_CASES"] = table_III["N_CASES"].round().astype(int)

    # create a new dataframe that groups the data by WEEK_LABEL, GEOGRAPHICAL_ORIGIN. Sums the N_CRITICAL, and N_CASES values and resets the index
    sum_DIAGNOSIS_CATEGORY = (
        table_III.groupby(["WEEK_LABEL", "GEOGRAPHICAL_ORIGIN"])[
            ["N_CRITICAL", "N_CASES"]
        ]
        .sum()
        .reset_index()
    )
    # add column DIAGNOSIS_CATEGORY to the sum_DIAGNOSIS_CATEGORY dataframe with value "all"
    sum_DIAGNOSIS_CATEGORY.insert(1, "DIAGNOSIS_CATEGORY", "all")
    # concatenate the sum_DIAGNOSIS_CATEGORY dataframe with the table_III dataframe, on axis 0
    table_III = pd.concat([table_III, sum_DIAGNOSIS_CATEGORY], axis=0)

    # create a new dataframe that groups the data by WEEK_LABEL, DIAGNOSIS_CATEGORY. Sums the N_CRITICAL, and N_CASES values and resets the index
    sum_GEOGRAPHICAL_ORIGIN = (
        table_III.groupby(["WEEK_LABEL", "DIAGNOSIS_CATEGORY"])[
            ["N_CRITICAL", "N_CASES"]
        ]
        .sum()
        .reset_index()
    )
    # add column GEOGRAPHICAL_ORIGIN to the sum_GEOGRAPHICAL_ORIGIN dataframe with value "all"
    sum_GEOGRAPHICAL_ORIGIN.insert(2, "GEOGRAPHICAL_ORIGIN", "all")
    # concatenate the sum_GEOGRAPHICAL_ORIGIN dataframe with the table_III dataframe, on axis 0
    table_III = pd.concat([table_III, sum_GEOGRAPHICAL_ORIGIN], axis=0)

    return table_III


def missing_data_prepro_I(result_df: pd.DataFrame) -> pd.DataFrame:
    """ """

    # Définissez les conditions et les valeurs correspondantes
    result_df["COMBI"] = (
        result_df["DIAGNOSIS_CATEGORY"]
        + " | "
        + result_df["AGE_CLASS"]
        + " | "
        + result_df["RSS_DURATION_CLASS"]
    )
    # Créer un graphique en bar plot pour le nombre de valeurs par année par combinaison DIAGNOSIS_CATEGORY et CLASS_AGE
    result_df["WEEK_LABEL"] = pd.to_datetime(result_df["WEEK_LABEL"], format="%d/%m/%y")

    result_df["YEAR"] = result_df["WEEK_LABEL"].dt.year

    result_df = result_df.sort_values(by="COMBI")
    result_df = result_df.drop(columns="COMBI")
    result_df = result_df.drop(columns="YEAR")

    weeks = pd.date_range(
        start=result_df["WEEK_LABEL"].min(),
        end=result_df["WEEK_LABEL"].max(),
        freq="W-Mon",
    )

    categories_age = result_df["AGE_CLASS"].unique()
    diagnosis_categories = result_df["DIAGNOSIS_CATEGORY"].unique()
    rss_duration_categories = result_df["RSS_DURATION_CLASS"].unique()

    combinations = list(
        product(weeks, diagnosis_categories, categories_age, rss_duration_categories)
    )

    columns = ["WEEK_LABEL", "DIAGNOSIS_CATEGORY", "AGE_CLASS", "RSS_DURATION_CLASS"]
    all_combinations_df = pd.DataFrame(combinations, columns=columns)

    merged_df = pd.merge(
        all_combinations_df,
        result_df,
        on=["WEEK_LABEL", "AGE_CLASS", "RSS_DURATION_CLASS", "DIAGNOSIS_CATEGORY"],
        how="outer",
    )

    merged_df = merged_df.sort_values(by="WEEK_LABEL")

    merged_df["N_CASES"] = merged_df["N_CASES"].fillna(0)

    return merged_df


def missing_data_prepro_II(result_df: pd.DataFrame) -> pd.DataFrame:
    """ """
    # Définissez les conditions et les valeurs correspondantes
    result_df["COMBI"] = result_df["DIAGNOSIS_CATEGORY"]

    # Créer un graphique en bar plot pour le nombre de valeurs par année par combinaison DIAGNOSIS_CATEGORY et CLASS_AGE
    result_df["WEEK_LABEL"] = pd.to_datetime(result_df["WEEK_LABEL"], format="%d/%m/%y")
    result_df["YEAR"] = result_df["WEEK_LABEL"].dt.year

    result_df = result_df.sort_values(by="COMBI")
    result_df = result_df.drop(columns="COMBI")
    result_df = result_df.drop(columns="YEAR")

    weeks = pd.date_range(
        start=result_df["WEEK_LABEL"].min(),
        end=result_df["WEEK_LABEL"].max(),
        freq="W-Mon",
    )

    diagnosis_categories = result_df["DIAGNOSIS_CATEGORY"].unique()

    combinations = list(product(weeks, diagnosis_categories))

    columns = ["WEEK_LABEL", "DIAGNOSIS_CATEGORY"]
    all_combinations_df = pd.DataFrame(combinations, columns=columns)

    merged_df = pd.merge(
        all_combinations_df,
        result_df,
        on=["WEEK_LABEL", "DIAGNOSIS_CATEGORY"],
        how="outer",
    )

    merged_df = merged_df.sort_values(by="WEEK_LABEL")

    merged_df["N_DEATH"] = merged_df["N_DEATH"].fillna(0)

    merged_df["N_CASES"] = merged_df["N_CASES"].fillna(0)

    return merged_df


def missing_data_prepro_III(result_df: pd.DataFrame) -> pd.DataFrame:

    # Définissez les conditions et les valeurs correspondantes
    result_df["COMBI"] = result_df["DIAGNOSIS_CATEGORY"]

    # Créer un graphique en bar plot pour le nombre de valeurs par année par combinaison DIAGNOSIS_CATEGORY et CLASS_AGE
    result_df["WEEK_LABEL"] = pd.to_datetime(result_df["WEEK_LABEL"], format="%d/%m/%y")
    result_df["YEAR"] = result_df["WEEK_LABEL"].dt.year

    result_df = result_df.sort_values(by="COMBI")
    result_df = result_df.drop(columns="COMBI")
    result_df = result_df.drop(columns="YEAR")

    weeks = pd.date_range(
        start=result_df["WEEK_LABEL"].min(),
        end=result_df["WEEK_LABEL"].max(),
        freq="W-Mon",
    )

    diagnosis_categories = result_df["DIAGNOSIS_CATEGORY"].unique()

    combinations = list(product(weeks, diagnosis_categories))

    columns = ["WEEK_LABEL", "DIAGNOSIS_CATEGORY"]
    all_combinations_df = pd.DataFrame(combinations, columns=columns)

    merged_df = pd.merge(
        all_combinations_df,
        result_df,
        on=["WEEK_LABEL", "DIAGNOSIS_CATEGORY"],
        how="outer",
    )

    merged_df = merged_df.sort_values(by="WEEK_LABEL")

    merged_df["N_CRITICAL"] = merged_df["N_CRITICAL"].fillna(0)

    merged_df["N_CASES"] = merged_df["N_CASES"].fillna(0)

    return merged_df


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

    merged_df["COMBI"] = (
        merged_df["DIAGNOSIS_CATEGORY"]
        + " | "
        + merged_df["AGE_CLASS"]
        + " | "
        + merged_df["RSS_DURATION_CLASS"]
    )
    unique_combinations = merged_df["COMBI"].unique()

    # List to store historical dataframes for each combination
    df_smoothed_dataframes = []

    # Set up the number of rows and columns for subplots
    num_rows = len(unique_combinations) // 2 + len(unique_combinations) % 2
    num_cols = 2

    for i, combo in enumerate(unique_combinations):

        filtered_df = merged_df.loc[merged_df["COMBI"] == combo]
        filtered_df = filtered_df.sort_values(by="WEEK_LABEL")
        filtered_df["N_CASES"] = filtered_df["N_CASES"].astype(int)
        # filtered_df['WEEK_LABEL'] = pd.to_datetime(filtered_df['WEEK_LABEL'])

        # Copy the filtered dataframe to avoid modifying the original
        smoothed_dataframes = filtered_df.copy()

        # Ensure 'WEEK_LABEL' is of type datetime
        smoothed_dataframes["WEEK_LABEL"] = pd.to_datetime(
            smoothed_dataframes["WEEK_LABEL"]
        )

        # Sort the DataFrame by date
        smoothed_dataframes = smoothed_dataframes.sort_values(by="WEEK_LABEL")

        # Define the Gaussian window
        window_size = 5
        STD = 4
        gaussian_window = gaussian(window_size, std=STD)
        gaussian_window /= gaussian_window.sum()

        # Apply convolution with 'same' option
        smoothed = convolve(
            smoothed_dataframes["N_CASES"].values, gaussian_window, mode="same"
        )

        # Add the 'N_CASES_SMOOTH' column to the DataFrame
        smoothed_dataframes["N_CASES_SMOOTH"] = smoothed

        # Calculate variance for confidence interval
        variance = convolve(
            (smoothed_dataframes["N_CASES"].values - smoothed) ** 2,
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
        smoothed_dataframes["LOWER_CI_N_CASES_SMOOTH"] = (
            smoothed_dataframes["N_CASES_SMOOTH"] - margin_of_error
        )
        smoothed_dataframes["UPPER_CI_N_CASES_SMOOTH"] = (
            smoothed_dataframes["N_CASES_SMOOTH"] + margin_of_error
        )

        # Sort the DataFrame by date
        smoothed_dataframes.sort_values(by="WEEK_LABEL", inplace=True)

        # Check and replace NaN values with 0 in the columns used
        smoothed_dataframes["LOWER_CI_N_CASES_SMOOTH"] = smoothed_dataframes[
            "LOWER_CI_N_CASES_SMOOTH"
        ].clip(lower=0)

        # Check for NaN values in the confidence interval columns
        lower_ci_smoothed = np.where(
            np.isnan(smoothed_dataframes["LOWER_CI_N_CASES_SMOOTH"]),
            0,
            smoothed_dataframes["LOWER_CI_N_CASES_SMOOTH"],
        )
        upper_ci_smoothed = np.where(
            np.isnan(smoothed_dataframes["UPPER_CI_N_CASES_SMOOTH"]),
            0,
            smoothed_dataframes["UPPER_CI_N_CASES_SMOOTH"],
        )

        # Append the smoothed dataframe to the list
        df_smoothed_dataframes.append(smoothed_dataframes)

    # Concatenate all historical dataframes in the list
    final_smoothed_dataframe = pd.concat(df_smoothed_dataframes, ignore_index=True)

    return final_smoothed_dataframe


def smooth_n_cases_n_death(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Smooth the 'N_CASES' and 'N_DEATH' columns in the DataFrame.

    Args:
        merged_df (pd.DataFrame): DataFrame containing the data to be smoothed.

    Returns:
        pd.DataFrame: DataFrame with smoothed data.
    """
    # Create a combination column
    merged_df["COMBI"] = merged_df["DIAGNOSIS_CATEGORY"]
    # Get unique combinations
    unique_combinations = merged_df["COMBI"].unique()

    # List to store smoothed dataframes for each combination
    df_smoothed_dataframes = []

    for i, combo in enumerate(unique_combinations):
        filtered_df = merged_df.loc[merged_df["COMBI"] == combo].copy()

        # Ensure 'WEEK_LABEL' is of type datetime
        filtered_df["WEEK_LABEL"] = pd.to_datetime(filtered_df["WEEK_LABEL"])

        # Sort the DataFrame by date
        filtered_df = filtered_df.sort_values(by="WEEK_LABEL")

        # Define the Gaussian window
        window_size = 10
        STD = 4
        gaussian_window = gaussian(window_size, std=STD)
        gaussian_window /= gaussian_window.sum()

        # Apply convolution with 'same' option for N_CASES
        smoothed_cases = convolve(
            filtered_df["N_CASES"].values, gaussian_window, mode="same"
        )
        # Add the 'N_CASES_SMOOTH' column to the DataFrame
        filtered_df["N_CASES_SMOOTH"] = smoothed_cases

        # Calculate variance for confidence interval for N_CASES
        variance_cases = convolve(
            (filtered_df["N_CASES"].values - smoothed_cases) ** 2,
            gaussian_window,
            mode="same",
        )
        std_dev_cases = variance_cases**0.5
        margin_of_error_cases = std_dev_cases * 1.96  # For a 95% confidence interval
        filtered_df["LOWER_CI_N_CASES_SMOOTH"] = smoothed_cases - margin_of_error_cases
        filtered_df["UPPER_CI_N_CASES_SMOOTH"] = smoothed_cases + margin_of_error_cases
        filtered_df["LOWER_CI_N_CASES_SMOOTH"] = np.where(
            filtered_df["LOWER_CI_N_CASES_SMOOTH"] < 0,
            0,
            filtered_df["LOWER_CI_N_CASES_SMOOTH"],
        )

        # Apply convolution with 'same' option for N_DEATH
        smoothed_deaths = convolve(
            filtered_df["N_DEATH"].values, gaussian_window, mode="same"
        )
        # Add the 'N_DEATH_SMOOTH' column to the DataFrame
        filtered_df["N_DEATH_SMOOTH"] = smoothed_deaths
        # Calculate variance for confidence interval for N_DEATH
        variance_deaths = convolve(
            (filtered_df["N_DEATH"].values - smoothed_deaths) ** 2,
            gaussian_window,
            mode="same",
        )
        std_dev_deaths = variance_deaths**0.5
        margin_of_error_deaths = std_dev_deaths * 1.96  # For a 95% confidence interval
        filtered_df["LOWER_CI_N_DEATH_SMOOTH"] = (
            smoothed_deaths - margin_of_error_deaths
        )
        filtered_df["UPPER_CI_N_DEATH_SMOOTH"] = (
            smoothed_deaths + margin_of_error_deaths
        )
        filtered_df["LOWER_CI_N_DEATH_SMOOTH"] = np.where(
            filtered_df["LOWER_CI_N_DEATH_SMOOTH"] < 0,
            0,
            filtered_df["LOWER_CI_N_DEATH_SMOOTH"],
        )
        # Sort the DataFrame by date
        filtered_df = filtered_df.sort_values(by="WEEK_LABEL")

        # Append the smoothed dataframe to the list
        df_smoothed_dataframes.append(filtered_df)

    # Concatenate all historical dataframes in the list
    final_smoothed_dataframe = pd.concat(df_smoothed_dataframes, ignore_index=True)

    return final_smoothed_dataframe


def smooth_n_cases_n_critical(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Smooth the 'N_CASES' and 'N_CRITICAL' columns in the DataFrame.

    Args:
        merged_df (pd.DataFrame): DataFrame containing the data to be smoothed.

    Returns:
        pd.DataFrame: DataFrame with smoothed data.
    """
    # Create a combination column
    merged_df["COMBI"] = merged_df["DIAGNOSIS_CATEGORY"]
    # Get unique combinations
    unique_combinations = merged_df["COMBI"].unique()

    # List to store smoothed dataframes for each combination
    df_smoothed_dataframes = []

    for i, combo in enumerate(unique_combinations):
        filtered_df = merged_df.loc[merged_df["COMBI"] == combo].copy()

        # Ensure 'WEEK_LABEL' is of type datetime
        filtered_df["WEEK_LABEL"] = pd.to_datetime(filtered_df["WEEK_LABEL"])

        # Sort the DataFrame by date
        filtered_df = filtered_df.sort_values(by="WEEK_LABEL")

        # Define the Gaussian window
        window_size = 5
        STD = 4
        gaussian_window = gaussian(window_size, std=STD)
        gaussian_window /= gaussian_window.sum()

        # Apply convolution with 'same' option for N_CASES
        smoothed_cases = convolve(
            filtered_df["N_CASES"].values, gaussian_window, mode="same"
        )
        # Add the 'N_CASES_SMOOTH' column to the DataFrame
        filtered_df["N_CASES_SMOOTH"] = smoothed_cases

        # Calculate variance for confidence interval for N_CASES
        variance_cases = convolve(
            (filtered_df["N_CASES"].values - smoothed_cases) ** 2,
            gaussian_window,
            mode="same",
        )
        std_dev_cases = variance_cases**0.5
        margin_of_error_cases = std_dev_cases * 1.96  # For a 95% confidence interval
        filtered_df["LOWER_CI_N_CASES_SMOOTH"] = smoothed_cases - margin_of_error_cases
        filtered_df["UPPER_CI_N_CASES_SMOOTH"] = smoothed_cases + margin_of_error_cases

        filtered_df["LOWER_CI_N_CASES_SMOOTH"] = np.where(
            filtered_df["LOWER_CI_N_CASES_SMOOTH"] < 0,
            0,
            filtered_df["LOWER_CI_N_CASES_SMOOTH"],
        )
        # Apply convolution with 'same' option for N_CRITICAL
        smoothed_critical = convolve(
            filtered_df["N_CRITICAL"].values, gaussian_window, mode="same"
        )
        # Add the 'N_CRITICAL_SMOOTH' column to the DataFrame
        filtered_df["N_CRITICAL_SMOOTH"] = smoothed_critical

        # Calculate variance for confidence interval for N_CRITICAL
        variance_critical = convolve(
            (filtered_df["N_CRITICAL"].values - smoothed_critical) ** 2,
            gaussian_window,
            mode="same",
        )
        std_dev_critical = variance_critical**0.5
        margin_of_error_critical = (
            std_dev_critical * 1.96
        )  # For a 95% confidence interval
        filtered_df["LOWER_CI_N_CRITICAL_SMOOTH"] = (
            smoothed_critical - margin_of_error_critical
        )
        filtered_df["UPPER_CI_N_CRITICAL_SMOOTH"] = (
            smoothed_critical + margin_of_error_critical
        )

        filtered_df["LOWER_CI_N_CRITICAL_SMOOTH"] = np.where(
            filtered_df["LOWER_CI_N_CRITICAL_SMOOTH"] < 0,
            0,
            filtered_df["LOWER_CI_N_CRITICAL_SMOOTH"],
        )
        # Sort the DataFrame by date
        filtered_df = filtered_df.sort_values(by="WEEK_LABEL")

        # Append the smoothed dataframe to the list
        df_smoothed_dataframes.append(filtered_df)

    # Concatenate all historical dataframes in the list
    final_smoothed_dataframe = pd.concat(df_smoothed_dataframes, ignore_index=True)

    return final_smoothed_dataframe


def table_I_baseline_creation(table_I: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess 'table_I' to create the baseline reference.

    Args:
        table_I (pd.DataFrame): DataFrame for 'table_I'.

    Returns:
        pd.DataFrame: The preprocessed 'table_I' with the baseline reference calculated with CI.
    """

    table_I_baseline = table_I.copy()
    table_I_baseline["WEEK_LABEL"] = pd.to_datetime(
        table_I_baseline["WEEK_LABEL"], format="%Y-%m-%d"
    )
    table_I_baseline["MONTH_YEAR"] = table_I_baseline["WEEK_LABEL"].dt.strftime("%m/%y")

    table_I_baseline["BASELINE_N_CASES"] = (
        table_I_baseline.groupby(
            ["MONTH_YEAR", "DIAGNOSIS_CATEGORY", "AGE_CLASS", "RSS_DURATION_CLASS"]
        )["N_CASES"]
        .transform("mean")
        .round()
        .astype(int)
    )
    table_I_baseline = table_I_baseline.drop("MONTH_YEAR", axis=1)

    # Define the confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the critical value using the normal distribution
    z = stats.norm.ppf((1 + confidence_level) / 2)

    grouped = table_I_baseline.groupby(
        ["DIAGNOSIS_CATEGORY", "AGE_CLASS", "RSS_DURATION_CLASS"]
    )

    def calculate_confidence_interval(data):
        std_dev = data["BASELINE_N_CASES"].std()
        lower_ci = data["BASELINE_N_CASES"] - z * std_dev
        upper_ci = data["BASELINE_N_CASES"] + z * std_dev
        data["LOWER_CI_BASELINE_N_CASES"] = lower_ci
        data["UPPER_CI_BASELINE_N_CASES"] = upper_ci
        data["LOWER_CI_BASELINE_N_CASES"] = data["LOWER_CI_BASELINE_N_CASES"].fillna(0)
        data["UPPER_CI_BASELINE_N_CASES"] = data["UPPER_CI_BASELINE_N_CASES"].fillna(0)
        return data

    table_I_baseline = grouped.apply(calculate_confidence_interval).reset_index(
        drop=True
    )
    table_I_baseline["LOWER_CI_BASELINE_N_CASES"] = table_I_baseline[
        "LOWER_CI_BASELINE_N_CASES"
    ].apply(lambda x: max(x, 0))

    return table_I_baseline


def table_II_baseline_creation(table_II: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess 'table_II' to create the baseline reference for deaths.

    Args:
        table_II (pd.DataFrame): DataFrame for 'table_II'.

    Returns:
        pd.DataFrame: The preprocessed 'table_II' with the baseline reference for deaths calculated.
    """

    table_II_baseline = table_II.copy()
    table_II_baseline["WEEK_LABEL"] = pd.to_datetime(
        table_II_baseline["WEEK_LABEL"], format="%Y-%m-%d"
    )
    table_II_baseline["MONTH_YEAR"] = table_II_baseline["WEEK_LABEL"].dt.strftime(
        "%m/%y"
    )

    table_II_baseline["BASELINE_DEATH"] = (
        table_II_baseline.groupby(["MONTH_YEAR", "DIAGNOSIS_CATEGORY"])["N_DEATH"]
        .transform("mean")
        .round()
        .astype(int)
    )
    table_II_baseline["BASELINE_N_CASES"] = (
        table_II_baseline.groupby(["MONTH_YEAR", "DIAGNOSIS_CATEGORY"])["N_CASES"]
        .transform("mean")
        .round()
        .astype(int)
    )
    table_II_baseline = table_II_baseline.drop("MONTH_YEAR", axis=1)

    # Define the confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the critical value using the normal distribution
    z = stats.norm.ppf((1 + confidence_level) / 2)

    grouped = table_II_baseline.groupby("DIAGNOSIS_CATEGORY")

    def calculate_confidence_interval(data):
        std_dev = data["BASELINE_DEATH"].std()
        lower_ci = data["BASELINE_DEATH"] - z * std_dev
        upper_ci = data["BASELINE_DEATH"] + z * std_dev
        data["LOWER_CI_BASELINE_DEATH"] = lower_ci
        data["UPPER_CI_BASELINE_DEATH"] = upper_ci
        data["LOWER_CI_BASELINE_DEATH"] = data["LOWER_CI_BASELINE_DEATH"].fillna(0)
        data["UPPER_CI_BASELINE_DEATH"] = data["UPPER_CI_BASELINE_DEATH"].fillna(0)

        std_dev = data["BASELINE_N_CASES"].std()
        lower_ci = data["BASELINE_N_CASES"] - z * std_dev
        upper_ci = data["BASELINE_N_CASES"] + z * std_dev
        data["LOWER_CI_BASELINE_N_CASES"] = lower_ci
        data["UPPER_CI_BASELINE_N_CASES"] = upper_ci
        data["LOWER_CI_BASELINE_N_CASES"] = data["LOWER_CI_BASELINE_N_CASES"].fillna(0)
        data["UPPER_CI_BASELINE_N_CASES"] = data["UPPER_CI_BASELINE_N_CASES"].fillna(0)

        return data

    table_II_baseline = grouped.apply(calculate_confidence_interval).reset_index(
        drop=True
    )

    # Set negative LOWER_CI values to zero
    table_II_baseline["LOWER_CI_BASELINE_N_CASES"] = table_II_baseline[
        "LOWER_CI_BASELINE_N_CASES"
    ].apply(lambda x: max(x, 0))
    table_II_baseline["LOWER_CI_BASELINE_DEATH"] = table_II_baseline[
        "LOWER_CI_BASELINE_DEATH"
    ].apply(lambda x: max(x, 0))

    return table_II_baseline


def table_III_baseline_creation(table_III: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess 'table_III' to create the baseline reference for critical cases.

    Args:
        table_III (pd.DataFrame): DataFrame for 'table_III'.

    Returns:
        pd.DataFrame: The preprocessed 'table_III' with the baseline reference for critical cases calculated.
    """

    table_III_baseline = table_III.copy()
    table_III_baseline["WEEK_LABEL"] = pd.to_datetime(
        table_III_baseline["WEEK_LABEL"], format="%Y-%m-%d"
    )
    table_III_baseline["MONTH_YEAR"] = table_III_baseline["WEEK_LABEL"].dt.strftime(
        "%m/%y"
    )

    table_III_baseline["BASELINE_CRITICAL"] = (
        table_III_baseline.groupby(
            ["MONTH_YEAR", "DIAGNOSIS_CATEGORY", "GEOGRAPHICAL_ORIGIN"]
        )["N_CRITICAL"]
        .transform("mean")
        .round()
        .astype(int)
    )
    table_III_baseline["BASELINE_N_CASES"] = (
        table_III_baseline.groupby(
            ["MONTH_YEAR", "DIAGNOSIS_CATEGORY", "GEOGRAPHICAL_ORIGIN"]
        )["N_CASES"]
        .transform("mean")
        .round()
        .astype(int)
    )
    table_III_baseline = table_III_baseline.drop("MONTH_YEAR", axis=1)

    # Define the confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the critical value using the normal distribution
    z = stats.norm.ppf((1 + confidence_level) / 2)

    grouped = table_III_baseline.groupby(["DIAGNOSIS_CATEGORY", "GEOGRAPHICAL_ORIGIN"])

    def calculate_confidence_interval(data):
        std_dev = data["BASELINE_N_CASES"].std()
        lower_ci = data["BASELINE_N_CASES"] - z * std_dev
        upper_ci = data["BASELINE_N_CASES"] + z * std_dev
        data["LOWER_CI_BASELINE_N_CASES"] = lower_ci
        data["UPPER_CI_BASELINE_N_CASES"] = upper_ci
        data["LOWER_CI_BASELINE_N_CASES"] = data["LOWER_CI_BASELINE_N_CASES"].fillna(0)
        data["UPPER_CI_BASELINE_N_CASES"] = data["UPPER_CI_BASELINE_N_CASES"].fillna(0)

        std_dev = data["BASELINE_CRITICAL"].std()
        lower_ci = data["BASELINE_CRITICAL"] - z * std_dev
        upper_ci = data["BASELINE_CRITICAL"] + z * std_dev
        data["LOWER_CI_BASELINE_CRITICAL"] = lower_ci
        data["UPPER_CI_BASELINE_CRITICAL"] = upper_ci
        data["LOWER_CI_BASELINE_CRITICAL"] = data["LOWER_CI_BASELINE_CRITICAL"].fillna(
            0
        )
        data["UPPER_CI_BASELINE_CRITICAL"] = data["UPPER_CI_BASELINE_CRITICAL"].fillna(
            0
        )

        return data

    table_III_baseline = grouped.apply(calculate_confidence_interval).reset_index(
        drop=True
    )

    # Set negative LOWER_CI values to zero
    table_III_baseline["LOWER_CI_BASELINE_N_CASES"] = table_III_baseline[
        "LOWER_CI_BASELINE_N_CASES"
    ].apply(lambda x: max(x, 0))
    table_III_baseline["LOWER_CI_BASELINE_CRITICAL"] = table_III_baseline[
        "LOWER_CI_BASELINE_CRITICAL"
    ].apply(lambda x: max(x, 0))

    return table_III_baseline


def baseline_n_cases(final_smoothed_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the baseline number of cases

    Args:
        final_smoothed_dataframe (pd.DataFrame): Processed DataFrame containing smoothed data.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data and calculated baseline number of cases.
    """
    # Unique list of diagnosis and age combinations
    final_smoothed_dataframe["COMBI"] = (
        final_smoothed_dataframe["DIAGNOSIS_CATEGORY"]
        + " | "
        + final_smoothed_dataframe["AGE_CLASS"]
        + " | "
        + final_smoothed_dataframe["RSS_DURATION_CLASS"]
    )
    unique_combinations = final_smoothed_dataframe["COMBI"].unique()

    final_smoothed_dataframe["YEAR"] = final_smoothed_dataframe["WEEK_LABEL"].dt.year

    # List to store resulting DataFrames for each combination
    result_dataframes = []

    # Define one year shift
    one_year_shift = pd.DateOffset(weeks=52)

    # Loop through each combination
    for combo in unique_combinations:
        # Filter DataFrame for current combination
        df = final_smoothed_dataframe[final_smoothed_dataframe["COMBI"] == combo].copy()

        # Add 'WEEK_LABEL_1YR_AGO' column to store corresponding date one year ago
        df["WEEK_LABEL_1YR_AGO"] = df["WEEK_LABEL"] - one_year_shift

        # Merge with itself to get values from the previous year
        merged_df = pd.merge(
            df,
            df[
                [
                    "WEEK_LABEL",
                    "N_CASES_SMOOTH",
                    "LOWER_CI_N_CASES_SMOOTH",
                    "UPPER_CI_N_CASES_SMOOTH",
                ]
            ],
            left_on="WEEK_LABEL_1YR_AGO",
            right_on="WEEK_LABEL",
            suffixes=("", "_1YR_AGO"),
            how="left",
        )

        # Replace missing values with values from the previous year
        merged_df["N_CASES_SMOOTH_1YR_AGO"].fillna(
            merged_df["N_CASES_SMOOTH"], inplace=True
        )
        merged_df["LOWER_CI_N_CASES_SMOOTH_1YR_AGO"].fillna(
            merged_df["LOWER_CI_N_CASES_SMOOTH"], inplace=True
        )
        merged_df["UPPER_CI_N_CASES_SMOOTH_1YR_AGO"].fillna(
            merged_df["UPPER_CI_N_CASES_SMOOTH"], inplace=True
        )

        # Drop 'WEEK_LABEL_1YR_AGO' column
        merged_df.drop(["WEEK_LABEL_1YR_AGO"], axis=1, inplace=True)

        # Add resulting DataFrame to the list
        result_dataframes.append(merged_df)

    # Concatenate all resulting DataFrames
    final_result_dataframe = pd.concat(result_dataframes, ignore_index=True)

    # Display the last 50 rows of the final resulting DataFrame
    final_result_dataframe.rename(
        columns={"N_CASES_SMOOTH_1YR_AGO": "BASELINE_N_CASES"}, inplace=True
    )
    final_result_dataframe.rename(
        columns={"LOWER_CI_N_CASES_SMOOTH_1YR_AGO": "LOWER_CI_BASELINE_N_CASES"},
        inplace=True,
    )
    final_result_dataframe.rename(
        columns={"UPPER_CI_N_CASES_SMOOTH_1YR_AGO": "UPPER_CI_BASELINE_N_CASES"},
        inplace=True,
    )

    final_result_dataframe = final_result_dataframe.drop(["COMBI", "YEAR"], axis=1)

    final_result_dataframe["ALERT_N_CASES"] = np.where(
        (
            final_result_dataframe["N_CASES"]
            > final_result_dataframe["UPPER_CI_BASELINE_N_CASES"]
        ),
        1,
        0,
    )

    return final_result_dataframe


import numpy as np
import pandas as pd


def baseline_n_cases_and_deaths(final_smoothed_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the baseline number of cases

    Args:
        final_smoothed_dataframe (pd.DataFrame): Processed DataFrame containing smoothed data.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data and calculated baseline number of cases.
    """
    # Unique list of diagnosis and age combinations
    final_smoothed_dataframe["COMBI"] = final_smoothed_dataframe["DIAGNOSIS_CATEGORY"]
    unique_combinations = final_smoothed_dataframe["COMBI"].unique()

    final_smoothed_dataframe["YEAR"] = final_smoothed_dataframe["WEEK_LABEL"].dt.year

    # List to store resulting DataFrames for each combination
    result_dataframes = []

    # Define one year shift
    one_year_shift = pd.DateOffset(weeks=52)

    # Loop through each combination
    for combo in unique_combinations:
        # Filter DataFrame for current combination
        df = final_smoothed_dataframe[final_smoothed_dataframe["COMBI"] == combo].copy()

        # Add 'WEEK_LABEL_1YR_AGO' column to store corresponding date one year ago
        df["WEEK_LABEL_1YR_AGO"] = df["WEEK_LABEL"] - one_year_shift

        # Merge with itself to get values from the previous year
        merged_df = pd.merge(
            df,
            df[
                [
                    "WEEK_LABEL",
                    "N_CASES_SMOOTH",
                    "LOWER_CI_N_CASES_SMOOTH",
                    "UPPER_CI_N_CASES_SMOOTH",
                ]
            ],
            left_on="WEEK_LABEL_1YR_AGO",
            right_on="WEEK_LABEL",
            suffixes=("", "_1YR_AGO"),
            how="left",
        )

        # Replace missing values with values from the previous year
        merged_df["N_CASES_SMOOTH_1YR_AGO"].fillna(
            merged_df["N_CASES_SMOOTH"], inplace=True
        )
        merged_df["LOWER_CI_N_CASES_SMOOTH_1YR_AGO"].fillna(
            merged_df["LOWER_CI_N_CASES_SMOOTH"], inplace=True
        )
        merged_df["UPPER_CI_N_CASES_SMOOTH_1YR_AGO"].fillna(
            merged_df["UPPER_CI_N_CASES_SMOOTH"], inplace=True
        )

        # Drop 'WEEK_LABEL_1YR_AGO' column
        merged_df.drop(["WEEK_LABEL_1YR_AGO"], axis=1, inplace=True)

        # Add resulting DataFrame to the list
        result_dataframes.append(merged_df)

    # Concatenate all resulting DataFrames
    final_result_dataframe = pd.concat(result_dataframes, ignore_index=True)
    result_dataframes = []
    # Loop through each combination
    for combo in unique_combinations:
        # Filter DataFrame for current combination
        df = final_result_dataframe[final_result_dataframe["COMBI"] == combo].copy()

        # Add 'WEEK_LABEL_1YR_AGO' column to store corresponding date one year ago
        df["WEEK_LABEL_1YR_AGO"] = df["WEEK_LABEL"] - one_year_shift

        # Merge with itself to get values from the previous year
        merged_df = pd.merge(
            df,
            df[
                [
                    "WEEK_LABEL",
                    "N_DEATH_SMOOTH",
                    "LOWER_CI_N_DEATH_SMOOTH",
                    "UPPER_CI_N_DEATH_SMOOTH",
                ]
            ],
            left_on="WEEK_LABEL_1YR_AGO",
            right_on="WEEK_LABEL",
            suffixes=("", "_1YR_AGO"),
            how="left",
        )

        # Replace missing values with values from the previous year
        merged_df["N_DEATH_SMOOTH_1YR_AGO"].fillna(
            merged_df["N_DEATH_SMOOTH"], inplace=True
        )
        merged_df["LOWER_CI_N_DEATH_SMOOTH_1YR_AGO"].fillna(
            merged_df["LOWER_CI_N_DEATH_SMOOTH"], inplace=True
        )
        merged_df["UPPER_CI_N_DEATH_SMOOTH_1YR_AGO"].fillna(
            merged_df["UPPER_CI_N_DEATH_SMOOTH"], inplace=True
        )

        # Drop 'WEEK_LABEL_1YR_AGO' column
        merged_df.drop(["WEEK_LABEL_1YR_AGO"], axis=1, inplace=True)

        # Add resulting DataFrame to the list
        result_dataframes.append(merged_df)

    # Concatenate all resulting DataFrames
    final_result_dataframe = pd.concat(result_dataframes, ignore_index=True)
    # Display the last 50 rows of the final resulting DataFrame
    final_result_dataframe.rename(
        columns={"N_CASES_SMOOTH_1YR_AGO": "BASELINE_N_CASES"}, inplace=True
    )
    final_result_dataframe.rename(
        columns={"LOWER_CI_N_CASES_SMOOTH_1YR_AGO": "LOWER_CI_BASELINE_N_CASES"},
        inplace=True,
    )
    final_result_dataframe.rename(
        columns={"UPPER_CI_N_CASES_SMOOTH_1YR_AGO": "UPPER_CI_BASELINE_N_CASES"},
        inplace=True,
    )

    final_result_dataframe.rename(
        columns={"N_DEATH_SMOOTH_1YR_AGO": "BASELINE_N_DEATH"}, inplace=True
    )
    final_result_dataframe.rename(
        columns={"LOWER_CI_N_DEATH_SMOOTH_1YR_AGO": "LOWER_CI_BASELINE_N_DEATH"},
        inplace=True,
    )
    final_result_dataframe.rename(
        columns={"UPPER_CI_N_DEATH_SMOOTH_1YR_AGO": "UPPER_CI_BASELINE_N_DEATH"},
        inplace=True,
    )
    final_result_dataframe = final_result_dataframe.drop(["COMBI", "YEAR"], axis=1)

    final_result_dataframe["ALERT_N_CASES"] = np.where(
        (
            final_result_dataframe["N_CASES"]
            > final_result_dataframe["UPPER_CI_BASELINE_N_CASES"]
        ),
        1,
        0,
    )
    final_result_dataframe["ALERT_N_DEATH"] = np.where(
        (
            final_result_dataframe["N_DEATH"]
            > final_result_dataframe["UPPER_CI_BASELINE_N_DEATH"]
        ),
        1,
        0,
    )
    return final_result_dataframe


def baseline_n_cases_and_crit(final_smoothed_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the baseline number of cases

    Args:
        final_smoothed_dataframe (pd.DataFrame): Processed DataFrame containing smoothed data.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data and calculated baseline number of cases.
    """
    # Unique list of diagnosis and age combinations
    final_smoothed_dataframe["COMBI"] = final_smoothed_dataframe["DIAGNOSIS_CATEGORY"]
    unique_combinations = final_smoothed_dataframe["COMBI"].unique()

    final_smoothed_dataframe["YEAR"] = final_smoothed_dataframe["WEEK_LABEL"].dt.year

    # List to store resulting DataFrames for each combination
    result_dataframes = []

    # Define one year shift
    one_year_shift = pd.DateOffset(weeks=52)

    # Loop through each combination
    for combo in unique_combinations:
        # Filter DataFrame for current combination
        df = final_smoothed_dataframe[final_smoothed_dataframe["COMBI"] == combo].copy()

        # Add 'WEEK_LABEL_1YR_AGO' column to store corresponding date one year ago
        df["WEEK_LABEL_1YR_AGO"] = df["WEEK_LABEL"] - one_year_shift

        # Merge with itself to get values from the previous year
        merged_df = pd.merge(
            df,
            df[
                [
                    "WEEK_LABEL",
                    "N_CASES_SMOOTH",
                    "LOWER_CI_N_CASES_SMOOTH",
                    "UPPER_CI_N_CASES_SMOOTH",
                ]
            ],
            left_on="WEEK_LABEL_1YR_AGO",
            right_on="WEEK_LABEL",
            suffixes=("", "_1YR_AGO"),
            how="left",
        )

        # Replace missing values with values from the previous year
        merged_df["N_CASES_SMOOTH_1YR_AGO"].fillna(
            merged_df["N_CASES_SMOOTH"], inplace=True
        )
        merged_df["LOWER_CI_N_CASES_SMOOTH_1YR_AGO"].fillna(
            merged_df["LOWER_CI_N_CASES_SMOOTH"], inplace=True
        )
        merged_df["UPPER_CI_N_CASES_SMOOTH_1YR_AGO"].fillna(
            merged_df["UPPER_CI_N_CASES_SMOOTH"], inplace=True
        )

        # Drop 'WEEK_LABEL_1YR_AGO' column
        merged_df.drop(["WEEK_LABEL_1YR_AGO"], axis=1, inplace=True)

        # Add resulting DataFrame to the list
        result_dataframes.append(merged_df)

    # Concatenate all resulting DataFrames
    final_result_dataframe = pd.concat(result_dataframes, ignore_index=True)
    result_dataframes = []
    # Loop through each combination
    for combo in unique_combinations:
        # Filter DataFrame for current combination
        df = final_result_dataframe[final_result_dataframe["COMBI"] == combo].copy()

        # Add 'WEEK_LABEL_1YR_AGO' column to store corresponding date one year ago
        df["WEEK_LABEL_1YR_AGO"] = df["WEEK_LABEL"] - one_year_shift

        # Merge with itself to get values from the previous year
        merged_df = pd.merge(
            df,
            df[
                [
                    "WEEK_LABEL",
                    "N_CRITICAL_SMOOTH",
                    "LOWER_CI_N_CRITICAL_SMOOTH",
                    "UPPER_CI_N_CRITICAL_SMOOTH",
                ]
            ],
            left_on="WEEK_LABEL_1YR_AGO",
            right_on="WEEK_LABEL",
            suffixes=("", "_1YR_AGO"),
            how="left",
        )

        # Replace missing values with values from the previous year
        merged_df["N_CRITICAL_SMOOTH_1YR_AGO"].fillna(
            merged_df["N_CRITICAL_SMOOTH"], inplace=True
        )
        merged_df["LOWER_CI_N_CRITICAL_SMOOTH_1YR_AGO"].fillna(
            merged_df["LOWER_CI_N_CRITICAL_SMOOTH"], inplace=True
        )
        merged_df["UPPER_CI_N_CRITICAL_SMOOTH_1YR_AGO"].fillna(
            merged_df["UPPER_CI_N_CRITICAL_SMOOTH"], inplace=True
        )

        # Drop 'WEEK_LABEL_1YR_AGO' column
        merged_df.drop(["WEEK_LABEL_1YR_AGO"], axis=1, inplace=True)

        # Add resulting DataFrame to the list
        result_dataframes.append(merged_df)

    # Concatenate all resulting DataFrames
    final_result_dataframe = pd.concat(result_dataframes, ignore_index=True)
    # Display the last 50 rows of the final resulting DataFrame
    final_result_dataframe.rename(
        columns={"N_CASES_SMOOTH_1YR_AGO": "BASELINE_N_CASES"}, inplace=True
    )
    final_result_dataframe.rename(
        columns={"LOWER_CI_N_CASES_SMOOTH_1YR_AGO": "LOWER_CI_BASELINE_N_CASES"},
        inplace=True,
    )
    final_result_dataframe.rename(
        columns={"UPPER_CI_N_CASES_SMOOTH_1YR_AGO": "UPPER_CI_BASELINE_N_CASES"},
        inplace=True,
    )

    final_result_dataframe.rename(
        columns={"N_CRITICAL_SMOOTH_1YR_AGO": "BASELINE_N_CRITICAL"}, inplace=True
    )
    final_result_dataframe.rename(
        columns={"LOWER_CI_N_CRITICAL_SMOOTH_1YR_AGO": "LOWER_CI_BASELINE_N_CRITICAL"},
        inplace=True,
    )
    final_result_dataframe.rename(
        columns={"UPPER_CI_N_CRITICAL_SMOOTH_1YR_AGO": "UPPER_CI_BASELINE_N_CRITICAL"},
        inplace=True,
    )
    final_result_dataframe = final_result_dataframe.drop(["COMBI", "YEAR"], axis=1)

    final_result_dataframe["ALERT_N_CASES"] = np.where(
        (
            final_result_dataframe["N_CASES"]
            > final_result_dataframe["UPPER_CI_BASELINE_N_CASES"]
        ),
        1,
        0,
    )
    final_result_dataframe["ALERT_N_CRITICAL"] = np.where(
        (
            final_result_dataframe["N_CRITICAL"]
            > final_result_dataframe["UPPER_CI_BASELINE_N_CRITICAL"]
        ),
        1,
        0,
    )
    return final_result_dataframe


def table_I_calculation_nowcasting(
    table_I_nowcasting: pd.DataFrame, biased_weeks: int = 5
) -> pd.DataFrame:
    """
    Perform nowcasting calculation for Table I.

    Args:
        table_I_nowcasting (pd.DataFrame): DataFrame containing Table I data.
        biased_weeks (int): Number of weeks considered for nowcasting correction factor. Default is 5.

    Returns:
        pd.DataFrame: DataFrame with nowcasting calculation results.

    """
    # Create the correction_factor_last_weeks vector
    correction_factor_last_weeks = np.linspace(1, 0.2, biased_weeks + 1)

    # Create a 'correction_factor' column with values of 1 throughout the DataFrame
    table_I_nowcasting["correction_factor"] = 1

    # List of unique categories for each column
    categories_diagnosis = table_I_nowcasting["DIAGNOSIS_CATEGORY"].unique()
    categories_age = table_I_nowcasting["AGE_CLASS"].unique()
    categories_rss_duration = table_I_nowcasting["RSS_DURATION_CLASS"].unique()

    # Loop over each combination of categories
    for diagnosis_category in categories_diagnosis:
        for age_class in categories_age:
            for rss_duration_class in categories_rss_duration:
                # Filter rows for the current combination of categories
                filtered_df = table_I_nowcasting.loc[
                    (table_I_nowcasting["DIAGNOSIS_CATEGORY"] == diagnosis_category)
                    & (table_I_nowcasting["AGE_CLASS"] == age_class)
                    & (table_I_nowcasting["RSS_DURATION_CLASS"] == rss_duration_class)
                ]

                # Assign values from the correction_factor_last_weeks vector to the last 10 rows of each sub-DataFrame
                table_I_nowcasting.loc[
                    filtered_df.index[-biased_weeks:], "correction_factor"
                ] = correction_factor_last_weeks[1:]

    # Calculate N_CASES_NOWCAST by dividing N_CASES by the correction factor
    table_I_nowcasting["N_CASES_NOWCAST"] = (
        table_I_nowcasting["N_CASES"] / table_I_nowcasting["correction_factor"]
    )

    # Drop the 'correction_factor' column
    table_I_nowcasting = table_I_nowcasting.drop(
        ["correction_factor", "LOWER_CI_N_CASES_SMOOTH", "UPPER_CI_N_CASES_SMOOTH"],
        axis=1,
    )

    # Add ALERT_N_CASES based on whether N_CASES_NOWCAST is greater than UPPER_CI_BASELINE_N_CASES
    table_I_nowcasting["ALERT_N_CASES"] = np.where(
        (
            table_I_nowcasting["N_CASES_NOWCAST"]
            > table_I_nowcasting["UPPER_CI_BASELINE_N_CASES"]
        ),
        1,
        0,
    )
    return table_I_nowcasting


def table_II_calculation_nowcasting(
    table_II_nowcasting: pd.DataFrame, biased_weeks: int = 5
) -> pd.DataFrame:
    """
    Perform nowcasting calculation for Table II.

    Args:
        table_II_nowcasting (pd.DataFrame): DataFrame containing Table II data.
        biased_weeks (int): Number of weeks considered for nowcasting correction factor. Default is 5.

    Returns:
        pd.DataFrame: DataFrame with nowcasting calculation results.

    """
    # Create the correction_factor_last_weeks vector
    correction_factor_last_weeks = np.linspace(1, 0.2, biased_weeks + 1)

    # Create a 'correction_factor' column with values of 1 throughout the DataFrame
    table_II_nowcasting["correction_factor"] = 1

    # List of unique categories for the DIAGNOSIS_CATEGORY column
    categories_diagnosis = table_II_nowcasting["DIAGNOSIS_CATEGORY"].unique()

    # Loop over each combination of categories
    for diagnosis_category in categories_diagnosis:
        # Filter rows for the current combination of categories
        filtered_df = table_II_nowcasting.loc[
            table_II_nowcasting["DIAGNOSIS_CATEGORY"] == diagnosis_category
        ]

        # Assign values from the correction_factor_last_weeks vector to the last 10 rows of each sub-DataFrame
        table_II_nowcasting.loc[
            filtered_df.index[-biased_weeks:], "correction_factor"
        ] = correction_factor_last_weeks[1:]

    # Calculate N_CASES_NOWCAST and N_DEATH_NOWCAST by dividing N_CASES and N_DEATH by the correction factor
    table_II_nowcasting["N_CASES_NOWCAST"] = (
        table_II_nowcasting["N_CASES"] / table_II_nowcasting["correction_factor"]
    )
    table_II_nowcasting["N_DEATH_NOWCAST"] = (
        table_II_nowcasting["N_DEATH"] / table_II_nowcasting["correction_factor"]
    )

    # Drop the 'correction_factor' column
    table_II_nowcasting = table_II_nowcasting.drop(
        [
            "correction_factor",
            "UPPER_CI_N_CASES_SMOOTH",
            "LOWER_CI_N_CASES_SMOOTH",
            "LOWER_CI_N_DEATH_SMOOTH",
            "UPPER_CI_N_DEATH_SMOOTH",
        ],
        axis=1,
    )

    # Add ALERT_N_CASES and ALERT_N_DEATH based on whether N_CASES_NOWCAST and N_DEATH_NOWCAST are greater than their respective upper confidence intervals
    table_II_nowcasting["ALERT_N_CASES"] = np.where(
        (
            table_II_nowcasting["N_CASES_NOWCAST"]
            > table_II_nowcasting["UPPER_CI_BASELINE_N_CASES"]
        ),
        1,
        0,
    )
    table_II_nowcasting["ALERT_N_DEATH"] = np.where(
        (
            table_II_nowcasting["N_DEATH_NOWCAST"]
            > table_II_nowcasting["UPPER_CI_BASELINE_N_DEATH"]
        ),
        1,
        0,
    )

    return table_II_nowcasting


def table_III_calculation_nowcasting(
    table_III_nowcasting: pd.DataFrame, biased_weeks: int = 5
) -> pd.DataFrame:
    """
    Perform nowcasting calculation for Table III.

    Args:
        table_III_nowcasting (pd.DataFrame): DataFrame containing Table III data.
        biased_weeks (int): Number of weeks considered for nowcasting correction factor. Default is 5.

    Returns:
        pd.DataFrame: DataFrame with nowcasting calculation results.

    """
    # Create the correction_factor_last_weeks vector
    correction_factor_last_weeks = np.linspace(1, 0.2, biased_weeks + 1)

    # Create a 'correction_factor' column with values of 1 throughout the DataFrame
    table_III_nowcasting["correction_factor"] = 1

    # List of unique categories for the DIAGNOSIS_CATEGORY column
    categories_diagnosis = table_III_nowcasting["DIAGNOSIS_CATEGORY"].unique()

    # Loop over each combination of categories
    for diagnosis_category in categories_diagnosis:
        # Filter rows for the current combination of categories
        filtered_df = table_III_nowcasting.loc[
            table_III_nowcasting["DIAGNOSIS_CATEGORY"] == diagnosis_category
        ]

        # Assign values from the correction_factor_last_weeks vector to the last 10 rows of each sub-DataFrame
        table_III_nowcasting.loc[
            filtered_df.index[-biased_weeks:], "correction_factor"
        ] = correction_factor_last_weeks[1:]

    # Calculate N_CASES_NOWCAST and N_CRITICAL_NOWCAST by dividing N_CASES and N_CRITICAL by the correction factor
    table_III_nowcasting["N_CASES_NOWCAST"] = (
        table_III_nowcasting["N_CASES"] / table_III_nowcasting["correction_factor"]
    )
    table_III_nowcasting["N_CRITICAL_NOWCAST"] = (
        table_III_nowcasting["N_CRITICAL"] / table_III_nowcasting["correction_factor"]
    )

    # Drop the 'correction_factor' column
    table_III_nowcasting = table_III_nowcasting.drop(
        [
            "correction_factor",
            "UPPER_CI_N_CASES_SMOOTH",
            "LOWER_CI_N_CASES_SMOOTH",
            "LOWER_CI_N_CRITICAL_SMOOTH",
            "UPPER_CI_N_CRITICAL_SMOOTH",
        ],
        axis=1,
    )

    # Add ALERT_N_CASES and ALERT_N_CRITICAL based on whether N_CASES_NOWCAST and N_CRITICAL_NOWCAST are greater than their respective upper confidence intervals
    table_III_nowcasting["ALERT_N_CASES"] = np.where(
        (
            table_III_nowcasting["N_CASES_NOWCAST"]
            > table_III_nowcasting["UPPER_CI_BASELINE_N_CASES"]
        ),
        1,
        0,
    )
    table_III_nowcasting["ALERT_N_CRITICAL"] = np.where(
        (
            table_III_nowcasting["N_CRITICAL_NOWCAST"]
            > table_III_nowcasting["UPPER_CI_BASELINE_N_CRITICAL"]
        ),
        1,
        0,
    )
    return table_III_nowcasting


def output_df(df: pd.DataFrame) -> None:
    """
    Save the processed data to an Excel file.

    Args:
        dataframe (pd.DataFrame): Processed DataFrame.
        filepath (str): Filepath to save the processed data.
    """
    df = pd.DataFrame(df)
    df_types = df.dtypes.to_frame().T

    # Réinitialiser l'index pour obtenir une colonne "Nom de la colonne"
    df_types = df_types.reset_index(drop=True)

    return df_types
