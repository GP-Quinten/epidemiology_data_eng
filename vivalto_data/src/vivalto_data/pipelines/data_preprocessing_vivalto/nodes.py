import re
from itertools import product

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import convolve
from scipy.signal.windows import gaussian


def upload_data(df):
    print(df.columns)
    return df

def get_most_recent_dataframe(vivalto_data: dict) -> pd.DataFrame:
    """
    Retrieves the most recent DataFrames from the vivalto_data dictionary based on the extraction date.

    Args:
        vivalto_data (dict): A dictionary containing Excel file data.

    Returns:
        tuple: A tuple containing the most recent DataFrames for different categories:
            - df_number_of_id_vivalto: DataFrame for 'TAB_I' category.
            - df_death_vivalto: DataFrame for 'TAB_II' category.
            - df_critical_health_vivalto: DataFrame for 'TAB_III' category.
    """
    latest_dataframe = None
    latest_extraction_date = pd.to_datetime('1900-01-01')

    # # Extract dates from file names and find the latest extraction date
    # pattern = r'_S\d+_(\d{8})_cleaned'

    # # Iterate through the Excel file names in vivalto_data
    # for file_name, excel_df in vivalto_data.items():
    #     match = re.search(pattern, file_name)

    #     if match:
    #         date_str = match.group(1)  # Get the date string from the matching pattern
    #         extraction_date = pd.to_datetime(date_str, format='%d%m%Y')

    #         # Compare the date with the previously recorded most recent extraction date
    #         if extraction_date > latest_extraction_date:
    #             latest_extraction_date = extraction_date
    #             latest_dataframe = excel_df  # Update the most recent DataFrame
    
    df_number_of_id_vivalto = vivalto_data['VIVALTO_s36-2023 s13-2024_09042024.xlsx']['TAB_I']
    df_death_vivalto = vivalto_data['VIVALTO_s36-2023 s13-2024_09042024.xlsx']['TAB_II']
    df_critical_health_vivalto = vivalto_data['VIVALTO_s36-2023 s13-2024_09042024.xlsx']['TAB_III']
    
    # df_number_of_id_vivalto = vivalto_data['TAB_I']
    # df_death_vivalto = vivalto_data['TAB_II']
    # df_critical_health_vivalto = vivalto_data['TAB_III']
    
    return df_number_of_id_vivalto, df_death_vivalto, df_critical_health_vivalto

def dates_formatting(df_number_of_id_vivalto, df_death_vivalto, df_critical_health_vivalto):
    """
    Preprocesses the given DataFrames by formatting date columns and extracting relevant information.

    Args:
        df_number_of_id_vivalto (pd.DataFrame): DataFrame for 'TAB_I' category.
        df_death_vivalto (pd.DataFrame): DataFrame for 'TAB_II' category.
        df_critical_health_vivalto (pd.DataFrame): DataFrame for 'TAB_III' category.

    Returns:
        tuple: A tuple containing the preprocessed DataFrames for each category.
            - df_number_of_id_vivalto_dates: DataFrame with formatted date columns for 'TAB_I'.
            - df_death_vivalto_dates: DataFrame with formatted date columns for 'TAB_II'.
            - df_critical_health_vivalto_dates: DataFrame with formatted date columns for 'TAB_III'.
    """
    # Preprocess 'LIB_SEM' column in each DataFrame
    for df in [df_number_of_id_vivalto, df_death_vivalto, df_critical_health_vivalto]:
        df['LIB_SEM'] = df['LIB_SEM'].str.extract(r'_(\d{8})_', expand=False)
        df['LIB_SEM'] = pd.to_datetime(df['LIB_SEM'], format='%d%m%Y', errors='coerce')

    # Preprocess 'CATEG_DIAG' column in each DataFrame
    for df in [df_number_of_id_vivalto, df_death_vivalto, df_critical_health_vivalto]:
        df['CATEG_DIAG'] = df['CATEG_DIAG'].str.split('-').str[1].str.strip()

    df_number_of_id_vivalto_dates = df_number_of_id_vivalto
    df_death_vivalto_dates= df_death_vivalto
    df_critical_health_vivalto_dates = df_critical_health_vivalto
    
    
    return df_number_of_id_vivalto_dates, df_death_vivalto_dates, df_critical_health_vivalto_dates

def extract_columns(df_number_of_id_vivalto_dates, df_death_vivalto_dates, df_critical_health_vivalto_dates):
    """
    Extracts specific columns from the preprocessed DataFrames.

    Args:
        df_number_of_id_vivalto_dates (pd.DataFrame): DataFrame with formatted date columns for 'TAB_I'.
        df_death_vivalto_dates (pd.DataFrame): DataFrame with formatted date columns for 'TAB_II'.
        df_critical_health_vivalto_dates (pd.DataFrame): DataFrame with formatted date columns for 'TAB_III'.

    Returns:
        tuple: A tuple containing DataFrames with extracted columns for each category.
            - df_number_of_id: DataFrame with selected columns for 'TAB_I'.
            - df_death: DataFrame with selected columns for 'TAB_II'.
            - df_critical_health: DataFrame with selected columns for 'TAB_III'.
    """
    df_number_of_id_columns = df_number_of_id_vivalto_dates[["LIB_SEM", "CATEG_DIAG", "CLAS_AGE", "CLAS_DUREERSS", "NRSS"]]
    # drop rows where CATEGORY_DIAG is '-' or NaN
    df_number_of_id_columns = df_number_of_id_columns[df_number_of_id_columns['CATEG_DIAG'].notna()]
    df_number_of_id_columns = df_number_of_id_columns[df_number_of_id_columns['CATEG_DIAG'] != '-']

    df_death_columns = df_death_vivalto_dates[["LIB_SEM", "CATEG_DIAG", "DECES", "NRSS"]]
    # same
    df_death_columns = df_death_columns[df_death_columns['CATEG_DIAG'].notna()]
    df_death_columns = df_death_columns[df_death_columns['CATEG_DIAG'] != '-']

    df_critical_health_columns = df_critical_health_vivalto_dates[["LIB_SEM", "CATEG_DIAG", "PASS_SOINSCRITIQUE","ORIG_GEO", "NRSS"]]
    # same
    df_critical_health_columns = df_critical_health_columns[df_critical_health_columns['CATEG_DIAG'].notna()]
    df_critical_health_columns = df_critical_health_columns[df_critical_health_columns['CATEG_DIAG'] != '-']

    return df_number_of_id_columns, df_death_columns, df_critical_health_columns

def translation_columns(df_number_of_id_columns, df_death_columns, df_critical_health_columns):
    """
    Translates and renames columns in the given DataFrames.

    Args:
        df_number_of_id_columns (pd.DataFrame): DataFrame for 'TAB_I' category with selected columns.
        df_death_columns (pd.DataFrame): DataFrame for 'TAB_II' category with selected columns.
        df_critical_health_columns (pd.DataFrame): DataFrame for 'TAB_III' category with selected columns.

    Returns:
        tuple: A tuple containing DataFrames with translated and renamed columns for each category.
            - df_number_of_id_translate: DataFrame with translated and renamed columns for 'TAB_I'.
            - df_death_translate: DataFrame with translated and renamed columns for 'TAB_II'.
            - df_critical_health_translate: DataFrame with translated and renamed columns for 'TAB_III'.
    """
    # Rename columns in each DataFrame
    df_number_of_id = df_number_of_id_columns.rename(columns={
        'CATEG_DIAG': 'DIAGNOSIS_CATEGORY',
        'LIB_SEM': 'WEEK_LABEL',
        'CLAS_AGE': 'AGE_CLASS',
        'CLAS_DUREERSS': 'RSS_DURATION_CLASS',
        'NRSS': 'N_CASES'
    })

    df_death = df_death_columns.rename(columns={
        'CATEG_DIAG': 'DIAGNOSIS_CATEGORY',
        'LIB_SEM': 'WEEK_LABEL',
        'DECES': 'DEATH_OR_NOT',
        'NRSS': 'TOTAL_RSS'
    })

    df_critical_health = df_critical_health_columns.rename(columns={
        'CATEG_DIAG': 'DIAGNOSIS_CATEGORY',
        'LIB_SEM': 'WEEK_LABEL',
        'ORIG_GEO': 'GEOGRAPHICAL_ORIGIN',
        'PASS_SOINSCRITIQUE': 'CRITICAL_CARE_OR_NOT',
        'NRSS': 'TOTAL_RSS'
    })
    
    # Translation for AGE_CLASS column
    age_class_translation = {
        '1-4 ans': '[1 - 5[ year(s)',
        '5-19 ans': '[5 - 19[ year(s)',
        '20-49 ans': '[20 - 49[ year(s)',
        '50-64 ans': '[50 - 64[ year(s)',
        '65 ans et plus': '65 years and older'
    }

    # Apply age class translation to the AGE_CLASS column in the DataFrame
    df_number_of_id['AGE_CLASS'] = df_number_of_id['AGE_CLASS'].replace(age_class_translation)

    # Translation for RSS_DURATION_CLASS column
    rss_duration_translation = {
        '<2 jours': 'Less than 2 days',
        '2-4 jours': '[2 - 5[ days',
        '5 jours et plus': '5 days and more'
    }

    # Apply RSS duration translation to the RSS_DURATION_CLASS column in the DataFrame
    df_number_of_id['RSS_DURATION_CLASS'] = df_number_of_id['RSS_DURATION_CLASS'].replace(rss_duration_translation)

    # Translation for DIAGNOSIS_CATEGORY column
    diagnosis_translation = {
        'COVID': 'COVID-19',
        'GRIPPE': 'FLU',
        "INFECTIONS RESPIRATOIRES LIEES A D'AUTRES VIRUS": 'RI_OTHER_VIRUS',
        'INFECTIONS RESPIRATOIRES EN GENERAL': 'GENERAL_RI'
    }

    # Apply diagnosis category translation to the DIAGNOSIS_CATEGORY column in multiple DataFrames
    df_number_of_id['DIAGNOSIS_CATEGORY'] = df_number_of_id['DIAGNOSIS_CATEGORY'].replace(diagnosis_translation)
    df_death['DIAGNOSIS_CATEGORY'] = df_death['DIAGNOSIS_CATEGORY'].replace(diagnosis_translation)
    df_critical_health['DIAGNOSIS_CATEGORY'] = df_critical_health['DIAGNOSIS_CATEGORY'].replace(diagnosis_translation)

    # Translation for DEATH_OR_NOT column in the df_death DataFrame
    death_translation = {'NON': 'No', 'OUI': 'Yes'}
    critical_care_translation = {'Non': 'No', 'Oui': 'Yes'}
    
    # Apply death or not translation to the DEATH_OR_NOT column in the df_death DataFrame
    df_death['DEATH_OR_NOT'] = df_death['DEATH_OR_NOT'].replace(death_translation)
    df_critical_health['CRITICAL_CARE_OR_NOT'] = df_critical_health['CRITICAL_CARE_OR_NOT'].replace(critical_care_translation)
    
    
    df_number_of_id = df_number_of_id.groupby(['WEEK_LABEL', 'DIAGNOSIS_CATEGORY', 'AGE_CLASS','RSS_DURATION_CLASS'])['N_CASES'].sum().reset_index()
    df_death = df_death.groupby(['WEEK_LABEL', 'DIAGNOSIS_CATEGORY', 'DEATH_OR_NOT'])['TOTAL_RSS'].sum().reset_index()
    df_critical_health = df_critical_health.groupby(['WEEK_LABEL', 'DIAGNOSIS_CATEGORY', 'CRITICAL_CARE_OR_NOT', 'GEOGRAPHICAL_ORIGIN'])['TOTAL_RSS'].sum().reset_index()
    
    
    table_I = df_number_of_id
    table_II = df_death
    table_III = df_critical_health

    return table_I, table_II, table_III

def missing_data_prepro_I(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    """
    # Définissez les conditions et les valeurs correspondantes
    result_df['COMBI'] = result_df['DIAGNOSIS_CATEGORY'] + ' | ' + result_df['AGE_CLASS']+ ' | ' + result_df['RSS_DURATION_CLASS']

    # Créer un graphique en bar plot pour le nombre de valeurs par année par combinaison DIAGNOSIS_CATEGORY et CLASS_AGE
    result_df['WEEK_LABEL'] = pd.to_datetime(result_df['WEEK_LABEL'])
    result_df['YEAR'] = result_df['WEEK_LABEL'].dt.year

    result_df = result_df.sort_values(by='COMBI')
    result_df = result_df.drop(columns='COMBI')
    result_df = result_df.drop(columns='YEAR')
    
    weeks = pd.date_range(start=result_df['WEEK_LABEL'].min(), end=result_df['WEEK_LABEL'].max(), freq='W-Mon')


    categories_age = result_df['AGE_CLASS'].unique()
    diagnosis_categories = ['COVID-19', 'FLU', 'RSV', 'GENERAL_RI', 'RI_OTHER_VIRUS']
    rss_duration_categories = result_df['RSS_DURATION_CLASS'].unique()

    combinations = list(product(weeks, diagnosis_categories, categories_age, rss_duration_categories))

    columns = ['WEEK_LABEL', 'DIAGNOSIS_CATEGORY', 'AGE_CLASS','RSS_DURATION_CLASS']
    all_combinations_df = pd.DataFrame(combinations, columns=columns)

    merged_df = pd.merge(all_combinations_df, result_df, on=['WEEK_LABEL', 'AGE_CLASS','RSS_DURATION_CLASS','DIAGNOSIS_CATEGORY'], how='outer')

    merged_df = merged_df.sort_values(by='WEEK_LABEL')
    merged_df['N_CASES'] = merged_df['N_CASES'].fillna(0)
    
    merged_df = merged_df.sort_values(by=['DIAGNOSIS_CATEGORY', 'AGE_CLASS', 'RSS_DURATION_CLASS', 'WEEK_LABEL'], ascending=[True, True, True, True])
    # drop rows where CATEGORY_DIAG is '-' or NaN
    merged_df = merged_df[merged_df['DIAGNOSIS_CATEGORY'].notna()]
    merged_df = merged_df[merged_df['DIAGNOSIS_CATEGORY'] != '-']
    merged_df = merged_df[merged_df['DIAGNOSIS_CATEGORY'] != '']

    table_I = merged_df

    #####################
    ### ADD SUBTOTALS ###

    #------------------------------------
    ### TABLE I ###
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

    # create a dataframe filering the table_I dataframe where the DIAGNOSIS_CATEGORY is not equal to "GENERAL_RI"
    table_I_filtered = table_I[table_I["DIAGNOSIS_CATEGORY"] != "GENERAL_RI"]
    # create a new dataframe that groups the data by WEEK_LABEL, RSS_DURATION_CLASS, AGE_CLASS. Sums the TOTAL_RSS values and resets the index
    sum_DIAGNOSIS_CATEGORY = (
        table_I_filtered.groupby(["WEEK_LABEL", "AGE_CLASS", "RSS_DURATION_CLASS"])["N_CASES"]
        .sum()
        .reset_index()
    )
    # add column DIAGNOSIS_CATEGORY to the sum_DIAGNOSIS_CATEGORY dataframe with value "all"
    sum_DIAGNOSIS_CATEGORY.insert(1, "DIAGNOSIS_CATEGORY", "all_resp_diags")
    # concatenate the sum_DIAGNOSIS_CATEGORY dataframe with the table_I dataframe, on axis 0
    table_I = pd.concat([table_I, sum_DIAGNOSIS_CATEGORY], axis=0)

    return table_I

def missing_data_prepro_II(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    """

    df_death_renamed = result_df[
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

    result_df = df_death_filtered.merge(
        sum_TOTAL_RSS, on=["WEEK_LABEL", "DIAGNOSIS_CATEGORY"], suffixes=("", "_SUM")
    )
    result_df = result_df.rename(columns={"TOTAL_RSS": "N_DEATH"})
    result_df = result_df.rename(columns={"TOTAL_RSS_SUM": "N_CASES"})
    result_df["N_DEATH"] = result_df["N_DEATH"].round().astype(int)
    result_df["N_CASES"] = result_df["N_CASES"].round().astype(int)
    
    
    # Définissez les conditions et les valeurs correspondantes
    result_df['COMBI'] = result_df['DIAGNOSIS_CATEGORY']

    # Créer un graphique en bar plot pour le nombre de valeurs par année par combinaison DIAGNOSIS_CATEGORY et CLASS_AGE
    result_df['WEEK_LABEL'] = pd.to_datetime(result_df['WEEK_LABEL'])
    result_df['YEAR'] = result_df['WEEK_LABEL'].dt.year

    result_df = result_df.sort_values(by='COMBI')
    result_df = result_df.drop(columns='COMBI')
    result_df = result_df.drop(columns='YEAR')
    
    weeks = pd.date_range(start=result_df['WEEK_LABEL'].min(), end=result_df['WEEK_LABEL'].max(), freq='W-Mon')


    diagnosis_categories = ['COVID-19', 'FLU', 'RSV', 'GENERAL_RI', 'RI_OTHER_VIRUS']
    combinations = list(product(weeks, diagnosis_categories))

    columns = ['WEEK_LABEL', 'DIAGNOSIS_CATEGORY']
    all_combinations_df = pd.DataFrame(combinations, columns=columns)

    merged_df = pd.merge(all_combinations_df, result_df, on=['WEEK_LABEL','DIAGNOSIS_CATEGORY'], how='outer')

    merged_df = merged_df.sort_values(by='WEEK_LABEL')
    
    merged_df['N_DEATH'] = merged_df['N_DEATH'].fillna(0)
    merged_df['N_CASES'] = merged_df['N_CASES'].fillna(0)
    merged_df = merged_df.sort_values(by=['DIAGNOSIS_CATEGORY', 'WEEK_LABEL'], ascending=[True, True])
    # drop rows where CATEGORY_DIAG is '-' or NaN
    merged_df = merged_df[merged_df['DIAGNOSIS_CATEGORY'].notna()]
    merged_df = merged_df[merged_df['DIAGNOSIS_CATEGORY'] != '-']
    merged_df = merged_df[merged_df['DIAGNOSIS_CATEGORY'] != '']

    table_II = merged_df

    #------------------------------------
    ### TABLE II ###
    # create a dataframe filering the table_II dataframe where the DIAGNOSIS_CATEGORY is not equal to "GENERAL_RI"
    table_II_filtered = table_II[table_II["DIAGNOSIS_CATEGORY"] != "GENERAL_RI"]
    # create a new dataframe that groups the data by WEEK_LABEL. Sums the N_DEATH, and N_CASES and  values and resets the index
    sum_DIAGNOSIS_CATEGORY = (
        table_II_filtered.groupby(["WEEK_LABEL"])[["N_DEATH", "N_CASES"]]
        .sum()
        .reset_index()
    )
    # add column DIAGNOSIS_CATEGORY to the sum_DIAGNOSIS_CATEGORY dataframe with value "all"
    sum_DIAGNOSIS_CATEGORY.insert(1, "DIAGNOSIS_CATEGORY", "all_resp_diags")
    # concatenate the sum_DIAGNOSIS_CATEGORY dataframe with the table_II dataframe, on axis 0
    table_II = pd.concat([table_II, sum_DIAGNOSIS_CATEGORY], axis=0)


    return table_II

def missing_data_prepro_III(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    """
    result_df = result_df[
        [
            "WEEK_LABEL",
            "DIAGNOSIS_CATEGORY",
            "CRITICAL_CARE_OR_NOT",
            "GEOGRAPHICAL_ORIGIN",
            "TOTAL_RSS",
        ]
    ]

    sum_TOTAL_RSS = (
        result_df.groupby(
            ["WEEK_LABEL", "DIAGNOSIS_CATEGORY","GEOGRAPHICAL_ORIGIN"]
        )["TOTAL_RSS"]
        .sum()
        .reset_index()
    )
    sum_TOTAL_RSS = sum_TOTAL_RSS.sort_values(
        by=["WEEK_LABEL", "DIAGNOSIS_CATEGORY","GEOGRAPHICAL_ORIGIN"]
    )

    df_critical_health_filtered = result_df[
        result_df["CRITICAL_CARE_OR_NOT"] == "Yes"
    ].drop("CRITICAL_CARE_OR_NOT", axis=1)
    
    sum_critical_health_TOTAL_RSS = (
        df_critical_health_filtered.groupby(
            ["WEEK_LABEL", "DIAGNOSIS_CATEGORY","GEOGRAPHICAL_ORIGIN"]
        )["TOTAL_RSS"]
        .sum()
        .reset_index()
    )

    result_df = sum_critical_health_TOTAL_RSS.merge(
        sum_TOTAL_RSS,
        on=["WEEK_LABEL", "DIAGNOSIS_CATEGORY","GEOGRAPHICAL_ORIGIN"],
        suffixes=("", "_SUM"),
    )
    result_df = result_df.rename(columns={"TOTAL_RSS": "N_CRITICAL"})
    result_df = result_df.rename(columns={"TOTAL_RSS_SUM": "N_CASES"})
    result_df["N_CRITICAL"] = result_df["N_CRITICAL"].round().astype(int)
    result_df["N_CASES"] = result_df["N_CASES"].round().astype(int)
    
    # Définissez les conditions et les valeurs correspondantes
    result_df['COMBI'] = result_df['DIAGNOSIS_CATEGORY'] + ' | ' + result_df['GEOGRAPHICAL_ORIGIN']

    # Créer un graphique en bar plot pour le nombre de valeurs par année par combinaison DIAGNOSIS_CATEGORY et CLASS_AGE
    result_df['WEEK_LABEL'] = pd.to_datetime(result_df['WEEK_LABEL'])
    result_df['YEAR'] = result_df['WEEK_LABEL'].dt.year

    result_df = result_df.sort_values(by='COMBI')
    result_df = result_df.drop(columns='COMBI')
    result_df = result_df.drop(columns='YEAR')
    
    weeks = pd.date_range(start=result_df['WEEK_LABEL'].min(), end=result_df['WEEK_LABEL'].max(), freq='W-Mon')


    categories_geo = result_df['GEOGRAPHICAL_ORIGIN'].unique()
    diagnosis_categories = ['COVID-19', 'FLU', 'RSV', 'GENERAL_RI', 'RI_OTHER_VIRUS']

    combinations = list(product(weeks, diagnosis_categories, categories_geo))

    columns = ['WEEK_LABEL', 'DIAGNOSIS_CATEGORY', 'GEOGRAPHICAL_ORIGIN']
    all_combinations_df = pd.DataFrame(combinations, columns=columns)

    merged_df = pd.merge(all_combinations_df, result_df, on=['WEEK_LABEL', 'GEOGRAPHICAL_ORIGIN','DIAGNOSIS_CATEGORY'], how='outer')

    merged_df = merged_df.sort_values(by='WEEK_LABEL')

    merged_df['N_CRITICAL'] = merged_df['N_CRITICAL'].fillna(0)
    merged_df['N_CASES'] = merged_df['N_CASES'].fillna(0)
    merged_df = merged_df.sort_values(by=['DIAGNOSIS_CATEGORY', 'GEOGRAPHICAL_ORIGIN', 'WEEK_LABEL'], ascending=[True, True, True])
    # drop rows where CATEGORY_DIAG is '-' or NaN
    merged_df = merged_df[merged_df['GEOGRAPHICAL_ORIGIN'] != '-']
    merged_df = merged_df[merged_df['DIAGNOSIS_CATEGORY'].notna()]
    merged_df = merged_df[merged_df['DIAGNOSIS_CATEGORY'] != '-']
    merged_df = merged_df[merged_df['DIAGNOSIS_CATEGORY'] != '']

    table_III = merged_df

    #------------------------------------
    ### TABLE III ###
    # create a dataframe filering the table_I dataframe where the DIAGNOSIS_CATEGORY is not equal to "GENERAL_RI"
    table_III_filtered = table_III[table_III["DIAGNOSIS_CATEGORY"] != "GENERAL_RI"]
    # create a new dataframe that groups the data by WEEK_LABEL, GEOGRAPHICAL_ORIGIN. Sums the N_CRITICAL, and N_CASES values and resets the index
    sum_DIAGNOSIS_CATEGORY = (
        table_III_filtered.groupby(["WEEK_LABEL", "GEOGRAPHICAL_ORIGIN"])[
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

def table_I_baseline_creation(df_number_of_id_vivalto_translate: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess 'df_number_of_id_vivalto_translate' to create the baseline reference.

    Args:
        df_number_of_id_vivalto_translate (pd.DataFrame): DataFrame for 'df_number_of_id_vivalto_translate'.

    Returns:
        pd.DataFrame: The preprocessed 'table_I' with the baseline reference calculated with CI.
    """

    table_I_baseline = df_number_of_id_vivalto_translate.copy()
    table_I_baseline["WEEK_LABEL"] = pd.to_datetime(
        table_I_baseline["WEEK_LABEL"], format="%Y-%m-%d"
    )
    table_I_baseline["MONTH_YEAR"] = table_I_baseline["WEEK_LABEL"].dt.strftime("%m-%y")

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
    
    grouped = table_I_baseline.groupby(["DIAGNOSIS_CATEGORY",'AGE_CLASS', 'RSS_DURATION_CLASS'])
    
    def calculate_confidence_interval(data):
        std_dev = data["BASELINE_N_CASES"].std()
        lower_ci = data["BASELINE_N_CASES"] - z * std_dev
        upper_ci = data["BASELINE_N_CASES"] + z * std_dev
        data['LOWER_CI_BASELINE_N_CASES'] = lower_ci
        data['UPPER_CI_BASELINE_N_CASES'] = upper_ci
        data['LOWER_CI_BASELINE_N_CASES'] = data['LOWER_CI_BASELINE_N_CASES'].fillna(0)
        data['UPPER_CI_BASELINE_N_CASES'] = data['UPPER_CI_BASELINE_N_CASES'].fillna(0)
        return data
    
    table_I_baseline = grouped.apply(calculate_confidence_interval).reset_index(drop=True)
    table_I_baseline['LOWER_CI_BASELINE_N_CASES'] = table_I_baseline['LOWER_CI_BASELINE_N_CASES'].apply(lambda x: max(x, 0))
    
    table_I_baseline_vivalto = table_I_baseline

    return table_I_baseline_vivalto

def table_II_baseline_creation(df_death_vivalto_translate: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess 'df_death_vivalto_translate' to create the baseline reference for deaths.

    Args:
        df_death_vivalto_translate (pd.DataFrame): DataFrame for 'df_death_vivalto_translate'.

    Returns:
        pd.DataFrame: The preprocessed 'df_death_vivalto_translate' with the baseline reference for deaths calculated.
    """
    

    table_II_baseline = df_death_vivalto_translate.copy()
    table_II_baseline["WEEK_LABEL"] = pd.to_datetime(
        table_II_baseline["WEEK_LABEL"], format="%Y-%m-%d"
    )
    table_II_baseline["MONTH_YEAR"] = table_II_baseline["WEEK_LABEL"].dt.strftime(
        "%m-%y"
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
        data['LOWER_CI_BASELINE_DEATH'] = lower_ci
        data['UPPER_CI_BASELINE_DEATH'] = upper_ci
        data['LOWER_CI_BASELINE_DEATH'] = data['LOWER_CI_BASELINE_DEATH'].fillna(0)
        data['UPPER_CI_BASELINE_DEATH'] = data['UPPER_CI_BASELINE_DEATH'].fillna(0)
        
        std_dev = data["BASELINE_N_CASES"].std()
        lower_ci = data["BASELINE_N_CASES"] - z * std_dev
        upper_ci = data["BASELINE_N_CASES"] + z * std_dev
        data['LOWER_CI_BASELINE_N_CASES'] = lower_ci
        data['UPPER_CI_BASELINE_N_CASES'] = upper_ci
        data['LOWER_CI_BASELINE_N_CASES'] = data['LOWER_CI_BASELINE_N_CASES'].fillna(0)
        data['UPPER_CI_BASELINE_N_CASES'] = data['UPPER_CI_BASELINE_N_CASES'].fillna(0)
        return data
    
    
    table_II_baseline = grouped.apply(calculate_confidence_interval).reset_index(drop=True)
    table_II_baseline['LOWER_CI_BASELINE_DEATH'] = table_II_baseline['LOWER_CI_BASELINE_DEATH'].apply(lambda x: max(x, 0))
    table_II_baseline['LOWER_CI_BASELINE_N_CASES'] = table_II_baseline['LOWER_CI_BASELINE_N_CASES'].apply(lambda x: max(x, 0))
    
    
    return table_II_baseline

def table_III_baseline_creation(df_critical_health_vivalto_translate: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess 'table_III' to create the baseline reference for critical cases.

    Args:
        table_III (pd.DataFrame): DataFrame for 'table_III'.

    Returns:
        pd.DataFrame: The preprocessed 'table_III' with the baseline reference for critical cases calculated.
    """

    table_III_baseline = df_critical_health_vivalto_translate.copy()
    table_III_baseline["WEEK_LABEL"] = pd.to_datetime(
        table_III_baseline["WEEK_LABEL"], format="%Y-%m-%d"
    )
    table_III_baseline["MONTH_YEAR"] = table_III_baseline["WEEK_LABEL"].dt.strftime(
        "%m-%y"
    )

    table_III_baseline["BASELINE_CRITICAL"] = (
        table_III_baseline.groupby(["MONTH_YEAR", "DIAGNOSIS_CATEGORY", "GEOGRAPHICAL_ORIGIN"])["N_CRITICAL"]
        .transform("mean")
        .round()
        .astype(int)
    )
    
    table_III_baseline["BASELINE_N_CASES"] = (
        table_III_baseline.groupby(["MONTH_YEAR", "DIAGNOSIS_CATEGORY","GEOGRAPHICAL_ORIGIN"])["N_CASES"]
        .transform("mean")
        .round()
        .astype(int)
    )
    table_III_baseline = table_III_baseline.drop("MONTH_YEAR", axis=1)
    
    # Define the confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the critical value using the normal distribution
    z = stats.norm.ppf((1 + confidence_level) / 2)

    grouped = table_III_baseline.groupby(["DIAGNOSIS_CATEGORY","GEOGRAPHICAL_ORIGIN"])
    
    def calculate_confidence_interval(data):
        std_dev = data["BASELINE_CRITICAL"].std()
        
        lower_ci = data["BASELINE_CRITICAL"] - z * std_dev
        upper_ci = data["BASELINE_CRITICAL"] + z * std_dev
        data['LOWER_CI_BASELINE_CRITICAL'] = lower_ci
        data['UPPER_CI_BASELINE_CRITICAL'] = upper_ci
        data['LOWER_CI_BASELINE_CRITICAL'] = data['LOWER_CI_BASELINE_CRITICAL'].fillna(0)
        data['UPPER_CI_BASELINE_CRITICAL'] = data['UPPER_CI_BASELINE_CRITICAL'].fillna(0)
        
        std_dev = data["BASELINE_N_CASES"].std()
        lower_ci = data["BASELINE_N_CASES"] - z * std_dev
        upper_ci = data["BASELINE_N_CASES"] + z * std_dev
        data['LOWER_CI_BASELINE_N_CASES'] = lower_ci
        data['UPPER_CI_BASELINE_N_CASES'] = upper_ci
        data['LOWER_CI_BASELINE_N_CASES'] = data['LOWER_CI_BASELINE_N_CASES'].fillna(0)
        data['UPPER_CI_BASELINE_N_CASES'] = data['UPPER_CI_BASELINE_N_CASES'].fillna(0)
        return data
    
    
    table_III_baseline = grouped.apply(calculate_confidence_interval).reset_index(drop=True)
    table_III_baseline['LOWER_CI_BASELINE_CRITICAL'] = table_III_baseline['LOWER_CI_BASELINE_CRITICAL'].apply(lambda x: max(x, 0))
    table_III_baseline['LOWER_CI_BASELINE_N_CASES'] = table_III_baseline['LOWER_CI_BASELINE_N_CASES'].apply(lambda x: max(x, 0))

    return table_III_baseline


def table_I_calculation_nowcasting(df: pd.DataFrame) -> pd.DataFrame:
    # Créer le vecteur correction_factor_last_weeks
    biased_weeks = 5
    correction_factor_last_weeks = np.linspace(1,0.5, biased_weeks + 1)

    # Créer une colonne correction_factor avec des valeurs 1 dans tout le DataFrame
    df['correction_factor'] = 1

    # Liste des catégories uniques pour chaque colonne
    categories_diagnosis = df['DIAGNOSIS_CATEGORY'].unique()
    categories_age = df['AGE_CLASS'].unique()
    categories_rss_duration = df['RSS_DURATION_CLASS'].unique()

    # Boucle sur chaque combinaison de catégories
    for diagnosis_category in categories_diagnosis:
        for age_class in categories_age:
            for rss_duration_class in categories_rss_duration:
                # Filtrer les lignes pour la combinaison actuelle de catégories
                filtered_df = df.loc[(df['DIAGNOSIS_CATEGORY'] == diagnosis_category) &
                                     (df['AGE_CLASS'] == age_class) &
                                     (df['RSS_DURATION_CLASS'] == rss_duration_class)]

                # Affecter les valeurs du vecteur correction_factor_last_weeks aux 10 dernières lignes de chaque sous-DataFrame
                nb_weeks = min(filtered_df.shape[0], biased_weeks)
                df.loc[filtered_df.index[-nb_weeks:], 'correction_factor'] = correction_factor_last_weeks[1:nb_weeks+1]


    df['N_CASES_NOWCAST'] = df['N_CASES'] / df['correction_factor'] #  TODO: May have to change to N_CASES_SMOOTHED_NOWCAST
    
    # TODO : move after smoothing
    df['ALERT_N_CASES'] = np.where((df['N_CASES_NOWCAST'] > df['UPPER_CI_BASELINE_N_CASES']),1,0) # TODO: May have to change to N_CASES_SMOOTHED_NOWCAST

    return df


def table_II_calculation_nowcasting(df: pd.DataFrame) -> pd.DataFrame:
    # Créer le vecteur correction_factor_last_weeks
    biased_weeks = 5
    correction_factor_last_weeks = np.linspace(1,0.5, biased_weeks + 1)

    # Créer une colonne correction_factor avec des valeurs 1 dans tout le DataFrame
    df['correction_factor'] = 1

    # Liste des catégories uniques pour chaque colonne
    categories_diagnosis = df['DIAGNOSIS_CATEGORY'].unique()

    # Boucle sur chaque combinaison de catégories
    for diagnosis_category in categories_diagnosis:
        # Filtrer les lignes pour la combinaison actuelle de catégories
        filtered_df = df.loc[(df['DIAGNOSIS_CATEGORY'] == diagnosis_category)]

        # Affecter les valeurs du vecteur correction_factor_last_weeks aux 10 dernières lignes de chaque sous-DataFrame
        nb_weeks = min(filtered_df.shape[0], biased_weeks)
        df.loc[filtered_df.index[-nb_weeks:], 'correction_factor'] = correction_factor_last_weeks[1:nb_weeks+1]


    df['N_CASES_NOWCAST'] = df['N_CASES'] / df['correction_factor'] # TODO: May have to change to N_CASES_SMOOTHED_NOWCAST
    df['N_DEATH_NOWCAST'] = df['N_DEATH'] / df['correction_factor'] # TODO: May have to change to N_CASES_SMOOTHED_NOWCAST
    
    # TODO : move after smoothing
    df['ALERT_N_DEATH'] = np.where((df['N_DEATH_NOWCAST'] > df['UPPER_CI_BASELINE_DEATH']),1,0) # TODO: May have to change to N_CASES_SMOOTHED_NOWCAST
    df['ALERT_N_CASES'] = np.where((df['N_CASES_NOWCAST'] > df['UPPER_CI_BASELINE_N_CASES']),1,0) # TODO: May have to change to N_CASES_SMOOTHED_NOWCAST

    return df


def table_III_calculation_nowcasting(df: pd.DataFrame) -> pd.DataFrame:
    # Créer le vecteur correction_factor_last_weeks
    biased_weeks = 5
    correction_factor_last_weeks = np.linspace(1,0.5, biased_weeks + 1)

    # Créer une colonne correction_factor avec des valeurs 1 dans tout le DataFrame
    df['correction_factor'] = 1

    # Liste des catégories uniques pour chaque colonne
    categories_diagnosis = df['DIAGNOSIS_CATEGORY'].unique()
    categories_geo = df['GEOGRAPHICAL_ORIGIN'].unique()


    # Boucle sur chaque combinaison de catégories
    for diagnosis_category in categories_diagnosis:
        for cat_geo in categories_geo:
            # Filtrer les lignes pour la combinaison actuelle de catégories
            filtered_df = df.loc[(df['DIAGNOSIS_CATEGORY'] == diagnosis_category) &
                                 (df['GEOGRAPHICAL_ORIGIN'] == cat_geo)]
            
            # Affecter les valeurs du vecteur correction_factor_last_weeks aux 10 dernières lignes de chaque sous-DataFrame
            nb_weeks = min(filtered_df.shape[0], biased_weeks)
            df.loc[filtered_df.index[-nb_weeks:], 'correction_factor'] = correction_factor_last_weeks[1:nb_weeks+1]

    df['N_CASES_NOWCAST'] = df['N_CASES'] / df['correction_factor'] # TODO: May have to change to N_CASES_SMOOTHED_NOWCAST
    df['N_CRITICAL_NOWCAST'] = df['N_CRITICAL'] / df['correction_factor'] # TODO: May have to change to N_CASES_SMOOTHED_NOWCAST
    
    # TODO : move after smoothing
    df['ALERT_N_CASES'] = np.where((df['N_CASES_NOWCAST'] > df['UPPER_CI_BASELINE_N_CASES']),1,0) # TODO: May have to change to N_CASES_SMOOTHED_NOWCAST
    df['ALERT_N_CRITICAL'] = np.where((df['N_CRITICAL_NOWCAST'] > df['UPPER_CI_BASELINE_CRITICAL']),1,0) # TODO: May have to change to N_CASES_SMOOTHED_NOWCAST

    return df


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
    merged_df['COMBI'] = merged_df['DIAGNOSIS_CATEGORY'] + ' | ' + merged_df['AGE_CLASS']+ ' | ' + merged_df['RSS_DURATION_CLASS']
    
    unique_combinations = merged_df['COMBI'].unique()

    # List to store historical dataframes for each combination
    df_smoothed_dataframes = []


    for i, combo in enumerate(unique_combinations):

        filtered_df = merged_df.loc[merged_df['COMBI'] == combo]
        filtered_df = filtered_df.sort_values(by='WEEK_LABEL')
        filtered_df['N_CASES_NOWCAST'] = filtered_df['N_CASES_NOWCAST'].astype(int)
        filtered_df['WEEK_LABEL'] = pd.to_datetime(filtered_df['WEEK_LABEL'])

        # Copy the filtered dataframe to avoid modifying the original
        smoothed_dataframes = filtered_df.copy()

        # Ensure 'WEEK_LABEL' is of type datetime
        smoothed_dataframes['WEEK_LABEL'] = pd.to_datetime(smoothed_dataframes['WEEK_LABEL'])

        # Sort the DataFrame by date
        smoothed_dataframes = smoothed_dataframes.sort_values(by='WEEK_LABEL')

        # Define the Gaussian window
        window_size = 5
        STD = 4
        gaussian_window = gaussian(window_size, std=STD)
        gaussian_window /= gaussian_window.sum()

        # Apply convolution with 'same' option
        smoothed = convolve(smoothed_dataframes['N_CASES_NOWCAST'].values, gaussian_window, mode='same')

        # Add the 'SMOOTHED_PASSAGE_DIAG_NB' column to the DataFrame
        smoothed_dataframes['N_CASES_NOWCAST_SMOOTH'] = smoothed

        smoothed_dataframes.sort_values(by='WEEK_LABEL', inplace=True)

        # Append the smoothed dataframe to the list
        df_smoothed_dataframes.append(smoothed_dataframes)


    # Concatenate all historical dataframes in the list
    final_smoothed_dataframe = pd.concat(df_smoothed_dataframes, ignore_index=True)
    # drop column COMBI
    final_smoothed_dataframe = final_smoothed_dataframe.drop(columns='COMBI')
    
    final_smoothed_dataframe= final_smoothed_dataframe.sort_values(
        by=["DIAGNOSIS_CATEGORY", "AGE_CLASS", "RSS_DURATION_CLASS", "WEEK_LABEL"],
        ascending=[True, True, True, True],
    )

    return final_smoothed_dataframe

def smooth_n_death(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join.

    Args:
        df_cas_pos (pd.DataFrame): Processed DataFrame for 'Cas_pos' sheets.
        df_nb_prelev (pd.DataFrame): Processed DataFrame for 'Nb_prélèvement' sheets.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data (number of pos tests + number of test).
    """
    # Assuming merged_df is your original DataFrame
    merged_df['COMBI'] = merged_df['DIAGNOSIS_CATEGORY'] 
    
    unique_combinations = merged_df['COMBI'].unique()

    # List to store historical dataframes for each combination
    df_smoothed_dataframes = []


    for i, combo in enumerate(unique_combinations):

        filtered_df = merged_df.loc[merged_df['COMBI'] == combo]
        filtered_df = filtered_df.sort_values(by='WEEK_LABEL')
        filtered_df['N_CASES_NOWCAST'] = filtered_df['N_CASES_NOWCAST'].astype(int)
        filtered_df['N_DEATH_NOWCAST'] = filtered_df['N_DEATH_NOWCAST'].astype(int)
        filtered_df['WEEK_LABEL'] = pd.to_datetime(filtered_df['WEEK_LABEL'])

        # Copy the filtered dataframe to avoid modifying the original
        smoothed_dataframes = filtered_df.copy()

        # Ensure 'WEEK_LABEL' is of type datetime
        smoothed_dataframes['WEEK_LABEL'] = pd.to_datetime(smoothed_dataframes['WEEK_LABEL'])

        # Sort the DataFrame by date
        smoothed_dataframes = smoothed_dataframes.sort_values(by='WEEK_LABEL')

        # Define the Gaussian window
        window_size = 5
        STD = 4
        gaussian_window = gaussian(window_size, std=STD)
        gaussian_window /= gaussian_window.sum()

        # Apply convolution with 'same' option
        smoothed = convolve(smoothed_dataframes['N_CASES_NOWCAST'].values, gaussian_window, mode='same')

        # Add the 'SMOOTHED_PASSAGE_DIAG_NB' column to the DataFrame
        smoothed_dataframes['N_CASES_NOWCAST_SMOOTH'] = smoothed
        
        # Apply convolution with 'same' option
        smoothed = convolve(smoothed_dataframes['N_DEATH_NOWCAST'].values, gaussian_window, mode='same')

        # Add the 'SMOOTHED_PASSAGE_DIAG_NB' column to the DataFrame
        smoothed_dataframes['N_DEATH_NOWCAST_SMOOTH'] = smoothed

        smoothed_dataframes.sort_values(by='WEEK_LABEL', inplace=True)

        # Append the smoothed dataframe to the list
        df_smoothed_dataframes.append(smoothed_dataframes)


    # Concatenate all historical dataframes in the list
    final_smoothed_dataframe = pd.concat(df_smoothed_dataframes, ignore_index=True)
    
    final_smoothed_dataframe= final_smoothed_dataframe.sort_values(
        by=["DIAGNOSIS_CATEGORY", "WEEK_LABEL"],
        ascending=[True, True],
    )
    # drop column COMBI
    final_smoothed_dataframe = final_smoothed_dataframe.drop(columns='COMBI')

    return final_smoothed_dataframe


def smooth_n_critical(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join.

    Args:
        df_cas_pos (pd.DataFrame): Processed DataFrame for 'Cas_pos' sheets.
        df_nb_prelev (pd.DataFrame): Processed DataFrame for 'Nb_prélèvement' sheets.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data (number of pos tests + number of test).
    """
    # Assuming merged_df is your original DataFrame
    merged_df['COMBI'] = merged_df['DIAGNOSIS_CATEGORY'] + ' | ' + merged_df['GEOGRAPHICAL_ORIGIN']
    unique_combinations = merged_df['COMBI'].unique()

    # List to store historical dataframes for each combination
    df_smoothed_dataframes = []


    for i, combo in enumerate(unique_combinations):

        filtered_df = merged_df.loc[merged_df['COMBI'] == combo]
        filtered_df = filtered_df.sort_values(by='WEEK_LABEL')
        filtered_df['N_CRITICAL_NOWCAST'] = filtered_df['N_CRITICAL_NOWCAST'].astype(int)
        filtered_df['N_CASES_NOWCAST'] = filtered_df['N_CASES_NOWCAST'].astype(int)
        filtered_df['WEEK_LABEL'] = pd.to_datetime(filtered_df['WEEK_LABEL'])

        # Copy the filtered dataframe to avoid modifying the original
        smoothed_dataframes = filtered_df.copy()

        # Ensure 'WEEK_LABEL' is of type datetime
        smoothed_dataframes['WEEK_LABEL'] = pd.to_datetime(smoothed_dataframes['WEEK_LABEL'])

        # Sort the DataFrame by date
        smoothed_dataframes = smoothed_dataframes.sort_values(by='WEEK_LABEL')

        # Define the Gaussian window
        window_size = 5
        STD = 4
        gaussian_window = gaussian(window_size, std=STD)
        gaussian_window /= gaussian_window.sum()

        # Apply convolution with 'same' option
        smoothed = convolve(smoothed_dataframes['N_CRITICAL_NOWCAST'].values, gaussian_window, mode='same')
        smoothed_dataframes['N_CRITICAL_NOWCAST_SMOOTH'] = smoothed
        
        smoothed = convolve(smoothed_dataframes['N_CASES_NOWCAST'].values, gaussian_window, mode='same')
        smoothed_dataframes['N_CASES_NOWCAST_SMOOTH'] = smoothed
        
        smoothed_dataframes.sort_values(by='WEEK_LABEL', inplace=True)

        # Append the smoothed dataframe to the list
        df_smoothed_dataframes.append(smoothed_dataframes)


    # Concatenate all historical dataframes in the list
    final_smoothed_dataframe = pd.concat(df_smoothed_dataframes, ignore_index=True)
    # drop column COMBI
    final_smoothed_dataframe = final_smoothed_dataframe.drop(columns='COMBI')
    
    final_smoothed_dataframe= final_smoothed_dataframe.sort_values(
        by=["DIAGNOSIS_CATEGORY", "GEOGRAPHICAL_ORIGIN", "WEEK_LABEL"],
        ascending=[True, True, True],
    )

    return final_smoothed_dataframe

def output_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Save the processed data to an Excel file.

    Args:
        df (pd.DataFrame): Processed DataFrame.

    Returns:
        pd.DataFrame: DataFrame with column names as the first row and column types as the second row.
    """
    df = pd.DataFrame(df)
    # Obtenez les noms de colonnes
    column_names = df.columns.tolist()

    # Créez un DataFrame contenant les noms de colonnes
    df_column_names = pd.DataFrame([column_names], columns=column_names)

    # Obtenez les types de données des colonnes
    column_types = df.dtypes.tolist()

    # Créez un DataFrame contenant les types de données
    df_column_types = pd.DataFrame([column_types], columns=column_names)

    # Concaténez les deux DataFrames
    df_with_column_names_and_types = pd.concat([df_column_names, df_column_types, df], ignore_index=True)

    return df_with_column_names_and_types