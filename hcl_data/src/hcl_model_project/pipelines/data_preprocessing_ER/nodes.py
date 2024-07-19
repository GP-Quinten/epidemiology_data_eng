from itertools import product

import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.signal.windows import gaussian


def upload_data(df):
    print(df.columns)
    return df


def preprocessing_by_sheets(HCL_data: dict) -> tuple:
    """
    Preprocesses data from multiple sheets in multiple Excel files.

    Args:
        HCL_data (dict): A dictionary containing Excel file data.

    Returns:
        tuple: A tuple containing two DataFrames:
            - df_cas_pos: Processed DataFrame for 'Cas_pos' sheets.
            - df_nb_prelev: Processed DataFrame for 'Nb_prélèvement' sheets.
    """
    concatenated_dataframes = []
    # Iterate through each Excel file in the dictionary
    for file_name, sheets in HCL_data.items():
        # List to store concatenated DataFrames for each sheet of the Excel file
        concatenated_sheets = []

        # Iterate through each sheet of the Excel file
        for sheet_name, dataframe in sheets.items():
            # Remove leading and trailing whitespaces from column names
            dataframe.rename(columns=lambda x: x.strip(), inplace=True)
            # Concatenate the tables from each sheet vertically
            concatenated_sheets.append(dataframe)
            # print(dataframe.columns)

        # Concatenate the tables from each sheet of each Excel file vertically
        concatenated_dataframes.append(
            pd.concat(concatenated_sheets, ignore_index=True, sort=False, axis=0)
        )

    # Concatenate the tables from each Excel file vertically
    final_df = pd.concat(
        concatenated_dataframes, ignore_index=True, sort=False, axis=0
    )
    # print(final_df.columns)
    final_df["DATE_EXTRACT"] = pd.to_datetime(
        final_df["DATE_EXTRACT"]
    )#.dt.date

    return final_df


def date_preprocessing(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join.

    Args:
        df_cas_pos (pd.DataFrame): Processed DataFrame for 'Cas_pos' sheets.
        df_nb_prelev (pd.DataFrame): Processed DataFrame for 'Nb_prélèvement' sheets.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data (number of pos tests + number of test).
    """
    final_df["WEEK_LABEL"] = final_df[
        "SEMAINE_PASSAGE_LIB"
    ].str.extract(r"(\d{2}/\d{2}/\d{2})")
    # print(final_df["WEEK_LABEL"])
    final_df["WEEK_LABEL"] = pd.to_datetime(
        final_df["WEEK_LABEL"], format="%d/%m/%y"
    )
    # print(final_df["WEEK_LABEL"])

    return final_df


def sum_same_week(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join.

    Args:
        df_cas_pos (pd.DataFrame): Processed DataFrame for 'Cas_pos' sheets.
        df_nb_prelev (pd.DataFrame): Processed DataFrame for 'Nb_prélèvement' sheets.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data (number of pos tests + number of test).
    """
    final_df = final_df.groupby(
        [
            "DATE_EXTRACT",
            "WEEK_LABEL",
            "CLASSE_AGE",
            "DIAGNOSTIC_CODE",
            "DIAGNOSTIC_LIB",
        ],
        as_index=False,
    ).agg({"PASSAGE_NB": "sum", "PASSAGE_DIAG_NB": "sum"})

    return final_df


def translate_rename(final_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join.

    Args:
        df_cas_pos (pd.DataFrame): Processed DataFrame for 'Cas_pos' sheets.
        df_nb_prelev (pd.DataFrame): Processed DataFrame for 'Nb_prélèvement' sheets.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data (number of pos tests + number of test).
    """
    final_dataframe.rename(columns={"CLASSE_AGE": "AGE_CLASS"}, inplace=True)

    # Translation for AGE_CLASS column
    age_class_translation = {
        "<1": "Less than 1 year",
        "1-4": "[1 - 5[ year(s)",
        "5-19": "[5 - 20[ years",
        "20-49": "[20 - 50[ years",
        "50-64": "[50 - 65[ years",
        "65+": "65 years and older",
    }
    final_dataframe["AGE_CLASS"] = final_dataframe["AGE_CLASS"].replace(
        age_class_translation
    )

    return final_dataframe


def ICD_code_processing(final_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join.

    Args:
        df_cas_pos (pd.DataFrame): Processed DataFrame for 'Cas_pos' sheets.
        df_nb_prelev (pd.DataFrame): Processed DataFrame for 'Nb_prélèvement' sheets.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data (number of pos tests + number of test).
    """
    # Définissez les conditions et les valeurs correspondantes
    conditions = [
        (
            final_dataframe["DIAGNOSTIC_CODE"].isin(
                ["U07.10", "U07.11", "U07.12", "U07.14", "U07.15"]
            )
        ),
        (final_dataframe["DIAGNOSTIC_CODE"].str.startswith(("J09", "J10", "J11"))),
        (
            final_dataframe["DIAGNOSTIC_CODE"].str.startswith(
                ("J12.1", "J20.5", "J21.0")
            )
        ),
        (final_dataframe["DIAGNOSTIC_CODE"].eq("J12.1")),
        (
            final_dataframe["DIAGNOSTIC_CODE"].eq("J17.1")
            | final_dataframe["DIAGNOSTIC_CODE"].str.startswith("J12")
            & ~final_dataframe["DIAGNOSTIC_CODE"].str.startswith("J12.1")
        ),
        (
            final_dataframe["DIAGNOSTIC_CODE"].str.startswith(
                (
                    "J00",
                    "J01",
                    "J02",
                    "J03",
                    "J04",
                    "J05",
                    "J06",
                    "J09",
                    "J10",
                    "J11",
                    "J12",
                    "J13",
                    "J14",
                    "J15",
                    "J16",
                    "J17",
                    "J18",
                    "J20",
                    "J21",
                    "J22",
                )
            )
        ),
        (
            final_dataframe["DIAGNOSTIC_CODE"].eq("J44.0")
            | final_dataframe["DIAGNOSTIC_CODE"].str.startswith(("J85", "J86"))
        ),
    ]

    values = [
        "COVID_19",
        "FLU",
        "RSV",
        "RSV",
        "RI_OTHER_VIRUS",
        "GENERAL_RI",
        "GENERAL_RI",
    ]

    # Utilisez la méthode numpy.select pour créer la nouvelle colonne
    new_diag_cat = np.select(conditions, values, default="UNKNOWN")
    final_dataframe.loc[:, "DIAGNOSIS_CATEGORY"] = new_diag_cat

    return final_dataframe


def preprocess_wrong_class(final_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join.

    Args:
        df_cas_pos (pd.DataFrame): Processed DataFrame for 'Cas_pos' sheets.
        df_nb_prelev (pd.DataFrame): Processed DataFrame for 'Nb_prélèvement' sheets.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data (number of pos tests + number of test).
    """
    # Définissez les conditions et les valeurs correspondantes
    result_df = (
        final_dataframe.groupby(["WEEK_LABEL", "AGE_CLASS", "DIAGNOSIS_CATEGORY"])[
            "PASSAGE_DIAG_NB"
        ]
        .sum()
        .reset_index()
    )
    result_df = result_df[result_df["AGE_CLASS"] != "121"]

    # create a dataframe called age_class_subtotals grouped by WEEK_LABEL, AGE_CLASS
    age_class_subtotals = (
        result_df.groupby(["WEEK_LABEL", "AGE_CLASS"])[
            "PASSAGE_DIAG_NB"
        ]
        .sum()
        .reset_index()
    )
    # add a new column called 'DIAGNOSIS_CATEGORY' with value 'ALL_RESP_DIAG' at second position
    age_class_subtotals.insert(1, "DIAGNOSIS_CATEGORY", "ALL_RESP_DIAG")
    # show the column names
    print("age_class_subtotals columns", age_class_subtotals.columns)

    # concatenate the two dataframes on axis 0
    result_df = pd.concat([result_df, age_class_subtotals], axis=0)

    # create a dataframe called diagnosis_category_subtotals grouped by WEEK_LABEL, DIAGNOSIS_CATEGORY
    diagnosis_category_subtotals = (
        result_df.groupby(["WEEK_LABEL", "DIAGNOSIS_CATEGORY"])[
            "PASSAGE_DIAG_NB"
        ]
        .sum()
        .reset_index()
    )

    # add a new column called 'AGE_CLASS' with value '0-200' at third position
    diagnosis_category_subtotals.insert(2, "AGE_CLASS", "0-200")
    # show the column names
    print("diagnosis_category_subtotals columns", diagnosis_category_subtotals.columns)
    
    # concatenate the two dataframes on axis 0
    result_df = pd.concat([result_df, diagnosis_category_subtotals], axis=0)

    return result_df


def missing_data_prepro(result_df: pd.DataFrame) -> pd.DataFrame:
    """ """
    # Définissez les conditions et les valeurs correspondantes
    result_df["COMBI_DIAG_AGE"] = (
        result_df["DIAGNOSIS_CATEGORY"] + " | " + result_df["AGE_CLASS"]
    )

    # Créer un graphique en bar plot pour le nombre de valeurs par année par combinaison DIAGNOSIS_CATEGORY et CLASS_AGE
    result_df["YEAR"] = result_df["WEEK_LABEL"].dt.year

    result_df = result_df.sort_values(by="COMBI_DIAG_AGE")
    result_df = result_df.drop(columns="COMBI_DIAG_AGE")
    result_df = result_df.drop(columns="YEAR")

    weeks = pd.date_range(
        start=result_df["WEEK_LABEL"].min(),
        end=result_df["WEEK_LABEL"].max(),
        freq="W-Mon",
    )

    categories_age = result_df["AGE_CLASS"].unique()
    diagnosis_categories = result_df["DIAGNOSIS_CATEGORY"].unique()

    combinations = list(product(weeks, diagnosis_categories, categories_age))

    columns = ["WEEK_LABEL", "DIAGNOSIS_CATEGORY", "AGE_CLASS"]
    all_combinations_df = pd.DataFrame(combinations, columns=columns)

    merged_df = pd.merge(
        all_combinations_df,
        result_df,
        on=["WEEK_LABEL", "AGE_CLASS", "DIAGNOSIS_CATEGORY"],
        how="outer",
    )

    merged_df = merged_df.sort_values(by="WEEK_LABEL")

    merged_df["PASSAGE_DIAG_NB"] = merged_df["PASSAGE_DIAG_NB"].fillna(0)

    merged_df["COMBI_DIAG_AGE"] = (
        merged_df["DIAGNOSIS_CATEGORY"] + " | " + merged_df["AGE_CLASS"]
    )

    # Créer un graphique en bar plot pour le nombre de valeurs par année par combinaison DIAGNOSIS_CATEGORY et CLASS_AGE
    merged_df["YEAR"] = merged_df["WEEK_LABEL"].dt.year

    merged_df = merged_df.sort_values(by="COMBI_DIAG_AGE")

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
    unique_combinations = merged_df["COMBI_DIAG_AGE"].unique()

    # List to store historical dataframes for each combination
    df_smoothed_dataframes = []

    # Set up the number of rows and columns for subplots
    num_rows = len(unique_combinations) // 2 + len(unique_combinations) % 2
    num_cols = 2

    for i, combo in enumerate(unique_combinations):

        filtered_df = merged_df.loc[merged_df["COMBI_DIAG_AGE"] == combo]
        filtered_df = filtered_df.sort_values(by="WEEK_LABEL")
        filtered_df["PASSAGE_DIAG_NB"] = filtered_df["PASSAGE_DIAG_NB"].astype(int)
        filtered_df["WEEK_LABEL"] = pd.to_datetime(filtered_df["WEEK_LABEL"])

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
            smoothed_dataframes["PASSAGE_DIAG_NB"].values, gaussian_window, mode="same"
        )

        # Add the 'SMOOTHED_PASSAGE_DIAG_NB' column to the DataFrame
        smoothed_dataframes["SMOOTHED_PASSAGE_DIAG_NB"] = smoothed

        # Calculate variance for confidence interval
        variance = convolve(
            (smoothed_dataframes["PASSAGE_DIAG_NB"].values - smoothed) ** 2,
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
        smoothed_dataframes["LOWER_CI_SMOOTHED_PASSAGE_DIAG_NB"] = (
            smoothed_dataframes["SMOOTHED_PASSAGE_DIAG_NB"] - margin_of_error
        )
        smoothed_dataframes["UPPER_CI_SMOOTHED_PASSAGE_DIAG_NB"] = (
            smoothed_dataframes["SMOOTHED_PASSAGE_DIAG_NB"] + margin_of_error
        )

        # Sort the DataFrame by date
        smoothed_dataframes.sort_values(by="WEEK_LABEL", inplace=True)

        # Check and replace NaN values with 0 in the columns used
        smoothed_dataframes["LOWER_CI_SMOOTHED_PASSAGE_DIAG_NB"] = smoothed_dataframes[
            "LOWER_CI_SMOOTHED_PASSAGE_DIAG_NB"
        ].clip(lower=0)

        # Check for NaN values in the confidence interval columns
        lower_ci_smoothed = np.where(
            np.isnan(smoothed_dataframes["LOWER_CI_SMOOTHED_PASSAGE_DIAG_NB"]),
            0,
            smoothed_dataframes["LOWER_CI_SMOOTHED_PASSAGE_DIAG_NB"],
        )
        upper_ci_smoothed = np.where(
            np.isnan(smoothed_dataframes["UPPER_CI_SMOOTHED_PASSAGE_DIAG_NB"]),
            0,
            smoothed_dataframes["UPPER_CI_SMOOTHED_PASSAGE_DIAG_NB"],
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
    unique_combinations = final_smoothed_dataframe["COMBI_DIAG_AGE"].unique()

    # Liste pour stocker les DataFrames résultants pour chaque combinaison
    result_dataframes = []

    # Définir le décalage d'un an
    one_year_shift = pd.DateOffset(weeks=52)

    # Boucle sur chaque combinaison
    for combo in unique_combinations:
        # Filtrer le DataFrame pour la combinaison actuelle
        df = final_smoothed_dataframe[
            final_smoothed_dataframe["COMBI_DIAG_AGE"] == combo
        ].copy()

        # make sure 'WEEK_LABEL' is of type datetime
        df["WEEK_LABEL"] = pd.to_datetime(df["WEEK_LABEL"])
        # Ajouter une colonne 'WEEK_LABEL_1YR_AGO' pour stocker la date correspondante il y a un an
        df["WEEK_LABEL_1YR_AGO"] = df["WEEK_LABEL"] - one_year_shift

        # Fusionner avec lui-même pour obtenir les valeurs de l'année précédente
        merged_df = pd.merge(
            df,
            df[
                [
                    "WEEK_LABEL",
                    "SMOOTHED_PASSAGE_DIAG_NB",
                    "LOWER_CI_SMOOTHED_PASSAGE_DIAG_NB",
                    "UPPER_CI_SMOOTHED_PASSAGE_DIAG_NB",
                ]
            ],
            left_on="WEEK_LABEL_1YR_AGO",
            right_on="WEEK_LABEL",
            suffixes=("", "_1YR_AGO"),
            how="left",
        )

        # Remplacer les valeurs manquantes par celles de l'année précédente
        merged_df["SMOOTHED_PASSAGE_DIAG_NB_1YR_AGO"].fillna(
            merged_df["SMOOTHED_PASSAGE_DIAG_NB"], inplace=True
        )
        merged_df["LOWER_CI_SMOOTHED_PASSAGE_DIAG_NB_1YR_AGO"].fillna(
            merged_df["LOWER_CI_SMOOTHED_PASSAGE_DIAG_NB"], inplace=True
        )
        merged_df["UPPER_CI_SMOOTHED_PASSAGE_DIAG_NB_1YR_AGO"].fillna(
            merged_df["UPPER_CI_SMOOTHED_PASSAGE_DIAG_NB"], inplace=True
        )

        # Supprimer la colonne 'WEEK_LABEL_1YR_AGO'
        merged_df.drop(["WEEK_LABEL_1YR_AGO"], axis=1, inplace=True)

        # Ajouter le DataFrame résultant à la liste
        result_dataframes.append(merged_df)

    # Concaténer tous les DataFrames résultants
    final_result_dataframe = pd.concat(result_dataframes, ignore_index=True)

    # Afficher les 50 dernières lignes du DataFrame final résultant
    final_result_dataframe.rename(
        columns={"SMOOTHED_PASSAGE_DIAG_NB_1YR_AGO": "BASELINE_PASSAGE_DIAG_NB"},
        inplace=True,
    )
    final_result_dataframe.rename(
        columns={
            "LOWER_CI_SMOOTHED_PASSAGE_DIAG_NB_1YR_AGO": "LOWER_CI_BASELINE_PASSAGE_DIAG_NB"
        },
        inplace=True,
    )
    final_result_dataframe.rename(
        columns={
            "UPPER_CI_SMOOTHED_PASSAGE_DIAG_NB_1YR_AGO": "UPPER_CI_BASELINE_PASSAGE_DIAG_NB"
        },
        inplace=True,
    )

    age_class_order = [
        "Less than 1 year",
        "[1 - 5[ year(s)",
        "[5 - 20[ years",
        "[20 - 50[ years",
        "[50 - 65[ years",
        "65 years and older",
    ]

    final_result_dataframe = final_result_dataframe.sort_values(
        by=["WEEK_LABEL", "DIAGNOSIS_CATEGORY", "AGE_CLASS"],
        key=lambda x: pd.Categorical(x, categories=age_class_order, ordered=True),
    )

    final_result_dataframe = final_result_dataframe.drop(
        ["COMBI_DIAG_AGE", "YEAR"], axis=1
    )

    final_result_dataframe["ALERT_N_CASES"] = np.where(
        (
            final_result_dataframe["PASSAGE_DIAG_NB"]
            > final_result_dataframe["UPPER_CI_BASELINE_PASSAGE_DIAG_NB"]
        ),
        1,
        0,
    )

    # print percentage of alerts
    print(
        "Percentage of alerts CGM ER data: ",
        final_result_dataframe["ALERT_N_CASES"].sum()
        / final_result_dataframe["ALERT_N_CASES"].count(),
    )

    return final_result_dataframe


def fraction_n_cases(final_result_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join.

    Args:
        df_cas_pos (pd.DataFrame): Processed DataFrame for 'Cas_pos' sheets.
        df_nb_prelev (pd.DataFrame): Processed DataFrame for 'Nb_prélèvement' sheets.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data (number of pos tests + number of test).
    """
    # Appliquer la fonction pour PASSAGE_DIAG_NB
    final_result_dataframe = calculate_fraction_and_total(
        final_result_dataframe,
        "PASSAGE_DIAG_NB",
        "TOTAL_NB_OF_PATIENTS",
        "FRACTION_OF_CASES",
    )

    # Appliquer la fonction pour SMOOTHED_PASSAGE_DIAG_NB
    final_result_dataframe = calculate_fraction_and_total(
        final_result_dataframe,
        "SMOOTHED_PASSAGE_DIAG_NB",
        "TOTAL_NB_OF_SMOOTHED",
        "FRACTION_OF_CASES_SMOOTHED",
    )

    # Appliquer la fonction pour BASELINE_PASSAGE_DIAG_NB
    final_result_dataframe = calculate_fraction_and_total(
        final_result_dataframe,
        "BASELINE_PASSAGE_DIAG_NB",
        "TOTAL_NB_OF_BASELINE",
        "FRACTION_OF_CASES_BASELINE",
    )

    return final_result_dataframe


def calculate_fraction_and_total(
    dataframe, column_name, total_column_name, fraction_column_name
):
    # Somme des colonnes pour chaque semaine et classe d'âge
    total_column = dataframe.groupby(["WEEK_LABEL", "AGE_CLASS"], as_index=False)[
        column_name
    ].sum()

    # Renommer la colonne pour refléter qu'il s'agit du total
    total_column = total_column.rename(columns={column_name: total_column_name})

    # Fusionner le dataframe principal avec le total
    dataframe = pd.merge(dataframe, total_column, on=["WEEK_LABEL", "AGE_CLASS"])

    # Créer la colonne FRACTION
    dataframe[fraction_column_name] = (
        dataframe[column_name] / dataframe[total_column_name]
    )

    dataframe = dataframe.drop(total_column_name, axis=1)

    return dataframe


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
    df_with_column_names_and_types = pd.concat(
        [df_column_names, df_column_types, df], ignore_index=True
    )

    return df_with_column_names_and_types
