import re

import numpy as np
import pandas as pd
import scipy.stats as stats


def upload_data(df):
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

    # Initialize lists to store DataFrames for 'Cas_pos' and 'Nb_prélèvement'
    cas_pos_dfs = []
    nb_prelev_dfs = []

    # Iterate through all DataFrames in the 'HCL_data' dictionary
    for excel_name, excel_df in HCL_data.items():
        # Copy the current dictionary keys into a list
        sheet_names = list(excel_df.keys())

        # Iterate through sheet names and rename them by replacing spaces with underscores
        for sheet_name in sheet_names:
            new_sheet_name = sheet_name.replace(" ", "_")
            # Temporarily store the current value under the old key
            sheet_data = excel_df[sheet_name]
            # Remove the old key (sheet name)
            del excel_df[sheet_name]
            # Add a new key with the new sheet name
            excel_df[new_sheet_name] = sheet_data

        # Fetch the DataFrame associated with the "total" sheet and assign it to df_total
        df_total = excel_df["total"]
        # Continue with your DataFrame operations here
        # Transpose the DataFrame, switching rows and columns
        df_total = df_total.transpose()
        df_total.reset_index(inplace=True)

        # Set the column names (headers) based on the first row of the DataFrame
        df_total.columns = df_total.iloc[0]

        # Drop the first row as it now contains the column names
        df_total = df_total.iloc[1:]

        if "Cas_pos" and "Nb_prélèvement" in excel_df:
            # For the 'Cas_pos' DataFrame
            # Rename the 'Cas pos' column to 'Disease'
            df = excel_df["Cas_pos"].rename(columns={"Cas pos": "Disease"})

            df["Disease"] = df["Disease"].replace("VRS", "RSV")

            # Replace spaces with underscores and add 'Cas_Pos_'
            df["Disease"] = "Cas_Pos_" + df["Disease"].str.replace(" ", "_")
            # Drop the 'Total' column
            df = df.drop(columns=["Total"])
            df_total["Année"] = df_total["Année"].astype(str)
            # Ajoutez une nouvelle colonne 'Annee_format' pour stocker le format de l'année
            df_total["Annee_format"] = df_total["Année"].apply(
                lambda x: "20" + x if len(x) == 2 else x
            )
            df_total["Date"] = pd.to_datetime(
                df_total["Annee_format"].astype(str)
                + df_total["N semaine"].astype(str)
                + "0",
                format="%Y%W%w",
            )
            df_total = df_total.drop(columns=["Annee_format"])

            # Create a 'Date' column
            # df_total['Date'] = pd.to_datetime(df_total['Année'].astype(str) + df_total['N semaine'].astype(str) + '0', format='%Y%W%w')
            df["Date"] = df_total["Date"].values[0]
            # Reshape the data
            df = df.melt(
                id_vars=["Date", "Disease"], var_name="Age_class", value_name="Cas_pos"
            )
            df = df.pivot(
                index=["Date", "Age_class"], columns="Disease", values="Cas_pos"
            )
            df.reset_index(inplace=True)
            df.columns.name = None
            cas_pos_dfs.append(df)

        if "Nb_prélèvement" in excel_df:
            # For the 'Nb_prélèvement' DataFrame
            # Rename the 'Nb prélèvement' column to 'Disease'
            df = excel_df["Nb_prélèvement"].rename(
                columns={"Nb prélèvement": "Disease"}
            )
            # Replace spaces with underscores and add 'Nb_Prelev_'
            df["Disease"] = df["Disease"].replace("VRS", "RSV")
            df["Disease"] = "Nb_Prelev_" + df["Disease"].str.replace(" ", "_")
            # Drop the 'Total' column
            df = df.drop(columns=["Total"])

            df_total["Année"] = df_total["Année"].astype(str)
            # Ajoutez une nouvelle colonne 'Annee_format' pour stocker le format de l'année
            df_total["Annee_format"] = df_total["Année"].apply(
                lambda x: "20" + x if len(x) == 2 else x
            )
            df_total["Date"] = pd.to_datetime(
                df_total["Annee_format"].astype(str)
                + df_total["N semaine"].astype(str)
                + "0",
                format="%Y%W%w",
            )
            df_total = df_total.drop(columns=["Annee_format"])

            # Create a 'Date' column
            # df_total['Date'] = pd.to_datetime(df_total['Année'].astype(str) + df_total['N semaine'].astype(str) + '0', format='%Y%W%w')
            df["Date"] = df_total["Date"].values[0]
            # Reshape the data
            df = df.melt(
                id_vars=["Date", "Disease"],
                var_name="Age_class",
                value_name="Nb_prelev",
            )
            df = df.pivot(
                index=["Date", "Age_class"], columns="Disease", values="Nb_prelev"
            )
            df.reset_index(inplace=True)
            df.columns.name = None
            nb_prelev_dfs.append(df)

    # Concatenate DataFrames for 'Cas_pos'
    df_cas_pos = pd.concat(cas_pos_dfs, axis=0, ignore_index=True)

    # Concatenate DataFrames for 'Nb_prélèvement'
    df_nb_prelev = pd.concat(nb_prelev_dfs, axis=0, ignore_index=True)
    return df_cas_pos, df_nb_prelev


def merge_cas_pos_and_nb_prelev(
    df_cas_pos: pd.DataFrame, df_nb_prelev: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join.

    Args:
        df_cas_pos (pd.DataFrame): Processed DataFrame for 'Cas_pos' sheets.
        df_nb_prelev (pd.DataFrame): Processed DataFrame for 'Nb_prélèvement' sheets.

    Returns:
        pd.DataFrame: Merged DataFrame containing combined data (number of pos tests + number of test).
    """
    # Merge 'df_cas_pos' and 'df_nb_prelev' DataFrames on ['Date', 'Age_class'] with an outer join
    merged_df = df_cas_pos.merge(df_nb_prelev, on=["Date", "Age_class"], how="outer")

    return merged_df


def translate_and_reorder(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate and reorder 'merged_df' DataFrame columns.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame to be processed.

    Returns:
        pd.DataFrame: Translated and reordered DataFrame.
    """

    # Create a translation dictionary for age class labels
    age_class_translation = {
        "<1an": "< 1 year",
        "1 à 5": "1 to 5",
        "6 à 11": "6 to 11",
        "11 à 18": "11 to 18",
        "18 à 40": "18 to 40",
        "40 à 65": "40 to 65",
        ">65ans": "> 65 years",
    }

    # Apply the translation using the .replace() method to update age class labels
    merged_df["Age_class"] = (
        merged_df["Age_class"].str.replace(" ", "").str.replace("à", " à ")
    )
    merged_df["Age_class"] = merged_df["Age_class"].replace(age_class_translation)

    # Rename the columns dynamically to match the desired format
    merged_df = merged_df.rename(
        columns=lambda x: x.replace("Nb_Prelev_", "Nb_of_test_")
    )
    merged_df = merged_df.rename(columns=lambda x: x.replace("Cas_Pos_", "Pos_test_"))

    # Define the desired order of Age_class
    age_class_order = [
        "< 1 year",
        "1 to 5",
        "6 to 11",
        "11 to 18",
        "18 to 40",
        "40 to 65",
        "> 65 years",
    ]

    # Create a Categorical data type for Age_class with the desired order
    merged_df["Age_class"] = pd.Categorical(
        merged_df["Age_class"], categories=age_class_order, ordered=True
    )

    # Sort the DataFrame by Date and Age_class
    hcl_lab_final = merged_df.sort_values(by=["Date", "Age_class"])

    return hcl_lab_final


def create_baseline(hcl_lab_final):

    hcl_lab_final_df = pd.DataFrame(hcl_lab_final)
    hcl_lab_final_df["Date"] = pd.to_datetime(hcl_lab_final_df["Date"])

    # Assuming 'Date' column is in datetime format
    hcl_lab_final_df["Month"] = hcl_lab_final_df["Date"].dt.to_period(
        "M"
    )  # Extract the month
    baseline_columns = [
        col
        for col in hcl_lab_final_df.columns
        if col.startswith("Pos_test_") or col.startswith("Nb_of_test_")
    ]
    baseline_data = hcl_lab_final_df[["Month", "Age_class"] + baseline_columns]

    # Group by both 'Month' and 'Age_class' and calculate the mean
    grouped = baseline_data.groupby(["Month", "Age_class"])

    # Use the aggregate method to calculate the mean
    agg_result = grouped.mean().reset_index()

    # Add the suffix "_BASELINE" to the column names
    agg_result.columns = [
        col + "_BASELINE" if col in baseline_columns else col
        for col in agg_result.columns
    ]

    # Merge the mean based on 'Month' and 'Age_class' with the original DataFrame
    hcl_lab_final_df = hcl_lab_final_df.merge(
        agg_result, on=["Month", "Age_class"], how="left"
    )
    hcl_lab_final_df = hcl_lab_final_df.drop("Month", axis=1)

    # Define the confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the critical value using the normal distribution
    z = stats.norm.ppf((1 + confidence_level) / 2)

    # Columns to calculate confidence intervals for (specify your baseline columns)
    baseline_columns = [
        col for col in hcl_lab_final_df.columns if col.endswith("_BASELINE")
    ]

    # Create new columns to store the results
    for col in baseline_columns:
        lower_col = "LOWER_CI_" + col
        upper_col = "UPPER_CI_" + col

        # Initialize columns with NaN
        hcl_lab_final_df[lower_col] = np.nan
        hcl_lab_final_df[upper_col] = np.nan

        # Calculate confidence intervals for each baseline column
        for age_class in hcl_lab_final_df["Age_class"].unique():
            # Select rows for the specific Age_Class
            subset = hcl_lab_final_df[hcl_lab_final_df["Age_class"] == age_class]
            # col.replace('_BASELINE', '')
            # Calculate the standard deviation for the specific column
            std_dev = subset[col].std()

            lower_ci = subset[col] - z * std_dev
            upper_ci = subset[col] + z * std_dev

            # Ensure that LOWER_CI is not less than zero
            hcl_lab_final_df.loc[
                hcl_lab_final_df["Age_class"] == age_class, lower_col
            ] = lower_ci.clip(lower=0)
            hcl_lab_final_df.loc[
                hcl_lab_final_df["Age_class"] == age_class, upper_col
            ] = upper_ci.clip(lower=0)

        col_patho = col.replace("_BASELINE", "")
        hcl_lab_final_df["ALERT_" + col_patho] = np.where(
            (hcl_lab_final_df[col_patho] > hcl_lab_final_df[upper_col]), 1, 0
        )

    hcl_lab_baseline_df = hcl_lab_final_df

    return hcl_lab_baseline_df


def create_nowcasting(hcl_lab_final):
    """
    This function applies nowcasting to selected columns of the input DataFrame and calculates
    confidence intervals for the nowcasted values.

    Parameters:
    hcl_lab_final (DataFrame): DataFrame containing lab data with translated and reordered columns.

    Returns:
    DataFrame: DataFrame containing nowcasted values with confidence intervals.
    """

    hcl_lab_final_df = pd.DataFrame(hcl_lab_final)
    hcl_lab_final_df["Date"] = pd.to_datetime(hcl_lab_final_df["Date"])

    # Define the correction factors (you can replace this with your correction_factor_last_weeks)
    correction_factor_last_weeks = np.linspace(1, 0.5, 12 + 1)[::-1]

    # List of columns to apply nowcasting to
    columns_to_nowcast = [
        col
        for col in hcl_lab_final_df.columns
        if re.match(r"^Pos_test_|^Nb_of_test_", col) and not col.endswith("_BASELINE")
    ]

    # Create an empty DataFrame to store the results
    hcl_lab_nowcasting_final = pd.DataFrame()

    # Iterate over each unique Age_class
    for age_class in hcl_lab_final_df["Age_class"].unique():
        mask = hcl_lab_final_df["Age_class"] == age_class
        df = hcl_lab_final_df[["Date", "Age_class"] + columns_to_nowcast].loc[mask]
        df = df.sort_values(by="Date", ascending=False)
        df["correction_factor"] = correction_factor_last_weeks[: len(df)]

        # Define a function to apply the correction factor to each column
        def apply_correction(col):
            return col / df["correction_factor"]

        # Apply the correction function to each column to calculate the nowcasted values
        for col in columns_to_nowcast:
            col_nowcasted = col + "_NOWCASTING"
            df[col_nowcasted] = apply_correction(df[col])

        hcl_lab_nowcasting_final = pd.concat(
            [hcl_lab_nowcasting_final, df], ignore_index=True
        )  # Use concat to combine DataFrames

    # Define the confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the critical value using the normal distribution
    z = stats.norm.ppf((1 + confidence_level) / 2)

    # Columns to calculate confidence intervals for
    baseline_columns = [
        col for col in hcl_lab_nowcasting_final.columns if col.endswith("_NOWCASTING")
    ]

    # Calculate confidence intervals for each baseline column
    for col in baseline_columns:
        lower_col = "LOWER_CI_" + col
        upper_col = "UPPER_CI_" + col

        lower_ci = hcl_lab_nowcasting_final[col] - z * hcl_lab_nowcasting_final[
            col
        ].apply(lambda x: max(x, 0))
        upper_ci = hcl_lab_nowcasting_final[col] + z * hcl_lab_nowcasting_final[
            col
        ].apply(lambda x: max(x, 0))

        # Ensure that LOWER_CI is not less than zero
        hcl_lab_nowcasting_final[lower_col] = lower_ci.apply(lambda x: max(x, 0))
        hcl_lab_nowcasting_final[upper_col] = upper_ci

    hcl_lab_nowcasting_final = hcl_lab_nowcasting_final.drop(
        "correction_factor", axis=1
    )

    return hcl_lab_nowcasting_final


def merge_nowcast_baseline(hcl_lab_nowcasting_final, hcl_lab_baseline_df):

    hcl_lab_nowcasting_final = pd.DataFrame(hcl_lab_nowcasting_final)
    hcl_lab_nowcasting_final["Date"] = pd.to_datetime(hcl_lab_nowcasting_final["Date"])

    hcl_lab_baseline_df = pd.DataFrame(hcl_lab_baseline_df)
    hcl_lab_baseline_df["Date"] = pd.to_datetime(hcl_lab_baseline_df["Date"])

    columns_to_select = ["Date", "Age_class"] + [
        col for col in hcl_lab_baseline_df.columns if col.endswith("_BASELINE")
    ]

    hcl_lab_baseline_df = hcl_lab_baseline_df[columns_to_select]

    hcl_lab_final_merged = hcl_lab_nowcasting_final.merge(
        hcl_lab_baseline_df, on=["Date", "Age_class"], how="left"
    )

    return hcl_lab_final_merged


def fraction_of_pos(hcl_lab_final_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the fraction of positive cases for diseases in 'hcl_lab_final_merged' DataFrame.

    Args:
        hcl_lab_final_merged (pd.DataFrame): DataFrame containing lab data with translated and reordered columns.

    Returns:
        pd.DataFrame: DataFrame with fractions of positive cases for each disease.
    """

    # Initialize an empty list to store disease names
    diseases = []
    hcl_lab_final = pd.DataFrame(hcl_lab_final_merged)

    # Iterate through the columns of the 'merged_df' DataFrame to extract disease names
    for column_name in hcl_lab_final.columns:
        if column_name.startswith("Pos_test_"):
            disease_name = column_name[len("Pos_test_") :]
            diseases.append(disease_name)

    # Initialize the 'hcl_lab_fraction_final' DataFrame with columns 'Date' and 'Age_class'
    hcl_lab_fraction_final = hcl_lab_final[["Date", "Age_class"]].copy()

    # Iterate through the dynamically extracted disease names and calculate fractions for each disease
    for disease in diseases:
        # Calculate the fraction column for this disease by dividing 'Pos_test_xxx' by 'Nb_of_test_xxx'
        cas_pos_column = f"Pos_test_{disease}"
        nb_prelev_column = f"Nb_of_test_{disease}"
        fraction_column = (
            hcl_lab_final[cas_pos_column] / hcl_lab_final[nb_prelev_column]
        )

        # Add this column to the 'hcl_lab_fraction_final' DataFrame with a descriptive name
        hcl_lab_fraction_final[f"Fraction_Pos_{disease}"] = fraction_column

    return hcl_lab_fraction_final


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
