from datetime import datetime

import numpy as np
import pandas as pd

# def data_extraction(paediatric_data_df: pd.DataFrame) -> pd.DataFrame:
#     return paediatric_data_df


def upload_data(df):
    return df

def data_extraction(paediatric_source: dict) -> pd.DataFrame:

    # Filtrer les fichiers par ceux qui commencent par "AIOLOS" et se terminent par ".xlsx"
    relevant_files = [file_name for file_name in paediatric_source.keys() if file_name.startswith("AIOLOS") and file_name.endswith(".xlsx")]
    print(relevant_files)
    # Sélectionner le fichier le plus récent en utilisant la date dans le nom du fichier 
    latest_file = None
    latest_date = None

    for file in relevant_files:
        if "_corrected" in file:  # Ignorer les fichiers avec le suffixe "_corrected"
            continue
        file_date = datetime.strptime(file.split('.')[0][7:], '%d-%m-%y')
        if latest_date is None or file_date > latest_date:
            latest_date = file_date
            latest_file = file
    
    #latest_file = max(relevant_files, key=lambda x: datetime.strptime(x.split('.')[0][7:], '%d-%m-%y'))

    # Récupérer les données du fichier le plus récent
    latest_data = paediatric_source[latest_file]

    # Vérifier si le fichier latest_file + "_corrected" existe
    corrected_file =  latest_file[:latest_file.rfind('.xlsx') ] + "_corrected.xlsx"
    if corrected_file in relevant_files:
        latest_file = corrected_file

    # Concaténer les feuilles du fichier le plus récent verticalement
    dataframe_aiolos = pd.concat([dataframe for dataframe in latest_data.values()], ignore_index=True, sort=False, axis=0)

    return dataframe_aiolos


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the first row called "semaine"
    df = df.drop(df.index[0])
    
    # Replace all NaN values by 0
    df = df.replace(np.nan, 0)

    # Replace cells containing "?" by NaN
    df = df.replace("?", np.nan)

    # Rename the first columns to correspond to the date, rename "Grippe" to "Flu", "Pneumopathie" tp "Pneumopathy" and "Bronchiolite" to "Bronchiolitis"

    df.rename(
        columns={
            "Unnamed: 0_level_0": "Date",
            "Unnamed: 0_level_1": "Date",
            "Unnamed: 2_level_0": "Bronchiolitis",
            "Unnamed: 3_level_0": "Bronchiolitis",
            "Unnamed: 4_level_0": "Bronchiolitis",
            "Unnamed: 6_level_0": "Covid",
            "Unnamed: 7_level_0": "Covid",
            "Unnamed: 9_level_0": "Flu",
            "Unnamed: 10_level_0": "Flu",
            "Unnamed: 11_level_0": "Flu",
            "Unnamed: 14_level_0": "Total Resp. Synd.",
            "Grippe": "Flu",
            "Pneumopathie": "Pneumopathy",
            "Bronchiolites": "Bronchiolitis",
            "Total Synd. resp.": "Total Resp. Synd.",
            "Nb consultations": "Number of consultations",
            "TDR indéterminé": "Undetermined Rapid Diagnostic Test",
            "% / Nb cons": "% of consultations",
            "TDR+": "Rapid Diagnostic Test +",
            "TDR-": "Rapid Diagnostic Test -",
            "Total": "current",
            "n": "current",
        },
        inplace=True,
    )
    

    week_year = df[("Date", "Date")].str.split("-w", expand=True)

    df["Week"] = week_year[1].astype(int)
    df["Year"] = week_year[0].astype(int)
    df.drop(("Date", "Date"), axis=1, inplace=True)
    
    # Change all data types to int except for the date
    df = df.astype(
        {
            "Flu": int,
            "Bronchiolitis": int,
            "Pneumopathy": int,
            "Total Resp. Synd.": int,
            "Covid": int,
            "Number of consultations": int,
        }
    )
    df[("Number of consultations", "current")] = df[("Number of consultations", "current")].astype('float64')
    df.set_index(["Year", "Week"], inplace=True)

    # Select only The Total Subcolumns Columns

    df = df.loc[
        :,
        [
            ("Covid", "current"),
            ("Flu", "current"),
            ("Bronchiolitis", "current"),
            ("Pneumopathy", "current"),
            ("Total Resp. Synd.", "current"),
            ("Number of consultations", "current"),
        ],
    ]

    # Delete the level 1 header :
    df_onelevel = df.copy()
    df_onelevel.columns = df.columns.droplevel(1)
    df_onelevel

    current_year = df.index[-1][0]

    # Remove current year to compute the baseline
    df_for_baseline = df.loc[df.index.get_level_values(0) != current_year]
    df_for_baseline

    df_agg = df_for_baseline.groupby("Week").agg(["mean", "std"])
    df_agg = df_agg.droplevel(1, axis=1)
    
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_agg.to_csv(os.path.join(script_dir, "baseline.csv"))
    
    # Compute the 95% confidence interval corresponding to the mean and std in each row
    # Number of years of data
    df_agg["Covid", "lower"] = df_agg["Covid", "mean"] - 1.96*df_agg["Covid", "std"]
    df_agg["Covid", "upper"] = df_agg["Covid", "mean"] + 1.96*df_agg["Covid", "std"]
    df_agg[("Flu", "lower")] = df_agg[("Flu", "mean")] - 1.96*df_agg[("Flu", "std")]
    df_agg[("Flu", "upper")] = df_agg[("Flu", "mean")] + 1.96*df_agg[("Flu", "std")]
    df_agg[("Bronchiolitis", "lower")] = (
        df_agg[("Bronchiolitis", "mean")] - 1.96*df_agg[("Bronchiolitis", "std")]
    )
    df_agg[("Bronchiolitis", "upper")] = (
        df_agg[("Bronchiolitis", "mean")] + 1.96*df_agg[("Bronchiolitis", "std")]
    )
    df_agg[("Pneumopathy", "lower")] = (
        df_agg[("Pneumopathy", "mean")] - 1.96*df_agg[("Pneumopathy", "std")]
    )
    df_agg[("Pneumopathy", "upper")] = (
        df_agg[("Pneumopathy", "mean")] + 1.96*df_agg[("Pneumopathy", "std")]
    )
    df_agg[("Total Resp. Synd.", "lower")] = (
        df_agg[("Total Resp. Synd.", "mean")] - 1.96*df_agg[("Total Resp. Synd.", "std")]
    )
    df_agg[("Total Resp. Synd.", "upper")] = (
        df_agg[("Total Resp. Synd.", "mean")] + 1.96*df_agg[("Total Resp. Synd.", "std")]
    )
    df_agg[("Number of consultations", "lower")] = (
        df_agg["Number of consultations"]["mean"]
        - 1.96*df_agg["Number of consultations"]["std"]
    )
    df_agg[("Number of consultations", "upper")] = (
        df_agg["Number of consultations"]["mean"]
        + 1.96*df_agg["Number of consultations"]["std"]
    )
    # Remove all the columns corresponding to the standard deviation

    df_agg = df_agg[
        [
            ("Covid", "mean"),
            ("Covid", "lower"),
            ("Covid", "upper"),
            ("Flu", "mean"),
            ("Flu", "lower"),
            ("Flu", "upper"),
            ("Bronchiolitis", "mean"),
            ("Bronchiolitis", "lower"),
            ("Bronchiolitis", "upper"),
            ("Pneumopathy", "mean"),
            ("Pneumopathy", "lower"),
            ("Pneumopathy", "upper"),
            ("Total Resp. Synd.", "mean"),
            ("Total Resp. Synd.", "lower"),
            ("Total Resp. Synd.", "upper"),
            ("Number of consultations", "mean"),
            ("Number of consultations", "lower"),
            ("Number of consultations", "upper"),
        ]
    ]

    # Merge dataframes on the 'week' column
    df_agg.reset_index(inplace=True)
    df.reset_index(inplace=True)
    result_df = pd.merge(df, df_agg, on="Week", how="inner")
    result_df.set_index(["Year", "Week"], inplace=True)
    # Sort by year and week
    result_df.sort_index(inplace=True)

    # Create, for each disease, an "alert" column that is True if the number of cases is above the upper confidence interval and False otherwise
    result_df[("Covid", "alert")] = (
        result_df[("Covid", "current")] > result_df[("Covid", "upper")]
    )
    result_df[("Flu", "alert")] = (
        result_df[("Flu", "current")] > result_df[("Flu", "upper")]
    )
    result_df[("Bronchiolitis", "alert")] = (
        result_df[("Bronchiolitis", "current")] > result_df[("Bronchiolitis", "upper")]
    )
    result_df[("Pneumopathy", "alert")] = (
        result_df[("Pneumopathy", "current")] > result_df[("Pneumopathy", "upper")]
    )
    result_df[("Total Resp. Synd.", "alert")] = (
        result_df[("Total Resp. Synd.", "current")]
        > result_df[("Total Resp. Synd.", "upper")]
    )
    # Same for the number of consultations
    result_df[("Number of consultations", "alert")] = (
        result_df[("Number of consultations", "current")]
        > result_df[("Number of consultations", "upper")]
    )

    # Sort the level 1 headers according to the level 0 header
    result_df.sort_index(axis=1, level=0, inplace=True)
    
    result_df.reset_index(inplace=True, col_level=0)
    
    result_df['Date'] = pd.to_datetime(result_df['Year'].astype(str) + result_df['Week'].astype(str) + '-1', format='%Y%U-%w')
    
    # Supprimer les colonnes 'Year' et 'Week'
    result_df.drop(['Year', 'Week'], axis=1, inplace=True)
    
    # Extraire la colonne 'Date'
    date_column = result_df.pop('Date')

    # Insérer la colonne 'Date' à la première position
    result_df.insert(0, 'Date', date_column)

    # # Réinitialiser l'index pour déplacer les niveaux de MultiIndex en colonnes
    df_reset = result_df.reset_index()
    # print(df_reset)
    # # Utiliser pivot_table pour obtenir un tableau simple avec les colonnes souhaitées
    # df_melted = df_reset.melt(id_vars=['index', 'Date'], var_name=['Diagnosis', 'Metric'])
    
        # Utiliser melt pour regrouper les colonnes liées à chaque maladie
    # Utiliser melt pour regrouper les colonnes liées à chaque maladie
    # Utiliser melt pour regrouper les colonnes liées à chaque maladie
    
    df_reset.reset_index(drop=True)
    
    df_melted = df_reset.melt(id_vars=['Date'], var_name=['Diagnosis', 'Metric'])
    

    # Créer une colonne combinée pour les colonnes 'Diagnosis' et 'Metric'
    df_melted['Diagnosis'] = df_melted['Diagnosis'].str.split('_', expand=True)[0]

    # Utiliser pivot_table pour obtenir un tableau avec les colonnes souhaitées
    df_final = df_melted.pivot_table(index=['Date', 'Diagnosis'], columns='Metric', values='value', aggfunc='first').reset_index()

    # Supprimer le nom de la colonne de l'index
    df_final.columns.name = None

    # Réinitialiser l'index
    df_final.reset_index(inplace=True)

    # Afficher le DataFrame résultant
    df_final = df_final.drop('index', axis=1)
    df_final = df_final.drop('', axis=1)
    df_final = df_final[~df_final['Diagnosis'].str.contains('index')]
    
    df_final = df_final.sort_values(by=['Diagnosis', 'Date'])
    
    consultations_df = df_final[df_final['Diagnosis'] == 'Number of consultations'].copy()

    consultations_df.rename(columns={'current': 'number_of_consultations', 'lower': 'lower_number_of_consultations', 'upper': 'upper_number_of_consultations'}, inplace=True)
    df_final = pd.merge(df_final, consultations_df[['Date', 'number_of_consultations']], on='Date', how='left')
    df_final['percentage_doctor_visit'] = df_final['current'] / df_final['number_of_consultations'] * 100

    consultations_df = df_final.loc[:, ['Date', 'Diagnosis','percentage_doctor_visit']]

    df_final['Week'] = df_final['Date'].apply(lambda x: x.isocalendar()[1]) 
    consultations_df['Week'] = consultations_df['Date'].apply(lambda x: x.isocalendar()[1])
    
    df_agg = consultations_df.groupby(["Week", "Diagnosis"])["percentage_doctor_visit"].agg(["mean", "std"])
    df_agg["lower_percentage_doctor_visit"] = df_agg["mean"] - 1.96*df_agg["std"]
    df_agg["upper_percentage_doctor_visit"] = df_agg["mean"] + 1.96*df_agg["std"]
    
    df_agg = df_agg.reset_index()
    
    df_final = pd.merge(df_final, df_agg[['Week', 'Diagnosis','lower_percentage_doctor_visit', 'upper_percentage_doctor_visit']], on=['Week',"Diagnosis"], how='left')
    
    df_final = df_final.drop(columns=['Week', 'number_of_consultations'])
    
    df_final['lower'] = df_final['lower'].apply(lambda x: max(0, x))
    df_final['lower_percentage_doctor_visit'] = df_final['lower_percentage_doctor_visit'].apply(lambda x: max(0, x))
    
    return df_final