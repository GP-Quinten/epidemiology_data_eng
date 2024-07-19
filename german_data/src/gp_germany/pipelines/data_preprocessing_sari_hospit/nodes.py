import numpy as np
import pandas as pd


def upload_data(df):
    return df


def download_data(df):

    return df


def preprocessing_hospit(SARI_hospit: pd.DataFrame) -> pd.DataFrame:
    """
    Perform preprocessing on RSV cases DataFrame.

    Parameters:
    - RSV_cases_SN: Pandas DataFrame
      DataFrame containing RSV cases for the SN region.

    Returns:
    - df_complete_reg: Pandas DataFrame
      RSV cases in Saxony, Germany on a weekly basis
    """

    def iso_week_to_date(iso_week):
        year, week = map(int, iso_week.split("-W"))
        first_day_of_week = pd.to_datetime(
            f"{year}-W{week}-1", format="%G-W%V-%u"
        )  # First day of the ISO week
        last_day_of_week = first_day_of_week + pd.DateOffset(
            days=6
        )  # Adding 6 days to get the last day of the week
        return last_day_of_week

    # Apply the function to the 'Kalenderwoche' column
    SARI_hospit["Date"] = SARI_hospit["Kalenderwoche"].apply(iso_week_to_date)

    SARI_hospit = SARI_hospit.loc[
        :, ["Date", "Altersgruppe", "SARI_Hospitalisierungsinzidenz"]
    ]

    # Pivoter la table pour avoir une colonne par catégorie d'âge et calculer la somme des valeurs pour chaque semaine
    pivot_df = SARI_hospit.pivot_table(
        index="Date",
        columns="Altersgruppe",
        values="SARI_Hospitalisierungsinzidenz",
        aggfunc="sum",
    )

    # Ajouter une colonne 'Total' contenant la somme des valeurs de chaque ligne
    # pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df.reset_index(inplace=True)

    desired_order = ["Date", "0-4", "5-14", "15-34", "35-59", "60-79", "80+", "00+"]
    pivot_df = pivot_df[desired_order]
    pivot_df.columns = [
        "date",
        "0-4",
        "5-14",
        "15-34",
        "35-59",
        "60-79",
        "80plus",
        "0-200",
    ]
    pivot_df.insert(pivot_df.shape[1], "geography", "DE")

    return pivot_df

def create_alerts(df):
    
    df_hospit = df.copy()
    df_hospit['date'] = pd.to_datetime(df_hospit['date'])
    df_hospit['week'] = df_hospit['date'].dt.isocalendar().week
    df_hospit['year'] = df_hospit['date'].dt.year
    df_hospit['date'] = df_hospit['date'].dt.date

    baseline_df = df_hospit[df_hospit['date'] < pd.to_datetime('2023-09-01').date()].drop(columns=['date'])
    df_hospit = df_hospit[df_hospit['date'] >= pd.to_datetime('2023-09-01').date()].drop(columns=['date'])

    # group by geography, year and week and sum the cases_% for each age_group that is in each column
    baseline_df = baseline_df.groupby(['geography', 'year', 'week']).sum().reset_index()
    # pivot all age_groups columns to rows
    baseline_df = pd.melt(baseline_df, id_vars=['geography', 'year', 'week'], var_name='age_group', value_name='cases_%')
    # show baseline_df head to see if it is correct

    df_hospit = df_hospit.groupby(['geography', 'year', 'week']).sum().reset_index()
    df_hospit = pd.melt(df_hospit, id_vars=['geography', 'year', 'week'], var_name='age_group', value_name='cases_%')

    baseline_df['mean'] = baseline_df.groupby(['geography', 'week', 'age_group'])['cases_%'].transform('mean')
    baseline_df['std'] = baseline_df.groupby(['geography', 'week', 'age_group'])['cases_%'].transform('std')
    baseline_df['upper'] = (baseline_df['mean'] + 1.96 * baseline_df['std']).clip(upper=100)
    baseline_df['lower'] = (baseline_df['mean'] - 1.96 * baseline_df['std']).clip(lower=0)

    df_hospit = df_hospit.merge(baseline_df[['geography', 'week', 'age_group', 'mean', 'upper', 'lower']], on=['geography', 'week', 'age_group'], how='left')
    df_hospit['alert'] = df_hospit['cases_%'] > df_hospit['upper']
    df_hospit['alert'] = df_hospit['alert'].astype(int)
    # create new column  'date' with format YYYY-MM-dd from 'year' and 'week', and set as first column
    df_hospit['date'] = pd.to_datetime(df_hospit['year'].astype(str) + ' ' + df_hospit['week'].astype(str) + ' 1', format='%G %V %u')
    df_hospit = df_hospit[['date', 'geography', 'age_group', 'cases_%', 'alert', 'mean', 'upper', 'lower']]
    
    return df_hospit

