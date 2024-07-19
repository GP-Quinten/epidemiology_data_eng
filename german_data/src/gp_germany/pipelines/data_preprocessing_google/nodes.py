import datetime
import math
import time
from itertools import chain, product
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def upload_data(df):
    print(df.columns)
    return df


def fetch_googletrend_raw(
    terms: list,
    geo_countries: list,
    geo_regions: list,
    geo_dmas: list,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
) -> None:
    """Fetch raw googletrend data.

    Args:
        terms (list): list of terms to fetch
        geo_countries (list): list of country codes to fetch
        geo_regions (list): list of region codes to fetch
        geo_dmas (list): list of DMA (Designated Market Area) codes to fetch
        date_start (pd.Timestamp): date of the start of the period to fetch
        date_end (pd.Timestamp): date of the end of the period to fetch

    Raises:
        ValueError
    """
    from googleapiclient.discovery import build

    SERVER = "https://trends.googleapis.com"
    API_VERSION = "v1beta"
    DISCOVERY_URL = SERVER + "/$discovery/rest?version=" + API_VERSION
    GT_API_KEY = "AIzaSyBr_wGP6CPEmAyR8HtqMmD8JLB49yomBC4"
    MAX_TERMS = 30  # the api limits the number of terms in the same request
    MAX_POINTS = 2000  # the api limits the number of points in the same response
    MAX_QUERIES = 5000  # the api limits the number of queries in the same day
    WAIT_TIME = 60 / 300  # the api limits the delay between queries

    # Open service
    service = build(
        "trends",
        API_VERSION,
        developerKey=GT_API_KEY,
        discoveryServiceUrl=DISCOVERY_URL,
    )

    # Initialize result
    result = []

    # Handle api's limitations
    nb_terms = len(terms)
    nb_days = (date_end - date_start).days + 1
    nb_geos = len(geo_countries + geo_regions + geo_dmas)
    batch_of_terms_size = min(
        nb_terms,
        MAX_TERMS,
        math.floor(MAX_POINTS / nb_days),
    )
    nb_batch_of_terms = math.ceil(nb_terms / batch_of_terms_size)
    nb_queries = nb_batch_of_terms * nb_geos
    if nb_days >= MAX_POINTS:
        raise ValueError(
            f"Invalid date range ({nb_days}), "
            f"should be less than MAX_POINTS={MAX_POINTS}"
        )
    if nb_queries >= MAX_QUERIES:
        raise ValueError(
            f"Invalid number of queries ({nb_queries}), "
            "should be less than MAX_QUERIES={MAX_QUERIES}"
        )

    # Iterate over queries
    with tqdm(total=nb_queries, leave=False, desc="Fetching - googletrend") as pbar:

        # Iterate over geos
        for geo_level, geo in chain(
            product(["geoRestriction_country"], geo_countries),
            product(["geoRestriction_region"], geo_regions),
            product(["geoRestriction_dma"], geo_dmas),
        ):

            # Iterate over batch of terms
            for batch in range(nb_batch_of_terms):

                # Compute batch of terms
                batch_start = batch * batch_of_terms_size
                batch_end = min(batch_start + batch_of_terms_size, len(terms))
                batch_of_terms = terms[batch_start:batch_end]

                # Request
                request = service.getTimelinesForHealth(
                    **{
                        "terms": batch_of_terms,
                        geo_level: geo,
                        "time_startDate": date_start.strftime("%Y-%m-%d"),
                        "time_endDate": date_end.strftime("%Y-%m-%d"),
                        "timelineResolution": "day",
                    }
                )

                # Response
                response = request.execute()
                time.sleep(WAIT_TIME)

                # Populate result
                response = [{**line, "geo": geo} for line in response["lines"]]
                result.extend(response)

                pbar.update(1)
                time.sleep(WAIT_TIME)

    pbar.close()

    # Close service
    service.close()

    df_googletrend_raw = (
        pd.DataFrame(
            [
                (point["date"], line["term"], point["value"])
                for line in result
                for point in line["points"]
            ],
            columns=["date", "term", "googletrend"],
        ).assign(date=lambda df: pd.to_datetime(df.date, format="%b %d %Y"))
        # .set_index(['date', 'geo', 'term']).sort_index()
        # .googletrend
    )

    return df_googletrend_raw


def preprocessing_google(symptoms: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess Google Trends data for symptoms in German and French.

    Args:
        symptoms (pd.DataFrame): DataFrame containing symptoms information.

    Returns:
    Google_symptoms_fr, Google_symptoms_de

        Tuple[pd.DataFrame, pd.DataFrame]:
        - Google_symptoms_fr : Counts of symptoms retrieved of the past 5 weeks from Google Trends in France
        - Google_symptoms_de : Counts of symptoms retrieved of the past 5 weeks from Google Trends in Germany
    """
    end_date = datetime.datetime.now()
    days=365*5
    print("Google days", days)
    delta = datetime.timedelta(days=days)
    start_date = end_date - delta
    print(start_date)

    terms = symptoms.german.tolist()
    terms_norm = list(set(list(map(str.lower, terms))))
    geo_countries = ["DE"]
    geo_regions = [] # ["BB"]
    geo_dmas = []
    date_start = start_date.date()
    print("date_start", date_start)
    date_end = end_date.date()
    print("date_end", date_end)

    df = fetch_googletrend_raw(
        terms_norm, geo_countries, geo_regions, geo_dmas, date_start, date_end
    )
    df_trans = df.pivot(index="date", columns="term", values="googletrend")
    df_trans.columns = symptoms.english.tolist()
    df_trans.insert(df_trans.shape[1], "sum", df_trans.sum(axis=1))
    df_trans.insert(0, "date", df_trans.index)
    df_trans.insert(1, "geography", "DE")
    Google_symptoms_de = df_trans

    terms = symptoms.french.tolist()
    terms_norm = list(set(list(map(str.lower, terms))))
    geo_countries = ["FR"]
    geo_regions = []
    geo_dmas = []
    date_start = start_date.date()
    date_end = end_date.date()

    df = fetch_googletrend_raw(
        terms_norm, geo_countries, geo_regions, geo_dmas, date_start, date_end
    )
    df_trans = df.pivot(index="date", columns="term", values="googletrend")
    df_trans.columns = symptoms.english.tolist()
    df_trans.insert(df_trans.shape[1], "sum", df_trans.sum(axis=1))
    df_trans.insert(0, "date", df_trans.index)
    df_trans.insert(1, "geography", "FR")
    Google_symptoms_fr = df_trans

    return Google_symptoms_fr, Google_symptoms_de


def create_alerts(df):
    '''
    Creates a table with alerts for each disease based on the number of cases. The first date of the table is 2023-09-01.
    It computes the mean and standard deviation of the number of cases for each disease for each week of the year, for each geogrpahy. It then computes the upper and lower confidence intervals for each disease. 
    Finally, it creates an "alert" column that is True if the number of cases is above the upper confidence interval and False otherwise. 

    Args:
        df (pd.DataFrame): DataFrame containing the number of cases for each disease (as columns) for each date (format YYYY-MM-dd) (as rows). First 2 columns are date (format YYYY-MM-dd) and geography

    Returns:
        pd.DataFrame: DataFrame containing the number of cases for each disease (as columns) for each week of the year (as rows). It also contains the upper and lower confidence intervals and the "alert" column.
    '''
    google_df = df.copy()
    google_df['date'] = pd.to_datetime(google_df['date'])
    google_df['week'] = google_df['date'].dt.isocalendar().week
    google_df['year'] = google_df['date'].dt.year
    google_df['date'] = google_df['date'].dt.date

    baseline_df = google_df[google_df['date'] < pd.to_datetime('2023-09-01').date()].drop(columns=['date'])
    google_df = google_df[google_df['date'] >= pd.to_datetime('2023-09-01').date()].drop(columns=['date'])

    # group by geography, year and week and sum the number of cases for each disease that is in each column
    baseline_df = baseline_df.groupby(['geography', 'year', 'week']).sum().reset_index()
    # pivot all diseases columns to rows
    baseline_df = pd.melt(baseline_df, id_vars=['geography', 'year', 'week'], var_name='disease', value_name='cases')
    # show baseline_df head to see if it is correct

    google_df = google_df.groupby(['geography', 'year', 'week']).sum().reset_index()
    google_df = pd.melt(google_df, id_vars=['geography', 'year', 'week'], var_name='disease', value_name='cases')

    baseline_df['mean'] = baseline_df.groupby(['geography', 'week', 'disease'])['cases'].transform('mean')
    baseline_df['std'] = baseline_df.groupby(['geography', 'week', 'disease'])['cases'].transform('std')
    baseline_df['upper'] = baseline_df['mean'] + 1.96 * baseline_df['std']
    baseline_df['lower'] = (baseline_df['mean'] - 1.96 * baseline_df['std']).clip(lower=0)

    google_df = google_df.merge(baseline_df[['geography', 'week', 'disease', 'mean', 'upper', 'lower']], on=['geography', 'week', 'disease'], how='left')
    google_df['alert'] = google_df['cases'] > google_df['upper']
    google_df['alert'] = google_df['alert'].astype(int)
    # create new column  'date' with format YYYY-MM-dd from 'year' and 'week', and set as first column
    google_df['date'] = pd.to_datetime(google_df['year'].astype(str) + ' ' + google_df['week'].astype(str) + ' 1', format='%G %V %u')
    google_df = google_df[['date', 'geography', 'disease', 'cases', 'alert', 'mean', 'upper', 'lower']]
    
    return google_df