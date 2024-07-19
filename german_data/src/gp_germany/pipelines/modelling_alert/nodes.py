import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress

# warnings.filterwarnings("ignore")


def upload_data(df):
    df = pd.DataFrame(df)
    return df


def log_linear_alert(size_FW, fit_y, time_dim, dates, alert_threshold=0):
    """
    Function to fit a linear regression on log transformed data.
    The output is the relative slope (including a 95% CI) of the past 5 weeks.
    Depending on the relative slope an arrow angle is computed which
    represents the slope.
    """

    if time_dim == "daily" or time_dim == "daily-alert":
        last_date = dates.values[-7]
        date = pd.date_range(start=last_date, periods=2, freq="W-MON")[1:]
    elif time_dim == "weekly":
        last_date = dates.values[-1]
        date = pd.date_range(start=last_date, periods=2, freq="W-MON")[1:]

    num_reg = len(fit_y.geography.unique())

    trends = pd.DataFrame()
    # looping over all regions
    for geo in fit_y.geography.unique():
        X_fit = np.arange(1, size_FW + 1)
        df_tmp = fit_y[fit_y.geography == geo]
        colname = df_tmp.columns.to_list()[0]
        Y_fit = df_tmp.iloc[:, 0].values
        Y_fit = np.vstack(Y_fit).astype(np.float64).reshape(1, size_FW)

        model = linregress
        slope, intercept, r, p, se = model(X_fit, Y_fit)

        # store relative slope (growth/decay rate) togehter with its confident intervall
        if time_dim == "daily-alert":
            slope = (
                slope * 7
            )  # with daily data the slope is computed per day, need it weekly
        slope_lwr, slope_upr = compute_slope_CI(
            alpha=0.05, size_FW=size_FW, reg_slope=slope, stderr=se
        )
        rel_slope = np.round((np.exp(slope) - 1) * 100, 2)
        rel_slope_lwr = np.round((np.exp(slope_lwr) - 1) * 100, 2)
        rel_slope_upr = np.round((np.exp(slope_upr) - 1) * 100, 2)
        if rel_slope_upr > alert_threshold:
            alert = 1
        else:
            alert = 0

        # significance is basically shown by the confidene of the slope

        trend = pd.DataFrame(
            np.array(
                [
                    rel_slope,
                    rel_slope_lwr,
                    rel_slope_upr,
                    compute_arrow_angle(rel_slope),
                    alert,
                ]
            ).reshape(-1, 5)
        )

        trends_df = pd.DataFrame()
        trends_df = pd.concat([trends_df, trend])

        trends_df.columns = [
            "rel_slope_mu: " + colname,
            "rel_slope_lwr: " + colname,
            "rel_slope_upr: " + colname,
            "arrow_angle: " + colname,
            "alert: " + colname,
        ]
        trends = pd.concat([trends, trends_df])

    return (date, trends)


def trends_prediction(
    data, size_FW_weekly, size_FW_daily, time_dim, size_FW_daily_alert
):
    data = pd.DataFrame(data).fillna(0)
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    data[numeric_cols] = np.log(data[numeric_cols] + 0.001)
    # make sure that data is sorted by th column date (most recent date first)
    data = data.sort_values(by=["date", "geography"], ascending=True).reset_index(drop=True)

    if time_dim == "weekly":
        size_FW = size_FW_weekly
    elif time_dim == "daily":
        size_FW = size_FW_daily
    elif time_dim == "daily-alert":
        size_FW = size_FW_daily_alert

    # retrieve all unique dates in the data
    date_list = data.date.unique()

    # # keep only 10 latest dates for pipeline testings
    # date_list = date_list[-10:]

    # show earliest date, and latest date
    print("Earliest date: ", date_list[0])
    print("Latest date: ", date_list[-1])

    first_date_for_trends = "2023-07-01" # reduce computation time
    # in date_list keep only dates after first_date_for_trends
    date_list = date_list[date_list >= first_date_for_trends]
    print("first date of trends computation: ", date_list[0])
    
    Trends_all_dates = pd.DataFrame()
    # loop on all dates and compute the relative slope for each date
    for date_of_trends in date_list:
        data_cut = data[data.date <= date_of_trends]

        new_df = pd.DataFrame(columns=data.columns)
        # chosse a random region among the unique regions
        region = data["geography"].unique()[0]
        window_size = min(size_FW, len(data_cut[data_cut["geography"] == region]))

        # Loop through unique geographies
        for geo in data["geography"].unique():
            new_df = pd.concat(
                [new_df, data_cut[data_cut["geography"] == geo].iloc[-window_size:,]]
            )  # only takes the last FW rows for fitting

        data_cut = new_df

        Trends = pd.DataFrame()
        if window_size > 2:
            for col in numeric_cols:
                date, trends = log_linear_alert(
                    window_size, data_cut[[col, "geography"]], time_dim, data_cut.date
                )
                Trends = pd.concat([Trends, trends], axis=1)
            Trends.insert(0, "geography", data_cut.geography.unique())
            Trends.insert(1, "date", date_of_trends)
        Trends_all_dates = pd.concat([Trends_all_dates, Trends], axis=0)

    return Trends_all_dates


def compute_slope_CI(alpha, size_FW, reg_slope, stderr):
    t_value = stats.t.ppf(1 - alpha / 2, size_FW)
    marg_err = t_value * stderr
    slope_lwr = reg_slope - marg_err
    slope_upr = reg_slope + marg_err

    return (slope_lwr, slope_upr)


def compute_arrow_angle(relative_slope):
    if relative_slope < -200:
        arrow_angle = -90

    elif relative_slope < -160 and relative_slope >= -200:
        arrow_angle = -75

    elif relative_slope < -120 and relative_slope >= -160:
        arrow_angle = -60

    elif relative_slope < -80 and relative_slope >= -120:
        arrow_angle = -45

    elif relative_slope < -40 and relative_slope >= -80:
        arrow_angle = -30

    elif relative_slope < 0 and relative_slope >= -40:
        arrow_angle = -15

    elif relative_slope == 0 or relative_slope == None or np.isnan(relative_slope) ==   True:
        arrow_angle = 0

    elif relative_slope <= 40 and relative_slope > 0:
        arrow_angle = 15

    elif relative_slope <= 80 and relative_slope > 40:
        arrow_angle = 30

    elif relative_slope <= 120 and relative_slope > 80:
        arrow_angle = 45

    elif relative_slope <= 160 and relative_slope > 120:
        arrow_angle = 60

    elif relative_slope <= 200 and relative_slope > 160:
        arrow_angle = 75

    elif relative_slope > 200:
        arrow_angle = 90

    return arrow_angle
