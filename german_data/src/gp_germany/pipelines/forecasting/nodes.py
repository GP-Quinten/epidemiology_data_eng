import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress

# warnings.filterwarnings("ignore")


def upload_data(df):
    print(df.columns)
    return df


def log_linear_alert(size_FW, fit_y, time_dim, dates):
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

        # significance is basically shown by the confidene of the slope

        trend = pd.DataFrame(
            np.array(
                [
                    rel_slope,
                    rel_slope_lwr,
                    rel_slope_upr,
                    compute_arrow_angle(rel_slope),
                ]
            ).reshape(-1, 4)
        )

        trends_df = pd.DataFrame()
        trends_df = pd.concat([trends_df, trend])

        trends_df.columns = [
            "rel_slope_mu: " + colname,
            "rel_slope_lwr: " + colname,
            "rel_slope_upr: " + colname,
            "arrow_angle: " + colname,
        ]
        trends = pd.concat([trends, trends_df])

    return (date, trends)


def forecast_prediction(
    data,
    size_PW,
    size_FW_weekly,
    size_FW_daily,
    time_dim,
    size_FW_daily_alert,
    age_strat,
):
    data = pd.DataFrame(data)
    if time_dim == "weekly":
        size_FW = size_FW_weekly
    elif time_dim == "daily":
        size_FW = size_FW_daily
    elif time_dim == "daily-alert":
        size_FW = size_FW_daily_alert

    Forecast = log_linear_forecast(
        size_FW, size_PW, data, time_dim, data.date, age_strat
    )
    Forecast = pd.DataFrame(Forecast)
    return Forecast


def log_linear_forecast(
    size_FW, size_PW, fit_y, time_dim, dates, age_strat, test_y=None
):
    """
    Function to fit a linear regression on log transformed data and
    to predict the next size_PW timepoints based on the regression parameter
    input data should have columns date, agegroup counts, total, geo
    """

    if time_dim == "daily":
        last_date = fit_y.date.values[-7]
        dates = pd.date_range(start=last_date, periods=3, freq="W-MON")[1:]
    elif time_dim == "weekly":
        last_date = fit_y.date.values[-1]
        dates = pd.date_range(start=last_date, periods=3, freq="W-MON")[1:]

    colnames = fit_y.columns.to_list()[1:-1]

    colnames_full = []
    for col in colnames:
        colnames_full.append(
            ["Pred_mu: " + col, "Pred_lwr: " + col, "Pred_upr: " + col]
        )

    fit_y_tot = fit_y[["geography", "total"]]

    Predictions = pd.DataFrame()
    for geo in fit_y.geography.unique():

        X_fit = np.arange(1, size_FW + 1)
        df_tmp = fit_y_tot[fit_y_tot.geography == geo]
        Y_fit = df_tmp.iloc[:, 1].values
        Y_fit = np.vstack(Y_fit).astype(np.float64).reshape(1, size_FW)

        model = linregress
        slope, intercept, r, p, se = model(X_fit, Y_fit)

        slope_lwr, slope_upr = compute_slope_CI(
            alpha=0.05, size_FW=size_FW, reg_slope=slope, stderr=se
        )

        X_test = np.arange(size_FW + 1, size_FW + 1 + size_PW)
        Y_pred = np.exp(intercept + slope * X_test)
        Y_pred_lwr = np.exp(intercept + slope_lwr * X_test)
        Y_pred_upr = np.exp(intercept + slope_upr * X_test)
        Y_pred_df = pd.DataFrame(
            [np.round(Y_pred, 2), np.round(Y_pred_lwr, 2), np.round(Y_pred_upr, 2)]
        ).transpose()

        # age strat is quite specific to COVID data rigth now.
        # if more age stratified data enters, it needs to be adjusted
        if age_strat:
            num_age_groups = fit_y.shape[1] - 3
            dataframe_age = fit_y.iloc[:, 1 : num_age_groups + 1].astype(float)
            # dataframe_age = fit_y.iloc[:,1:7].astype(float)
            AG_temp = np.exp(dataframe_age).sum()
            AG_temp_sum = AG_temp.sum()
            AG_temp_relative = AG_temp / AG_temp_sum
            Y_pred_age = np.outer(Y_pred[:], AG_temp_relative)
            Y_pred_age_lwr = np.outer(Y_pred_lwr[:], AG_temp_relative)
            Y_pred_age_upr = np.outer(Y_pred_upr[:], AG_temp_relative)

            Y_pred_age_mat = np.dstack(
                (
                    np.round(Y_pred_age.astype(float), 2),
                    np.round(Y_pred_age_lwr.astype(float), 2),
                    np.round(Y_pred_age_upr.astype(float), 2),
                )
            ).reshape(size_PW, num_age_groups * 3)
            Y_pred_age_df = pd.DataFrame(Y_pred_age_mat)

            Prediction_df = pd.concat([Y_pred_age_df, Y_pred_df], axis=1)

        else:

            Prediction_df = Y_pred_df

        Prediction_df.columns = sum(colnames_full, [])

        if time_dim == "daily":
            Prediction_df = (
                Prediction_df.groupby(np.arange(len(Prediction_df)) // 7)
                .mean(0)
                .round(2)
            )

        Prediction_df.insert(Prediction_df.shape[1], "geography", geo)

        Prediction_df.insert(0, "date", dates)
        Predictions = pd.concat([Predictions, Prediction_df], axis=0)

    return Predictions


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

    elif relative_slope == 0:
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
