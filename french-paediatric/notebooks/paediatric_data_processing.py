# %%
# Load the paediatric data :AIOLOS19-12-2023
import numpy as np
import pandas as pd

# Load the data in xlsx file
df = pd.read_excel("../data/01_raw/AIOLOS19-12-2023.csv", skiprows=1, header=[0, 1])
# Remove the first row called "semaine"
df = df.drop(df.index[0])
df

# %%
# Replace all NaN values by 0
df = df.replace(np.nan, 0)

# Replace cells containing "?" by NaN
df = df.replace("?", np.nan)


# %%
# Rename the first columns to correspond to the date, rename "Grippe" to "Flu", "Pneumopathie" tp "Pneumopathy" and "Bronchiolite" to "Bronchiolitis"

df.rename(
    columns={
        "Unnamed: 0_level_0": "Date",
        "Unnamed: 0_level_1": "Date",
        "Grippe": "Flu",
        "Pneumopathie": "Pneumopathy",
        "Bronchiolites": "Bronchiolitis",
        "Total Synd. resp.": "Total Resp. Synd.",
        "Nb consultations": "Number of consultations",
        "TDR indéterminé": "Undetermined Rapid Diagnostic Test",
        "% / Nb cons": "% of consulations",
        "TDR+": "Rapid Diagnostic Test +",
        "TDR-": "Rapid Diagnostic Test -",
        "Total": "current",
        "n": "current",
    },
    inplace=True,
)

# %%
week_year = df[("Date", "Date")].str.split("-w", expand=True)

df["Week"] = week_year[1].astype(int)
df["Year"] = week_year[0].astype(int)
df.drop(("Date", "Date"), axis=1, inplace=True)

# %%
# Remove week 53
df = df.loc[df[("Week")] != 53]
# Change all data types to int except for the date
df = df.astype(
    {
        "Flu": int,
        "Bronchiolitis": int,
        "Pneumopathy": int,
        "Total Resp. Synd.": int,
        "Covid": int,
    }
)

# %%
df.set_index(["Year", "Week"], inplace=True)

# %%
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


# %%
# Delete the level 1 header :
df_onelevel = df.copy()
df_onelevel.columns = df.columns.droplevel(1)
df_onelevel

# %%
current_year = df.index[-1][0]

# %%
# Remove current year to compute the baseline
df_for_baseline = df.loc[df.index.get_level_values(0) != current_year]
df_for_baseline

# %%
df_agg = df_for_baseline.groupby("Week").agg(["mean", "std"])
df_agg = df_agg.droplevel(1, axis=1)

# %%
# Compute the 95% confidence interval corresponding to the mean and std in each row
# Number of years of data
df_agg["Covid", "lower"] = df_agg["Covid", "mean"] - df_agg["Covid", "std"]
df_agg["Covid", "upper"] = df_agg["Covid", "mean"] + df_agg["Covid", "std"]
df_agg[("Flu", "lower")] = df_agg[("Flu", "mean")] - df_agg[("Flu", "std")]
df_agg[("Flu", "upper")] = df_agg[("Flu", "mean")] + df_agg[("Flu", "std")]
df_agg[("Bronchiolitis", "lower")] = (
    df_agg[("Bronchiolitis", "mean")] - df_agg[("Bronchiolitis", "std")]
)
df_agg[("Bronchiolitis", "upper")] = (
    df_agg[("Bronchiolitis", "mean")] + df_agg[("Bronchiolitis", "std")]
)
df_agg[("Pneumopathy", "lower")] = (
    df_agg[("Pneumopathy", "mean")] - df_agg[("Pneumopathy", "std")]
)
df_agg[("Pneumopathy", "upper")] = (
    df_agg[("Pneumopathy", "mean")] + df_agg[("Pneumopathy", "std")]
)
df_agg[("Total Resp. Synd.", "lower")] = (
    df_agg[("Total Resp. Synd.", "mean")] - df_agg[("Total Resp. Synd.", "std")]
)
df_agg[("Total Resp. Synd.", "upper")] = (
    df_agg[("Total Resp. Synd.", "mean")] + df_agg[("Total Resp. Synd.", "std")]
)
df_agg[("Number of consultations", "lower")] = (
    df_agg["Number of consultations"]["mean"] - df_agg["Number of consultations"]["std"]
)
df_agg[("Number of consultations", "upper")] = (
    df_agg["Number of consultations"]["mean"] - df_agg["Number of consultations"]["std"]
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


# %%
# Merge dataframes on the 'week' column
df_agg.reset_index(inplace=True)
df.reset_index(inplace=True)
result_df = pd.merge(df, df_agg, on="Week", how="inner")
result_df.set_index(["Year", "Week"], inplace=True)
# Sort by year and week
result_df.sort_index(inplace=True)

# %%
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

# %%
# Sort the level 1 headers according to the level 0 header
result_df.sort_index(axis=1, level=0, inplace=True)
