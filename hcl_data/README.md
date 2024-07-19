# AIOLOS: Forecasting model for HCL Data

In this project, our goal is to create predictive models for estimating the number of patients diagnosed with various conditions, including COVID-19, influenza (grippe), respiratory syncytial virus (RSV), and other respiratory infections. Additionally, we aim to predict hospital admissions, deaths, and patients in critical care units over a two-week period.

## Pipeline inputs

The pipeline takes the following dataframes from the HCL dataset as input:

- `COVID_19_I.csv`: DataFrame for COVID-19 data type I (number of patients data).
- `COVID_19_II.csv`: DataFrame for COVID-19 data type II (death-related data).
- `COVID_19_III.csv`: DataFrame for COVID-19 data type III (critical health-related data).
- `GRIPPE_I.csv`: DataFrame for grippe data type I (number of patients data).
- `GRIPPE_II.csv`: DataFrame for grippe data type II (death-related data).
- `GRIPPE_III.csv`: DataFrame for grippe data type III (critical health-related data).
- `IR_AUTVIRUS_I.csv`: DataFrame for other virus data type I (number of patients data).
- `IR_AUTVIRUS_II.csv`: DataFrame for other virus data type II (death-related data).
- `IR_AUTVIRUS_III.csv`: DataFrame for other virus data type III (critical health-related data).
- `IR_GENERAL_I.csv`: DataFrame for general virus data type I (number of patients data).
- `IR_GENERAL_II.csv`: DataFrame for general virus data type II (death-related data).
- `IR_GENERAL_III.csv`: DataFrame for general virus data type III (critical health-related data).
- `RSV_I.csv`: DataFrame for RSV data type I (number of patients data).
- `RSV_II.csv`: DataFrame for RSV data type II (death-related data).
- `RSV_III.csv`: DataFrame for RSV data type III (critical health-related data).

## Pipeline Outputs

The pipeline generates the following output for each of the mentioned conditions (COVID-19, influenza, RSV, and other respiratory infections):

- `forecast_df.csv`: A DataFrame containing forecasts and confidence intervals for a two-day horizon. This includes predictions for the following categories:
  - Number of patients entering the hospital.
  - Number of deaths.
  - Number of patients in critical care units.

## Code Usage

### Building Docker Container

To build a Docker container based on the project's Dockerfile, follow these steps:

1. Clone this repository
2. Navigate to the project directory
3. Install the kedro-docker plugin:

```bash
pip install kedro-docker
```

4.  Build the Docker container:

```bash
kedro docker build
```

Don't forget to rebuild the container after making changes!

### Running the Entire Pipeline

To execute the entire pipeline in a specific environment, use the following command:

```bash
kedro docker run --env=<conf name> --docker-args="--env MongoDBUsername="XXX" --env MongoDBPassword="XXX" --env CGM_PASSWORD="XXX" CD_PASSWORD="XXX""
```

- `dev`: QH infra.
- `prod`: Umlaut infra
  By omitting the environment specification, the data will be stored locally, in the container:

```bash
kedro docker run
```

### Downloading the data from CGM's server

To get the data from CGM's server

```bash
kedro docker run --pipeline=data_extraction --env=<conf name>
```

### Preprocessing

For raw data preprocessing, you can run:

```bash
kedro docker run --pipeline=data_preprocessing --env=<conf name>
```
