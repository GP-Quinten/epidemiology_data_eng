# Vivalto Data Preprocessing

In this project, we aim to preprocess data obtained from various FINESS codes of private hospitals in France. The data focuses on the number of NRSS (Nombre de Séjours hospitaliers complets) for different categories of respiratory infectious diseases, segmented by age group and duration of the RSS. Additionally, the dataset provides information on the number of deaths due to respiratory diseases and the number of critical cases per geographical region.

## Pipeline Inputs

The pipeline requires the following dataframes from the Vivalto dataset:

- Dataframes starting with "Données Vivalto" in the server.

## Pipeline Outputs

The pipeline generates three output dataframes stored in MongoDB:

1. `table_I_final_vivalto`: Provides the number of NRSS per disease, age class, and RSS duration. It includes a fictitious baseline and an alert when the number of NRSS exceeds the baseline.

2. `table_II_final_vivalto`: Presents the number of deaths per disease with a fictitious baseline and an alert when the death count surpasses the baseline.

3. `table_III_final_vivalto`: Offers the number of critical cases per disease and geographical region. It includes a baseline and an alert when the number of critical cases exceeds the baseline.


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
kedro docker run --pipeline=data_preprocessing_vivalto --env=<conf name>
```