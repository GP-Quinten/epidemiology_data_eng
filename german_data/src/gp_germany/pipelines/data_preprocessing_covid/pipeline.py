from kedro.pipeline import Pipeline, node

from .nodes import create_alerts, download_data, preprocessing_covid, upload_data


def create_pipeline(**kwargs):
    return Pipeline(
        nodes=[
            node(
                download_data,
                inputs="Hospitalization_ger.source",
                outputs="Hospitalization_ger.extract",
                name="download_Hospitalization_ger_node",
            ),
            node(
                download_data,
                inputs="RKI_full_data.source",
                outputs="RKI_full_data.extract",
                name="download_RKI_full_data_node",
            ),
            node(
                preprocessing_covid,
                inputs=["RKI_full_data.extract", "Hospitalization_ger.extract"],
                outputs=[
                    "COVID_cases",
                    "COVID_deaths",
                    "COVID_hosp",
                    "COVID_cases_weekly",
                    "COVID_deaths_weekly",
                    "COVID_hosp_weekly",
                ],
                name="preprocessing_covid_node",
            ),
            node(
                create_alerts,
                inputs="COVID_cases",
                outputs="COVID_cases_alerts_Quinten",
                name="COVID_cases_quinten_alerts_node",
            ),
            node(
                create_alerts,
                inputs="COVID_deaths",
                outputs="COVID_deaths_alerts_Quinten",
                name="COVID_deaths_quinten_alerts_node",
            ),
            node(
                create_alerts,
                inputs="COVID_hosp",
                outputs="COVID_hosp_alerts_Quinten",
                name="COVID_hosp_quinten_alerts_node",
            ),
            node(
                upload_data,
                inputs="COVID_cases",
                outputs="COVID_cases_MongoDB",
                name="upload_COVID_cases_MongoDB_node",
            ),
            node(
                upload_data,
                inputs="COVID_deaths",
                outputs="COVID_deaths_MongoDB",
                name="upload_COVID_deaths_MongoDB_node",
            ),
            node(
                upload_data,
                inputs="COVID_hosp",
                outputs="COVID_hosp_MongoDB",
                name="upload_COVID_hosp_MongoDB_node",
            ),
            node(
                upload_data,
                inputs="COVID_cases_alerts_Quinten",
                outputs="COVID_cases_alerts_Quinten_MongoDB",
                name="upload_COVID_cases_alerts_Quinten_MongoDB_node",
            ),
            node(
                upload_data,
                inputs="COVID_deaths_alerts_Quinten",
                outputs="COVID_deaths_alerts_Quinten_MongoDB",
                name="upload_COVID_deaths_alerts_Quinten_MongoDB_node",
            ),
            node(
                upload_data,
                inputs="COVID_hosp_alerts_Quinten",
                outputs="COVID_hosp_alerts_Quinten_MongoDB",
                name="upload_COVID_hosp_alerts_Quinten_MongoDB_node",
            ),
        ]
    )
