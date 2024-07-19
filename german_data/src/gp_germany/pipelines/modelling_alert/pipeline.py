from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import trends_prediction, upload_data


def create_pipeline(**kwargs) -> Pipeline:
    alert_pipeline = pipeline(
        [
            node(
                trends_prediction,
                inputs=[
                    "German_GP_Diagnoses_counts_for_trends",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_cgm",
                    "params:size_FW_daily_alert",
                ],
                outputs="German_GP_Alert_Diagnoses",
                name="alert_diagnoses_node",
            ),
            node(
                trends_prediction,
                inputs=[
                    "German_GP_Prescriptions_counts_for_trends",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_cgm",
                    "params:size_FW_daily_alert",
                ],
                outputs="German_GP_Alert_Prescriptions",
                name="alert_prescriptions_node",
            ),
            node(
                trends_prediction,
                inputs=[
                    "German_Pharmacy_Sales",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_cgm",
                    "params:size_FW_daily_alert",
                ],
                outputs="German_Pharmacy_Alert_Sales",
                name="alert_Sales_node",
            ),
            node(
                trends_prediction,
                inputs=[
                    "German_Pharmacy_Sales_rel",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_cgm",
                    "params:size_FW_daily_alert",
                ],
                outputs="German_Pharmacy_Alert_Sales_rel",
                name="alert_Sales_rel_node",
            ),
            node(
                trends_prediction,
                inputs=[
                    "Google_symptoms_fr",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_google",
                    "params:size_FW_daily_alert",
                ],
                outputs="Alert_Google_symptoms_fr",
                name="Alert_Google_symptoms_fr_node",
            ),
            node(
                trends_prediction,
                inputs=[
                    "Google_symptoms_de",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_google",
                    "params:size_FW_daily_alert",
                ],
                outputs="Alert_Google_symptoms_de",
                name="Alert_Google_symptoms_de_node",
            ),
            node(
                trends_prediction,
                inputs=[
                    "COVID_cases",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_covid",
                    "params:size_FW_daily_alert",
                ],
                outputs="Alert_COVID_cases",
                name="Alert_COVID_cases_node",
            ),
            node(
                trends_prediction,
                inputs=[
                    "COVID_deaths",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_covid",
                    "params:size_FW_daily_alert",
                ],
                outputs="Alert_COVID_deaths",
                name="Alert_COVID_deaths_node",
            ),
            node(
                trends_prediction,
                inputs=[
                    "COVID_hosp",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_covid",
                    "params:size_FW_daily_alert",
                ],
                outputs="Alert_COVID_hosp",
                name="Alert_COVID_hosp_node",
            ),
            node(
                trends_prediction,
                inputs=[
                    "Influenza_cases",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_influenza",
                    "params:size_FW_daily_alert",
                ],
                outputs="Alert_Influenza_cases",
                name="Alert_Influenza_cases_node",
            ),
            node(
                trends_prediction,
                inputs=[
                    "RSV_cases",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_rsv",
                    "params:size_FW_daily_alert",
                ],
                outputs="Alert_RSV_cases",
                name="Alert_RSV_cases_node",
            ),
            node(
                trends_prediction,
                inputs=[
                    "SARI_hospit_preprocessed",
                    "params:size_FW_weekly",
                    "params:size_FW_daily",
                    "params:time_dim_SARI",
                    "params:size_FW_daily_alert",
                ],
                outputs="Alert_SARI_hospit",
                name="Alert_SARI_hospit_node",
            ),
            node(
                upload_data,
                inputs="German_GP_Alert_Diagnoses",
                outputs="German_GP_Alert_Diagnoses_MongoDB",
                name="upload_alert_diagnoses_node",
            ),
            node(
                upload_data,
                inputs="German_GP_Alert_Prescriptions",
                outputs="German_GP_Alert_Prescriptions_MongoDB",
                name="upload_alert_prescriptions_node",
            ),
            node(
                upload_data,
                inputs="German_Pharmacy_Alert_Sales",
                outputs="German_Pharmacy_Alert_Sales_MongoDB",
                name="upload_alert_Sales_node",
            ),
            node(
                upload_data,
                inputs="German_Pharmacy_Alert_Sales_rel",
                outputs="German_Pharmacy_Alert_Sales_rel_MongoDB",
                name="upload_alert_Sales_rel_node",
            ),
            node(
                upload_data,
                inputs="Alert_Google_symptoms_fr",
                outputs="Alert_Google_symptoms_fr_MongoDB",
                name="upload_Alert_Google_symptoms_fr",
            ),
            node(
                upload_data,
                inputs="Alert_Google_symptoms_de",
                outputs="Alert_Google_symptoms_de_MongoDB",
                name="upload_Alert_Google_symptoms_de",
            ),
            node(
                upload_data,
                inputs="Alert_COVID_cases",
                outputs="Alert_COVID_cases_MongoDB",
                name="upload_alert_covid_cases_node",
            ),
            node(
                upload_data,
                inputs="Alert_COVID_deaths",
                outputs="Alert_COVID_deaths_MongoDB",
                name="upload_alert_covid_deaths_node",
            ),
            node(
                upload_data,
                inputs="Alert_COVID_hosp",
                outputs="Alert_COVID_hosp_MongoDB",
                name="upload_alert_covid_hosp_node",
            ),
            node(
                upload_data,
                inputs="Alert_Influenza_cases",
                outputs="Alert_Influenza_cases_MongoDB",
                name="upload_alert_influenza_cases_node",
            ),
            node(
                upload_data,
                inputs="Alert_RSV_cases",
                outputs="Alert_RSV_cases_MongoDB",
                name="upload_alert_RSV_cases_node",
            ),
            # node(
            #     upload_data,
            #     inputs="Alert_SARI_hospit",
            #     outputs="Alert_SARI_hospit_MongoDB",
            #     name="upload_alert_SARI_hospit_node",
            # ),
        ]
    )
    return alert_pipeline
