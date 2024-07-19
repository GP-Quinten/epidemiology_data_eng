# src/data_project/pipeline.py
from kedro.pipeline import Pipeline, node

from .nodes import create_alerts, preprocessing_google, upload_data


def create_pipeline(**kwargs):
    return Pipeline(
        nodes=[
            node(
                preprocessing_google,
                inputs="symptoms",
                outputs=["Google_symptoms_fr", "Google_symptoms_de"],
                name="processing_google_node",
            ),
            node(
                create_alerts,
                inputs="Google_symptoms_fr",
                outputs="Google_symptoms_fr_alerts",
                name="create_alerts_google_fr_node",
            ),
            node(
                create_alerts,
                inputs="Google_symptoms_de",
                outputs="Google_symptoms_de_alerts",
                name="create_alerts_google_de_node",
            ),
            node(
                upload_data,
                inputs="Google_symptoms_fr",
                outputs="Google_symptoms_fr_MongoDB",
                name="Google_symptoms_fr_MongoDB_node",
            ),
            node(
                upload_data,
                inputs="Google_symptoms_de",
                outputs="Google_symptoms_de_MongoDB",
                name="Google_symptoms_de_MongoDB_node",
            ),
            node(
                upload_data,
                inputs="Google_symptoms_fr_alerts",
                outputs="Google_symptoms_fr_alerts_quinten_MongoDB",
                name="Google_symptoms_fr_alerts_MongoDB_node",
            ),
            node(
                upload_data,
                inputs="Google_symptoms_de_alerts",
                outputs="Google_symptoms_de_alerts_quinten_MongoDB",
                name="Google_symptoms_de_alerts_MongoDB_node",
            ),
        ]
    )
