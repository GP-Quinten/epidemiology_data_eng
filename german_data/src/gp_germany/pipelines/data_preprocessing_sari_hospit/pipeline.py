from kedro.pipeline import Pipeline, node

from .nodes import create_alerts, download_data, preprocessing_hospit, upload_data


def create_pipeline(**kwargs):
    return Pipeline(
        nodes=[
            node(
                download_data,
                inputs="SARI_hospit.source",
                outputs="SARI_hospit.extract",
                name="download_SARI_hosp_node",
            ),
            node(
                preprocessing_hospit,
                inputs="SARI_hospit.extract",
                outputs="SARI_hospit_preprocessed",
                name="preprocessing_hospit_node",
            ),
            node(
                create_alerts,
                inputs="SARI_hospit_preprocessed",
                outputs="SARI_hospit_alerts_quinten",
                name="create_sari_hospit_alerts_quinten_node",
            ),
            node(
                upload_data,
                inputs="SARI_hospit_preprocessed",
                outputs="SARI_hospit_preprocessed_MongoDB",
                name="SARI_hospit_MongoDB_node",
            ),
            node(
                upload_data,
                inputs="SARI_hospit_alerts_quinten",
                outputs="SARI_hospit_alerts_quinten_MongoDB",
                name="SARI_hospit_alerts_quinten_MongoDB_node",
            ),
        ]
    )
