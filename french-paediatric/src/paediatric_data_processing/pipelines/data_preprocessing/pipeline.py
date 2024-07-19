# src/data_project/pipeline.py
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import data_extraction, data_preprocessing, upload_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                data_extraction,
                inputs="paediatric_source",
                outputs="paediatric_extract",
                name="paediatric_data_extraction_node",
            ),
            node(
                data_preprocessing,
                inputs="paediatric_extract",
                outputs="paediatric_export",
                name="paediatric_data_preprocessing_node",
            ),
            node(
                upload_data,
                inputs="paediatric_export",
                outputs="paediatric_local",
                name="paediatric_local_save_node",
            ),
        ]
    )
