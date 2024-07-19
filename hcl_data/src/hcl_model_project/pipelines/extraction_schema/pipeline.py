from kedro.pipeline import Pipeline, node, pipeline

from .nodes import infer_data_schema


def create_pipeline(**kwargs) -> Pipeline:
    infer_schema_pipeline = pipeline(
        [
            node(
                func=infer_data_schema,
                inputs="",
                outputs="extract_schema",
                name="infer_data_schema",
            ),
        ]
    )
