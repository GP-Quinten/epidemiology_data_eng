from kedro.pipeline import Pipeline, node, pipeline

from .nodes import validate_extract_schema


def create_pipeline(**kwargs) -> Pipeline:
    validate_extract_pipe = pipeline(
        [
            node(
                func=validate_extract_schema,
                inputs=["extract", "extract_schema"],
                outputs="data",
                name="validate_extract_schema",
            ),
        ]
    )
