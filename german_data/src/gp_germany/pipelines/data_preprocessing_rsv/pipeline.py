from kedro.pipeline import Pipeline, node

from .nodes import preprocessing_rsv, upload_data


def create_pipeline(**kwargs):
    return Pipeline(
        nodes=[
            node(
                preprocessing_rsv,
                inputs="RSV_cases_SN",
                outputs="RSV_cases",
                name="preprocessing_rsv_node",
            ),
            node(
                upload_data,
                inputs="RSV_cases",
                outputs="RSV_cases_MongoDB",
                name="RSV_cases_MongoDB_node",
            ),
        ]
    )
