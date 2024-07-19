from kedro.pipeline import Pipeline, node

from .nodes import preprocessing_influenza, upload_data


def create_pipeline(**kwargs):
    return Pipeline(
        nodes=[
            node(
                preprocessing_influenza,
                inputs=[
                    "Influenza_cases_BB",
                    "Influenza_cases_BE",
                    "Influenza_cases_BW",
                    "Influenza_cases_BY",
                    "Influenza_cases_DE",
                    "Influenza_cases_HB",
                    "Influenza_cases_HE",
                    "Influenza_cases_HH",
                    "Influenza_cases_MV",
                    "Influenza_cases_NI",
                    "Influenza_cases_NW",
                    "Influenza_cases_RP",
                    "Influenza_cases_SH",
                    "Influenza_cases_SL",
                    "Influenza_cases_SN",
                    "Influenza_cases_ST",
                    "Influenza_cases_TH",
                ],
                outputs="Influenza_cases",
                name="preprocessing_influenza_node",
            ),
            node(
                upload_data,
                inputs="Influenza_cases",
                outputs="Influenza_cases_MongoDB",
                name="Influenza_cases_MongoDB_node",
            ),
        ]
    )
