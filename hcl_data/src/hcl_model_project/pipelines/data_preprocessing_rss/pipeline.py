from kedro.pipeline import Pipeline, node

from .nodes import (  # concat_preprocessing,
    add_new_category_to_dataframes,
    baseline_n_cases,
    baseline_n_cases_and_crit,
    baseline_n_cases_and_deaths,
    missing_data_prepro_I,
    missing_data_prepro_II,
    missing_data_prepro_III,
    preprocess_table_I,
    preprocess_table_II,
    preprocess_table_III,
    smooth_n_cases,
    smooth_n_cases_n_critical,
    smooth_n_cases_n_death,
    table_I_calculation_nowcasting,
    table_II_calculation_nowcasting,
    table_III_calculation_nowcasting,
    translate_dataframes,
    update_dataframes_SEMX_A_X,
    upload_data,
)


def create_pipeline(**kwargs):
    return Pipeline(
        nodes=[
            node(
                update_dataframes_SEMX_A_X,
                inputs="HCL_update_SEM",
                outputs=["df_number_of_id", "df_death", "df_critical_health"],
                name="HCL_update_SEM_node",
            ),
            node(
                translate_dataframes,
                inputs=["df_number_of_id", "df_death", "df_critical_health"],
                outputs=[
                    "df_number_of_id_translate",
                    "df_death_translate",
                    "df_critical_health_translate",
                ],
                name="translate_dataframes_node",
            ),
            node(
                add_new_category_to_dataframes,
                inputs=[
                    "df_number_of_id_translate",
                    "df_death_translate",
                    "df_critical_health_translate",
                ],
                outputs=[
                    "df_table_I_newcat",
                    "df_table_II_newcat",
                    "df_table_III_newcat",
                ],
                name="add_new_category_to_dataframes_node",
            ),
            node(
                preprocess_table_I,
                inputs="df_table_I_newcat",
                outputs="table_I",
                name="preprocess_table_I_node",
            ),
            node(
                preprocess_table_II,
                inputs="df_table_II_newcat",
                outputs="table_II",
                name="preprocess_table_II_node",
            ),
            node(
                preprocess_table_III,
                inputs="df_table_III_newcat",
                outputs="table_III",
                name="preprocess_table_III_node",
            ),
            node(
                missing_data_prepro_I,
                inputs="table_I",
                outputs="table_I_missing_data",
                name="missing_data_prepro_I_node",
            ),
            node(
                missing_data_prepro_II,
                inputs="table_II",
                outputs="table_II_missing_data",
                name="missing_data_prepro_II_node",
            ),
            node(
                missing_data_prepro_III,
                inputs="table_III",
                outputs="table_III_missing_data",
                name="missing_data_prepro_III_node",
            ),
            node(
                smooth_n_cases,
                inputs="table_I_missing_data",
                outputs="table_I_historical_smooth",
                name="table_I_historical_smooth_node",
            ),
            node(
                smooth_n_cases_n_death,
                inputs="table_II_missing_data",
                outputs="table_II_historical_smooth",
                name="table_II_historical_smooth_node",
            ),
            node(
                smooth_n_cases_n_critical,
                inputs="table_III_missing_data",
                outputs="table_III_historical_smooth",
                name="table_III_historical_smooth_node",
            ),
            node(
                baseline_n_cases,
                inputs="table_I_historical_smooth",
                outputs="table_I_historical_baseline",
                name="table_I_historical_baseline_node",
            ),
            node(
                baseline_n_cases_and_deaths,
                inputs="table_II_historical_smooth",
                outputs="table_II_historical_baseline",
                name="table_II_historical_baseline_node",
            ),
            node(
                baseline_n_cases_and_crit,
                inputs="table_III_historical_smooth",
                outputs="table_III_historical_baseline",
                name="table_III_historical_baseline_node",
            ),
            node(
                table_I_calculation_nowcasting,
                inputs="table_I_historical_baseline",
                outputs="table_I_final",
                name="table_I_final_node",
            ),
            node(
                table_II_calculation_nowcasting,
                inputs="table_II_historical_baseline",
                outputs="table_II_final",
                name="table_II_final_node",
            ),
            node(
                table_III_calculation_nowcasting,
                inputs="table_III_historical_baseline",
                outputs="table_III_final",
                name="table_III_final_node",
            ),
            node(
                upload_data,
                inputs="table_I_final",
                outputs="table_I_final.local",
                name="table_I_final_extract_node",
            ),
            node(
                upload_data,
                inputs="table_II_final",
                outputs="table_II_final.local",
                name="table_II_final_extract_node",
            ),
            node(
                upload_data,
                inputs="table_III_final",
                outputs="table_III_final.local",
                name="table_III_final_extract_node",
            ),
        ]
    )
