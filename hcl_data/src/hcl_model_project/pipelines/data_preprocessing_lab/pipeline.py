from kedro.pipeline import Pipeline, node

from .nodes import (  # create_nowcasting,; merge_nowcast_baseline
    create_baseline,
    fraction_of_pos,
    merge_cas_pos_and_nb_prelev,
    preprocessing_by_sheets,
    translate_and_reorder,
    upload_data,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                preprocessing_by_sheets,
                inputs="HCL_data",
                outputs=["df_cas_pos", "df_nb_prelev"],
                name="list_HCL_data_files",
            ),
            node(
                merge_cas_pos_and_nb_prelev,
                inputs=["df_cas_pos", "df_nb_prelev"],
                outputs="merged_df",
                name="merge_cas_pos_and_nb_prelev",
            ),
            node(
                translate_and_reorder,
                inputs="merged_df",
                outputs="hcl_lab_final",
                name="translate_and_reorder",
            ),
            node(
                fraction_of_pos,
                inputs="hcl_lab_baseline_final",
                outputs="hcl_lab_fraction_final",
                name="fraction_of_pos",
            ),
            node(
                create_baseline,
                inputs="hcl_lab_final",
                outputs="hcl_lab_baseline_final",
                name="create_baseline",
            ),
            node(
                upload_data,
                inputs="hcl_lab_fraction_final",
                outputs="hcl_lab_fraction_final.local",
                name="hcl_lab_fraction_final_extract_node",
            ),
            node(
                upload_data,
                inputs="hcl_lab_baseline_final",
                outputs="hcl_lab_baseline_final.local",
                name="hcl_lab_baseline_final_extract_node",
            ),
        ]
    )
