import pandas as pd
import pandera as pa


def infer_data_schema(df: pd.DataFrame) -> str:
    """
    Infer the data schema from a DataFrame and return it as a YAML string.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        str: YAML schema representation.
    """
    schema = pa.infer_schema(df)
    yaml_schema = schema.to_yaml()
    return yaml_schema
