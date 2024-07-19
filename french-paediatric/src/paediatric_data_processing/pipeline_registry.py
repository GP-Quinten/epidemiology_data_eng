#from .pipelines.data_preprocessing.pipeline import create_pipeline

#from .pipelines.modelling.pipeline import create_pipeline as create_data_preprocessing_pipeline


"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

#def register_pipelines():
#    return {
#        "data_preprocessing": create_data_preprocessing_pipeline(),
        # Ajoutez d'autres pipelines ici si nécessaire
#    }


"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
