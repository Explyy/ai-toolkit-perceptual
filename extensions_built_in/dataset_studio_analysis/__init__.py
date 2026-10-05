from toolkit.extension import Extension

class DatasetStudioAnalysisExtension(Extension):
    uid = "dataset_studio_analysis"
    name = "Dataset Studio automatic preselection"

    @classmethod
    def get_process(cls):
        from .process import DatasetStudioAnalysisProcess
        return DatasetStudioAnalysisProcess

AI_TOOLKIT_EXTENSIONS = [DatasetStudioAnalysisExtension]
