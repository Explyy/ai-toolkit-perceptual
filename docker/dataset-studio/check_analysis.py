"""Actual pinned image import/interface gate, no models loaded or downloaded."""
import inspect
import importlib.metadata
import sys
sys.path.insert(0,'/app/ai-toolkit')
from jobs.process import BaseExtensionProcess
from extensions_built_in.dataset_studio_analysis import AI_TOOLKIT_EXTENSIONS
from extensions_built_in.dataset_studio_analysis.process import DatasetStudioAnalysisProcess
from extensions_built_in.dataset_studio_analysis.artifacts import configuration
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, RTDetrForObjectDetection, VitPoseForPoseEstimation
import cv2
import onnxruntime
assert issubclass(DatasetStudioAnalysisProcess,BaseExtensionProcess)
assert list(inspect.signature(BaseExtensionProcess.__init__).parameters)==['self','process_id','job','config']
assert AI_TOOLKIT_EXTENSIONS[0].get_process() is DatasetStudioAnalysisProcess
assert configuration()[0]['version']=='arcface-depth-pose-v1'
assert importlib.metadata.version('transformers')=='5.5.3'
assert onnxruntime.__version__=='1.30.0'
assert cv2.__version__=='4.11.0'
assert hasattr(cv2,'FaceDetectorYN') and hasattr(cv2.FaceDetectorYN,'create')
import numpy
import torch
assert numpy.__version__=='1.26.4'
assert torch.__version__=='2.13.0+cu130'
assert importlib.metadata.version('flatbuffers')=='25.12.19'
print('Studio automatic-analysis actual runtime imports passed; inference not run')
