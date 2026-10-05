"""Image-build compatibility gate; imports only, never downloads a model."""
import inspect
from pathlib import Path
import sys
sys.path.insert(0, '/app/ai-toolkit')
from version import VERSION
from extensions_built_in.captioner.Qwen3VLCaptioner import Qwen3VLCaptioner
from extensions_built_in.captioner.BaseCaptioner import BaseCaptioner, CaptionConfig
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText

assert VERSION == "0.13.23", "Unexpected upstream runtime version"
assert issubclass(Qwen3VLCaptioner, BaseCaptioner)
source = inspect.getsource(Qwen3VLCaptioner)
assert "AutoModelForImageTextToText" in source or "Qwen3VLForConditionalGeneration" in source
base = inspect.getsource(BaseCaptioner)
for field in ("path_to_caption", "caption_prompt", "max_new_tokens", "caption_extension"):
    assert field in base or field in source, f"Missing native caption field: {field}"
assert Path('/app/ai-toolkit/run.py').is_file()
config = CaptionConfig(model_name_or_path='Qwen/Qwen3-VL-2B-Instruct', path_to_caption='/workspace/dataset-studio/data/synthetic-scope',
                       extensions=['jpg', 'jpeg', 'png', 'webp'], caption_extension='txt', recaption=True,
                       caption_prompt='Describe the synthetic image.', max_res=512, max_new_tokens=128)
assert config.device == 'cuda' and config.model_name_or_path == 'Qwen/Qwen3-VL-2B-Instruct'
print("Studio runtime 0.13.23 Qwen3-VL compatibility imports passed (inference not run)")
