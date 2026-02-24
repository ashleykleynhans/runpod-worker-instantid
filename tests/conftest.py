import sys
import os
import types
from unittest.mock import MagicMock

# Add src/ to path so handler's local imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ---------------------------------------------------------------------------
# Mock heavy third-party modules before any src module is imported.
# This allows the module-level initialisation code to execute without
# GPU hardware, model files, or large dependencies being present.
# ---------------------------------------------------------------------------


# -- Stub base class for diffusers pipelines --------------------------------
# MagicMock cannot be used as a base class for inheritance (class methods like
# from_pretrained are not resolvable).  We provide a lightweight stub instead.

class _PipelineStub:
    """Minimal stand-in for any diffusers pipeline base class."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        inst = cls()
        for k, v in kwargs.items():
            setattr(inst, k, v)
        return inst

    @classmethod
    def from_single_file(cls, *args, **kwargs):
        inst = cls()
        for k, v in kwargs.items():
            setattr(inst, k, v)
        return inst

    def to(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        # Return a MagicMock for any attribute not explicitly set so that
        # downstream code (e.g. self.unet, self.vae, ...) keeps working.
        mock = MagicMock()
        object.__setattr__(self, name, mock)
        return mock


# -- torch ------------------------------------------------------------------
mock_torch = MagicMock()
mock_torch.float16 = 'float16'
mock_torch.float32 = 'float32'
mock_torch.__contains__ = lambda self, x: False
mock_torch.device.return_value = 'cpu'
# Make @torch.no_grad() a no-op decorator so the wrapped function is preserved
mock_torch.no_grad.return_value = lambda fn: fn
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.mps'] = MagicMock()

# -- cv2 --------------------------------------------------------------------
mock_cv2 = MagicMock()
mock_cv2.COLOR_BGR2RGB = 4
mock_cv2.COLOR_RGB2BGR = 16
sys.modules['cv2'] = mock_cv2

# -- diffusers (and all submodules) -----------------------------------------
# Build a real module object for 'diffusers' so that attribute access from
# both ``import diffusers`` and ``from diffusers import X`` works correctly.

_diffusers = types.ModuleType('diffusers')

# Pipeline stubs that can be subclassed
_diffusers.StableDiffusionXLControlNetPipeline = _PipelineStub
_diffusers.StableDiffusionPipeline = _PipelineStub
_diffusers.StableDiffusionXLPipeline = _PipelineStub

# Scheduler stubs (plain MagicMock is fine – they are instantiated, not subclassed)
_diffusers.EulerDiscreteScheduler = MagicMock()

# Model stubs
_diffusers.UNet2DConditionModel = MagicMock()
_diffusers.SchedulerMixin = MagicMock()
_diffusers.AutoencoderKL = MagicMock()

sys.modules['diffusers'] = _diffusers

# diffusers.models – ControlNetModel must be a real class so isinstance() works
class _ControlNetModelStub:
    """Stand-in for diffusers.models.ControlNetModel."""
    config = MagicMock()

    def __call__(self, *args, **kwargs):
        return MagicMock(), MagicMock()

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

_diffusers_models = types.ModuleType('diffusers.models')
_diffusers_models.ControlNetModel = _ControlNetModelStub
sys.modules['diffusers.models'] = _diffusers_models

# diffusers.image_processor
sys.modules['diffusers.image_processor'] = MagicMock()

# diffusers.utils
_diffusers_utils = MagicMock()
# Make @replace_example_docstring(...) a no-op decorator
_diffusers_utils.replace_example_docstring.return_value = lambda fn: fn
sys.modules['diffusers.utils'] = _diffusers_utils
sys.modules['diffusers.utils.torch_utils'] = MagicMock()
sys.modules['diffusers.utils.import_utils'] = MagicMock()

# diffusers.pipelines
_diffusers_pipelines = types.ModuleType('diffusers.pipelines')
sys.modules['diffusers.pipelines'] = _diffusers_pipelines

_diffusers_pipelines_sd = types.ModuleType('diffusers.pipelines.stable_diffusion')
_diffusers_pipelines_sd.convert_from_ckpt = MagicMock()
sys.modules['diffusers.pipelines.stable_diffusion'] = _diffusers_pipelines_sd
sys.modules['diffusers.pipelines.stable_diffusion.convert_from_ckpt'] = MagicMock()

_diffusers_pipelines_sdxl = types.ModuleType('diffusers.pipelines.stable_diffusion_xl')
_diffusers_pipelines_sdxl.StableDiffusionXLPipelineOutput = MagicMock()
sys.modules['diffusers.pipelines.stable_diffusion_xl'] = _diffusers_pipelines_sdxl

_diffusers_pipelines_cn = types.ModuleType('diffusers.pipelines.controlnet')
_diffusers_pipelines_cn_multi = types.ModuleType('diffusers.pipelines.controlnet.multicontrolnet')

class _MultiControlNetModelStub:
    """Stand-in for MultiControlNetModel."""
    def __call__(self, *args, **kwargs):
        return MagicMock(), MagicMock()

_diffusers_pipelines_cn_multi.MultiControlNetModel = _MultiControlNetModelStub
sys.modules['diffusers.pipelines.controlnet'] = _diffusers_pipelines_cn
sys.modules['diffusers.pipelines.controlnet.multicontrolnet'] = _diffusers_pipelines_cn_multi

# diffusers.schedulers
_diffusers_schedulers = types.ModuleType('diffusers.schedulers')
for _sched in ['DDIMScheduler', 'DDPMScheduler', 'LMSDiscreteScheduler',
               'EulerDiscreteScheduler', 'EulerAncestralDiscreteScheduler',
               'UniPCMultistepScheduler']:
    setattr(_diffusers_schedulers, _sched, MagicMock())
sys.modules['diffusers.schedulers'] = _diffusers_schedulers

sys.modules['diffusers.loaders'] = MagicMock()
sys.modules['diffusers.loaders.ip_adapter'] = MagicMock()

# -- insightface -------------------------------------------------------------
sys.modules['insightface'] = MagicMock()
sys.modules['insightface.app'] = MagicMock()

# -- runpod ------------------------------------------------------------------
sys.modules['runpod'] = MagicMock()
sys.modules['runpod.serverless'] = MagicMock()
sys.modules['runpod.serverless.utils'] = MagicMock()
sys.modules['runpod.serverless.utils.rp_validator'] = MagicMock()
sys.modules['runpod.serverless.modules'] = MagicMock()
sys.modules['runpod.serverless.modules.rp_logger'] = MagicMock()

# -- huggingface_hub ---------------------------------------------------------
sys.modules['huggingface_hub'] = MagicMock()

# -- ip_adapter (local src but lightweight once torch is mocked) -------------
sys.modules['ip_adapter'] = MagicMock()
sys.modules['ip_adapter.resampler'] = MagicMock()
sys.modules['ip_adapter.utils'] = MagicMock()

# ip_adapter.attention_processor – IPAttnProcessor must be a real class so
# isinstance() checks in the pipeline code work correctly.
class _IPAttnProcessorStub:
    def __init__(self, *args, **kwargs):
        self.scale = kwargs.get('scale', 1.0)

    def to(self, *args, **kwargs):
        return self

class _AttnProcessorStub:
    def to(self, *args, **kwargs):
        return self

_ip_adapter_attn = types.ModuleType('ip_adapter.attention_processor')
_ip_adapter_attn.IPAttnProcessor = _IPAttnProcessorStub
_ip_adapter_attn.IPAttnProcessor2_0 = _IPAttnProcessorStub
_ip_adapter_attn.AttnProcessor = _AttnProcessorStub
_ip_adapter_attn.AttnProcessor2_0 = _AttnProcessorStub
sys.modules['ip_adapter.attention_processor'] = _ip_adapter_attn

# -- other heavy deps used by model_util / pipeline --------------------------
sys.modules['transformers'] = MagicMock()
sys.modules['safetensors'] = MagicMock()
sys.modules['safetensors.torch'] = MagicMock()
sys.modules['omegaconf'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['onnxruntime'] = MagicMock()
_mock_xformers = MagicMock()
_mock_xformers.__version__ = '0.0.20'
sys.modules['xformers'] = _mock_xformers
sys.modules['packaging'] = MagicMock()
sys.modules['packaging.version'] = MagicMock()
sys.modules['intel_extension_for_pytorch'] = MagicMock()

# -- PIL (use real Pillow if available, otherwise mock) ----------------------
try:
    import PIL          # noqa: F401
except ImportError:
    sys.modules['PIL'] = MagicMock()
    sys.modules['PIL.Image'] = MagicMock()
    sys.modules['PIL.ImageOps'] = MagicMock()

# -- numpy (use real numpy if available, otherwise mock) ---------------------
try:
    import numpy        # noqa: F401
except ImportError:
    mock_np = MagicMock()
    mock_np.iinfo.return_value.max = 2147483647
    sys.modules['numpy'] = mock_np
