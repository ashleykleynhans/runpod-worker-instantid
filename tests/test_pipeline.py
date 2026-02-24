from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
from PIL import Image

import pipeline_stable_diffusion_xl_instantid as pipeline_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipe():
    """Create a StableDiffusionXLInstantIDPipeline instance with mocked internals."""
    pipe = pipeline_mod.StableDiffusionXLInstantIDPipeline()
    pipe.unet = MagicMock()
    pipe.unet.config.cross_attention_dim = 768
    pipe.unet.config.block_out_channels = [320, 640, 1280, 1280]
    pipe.unet.device = 'cpu'
    pipe.unet.dtype = 'float32'
    pipe.unet.attn_processors = MagicMock()
    pipe.vae = MagicMock()
    pipe.controlnet = MagicMock()
    pipe.scheduler = MagicMock()
    pipe.image_proj_model = MagicMock()
    pipe.image_proj_model_in_features = 512
    pipe.device = 'cpu'
    pipe.dtype = 'float32'
    return pipe


def _make_rgb_image(w=64, h=64):
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# draw_kps (pipeline module's own copy)
# ---------------------------------------------------------------------------

class TestPipelineDrawKps:
    def test_returns_pil_image(self):
        image_pil = _make_rgb_image(100, 100)
        kps = np.array([
            [20, 20], [40, 20], [30, 40], [20, 60], [40, 60],
        ], dtype=np.float32)

        pipeline_mod.cv2.ellipse2Poly.return_value = np.array([[25, 30], [30, 35], [35, 30]])
        pipeline_mod.cv2.fillConvexPoly.return_value = np.zeros((100, 100, 3), dtype=np.float64)
        pipeline_mod.cv2.circle.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        result = pipeline_mod.draw_kps(image_pil, kps)
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# cuda
# ---------------------------------------------------------------------------

class TestCuda:
    def test_basic_cuda(self):
        pipe = _make_pipe()
        pipe.cuda()

    def test_cuda_with_xformers_available(self):
        pipe = _make_pipe()
        with patch.object(pipeline_mod, 'is_xformers_available', return_value=True):
            pipe.cuda(use_xformers=True)

    def test_cuda_with_xformers_unavailable(self):
        pipe = _make_pipe()
        with patch.object(pipeline_mod, 'is_xformers_available', return_value=False):
            try:
                pipe.cuda(use_xformers=True)
                assert False, 'Expected ValueError'
            except ValueError as e:
                assert 'xformers is not available' in str(e)

    def test_cuda_without_image_proj_model(self):
        pipe = _make_pipe()
        del pipe.image_proj_model
        pipe.cuda()


# ---------------------------------------------------------------------------
# load_ip_adapter_instantid
# ---------------------------------------------------------------------------

class TestLoadIpAdapterInstantid:
    def test_calls_set_methods(self):
        pipe = _make_pipe()
        with patch.object(pipe, 'set_image_proj_model') as mock_proj, \
             patch.object(pipe, 'set_ip_adapter') as mock_adapter:
            pipe.load_ip_adapter_instantid('ckpt.bin', image_emb_dim=512, num_tokens=16, scale=0.5)
            mock_proj.assert_called_once_with('ckpt.bin', 512, 16)
            mock_adapter.assert_called_once_with('ckpt.bin', 16, 0.5)


# ---------------------------------------------------------------------------
# set_image_proj_model
# ---------------------------------------------------------------------------

class TestSetImageProjModel:
    def test_sets_model(self):
        pipe = _make_pipe()
        pipeline_mod.torch.load.return_value = {'image_proj': {'key': 'value'}}
        pipe.set_image_proj_model('ckpt.bin')
        assert pipe.image_proj_model_in_features == 512

    def test_without_image_proj_key(self):
        pipe = _make_pipe()
        pipeline_mod.torch.load.return_value = {'key': 'value'}
        pipe.set_image_proj_model('ckpt.bin')


# ---------------------------------------------------------------------------
# set_ip_adapter
# ---------------------------------------------------------------------------

class TestSetIpAdapter:
    def test_processes_attn_processors(self):
        pipe = _make_pipe()
        pipe.unet.attn_processors.keys.return_value = [
            'mid_block.attn1.processor',
            'up_blocks.0.attn2.processor',
            'down_blocks.1.attn2.processor',
        ]
        pipeline_mod.torch.load.return_value = {'ip_adapter': {}}
        pipeline_mod.torch.nn.ModuleList.return_value = MagicMock()
        pipe.set_ip_adapter('ckpt.bin', num_tokens=16, scale=0.5)

    def test_without_ip_adapter_key(self):
        pipe = _make_pipe()
        pipe.unet.attn_processors.keys.return_value = ['mid_block.attn1.processor']
        pipeline_mod.torch.load.return_value = {'key': 'value'}
        pipeline_mod.torch.nn.ModuleList.return_value = MagicMock()
        pipe.set_ip_adapter('ckpt.bin', num_tokens=16, scale=0.5)


# ---------------------------------------------------------------------------
# set_ip_adapter_scale
# ---------------------------------------------------------------------------

class TestSetIpAdapterScale:
    def test_sets_scale(self):
        pipe = _make_pipe()
        ip_proc = pipeline_mod.IPAttnProcessor(scale=1.0)
        pipe.unet.attn_processors.values.return_value = [ip_proc, MagicMock()]
        pipe.set_ip_adapter_scale(0.7)
        assert ip_proc.scale == 0.7


# ---------------------------------------------------------------------------
# _encode_prompt_image_emb
# ---------------------------------------------------------------------------

class TestEncodePromptImageEmb:
    def test_with_tensor(self):
        pipe = _make_pipe()
        mock_emb = MagicMock()
        pipeline_mod.torch.Tensor = type(mock_emb)
        # image_proj_model(prompt_image_emb) must return something with shape=(bs, seq, dim)
        proj_output = MagicMock()
        proj_output.shape = (1, 16, 512)
        pipe.image_proj_model.return_value = proj_output
        pipe._encode_prompt_image_emb(mock_emb, 'cpu', 1, 'float32', True)

    def test_without_tensor(self):
        pipe = _make_pipe()
        pipeline_mod.torch.Tensor = type(None)  # won't match a list
        proj_output = MagicMock()
        proj_output.shape = (1, 16, 512)
        pipe.image_proj_model.return_value = proj_output
        pipe._encode_prompt_image_emb([1, 2, 3], 'cpu', 1, 'float32', False)


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------

class TestPipelineCall:
    def _setup_pipe(self):
        pipe = _make_pipe()

        pipe._execution_device = 'cpu'
        pipe.do_classifier_free_guidance = True
        pipe.cross_attention_kwargs = None
        pipe._guidance_scale = 7.5
        pipe._clip_skip = None
        pipe._cross_attention_kwargs = None
        pipe.clip_skip = None
        pipe.text_encoder_2 = MagicMock()
        pipe.text_encoder_2.config.projection_dim = 1280
        pipe._num_timesteps = 2
        pipe.watermark = None
        pipe.unet_name = 'unet'

        pipe.check_inputs = MagicMock()
        pipe.encode_prompt = MagicMock(return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        ))
        pipe._encode_prompt_image_emb = MagicMock(return_value=MagicMock())
        pipe.prepare_image = MagicMock(return_value=MagicMock(shape=[1, 3, 64, 64]))
        pipe.scheduler.set_timesteps = MagicMock()
        pipe.scheduler.timesteps = [1]
        pipe.scheduler.order = 1
        pipe.scheduler.scale_model_input = MagicMock(return_value=MagicMock())
        pipe.scheduler.step = MagicMock(return_value=[MagicMock()])
        # unet(...)[0] must support .chunk(2) → (uncond, cond)
        noise_pred_mock = MagicMock()
        noise_pred_mock.chunk.return_value = (MagicMock(), MagicMock())
        pipe.unet.return_value = [noise_pred_mock]
        pipe.unet.config.in_channels = 4
        pipe.unet.config.time_cond_proj_dim = None
        pipe.prepare_latents = MagicMock(return_value=MagicMock())
        pipe.prepare_extra_step_kwargs = MagicMock(return_value={})
        pipe.progress_bar = MagicMock()
        pipe._get_add_time_ids = MagicMock(return_value=MagicMock())
        pipe.controlnet = pipeline_mod.ControlNetModel()
        pipe.controlnet.config = MagicMock()
        pipe.controlnet.config.global_pool_conditions = False
        pipe.controlnet._orig_mod = pipe.controlnet
        pipe.controlnet.dtype = 'float32'
        pipe.maybe_free_model_hooks = MagicMock()
        pipe.image_processor = MagicMock()
        pipe.vae.dtype = 'float32'
        pipe.vae.config.force_upcast = False
        pipe.vae.config.scaling_factor = 0.18215
        pipe.vae.decode = MagicMock(return_value=[MagicMock()])

        pipeline_mod.is_compiled_module.return_value = False
        pipeline_mod.is_torch_version.return_value = False

        return pipe

    def test_basic_call(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
        )
        assert result is not None

    def test_call_with_return_dict(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=True,
        )
        assert result is not None

    def test_call_with_callback(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
            callback=MagicMock(),
            callback_steps=1,
        )
        assert result is not None

    def test_call_with_latent_output(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
            output_type='latent',
        )
        assert result is not None

    def test_call_with_prompt_list(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt=['a person', 'a dog'],
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
        )
        assert result is not None

    def test_call_with_prompt_embeds(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        prompt_embeds = MagicMock()
        prompt_embeds.shape = [1, 77, 768]

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt=None,
            prompt_embeds=prompt_embeds,
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
        )
        assert result is not None

    def test_call_with_ip_adapter_scale(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
            ip_adapter_scale=0.5,
        )
        assert result is not None

    def test_call_with_negative_size_overrides(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
            negative_original_size=(512, 512),
            negative_target_size=(1024, 1024),
        )
        assert result is not None

    def test_call_with_vae_upcasting(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)
        pipe.vae.dtype = 'float16'
        pipe.vae.config.force_upcast = True
        pipe.upcast_vae = MagicMock()
        pipe.vae.post_quant_conv = MagicMock()
        pipe.vae.post_quant_conv.parameters.return_value = iter([MagicMock(dtype='float32')])

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
        )
        assert result is not None

    def test_call_guidance_start_end_list_mismatch(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        # control_guidance_start is not list, control_guidance_end is list
        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
            control_guidance_start=0.0,
            control_guidance_end=[1.0],
        )
        assert result is not None

    def test_call_guidance_start_list_end_scalar(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
            control_guidance_start=[0.0],
            control_guidance_end=1.0,
        )
        assert result is not None

    def test_call_with_watermark(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)
        pipe.watermark = MagicMock()

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
        )
        assert result is not None

    def test_call_without_text_encoder_2(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)
        pipe.text_encoder_2 = None
        # pooled_prompt_embeds shape needs to work
        mock_ppe = MagicMock()
        mock_ppe.shape = [1, 1280]
        pipe.encode_prompt.return_value = (MagicMock(), MagicMock(), mock_ppe, MagicMock())

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
        )
        assert result is not None

    def test_call_with_callback_on_step_end(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        cb = MagicMock(return_value={})
        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=[],
        )
        assert result is not None

    def test_call_with_guess_mode(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
            guess_mode=True,
        )
        assert result is not None

    def test_call_with_timestep_cond(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)
        pipe.unet.config.time_cond_proj_dim = 256
        pipe.get_guidance_scale_embedding = MagicMock(return_value=MagicMock())
        pipe.guidance_scale = 7.5

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
        )
        assert result is not None

    def test_call_with_compiled_modules(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)
        pipeline_mod.is_compiled_module.return_value = True
        pipeline_mod.is_torch_version.return_value = True

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
        )
        assert result is not None
        # Reset for other tests
        pipeline_mod.is_compiled_module.return_value = False
        pipeline_mod.is_torch_version.return_value = False

    def test_call_with_image_list_original_size(self):
        """Cover the isinstance(image, list) branch for original_size."""
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)
        pipe.prepare_image = MagicMock(return_value=MagicMock(shape=[1, 3, 64, 64]))

        # Use MultiControlNetModel for the image list branch
        multi_cn = pipeline_mod.MultiControlNetModel()
        multi_cn.nets = [pipeline_mod.ControlNetModel()]
        multi_cn.nets[0].config = MagicMock()
        multi_cn.nets[0].config.global_pool_conditions = False
        multi_cn.dtype = 'float32'
        multi_cn._orig_mod = multi_cn
        pipe.controlnet = multi_cn

        mock_img = MagicMock()
        mock_img.shape = [1, 3, 64, 64]
        pipe.prepare_image.return_value = mock_img

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=[MagicMock()],
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
            controlnet_conditioning_scale=1.0,
        )
        assert result is not None

    def test_call_without_classifier_free_guidance(self):
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)
        pipe.do_classifier_free_guidance = False

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
        )
        assert result is not None

    def test_call_with_list_conditioning_scale(self):
        """Cover controlnet_conditioning_scale as list with single ControlNetModel."""
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
            controlnet_conditioning_scale=[1.0],
        )
        assert result is not None

    def test_call_with_callback_on_step_end_tensor_inputs(self):
        """Cover callback_on_step_end with non-empty tensor_inputs."""
        pipe = self._setup_pipe()
        pipe.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        pipe.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

        cb = MagicMock(return_value={})
        result = pipeline_mod.StableDiffusionXLInstantIDPipeline.__call__(
            pipe,
            prompt='a person',
            image=MagicMock(),
            image_embeds=MagicMock(),
            num_inference_steps=1,
            return_dict=False,
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=['latents'],
        )
        assert result is not None
