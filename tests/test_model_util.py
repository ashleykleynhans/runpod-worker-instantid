from unittest.mock import MagicMock, patch, call
import model_util


# ---------------------------------------------------------------------------
# load_checkpoint_with_text_encoder_conversion
# ---------------------------------------------------------------------------

class TestLoadCheckpointWithTextEncoderConversion:
    def test_safetensors_path(self):
        model_util.load_file.return_value = {
            'cond_stage_model.transformer.embeddings.weight': 'v1',
            'some_other_key': 'v2',
        }
        checkpoint, state_dict = model_util.load_checkpoint_with_text_encoder_conversion(
            'model.safetensors'
        )
        assert checkpoint is None
        assert 'cond_stage_model.transformer.text_model.embeddings.weight' in state_dict
        assert 'some_other_key' in state_dict

    def test_ckpt_path_with_state_dict_key(self):
        inner = {'cond_stage_model.transformer.encoder.layer': 'v1'}
        model_util.torch.load.return_value = {'state_dict': inner}
        checkpoint, state_dict = model_util.load_checkpoint_with_text_encoder_conversion(
            'model.ckpt'
        )
        assert checkpoint == {'state_dict': inner}
        assert 'cond_stage_model.transformer.text_model.encoder.layer' in state_dict

    def test_ckpt_path_without_state_dict_key(self):
        raw = {'cond_stage_model.transformer.final_layer_norm.bias': 'v1'}
        model_util.torch.load.return_value = raw.copy()
        checkpoint, state_dict = model_util.load_checkpoint_with_text_encoder_conversion(
            'model.ckpt'
        )
        assert checkpoint is None
        assert 'cond_stage_model.transformer.text_model.final_layer_norm.bias' in state_dict

    def test_no_matching_keys(self):
        model_util.torch.load.return_value = {'unrelated_key': 'v'}
        checkpoint, state_dict = model_util.load_checkpoint_with_text_encoder_conversion(
            'model.ckpt'
        )
        assert 'unrelated_key' in state_dict


# ---------------------------------------------------------------------------
# create_unet_diffusers_config
# ---------------------------------------------------------------------------

class TestCreateUnetDiffusersConfig:
    def test_v1_config(self):
        config = model_util.create_unet_diffusers_config(v2=False)
        assert config['sample_size'] == model_util.UNET_PARAMS_IMAGE_SIZE
        assert config['cross_attention_dim'] == model_util.UNET_PARAMS_CONTEXT_DIM
        assert config['attention_head_dim'] == model_util.UNET_PARAMS_NUM_HEADS
        assert 'use_linear_projection' not in config

    def test_v2_config(self):
        config = model_util.create_unet_diffusers_config(v2=True)
        assert config['cross_attention_dim'] == model_util.V2_UNET_PARAMS_CONTEXT_DIM
        assert config['attention_head_dim'] == model_util.V2_UNET_PARAMS_ATTENTION_HEAD_DIM
        assert 'use_linear_projection' not in config

    def test_v2_with_linear_projection(self):
        config = model_util.create_unet_diffusers_config(v2=True, use_linear_projection_in_v2=True)
        assert config['use_linear_projection'] is True


# ---------------------------------------------------------------------------
# load_diffusers_model
# ---------------------------------------------------------------------------

class TestLoadDiffusersModel:
    def test_v1(self):
        result = model_util.load_diffusers_model('model_path', v2=False)
        assert len(result) == 4  # tokenizer, text_encoder, unet, vae

    def test_v2(self):
        result = model_util.load_diffusers_model('model_path', v2=True)
        assert len(result) == 4

    def test_v1_with_clip_skip(self):
        result = model_util.load_diffusers_model('model_path', v2=False, clip_skip=2)
        assert len(result) == 4

    def test_v2_with_clip_skip(self):
        result = model_util.load_diffusers_model('model_path', v2=True, clip_skip=2)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# load_checkpoint_model
# ---------------------------------------------------------------------------

class TestLoadCheckpointModel:
    def test_basic(self):
        model_util.load_file.return_value = {}
        model_util.torch.load.return_value = {}
        model_util.convert_ldm_unet_checkpoint.return_value = {}
        model_util.UNet2DConditionModel.return_value = MagicMock()

        result = model_util.load_checkpoint_model('model.ckpt')
        assert len(result) == 4

    def test_with_clip_skip_v1(self):
        model_util.load_file.return_value = {}
        model_util.torch.load.return_value = {}
        model_util.convert_ldm_unet_checkpoint.return_value = {}
        model_util.UNet2DConditionModel.return_value = MagicMock()

        result = model_util.load_checkpoint_model('model.ckpt', clip_skip=2, v2=False)
        assert len(result) == 4

    def test_with_clip_skip_v2(self):
        model_util.load_file.return_value = {}
        model_util.torch.load.return_value = {}
        model_util.convert_ldm_unet_checkpoint.return_value = {}
        model_util.UNet2DConditionModel.return_value = MagicMock()

        result = model_util.load_checkpoint_model('model.ckpt', clip_skip=2, v2=True)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# load_models
# ---------------------------------------------------------------------------

class TestLoadModels:
    def test_diffusers_with_scheduler(self):
        model_util.OmegaConf.to_container.return_value = {}
        result = model_util.load_models('model_path', scheduler_name='ddim')
        assert len(result) == 5

    def test_diffusers_without_scheduler(self):
        result = model_util.load_models('model_path', scheduler_name=None)
        assert result[3] is None

    def test_checkpoint_path(self):
        model_util.load_file.return_value = {}
        model_util.torch.load.return_value = {}
        model_util.convert_ldm_unet_checkpoint.return_value = {}
        model_util.UNet2DConditionModel.return_value = MagicMock()

        result = model_util.load_models('model.safetensors', scheduler_name=None)
        assert len(result) == 5

    def test_v_pred(self):
        model_util.OmegaConf.to_container.return_value = {}
        result = model_util.load_models('model_path', scheduler_name='euler', v_pred=True)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# load_diffusers_model_xl
# ---------------------------------------------------------------------------

class TestLoadDiffusersModelXl:
    def test_basic(self):
        result = model_util.load_diffusers_model_xl('model_path')
        tokenizers, text_encoders, unet, vae = result
        assert len(tokenizers) == 2
        assert len(text_encoders) == 2


# ---------------------------------------------------------------------------
# load_checkpoint_model_xl
# ---------------------------------------------------------------------------

class TestLoadCheckpointModelXl:
    def test_basic(self):
        mock_pipe = MagicMock()
        mock_pipe.tokenizer = 'tok1'
        mock_pipe.tokenizer_2 = 'tok2'
        mock_pipe.text_encoder = MagicMock()
        mock_pipe.text_encoder_2 = MagicMock()
        mock_pipe.unet = 'unet'
        mock_pipe.vae = 'vae'
        with patch.object(model_util.StableDiffusionXLPipeline,
                          'from_single_file', return_value=mock_pipe):
            result = model_util.load_checkpoint_model_xl('model.ckpt')
        tokenizers, text_encoders, unet, vae = result
        assert len(tokenizers) == 2
        assert len(text_encoders) == 2
        assert text_encoders[1].pad_token_id == 0


# ---------------------------------------------------------------------------
# load_models_xl
# ---------------------------------------------------------------------------

class TestLoadModelsXl:
    def test_diffusers_with_scheduler(self):
        model_util.OmegaConf.to_container.return_value = {}
        result = model_util.load_models_xl('model_path', scheduler_name='ddpm')
        assert len(result) == 5

    def test_diffusers_without_scheduler(self):
        result = model_util.load_models_xl('model_path', scheduler_name=None)
        assert result[3] is None

    def test_checkpoint_path(self):
        mock_pipe = MagicMock()
        mock_pipe.tokenizer = 'tok1'
        mock_pipe.tokenizer_2 = 'tok2'
        mock_pipe.text_encoder = MagicMock()
        mock_pipe.text_encoder_2 = MagicMock()
        mock_pipe.unet = 'unet'
        mock_pipe.vae = 'vae'
        with patch.object(model_util.StableDiffusionXLPipeline,
                          'from_single_file', return_value=mock_pipe):
            result = model_util.load_models_xl('model.safetensors', scheduler_name=None)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# create_noise_scheduler
# ---------------------------------------------------------------------------

class TestCreateNoiseScheduler:
    def test_ddim(self):
        model_util.OmegaConf.to_container.return_value = {}
        model_util.create_noise_scheduler('ddim', noise_scheduler_kwargs={})

    def test_ddpm(self):
        model_util.OmegaConf.to_container.return_value = {}
        model_util.create_noise_scheduler('ddpm', noise_scheduler_kwargs={})

    def test_lms(self):
        model_util.OmegaConf.to_container.return_value = {}
        model_util.create_noise_scheduler('lms', noise_scheduler_kwargs={})

    def test_euler_a(self):
        model_util.OmegaConf.to_container.return_value = {}
        model_util.create_noise_scheduler('euler_a', noise_scheduler_kwargs={})

    def test_euler(self):
        model_util.OmegaConf.to_container.return_value = {}
        model_util.create_noise_scheduler('euler', noise_scheduler_kwargs={})

    def test_unipc(self):
        model_util.OmegaConf.to_container.return_value = {}
        model_util.create_noise_scheduler('unipc', noise_scheduler_kwargs={})

    def test_unknown_raises(self):
        try:
            model_util.create_noise_scheduler('unknown_scheduler', noise_scheduler_kwargs={})
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'Unknown scheduler' in str(e)


# ---------------------------------------------------------------------------
# torch_gc
# ---------------------------------------------------------------------------

class TestTorchGc:
    def test_with_cuda_available(self):
        model_util.torch.cuda.is_available.return_value = True
        model_util.torch_gc()
        model_util.torch.cuda.empty_cache.assert_called()

    def test_without_cuda(self):
        model_util.torch.cuda.is_available.return_value = False
        model_util.torch_gc()


# ---------------------------------------------------------------------------
# is_intel_xpu
# ---------------------------------------------------------------------------

class TestIsIntelXpu:
    def test_gpu_with_xpu(self):
        model_util.cpu_state = model_util.CPUState.GPU
        model_util.xpu_available = True
        assert model_util.is_intel_xpu() is True

    def test_gpu_without_xpu(self):
        model_util.cpu_state = model_util.CPUState.GPU
        model_util.xpu_available = False
        assert model_util.is_intel_xpu() is False

    def test_cpu_state(self):
        model_util.cpu_state = model_util.CPUState.CPU
        assert model_util.is_intel_xpu() is False


# ---------------------------------------------------------------------------
# get_torch_device
# ---------------------------------------------------------------------------

class TestGetTorchDevice:
    def test_mps(self):
        model_util.directml_enabled = False
        model_util.cpu_state = model_util.CPUState.MPS
        result = model_util.get_torch_device()
        model_util.torch.device.assert_called_with("mps")

    def test_cpu(self):
        model_util.directml_enabled = False
        model_util.cpu_state = model_util.CPUState.CPU
        result = model_util.get_torch_device()
        model_util.torch.device.assert_called_with("cpu")

    def test_gpu_cuda(self):
        model_util.directml_enabled = False
        model_util.cpu_state = model_util.CPUState.GPU
        model_util.xpu_available = False
        result = model_util.get_torch_device()
        model_util.torch.device.assert_called()

    def test_gpu_xpu(self):
        model_util.directml_enabled = False
        model_util.cpu_state = model_util.CPUState.GPU
        model_util.xpu_available = True
        result = model_util.get_torch_device()
        model_util.torch.device.assert_called_with("xpu")

    def test_directml(self):
        model_util.directml_enabled = True
        model_util.directml_device = 'directml_dev'
        result = model_util.get_torch_device()
        assert result == 'directml_dev'
        model_util.directml_enabled = False
