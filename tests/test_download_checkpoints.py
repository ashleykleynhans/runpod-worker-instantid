from unittest.mock import MagicMock, patch

import download_checkpoints
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline


# ---------------------------------------------------------------------------
# fetch_instantid_checkpoints
# ---------------------------------------------------------------------------

class TestFetchInstantidCheckpoints:
    def test_downloads_three_files(self):
        download_checkpoints.hf_hub_download.reset_mock()
        download_checkpoints.fetch_instantid_checkpoints()
        assert download_checkpoints.hf_hub_download.call_count == 3

        calls = download_checkpoints.hf_hub_download.call_args_list
        filenames = [c.kwargs.get('filename') or c[1].get('filename') for c in calls]
        assert 'ControlNetModel/config.json' in filenames
        assert 'ControlNetModel/diffusion_pytorch_model.safetensors' in filenames
        assert 'ip-adapter.bin' in filenames


# ---------------------------------------------------------------------------
# fetch_pretrained_model
# ---------------------------------------------------------------------------

class TestFetchPretrainedModel:
    def test_success_first_try(self):
        mock_pipe = MagicMock()
        with patch.object(StableDiffusionXLInstantIDPipeline,
                          'from_pretrained', return_value=mock_pipe):
            result = download_checkpoints.fetch_pretrained_model('model_name', torch_dtype='f16')
        assert result is mock_pipe

    def test_retries_on_os_error(self):
        mock_pipe = MagicMock()
        with patch.object(StableDiffusionXLInstantIDPipeline,
                          'from_pretrained',
                          side_effect=[OSError('err'), OSError('err'), mock_pipe]):
            result = download_checkpoints.fetch_pretrained_model('model_name')
        assert result is mock_pipe

    def test_raises_after_max_retries(self):
        with patch.object(StableDiffusionXLInstantIDPipeline,
                          'from_pretrained', side_effect=OSError('fail')):
            try:
                download_checkpoints.fetch_pretrained_model('model_name')
                assert False, 'Expected OSError'
            except OSError:
                pass


# ---------------------------------------------------------------------------
# get_instantid_pipeline
# ---------------------------------------------------------------------------

class TestGetInstantidPipeline:
    def test_calls_fetch(self):
        mock_pipe = MagicMock()
        with patch.object(download_checkpoints, 'fetch_pretrained_model',
                          return_value=mock_pipe) as mock_fetch:
            result = download_checkpoints.get_instantid_pipeline()
            mock_fetch.assert_called_once()
        assert result is mock_pipe
