import base64
import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

import handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(w=64, h=64):
    """Return a small RGB PIL Image."""
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def _image_to_base64(img):
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ---------------------------------------------------------------------------
# load_image_from_base64
# ---------------------------------------------------------------------------

class TestLoadImageFromBase64:
    def test_returns_pil_image(self):
        img = _make_rgb_image()
        b64 = _image_to_base64(img)
        result = handler.load_image_from_base64(b64)
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_http_url(self):
        img = _make_rgb_image()
        buf = BytesIO()
        img.save(buf, format='PNG')
        mock_resp = MagicMock()
        mock_resp.content = buf.getvalue()

        with patch.object(handler.requests, 'get', return_value=mock_resp) as mock_get:
            result = handler.load_image('http://example.com/face.png')
            mock_get.assert_called_once_with('http://example.com/face.png')
        assert result.mode == 'RGB'

    def test_https_url(self):
        img = _make_rgb_image()
        buf = BytesIO()
        img.save(buf, format='PNG')
        mock_resp = MagicMock()
        mock_resp.content = buf.getvalue()

        with patch.object(handler.requests, 'get', return_value=mock_resp):
            result = handler.load_image('https://example.com/face.png')
        assert result.mode == 'RGB'

    def test_base64_string(self):
        img = _make_rgb_image()
        b64 = _image_to_base64(img)
        result = handler.load_image(b64)
        assert result.mode == 'RGB'


# ---------------------------------------------------------------------------
# determine_file_extension
# ---------------------------------------------------------------------------

class TestDetermineFileExtension:
    def test_jpg_prefix(self):
        assert handler.determine_file_extension('/9j/abc') == '.jpg'

    def test_png_prefix(self):
        assert handler.determine_file_extension('iVBORw0Kgxyz') == '.png'

    def test_unknown_defaults_to_png(self):
        assert handler.determine_file_extension('AAAA') == '.png'

    def test_exception_defaults_to_png(self):
        bad = MagicMock()
        bad.startswith = MagicMock(side_effect=Exception('boom'))
        assert handler.determine_file_extension(bad) == '.png'


# ---------------------------------------------------------------------------
# randomize_seed_fn
# ---------------------------------------------------------------------------

class TestRandomizeSeedFn:
    def test_no_randomize(self):
        assert handler.randomize_seed_fn(42, False) == 42

    def test_randomize(self):
        seed = handler.randomize_seed_fn(42, True)
        assert isinstance(seed, int)
        assert 0 <= seed <= handler.MAX_SEED


# ---------------------------------------------------------------------------
# convert_from_cv2_to_image / convert_from_image_to_cv2
# ---------------------------------------------------------------------------

class TestConvertFromCv2ToImage:
    def test_calls_cvtcolor(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        handler.cv2.cvtColor.return_value = arr
        result = handler.convert_from_cv2_to_image(arr)
        handler.cv2.cvtColor.assert_called_once_with(arr, handler.cv2.COLOR_BGR2RGB)
        assert isinstance(result, Image.Image)


class TestConvertFromImageToCv2:
    def test_calls_cvtcolor(self):
        img = _make_rgb_image(10, 10)
        expected = np.zeros((10, 10, 3), dtype=np.uint8)
        handler.cv2.cvtColor.return_value = expected
        result = handler.convert_from_image_to_cv2(img)
        assert handler.cv2.cvtColor.called
        assert np.array_equal(result, expected)


# ---------------------------------------------------------------------------
# draw_kps
# ---------------------------------------------------------------------------

class TestDrawKps:
    def test_returns_pil_image(self):
        image_pil = _make_rgb_image(100, 100)
        kps = np.array([
            [20, 20], [40, 20], [30, 40], [20, 60], [40, 60],
        ], dtype=np.float32)

        handler.cv2.ellipse2Poly.return_value = np.array([[25, 30], [30, 35], [35, 30]])
        handler.cv2.fillConvexPoly.return_value = np.zeros((100, 100, 3), dtype=np.float64)
        handler.cv2.circle.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        result = handler.draw_kps(image_pil, kps)
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# resize_img
# ---------------------------------------------------------------------------

class TestResizeImg:
    def test_with_explicit_size(self):
        img = _make_rgb_image(200, 100)
        result = handler.resize_img(img, size=(64, 64))
        assert result.size == (64, 64)

    def test_without_size_uses_defaults(self):
        img = _make_rgb_image(200, 100)
        result = handler.resize_img(img)
        w, h = result.size
        assert w % 64 == 0
        assert h % 64 == 0

    def test_pad_to_max_side(self):
        img = _make_rgb_image(200, 100)
        result = handler.resize_img(img, max_side=256, min_side=128,
                                    size=(100, 80), pad_to_max_side=True)
        assert result.size == (256, 256)


# ---------------------------------------------------------------------------
# apply_style
# ---------------------------------------------------------------------------

class TestApplyStyle:
    def test_known_style(self):
        p, n = handler.apply_style('Watercolor', 'a cat')
        assert 'a cat' in p
        assert 'watercolor' in p.lower()

    def test_unknown_style_falls_back_to_default(self):
        p, n = handler.apply_style('NonExistentStyle', 'a dog')
        assert 'a dog' in p

    def test_negative_prompt_appended(self):
        _, n = handler.apply_style('Watercolor', 'a cat', 'ugly')
        assert 'ugly' in n


# ---------------------------------------------------------------------------
# get_instantid_pipeline
# ---------------------------------------------------------------------------

class TestGetInstantidPipeline:
    def test_pretrained_path(self):
        mock_pipe = MagicMock()
        with patch.object(handler.StableDiffusionXLInstantIDPipeline,
                          'from_pretrained', return_value=mock_pipe) as mock_fp:
            mock_pipe.to.return_value = mock_pipe
            result = handler.get_instantid_pipeline('some/model')
            mock_fp.assert_called_once()
        mock_pipe.load_ip_adapter_instantid.assert_called_once_with(handler.face_adapter)
        assert result is mock_pipe

    def test_checkpoint_path(self):
        mock_pipe = MagicMock()
        handler.load_models_xl.return_value = (
            ['tok1', 'tok2'], ['enc1', 'enc2'], 'unet', None, 'vae'
        )
        with patch.object(handler.StableDiffusionXLInstantIDPipeline,
                          '__init__', return_value=None):
            with patch.object(handler.StableDiffusionXLInstantIDPipeline,
                              'to', return_value=mock_pipe):
                result = handler.get_instantid_pipeline('model.safetensors')
        mock_pipe.load_ip_adapter_instantid.assert_called_once_with(handler.face_adapter)
        assert result is mock_pipe

    def test_ckpt_path(self):
        mock_pipe = MagicMock()
        handler.load_models_xl.return_value = (
            ['tok1', 'tok2'], ['enc1', 'enc2'], 'unet', None, 'vae'
        )
        with patch.object(handler.StableDiffusionXLInstantIDPipeline,
                          '__init__', return_value=None):
            with patch.object(handler.StableDiffusionXLInstantIDPipeline,
                              'to', return_value=mock_pipe):
                result = handler.get_instantid_pipeline('model.ckpt')
        assert result is mock_pipe


# ---------------------------------------------------------------------------
# generate_image
# ---------------------------------------------------------------------------

class TestGenerateImage:
    def _setup_mocks(self):
        """Set up common mocks for generate_image tests."""
        img = _make_rgb_image(64, 64)
        b64 = _image_to_base64(img)

        face_info_entry = {
            'bbox': [0, 0, 64, 64],
            'embedding': np.zeros(512),
            'kps': np.array([[20, 20], [40, 20], [30, 40], [20, 60], [40, 60]],
                            dtype=np.float32),
        }
        handler.app.get.return_value = [face_info_entry]

        handler.cv2.cvtColor.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        handler.cv2.ellipse2Poly.return_value = np.array([[25, 30], [30, 35], [35, 30]])
        handler.cv2.fillConvexPoly.return_value = np.zeros((64, 64, 3), dtype=np.float64)
        handler.cv2.circle.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

        result_img = _make_rgb_image(64, 64)
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.images = [result_img]
        handler.PIPELINE = mock_pipeline

        return b64

    def test_basic_generation(self):
        b64 = self._setup_mocks()
        handler.CURRENT_MODEL = handler.DEFAULT_MODEL

        images = handler.generate_image(
            'job-1', handler.DEFAULT_MODEL, b64, None,
            'a person', '', 'Watercolor', 30, 0.8, 0.8, 5.0, 42, 0, 0
        )
        assert len(images) == 1

    def test_with_explicit_dimensions(self):
        b64 = self._setup_mocks()
        handler.CURRENT_MODEL = handler.DEFAULT_MODEL

        images = handler.generate_image(
            'job-2', handler.DEFAULT_MODEL, b64, None,
            'a person', '', 'Watercolor', 30, 0.8, 0.8, 5.0, 42, 512, 512
        )
        assert len(images) == 1

    def test_with_pose_image(self):
        b64 = self._setup_mocks()
        handler.CURRENT_MODEL = handler.DEFAULT_MODEL

        images = handler.generate_image(
            'job-3', handler.DEFAULT_MODEL, b64, b64,
            'a person', '', 'Watercolor', 30, 0.8, 0.8, 5.0, 42, 0, 0
        )
        assert len(images) == 1

    def test_none_face_image_raises(self):
        try:
            handler.generate_image(
                'job-4', handler.DEFAULT_MODEL, None, None,
                'a person', '', 'Watercolor', 30, 0.8, 0.8, 5.0, 42, 0, 0
            )
            assert False, 'Expected exception'
        except Exception as e:
            assert 'face image' in str(e)

    def test_none_prompt_defaults(self):
        b64 = self._setup_mocks()
        handler.CURRENT_MODEL = handler.DEFAULT_MODEL

        images = handler.generate_image(
            'job-5', handler.DEFAULT_MODEL, b64, None,
            None, '', 'Watercolor', 30, 0.8, 0.8, 5.0, 42, 0, 0
        )
        assert len(images) == 1

    def test_model_change_reloads_pipeline(self):
        b64 = self._setup_mocks()
        handler.CURRENT_MODEL = 'old_model'

        mock_pipe = MagicMock()
        mock_pipe.return_value.images = [_make_rgb_image()]
        with patch.object(handler, 'get_instantid_pipeline', return_value=mock_pipe):
            handler.generate_image(
                'job-6', 'new_model', b64, None,
                'a person', '', 'Watercolor', 30, 0.8, 0.8, 5.0, 42, 0, 0
            )
        assert handler.CURRENT_MODEL == 'new_model'

    def test_no_face_in_image_raises(self):
        b64 = self._setup_mocks()
        handler.app.get.return_value = []
        handler.CURRENT_MODEL = handler.DEFAULT_MODEL

        try:
            handler.generate_image(
                'job-7', handler.DEFAULT_MODEL, b64, None,
                'a person', '', 'Watercolor', 30, 0.8, 0.8, 5.0, 42, 0, 0
            )
            assert False, 'Expected exception'
        except Exception as e:
            assert 'Cannot find any face' in str(e)

    def test_no_face_in_pose_image_raises(self):
        b64 = self._setup_mocks()
        handler.CURRENT_MODEL = handler.DEFAULT_MODEL

        face_info_entry = {
            'bbox': [0, 0, 64, 64],
            'embedding': np.zeros(512),
            'kps': np.array([[20, 20], [40, 20], [30, 40], [20, 60], [40, 60]],
                            dtype=np.float32),
        }
        handler.app.get.side_effect = [[face_info_entry], []]

        try:
            handler.generate_image(
                'job-8', handler.DEFAULT_MODEL, b64, b64,
                'a person', '', 'Watercolor', 30, 0.8, 0.8, 5.0, 42, 0, 0
            )
            assert False, 'Expected exception'
        except Exception as e:
            assert 'reference image' in str(e)
        finally:
            handler.app.get.side_effect = None


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------

class TestHandler:
    def test_validation_errors(self):
        handler.validate.return_value = {'errors': ['face_image is required']}
        result = handler.handler({'id': 'j1', 'input': {}})
        assert 'error' in result
        assert result['error'] == ['face_image is required']

    def test_successful_run(self):
        result_img = _make_rgb_image(64, 64)
        handler.validate.return_value = {
            'validated_input': {
                'model': handler.DEFAULT_MODEL,
                'face_image': _image_to_base64(_make_rgb_image()),
                'pose_image': None,
                'prompt': 'a person',
                'negative_prompt': '',
                'style_name': 'Watercolor',
                'num_steps': 30,
                'identitynet_strength_ratio': 0.8,
                'adapter_strength_ratio': 0.8,
                'guidance_scale': 5.0,
                'seed': 42,
                'width': 0,
                'height': 0,
            }
        }

        with patch.object(handler, 'generate_image', return_value=[result_img]):
            result = handler.handler({'id': 'j2', 'input': {}})
        assert 'image' in result
        decoded = base64.b64decode(result['image'])
        assert len(decoded) > 0

    def test_exception_returns_error(self):
        handler.validate.return_value = {
            'validated_input': {
                'model': handler.DEFAULT_MODEL,
                'face_image': None,
                'pose_image': None,
                'prompt': 'a person',
                'negative_prompt': '',
                'style_name': 'Watercolor',
                'num_steps': 30,
                'identitynet_strength_ratio': 0.8,
                'adapter_strength_ratio': 0.8,
                'guidance_scale': 5.0,
                'seed': 42,
                'width': 0,
                'height': 0,
            }
        }

        with patch.object(handler, 'generate_image', side_effect=Exception('test error')):
            result = handler.handler({'id': 'j3', 'input': {}})
        assert 'error' in result
        assert result['error'] == 'test error'
        assert result['refresh_worker'] is True
        assert 'output' in result


# ---------------------------------------------------------------------------
# Module-level __main__ block
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_block(self):
        with patch.object(handler.runpod.serverless, 'start') as mock_start:
            exec(
                compile(
                    "if __name__ == '__main__':\n"
                    "    logger.info('Starting RunPod Serverless...')\n"
                    "    runpod.serverless.start({'handler': handler})\n",
                    '<test>',
                    'exec',
                ),
                {
                    '__name__': '__main__',
                    'logger': handler.logger,
                    'runpod': handler.runpod,
                    'handler': handler.handler,
                },
            )
            mock_start.assert_called_once()
