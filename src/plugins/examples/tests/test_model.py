"""Tests for models.py module."""

import numpy as np
from plugins.examples.model import VelocityTuningCurvesModel
import pytest

from nds.samples import Samples


def _create_model_weights_file(tmp_path, b0, m, pd, bs):
    filepath = tmp_path / "weights"
    np.savez(str(filepath), b0=b0, m=m, pd=pd, bs=bs)
    return str(filepath) + ".npz"


class TestVelocityTuningCurvesModel:
    """Tests for VelocityTuningCurvesModel class."""

    def test_build_model_from_array_params(self, tmp_path):
        """Test that we can instantiate a model with model weights."""
        model_weights = _create_model_weights_file(
            tmp_path,
            b0=np.random.random(3),
            m=np.random.random(3),
            pd=np.random.random(3),
            bs=np.random.random(3),
        )
        VelocityTuningCurvesModel(model_weights)

    def test_build_model_from_list_params_equals_array_params(self, tmp_path):
        """Test that we can build a model from lists passed as parameters."""
        model_weights = _create_model_weights_file(
            tmp_path,
            b0=list(range(3)),
            m=list(range(3)),
            pd=list(range(3)),
            bs=list(range(3)),
        )
        model_list = VelocityTuningCurvesModel(model_weights)

        model_weights = _create_model_weights_file(
            tmp_path, b0=np.arange(3), m=np.arange(3), pd=np.arange(3), bs=np.arange(3)
        )
        model_array = VelocityTuningCurvesModel(model_weights)
        np.testing.assert_array_equal(model_list.b0, model_array.b0)
        np.testing.assert_array_equal(model_list.m, model_array.m)
        np.testing.assert_array_equal(model_list.pd, model_array.pd)
        np.testing.assert_array_equal(model_list.bs, model_array.bs)

    def test_encode(self, tmp_path):
        """Test if encoding is outputting expected values."""
        model_weights = _create_model_weights_file(
            tmp_path,
            b0=np.array([-10, 0, 10, 10]),
            m=np.array([-10, 0, 10, 10]),
            pd=np.array([0, np.pi / 4, np.pi, 3 * np.pi]),
            bs=np.array([-10, 0, 10, 10]),
        )
        model = VelocityTuningCurvesModel(model_weights)
        input_samples = Samples(
            timestamps=np.arange(3),
            data=np.vstack((np.array([0, 1, 1]), np.array([0, 0, 1]))).T,
        )
        encoded_data = model.encode(input_samples).data
        expected_data = np.array(
            [[-10, 0, 10, 10], [-30, 0, 10, 10], [-34.14, 0.0, 14.14, 14.14]]
        )

        np.testing.assert_allclose(encoded_data, expected_data, atol=1e-2)

    def test_build_model_from_params_wrong_shape(self, tmp_path):
        """Test if exception is raised if parameters of different shapes are passed."""
        model_weights = _create_model_weights_file(
            tmp_path,
            b0=np.random.random(4),
            m=np.random.random(3),
            pd=np.random.random(3),
            bs=np.random.random(3),
        )
        with pytest.raises(ValueError):
            VelocityTuningCurvesModel(model_weights)

    def test_build_model_from_2d_params(self, tmp_path):
        """Test if exception is raised if 2d parameter is passed."""
        model_weights = _create_model_weights_file(
            tmp_path,
            b0=np.random.random(6).reshape(3, 2),
            m=np.random.random(3),
            pd=np.random.random(3),
            bs=np.random.random(3),
        )
        with pytest.raises(ValueError):
            VelocityTuningCurvesModel(model_weights)

    def test_build_model_from_empty_params(self, tmp_path):
        """Test if exception is raised if empty parameters are passed."""
        model_weights = _create_model_weights_file(
            tmp_path,
            b0=np.random.random(0),
            m=np.random.random(0),
            pd=np.random.random(0),
            bs=np.random.random(0),
        )
        with pytest.raises(ValueError):
            VelocityTuningCurvesModel(model_weights)
