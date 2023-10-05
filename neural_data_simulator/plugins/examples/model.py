"""Example of a custom EncoderModel."""
import logging
from typing import Union

from numpy import ndarray
import numpy as np

from neural_data_simulator.models import EncoderModel
from neural_data_simulator.samples import Samples
from neural_data_simulator.util.runtime import get_abs_path


class VelocityTuningCurvesModel(EncoderModel):
    r"""Two-dimensional tuning curve model.

    A two-dimensional tuning curve model that takes into account both preferred
    direction and magnitude of the velocity, following the equation from `decoding arm
    speed during reaching
    <https://ncbi.nlm.nih.gov/pmc/articles/PMC6286377/pdf/41467_2018_Article_7647.pdf>`_:

    :math:`spike rate = b_0 + m * |v| * cos(\theta - \theta_{pd}) + b_s*|v|`

    In the paper, this model is referenced as `offset model`.
    """

    def __init__(self, model_weights_file: str):
        """Initialize VelocityTurningCurvesModel class.

        Args:
            model_weights_file: Path to the model weights file.
        """
        if not model_weights_file:
            raise Exception(
                "Missing model_weights_file argument for VelocityTuningCurvesModel"
            )
        self.logger = logging.getLogger(__name__)

        try:
            filepath = get_abs_path(model_weights_file)
            params = np.load(filepath)
        except ValueError:
            print(f"Can not load model weights file {model_weights_file}")
            raise

        self.build_model_from_params(
            b0=params["b0"], m=params["m"], pd=params["pd"], bs=params["bs"]
        )

    def encode(self, X: Samples) -> Samples:
        r"""Encode 2D behavior data into spiking data.

        Calculate spiking rates from the equation:
        :math:`spike rate = b_0 + m * |v| * cos(\theta - \theta_{pd}) + b_s*|v|`
        with v = velocity, theta = velocity direction.

        Args:
            X: :class:`neural_data_simulator.samples.Samples` dataclass with timestamps
              and data points for velocity data (x and y direction, respectively) in
              the same unit as parameters were fit for. Each element of samples
              timestamps has 1 column where the acquisition time of the data point
              is stored. Each element of samples data has 2 columns, one for each
              velocity direction. Each row of samples corresponds to a data point
              acquisition.

        Returns:
            :class:`neural_data_simulator.samples.Samples` with the same timestamps as
            input, but data matching the size of the model parameters (which represent
            the units spikes are generated for).
        """
        velocities = X.data
        magnitudes = VelocityTuningCurvesModel._get_magnitude(velocities)
        angles = VelocityTuningCurvesModel._get_angle(velocities)

        spike_rates = self._encode(magnitudes, angles)
        return Samples(timestamps=X.timestamps, data=spike_rates)

    def build_model_from_params(
        self,
        b0: Union[list, ndarray],
        m: Union[list, ndarray],
        pd: Union[list, ndarray],
        bs: Union[list, ndarray],
    ):
        """Build a tuning curve model based on pre-calculated parameters.

        Inputs are the lists of parameters already fit for the data.
        """
        self.b0 = np.array(b0)
        self.m = np.array(m)
        self.pd = np.array(pd)
        self.bs = np.array(bs)
        self._validate_parameters_shape()

    @staticmethod
    def _get_magnitude(X: ndarray) -> ndarray:
        """Calculate L2 norm of each sample for an input of type n_samples by n_axes.

        Returns an array of shape n_samples by 1.
        """
        mag = np.linalg.norm(X, axis=1)
        mag = np.expand_dims(mag, axis=1)
        return mag

    @staticmethod
    def _get_angle(X: ndarray) -> ndarray:
        """Calculate the angle of an input.

        Input of type n_samples by 2 with columns [x, y] using arctan2.
        Returns an array of shape n_samples by 1 with angles in radians
        """
        angle = np.arctan2(X[:, 1], X[:, 0])
        angle = np.expand_dims(angle, axis=1)
        return angle

    def _encode(self, velocity_magnitude: ndarray, velocity_angle: ndarray) -> ndarray:
        """Calculate spiking rates using the tuning curve equation.

        Spiking rates are calculated for the input velocity magnitude and angle. Both
        inputs and output are in the shape of n_samples by 1.
        """
        spike_rates = (
            self.b0
            + self.m * velocity_magnitude * np.cos(velocity_angle - self.pd)
            + self.bs * velocity_magnitude
        )
        return spike_rates

    def _validate_parameters_shape(self):
        if not (self.b0.shape == self.m.shape == self.pd.shape == self.bs.shape):
            raise ValueError("All parameters must have the same length")

        if len(self.b0.shape) > 1:
            raise ValueError("Parameters must be a 1D array")

        if len(self.b0) < 1:
            raise ValueError("Parameters must have length >= 1")


def create_model() -> EncoderModel:
    """Instantiate the ExampleModel."""
    model_weights_file = "sample_data/session_4_tuning_curves_params.npz"
    return VelocityTuningCurvesModel(model_weights_file=model_weights_file)
