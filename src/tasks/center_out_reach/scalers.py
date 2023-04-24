"""Scalers and unit converters."""
from typing import Optional, Union

from numpy import ndarray
from sklearn.preprocessing import StandardScaler
from tasks.center_out_reach.screen_info import get_monitors
from tasks.center_out_reach.screen_info import get_ppmm


class PixelsToMetersConverter:
    """Unit conversion between pixels and meters."""

    def __init__(self, ppi: Optional[float]):
        """Initialize the PixelsToMetersConverter class.

        Args:
            ppi: The pixels per inch of the display or None. If ppi is None, the
              function will try to calculate it based on the default monitor.
        """
        if ppi is None:
            if monitors := get_monitors():
                first_monitor = monitors[0]
                if ppmm := get_ppmm(first_monitor):
                    ppi = ppmm[0] * 25.4
        if ppi is None:
            raise ValueError("Could not detect monitor ppi. Please configure it.")

        self.ppi = ppi
        self.pixels_per_meter = ppi / 0.0254
        self.pixels_per_millimeter = self.pixels_per_meter / 1000

    def pixels_to_millimeters(self, X: ndarray) -> ndarray:
        """Convert pixels to millimeters.

        Args:
            X: An array of pixel values.

        Returns:
            The array resulted from the conversion.
        """
        return X / self.pixels_per_millimeter

    def millimeters_to_pixels(self, X: ndarray) -> ndarray:
        """Convert millimeters to pixels.

        Args:
            X: An array of millimeter values.

        Returns:
            The array resulted from the conversion.
        """
        return X * self.pixels_per_millimeter

    def pixels_to_meters(self, X: ndarray) -> ndarray:
        """Convert pixels to meters.

        Args:
            X: An array of pixel values.

        Returns:
            The array resulted from the conversion.
        """
        return X / self.pixels_per_meter

    def meters_to_pixels(self, X: Union[float, ndarray]) -> Union[float, ndarray]:
        """Convert meters to pixels.

        Args:
            X: An array of meter values or a single float value.

        Returns:
            The array resulted from the conversion.
        """
        return X * self.pixels_per_meter


class StandardVelocityScaler:
    """A simple standard scaler for velocities."""

    def __init__(
        self, scale: ndarray, mean: ndarray, unit_converter: PixelsToMetersConverter
    ):
        """Instantiate a new scaler for velocities.

        The purpose of this scaler is to scale the velocity to a standard
        deviation of 1 and a mean of 0 when calling the transform function.

        Args:
            scale: The scale to apply to the velocities.
            mean: The mean to offset the velocities.
            unit_converter: The unit converter to use to convert between
              pixels and millimeters.
        """
        self.standard_scaler = StandardScaler()
        self.standard_scaler.scale_ = scale
        self.standard_scaler.mean_ = mean

        self.unit_converter = unit_converter

    def transform(self, X: ndarray) -> ndarray:
        """Apply the transformation required to standardize the velocity.

        This transformation should be applied to the velocity before sending it
        to the encoder.

        Args:
            X: The velocity to transform.

        Returns:
            Standardized velocity.

        """
        velocity = self.standard_scaler.transform(X)
        return self.unit_converter.pixels_to_millimeters(velocity)

    def inverse_transform(self, X: ndarray) -> ndarray:
        """Apply the transformation required to reverse the scaling of the velocity.

        This transformation should be applied to the velocity after receiving it
        from the decoder.

        Args:
            X: The velocity to transform.

        Returns:
            Scaled velocity.
        """
        velocity = self.unit_converter.millimeters_to_pixels(X)
        return self.standard_scaler.inverse_transform(velocity)
