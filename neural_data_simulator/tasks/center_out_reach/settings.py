"""Settings schema for the center-out reach task."""
from typing import Optional

from pydantic import BaseModel
from pydantic import validator

from neural_data_simulator.settings import LSLInputModel
from neural_data_simulator.settings import LSLOutputModel


class CenterOutReach(BaseModel):
    """Center-out reach settings."""

    class Input(BaseModel):
        """Input settings."""

        enabled: bool
        lsl: Optional[LSLInputModel]

        @validator("lsl")
        def _lsl_config_is_set_for_input_enabled(cls, v, values):
            if not v and values.get("enabled"):
                raise ValueError("lsl needs to be configured when input is enabled")

            return v

    class Output(BaseModel):
        """Output settings."""

        lsl: LSLOutputModel

    class Window(BaseModel):
        """Window settings."""

        class Colors(BaseModel):
            """Colors used in the GUI."""

            background: str
            decoded_cursor: str
            actual_cursor: str
            target: str
            target_waiting_for_cue: str
            decoded_cursor_on_target: str

        width: Optional[float]
        height: Optional[float]
        ppi: Optional[float]

        colors: Colors

    class StandardScaler(BaseModel):
        """Velocity scaler settings."""

        scale: list[float]
        mean: list[float]

    class Task(BaseModel):
        """Task settings."""

        target_radius: float
        cursor_radius: float
        radius_to_target: float
        number_of_targets: int

        delay_to_begin: float
        delay_waiting_for_cue: float
        target_holding_time: float
        max_trial_time: int

    input: Input
    output: Output
    sampling_rate: float
    window: Window
    with_metrics: bool
    standard_scaler: StandardScaler
    task: Task
