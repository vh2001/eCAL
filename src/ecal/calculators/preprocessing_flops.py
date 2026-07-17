"""FLOP calculators for data preprocessing steps (normalization, min-max
scaling, and Gramian Angular/Difference Field encoding)."""

from abc import ABC, abstractmethod
from typing import Dict, Union


class PreprocessingFLOPCalculator(ABC):
    """Abstract base class for data-preprocessing FLOP calculators."""

    @abstractmethod
    def calculate_flops(self, data_size: int) -> Dict[str, Union[int, Dict]]:
        """Calculate FLOPs required to preprocess a batch of data.

        Args:
            data_size: Total number of scalar data points to preprocess.

        Returns:
            Dict[str, Union[int, Dict]]: ``"total_flops"`` and ``"data_shape"``.
        """
        pass


class NormalizationCalculator(PreprocessingFLOPCalculator):
    """FLOP calculator for z-score normalization (mean/std standardization)."""

    def calculate_flops(self, data_size: int) -> Dict[str, Union[int, Dict]]:
        """Calculate FLOPs for z-score normalization of ``data_size`` points.

        Accounts for computing the mean, standard deviation, and applying
        ``(x - mean) / std`` to every point.

        Args:
            data_size: Total number of scalar data points to normalize.

        Returns:
            Dict[str, Union[int, Dict]]: ``"total_flops"`` (``6 * data_size + 1``)
            and ``"data_shape"`` (``None``, shape is unchanged by normalization).
        """
        # calculating mean:
        # 1. add all data points -> data_size - 1
        # 2. divide by data_size -> 1
        # Mean calculation FLOPS: data_size - 1 + 1 = data_size
        # ------------------------------------------------------------
        # calculating std:
        # 1. subtract mean from each data point -> data_size
        # 2. square the result -> data_size
        # 3. add the squares -> data_size - 1
        # 4. divide by data_size -> 1
        # 5. take the square root -> 1
        # Std. calculation FLOPS: data_size + data_size + (data_size - 1) + 1 + 1 = 3 * data_size + 1
        # ------------------------------------------------------------
        # normalization:
        # 1. subtract mean from each data point -> data_size
        # 2. divide by std -> data_size
        # normalization FLOPS: data_size + data_size = 2 * data_size
        # ------------------------------------------------------------

        # FINAL total FLOPS calculation: data_size + 3 * data_size + 1 + 2 * data_size = 6 * data_size + 1

        total_flops = (6 * data_size) + 1

        return {"total_flops": total_flops,
                "data_shape": None
                }


class MinMaxScalingCalculator(PreprocessingFLOPCalculator):
    """FLOP calculator for min-max scaling."""

    def calculate_flops(self, data_size: int) -> Dict[str, Union[int, Dict]]:
        """Calculate FLOPs for min-max scaling of ``data_size`` points.

        Accounts for computing ``max - min`` once and applying
        ``(x - min) / (max - min)`` to every point.

        Args:
            data_size: Total number of scalar data points to scale.

        Returns:
            Dict[str, Union[int, Dict]]: ``"total_flops"`` (``2 * data_size + 1``)
            and ``"data_shape"`` (``None``, shape is unchanged by scaling).
        """
        # Min-Max scaling:
        # 0. find max and min -> 0
        # 1. calculate max-min -> 1
        # 1. subtract min from each data point -> data_size
        # 2. divide by (max - min) -> data_size
        # Total FLOPS: 1 + data_size + data_size = 2 * data_size + 1

        scaling_flops = data_size * 2 + 1  #

        return {"total_flops": scaling_flops,
                "data_shape": None
                }


class GramianDifferenceFieldCalculator(PreprocessingFLOPCalculator):
    """FLOP calculator for Gramian Angular/Difference Field (GADF) encoding,
    following the pyts implementation's FLOP profile."""

    def calculate_flops(self, data_size: int, time_steps: int) -> Dict[str, Union[int, Dict]]:
        """Calculate FLOPs for GADF encoding of time-series data.

        Accounts for two min-max scaling passes over the data followed by the
        pairwise GADF computation across time steps.

        Args:
            data_size: Number of independent time-series samples.
            time_steps: Number of time steps per sample.

        Returns:
            Dict[str, Union[int, Dict]]: ``"total_flops"`` and ``"data_shape"``
            (``(data_size, time_steps, time_steps)``, the shape of the resulting
            GADF matrices).
        """
        #
        # 1. perform minmax 2 times -> 2 * data_size +1
        # 2. compute GADF flops based on pyTS implementation - > (5 * time_steps + time_steps * time_steps) * data_size
        minmax_calculator = MinMaxScalingCalculator()
        minmax_flops = minmax_calculator.calculate_flops(data_size * time_steps)["total_flops"]

        gadf_flops = (5 * time_steps + time_steps * time_steps) * data_size
        total_flops = minmax_flops + gadf_flops
        return {
            "total_flops": total_flops,
            "data_shape": (data_size, time_steps, time_steps)
        }
