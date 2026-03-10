from abc import ABC, abstractmethod
from typing import Dict, Union


class PreprocessingFLOPCalculator(ABC):
    @abstractmethod
    def calculate_flops(self, data_size: int) -> Dict[str, Union[int, Dict]]:
        pass


class NormalizationCalculator(PreprocessingFLOPCalculator):
    def calculate_flops(self, data_size: int) -> Dict[str, Union[int, Dict]]:
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
    def calculate_flops(self, data_size: int) -> Dict[str, Union[int, Dict]]:
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
    def calculate_flops(self, data_size: int, time_steps: int) -> Dict[str, Union[int, Dict]]:
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
