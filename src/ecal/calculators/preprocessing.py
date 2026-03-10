from ecal.calculators.preprocessing_flops import (
    NormalizationCalculator,
    MinMaxScalingCalculator,
    GramianDifferenceFieldCalculator,
)


class DataPreprocessing:
    """Data preprocessing class that calculates the FLOPs for various data preprocessing tasks"""

    def __init__(self, preprocessing_type: str = 'normalization', processor_flops_per_second: float = 1e12,
                 processor_max_power: int = 100, time_steps: int = 1):
        """
        Initialize DataPreprocessing class

        Args:
            preprocessing_type: Type of preprocessing to perform
        """
        self.calculators = {
            'normalization': NormalizationCalculator(),
            'min_max_scaling': MinMaxScalingCalculator(),
            'GADF': GramianDifferenceFieldCalculator()
        }
        self.preprocessing_type = preprocessing_type
        self.set_preprocessing_type(preprocessing_type)
        self.processor_flops_per_second = processor_flops_per_second
        self.processor_max_power = processor_max_power

    def set_preprocessing_type(self, preprocessing_type: str) -> None:
        """Set the preprocessing type"""
        if preprocessing_type not in self.calculators:
            raise ValueError(f"Unsupported preprocessing type: {preprocessing_type}")
        self.calculator = self.calculators[preprocessing_type]

    def calculate_flops(self, data_bits: int, time_steps=1) -> float:
        """
        Calculate FLOPs for the current preprocessing type

        Args:
            data_bits: Number of bits in the input data
            time_steps: Number of time steps in the input time series data

        Returns:
            Total FLOPs for the current preprocessing type
        """
        if self.preprocessing_type == 'GADF':
            return self.calculator.calculate_flops(data_bits, time_steps)

        return self.calculator.calculate_flops(data_bits)

    def calculate_energy(self, data_bits: int, time_steps: int) -> float:
        # Calculate the total number of flops
        if self.preprocessing_type == 'GADF':
            calc_dict = self.calculate_flops(data_bits, time_steps)
        else:
            calc_dict = self.calculate_flops(data_bits * time_steps)
        total_flops = calc_dict['total_flops']

        total_time = total_flops / self.processor_flops_per_second
        total_energy = total_time * self.processor_max_power
        return {
            "total_energy": total_energy,
            "total_bits": data_bits * time_steps,
        }
