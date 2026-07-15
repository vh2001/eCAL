from typing import Tuple, Optional
from ecal.calculators.model_flops import FLOPCalculator, FlopsCalculatorFactory
from torchvision.models import resnet18


class Inference:
    """
    This class is used to estimate the flops of the model inference, which is then used to estimate
    the energy consumption of the model inference.
    """

    def __init__(self, model_name: str, input_size: Tuple, num_samples: int, processor_flops_per_second: float,
                 processor_max_power: int, calculator: Optional[FLOPCalculator] = None):
        """
        Initialize Inference class
        Args:
            calculator: FLOPCalculator implementation
            model_name: PyTorch model or model name
            input_size: Tuple of input size
            num_samples: int of number of samples
            processor_flops_per_second: float of processor flops per second
            processor_max_power: int of processor max power in watts
        """
        if model_name == 'resnet18':
            self.model = resnet18()
        else:
            self.model = model_name

        if calculator is not None:
            self.calculator = calculator
        else:
            self.calculator = FlopsCalculatorFactory.create_calculator(self.model)
        self.input_size = input_size
        self.num_samples = num_samples
        # hardware parameters
        self.processor_flops_per_second = processor_flops_per_second
        self.processor_max_power = processor_max_power

    def calculate_flops(self) -> float:
        """
        Calculate total FLOPs for the current inference workload

        Returns:
            Total FLOPs across num_samples inference calls (a single forward
            pass's FLOPs multiplied by num_samples)
        """
        forward_flops = self.calculator.calculate(self.model, self.input_size)['total_flops']
        total_flops = forward_flops * self.num_samples
        return total_flops

    def calculate_energy(self) -> float:
        """
        Calculate the energy usage of the current inference

        Returns:
            Total energy usage of the current inference in Joules
        """
        # Calculate the total number of flops
        total_flops = self.calculate_flops()
        # Calculate the total energy usage
        total_time = total_flops / self.processor_flops_per_second
        total_energy = total_time * self.processor_max_power
        return total_energy
