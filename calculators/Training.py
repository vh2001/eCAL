from typing import Dict, Union, Tuple, Optional

# import resnet18
from torchvision.models import resnet18

from .ModelFLOPS import FLOPCalculator, FlopsCalculatorFactory


class Training:
    """
    This class is used to estimate the flops of the model training, which is then used to estimate 
    the energy consumption of the model training.
    """

    def __init__(self, model_name: str,
                 batch_size: int, num_epochs: int, num_samples: int,
                 processor_flops_per_second: float, processor_max_power: int, input_size: Tuple,
                 evaluation_strategy: str, k_folds: int, split_ratio: float,
                 calculator: Optional[FLOPCalculator] = None):
        """
        Initialize Training class with optional custom FLOP calculator
        
        Args:
            calculator: Optional custom FLOPCalculator implementation
            input_size: Tuple of input size
            batch_size: int of batch size
            num_epochs: int of number of epochs
            num_samples: int of number of samples
            processor_flops_per_second: float of processor flops per second
            processor_max_power: int of processor max power in watts
            evaluation_strategy: str of evaluation strategy
            k_folds: int of number of folds for cross-validation
            split_ratio: float of split ratio for train-test split
        """
        if model_name == 'resnet18':
            self.model = resnet18()
        else:
            self.model = model_name

        if calculator is not None:
            self.calculator = calculator
        else:
            self.calculator = FlopsCalculatorFactory.create_calculator(self.model)

        if evaluation_strategy == 'train_test_split':
            self.evaluation_strategy = 'train_test_split'
            self.split_ratio = split_ratio
        elif evaluation_strategy == 'cross_validation':
            self.evaluation_strategy = 'cross_validation'
            self.k_folds = k_folds
        else:
            raise ValueError(f"Unsupported evaluation strategy: {evaluation_strategy}")

        self.input_size = input_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        # hardware parameters
        self.processor_flops_per_second = processor_flops_per_second
        self.processor_max_power = processor_max_power

    def calculate_flops_training(self) -> Dict[str, Union[int, Dict]]:

        forward_flops = self.calculator.calculate(self.model, self.input_size)['total_flops']
        # 1 training pass takes roughly 3x a single forward pass
        training_flops = forward_flops * 3
        # Calculate the number of batches
        if self.evaluation_strategy == 'train_test_split':
            training_samples = self.num_samples * self.split_ratio
        elif self.evaluation_strategy == 'cross_validation':
            percentage_of_samples = 1 - (1 / self.k_folds)  # percentage of samples used for training
            number_of_folds = self.k_folds
            training_samples = self.num_samples * percentage_of_samples * number_of_folds

        else:
            raise ValueError(f"Unsupported evaluation strategy: {self.evaluation_strategy}")
        # Calculate the total number of flops
        total_flops = training_flops * training_samples * self.num_epochs
        return total_flops

    def calculate_flops_evaluation(self) -> float:
        # Calculate the total number of flops
        forward_flops = self.calculator.calculate(self.model, self.input_size)['total_flops']
        if self.evaluation_strategy == 'train_test_split':
            evaluation_samples = self.num_samples * (1 - self.split_ratio)
        elif self.evaluation_strategy == 'cross_validation':
            percentage_of_samples = 1 / self.k_folds  # percentage of samples used for evaluation
            number_of_folds = self.k_folds
            evaluation_samples = self.num_samples * percentage_of_samples * number_of_folds
        else:
            raise ValueError(f"Unsupported evaluation strategy: {self.evaluation_strategy}")

        total_flops = forward_flops * evaluation_samples
        return total_flops

    def calculate_energy(self) -> float:
        # Calculate the total number of flops
        training_flops = self.calculate_flops_training()
        evaluation_flops = self.calculate_flops_evaluation()

        training_energy = training_flops / self.processor_flops_per_second * self.processor_max_power
        evaluation_energy = evaluation_flops / self.processor_flops_per_second * self.processor_max_power

        # Calculate the total energy usage
        total_energy = training_energy + evaluation_energy

        return {
            "total_energy": total_energy,
            "training_energy": training_energy,
            "evaluation_energy": evaluation_energy,
            "training_flops": training_flops,
            "evaluation_flops": evaluation_flops,
            "train_time": self.processor_flops_per_second / training_flops,
            "eval_time": self.processor_flops_per_second / evaluation_flops
        }
