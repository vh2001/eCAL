from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple

from calflops import calculate_flops_hf, calculate_flops
from torch import nn


class FlopsCalculatorFactory:
    @staticmethod
    def create_calculator(model: Union[nn.Module, str]) -> 'FLOPCalculator':
        if isinstance(model, str):
            return CalFlopsCalculatorHF()
        elif isinstance(model, nn.Module):
            return CalFlopsCalculatorPT()
        else:
            raise ValueError("Model must be either a string (HuggingFace model name) or nn.Module (PyTorch model)")


class FLOPCalculator(ABC):
    @abstractmethod
    def calculate(self, model: Union[nn.Module, str], input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        pass


class CalFlopsCalculatorHF(FLOPCalculator):
    def calculate(self, model: str, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        flops, macs, params = calculate_flops_hf(model_name=model, input_shape=input_size, print_results=False,
                                                 output_as_string=False)
        return {"total_flops": flops, "total_params": params}


class CalFlopsCalculatorPT(FLOPCalculator):
    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        flops, macs, params = calculate_flops(model=model,
                                              input_shape=input_size,
                                              output_as_string=False,
                                              output_precision=4,
                                              print_results=False)
        return {"total_flops": flops, "total_params": params}

class MLPCalculator(FLOPCalculator):
    def __init__(self, num_layers: int, din: int, dout: int, num_samples: int = 1,
                 num_classes: int = 2):
        self.din = din
        self.dout = dout
        self.L = num_layers
        self.T = num_samples
        self.C = num_classes

    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        """
        Calculate FLOPs and parameters for a Multilayer Perceptron
        
        Parameters:
        - L: Number of layers
        - M_l-1: Input dimension of the layer
        - M_l: Output dimension of the layer
        """
        L = self.L  # Number of layers

        # Total FLOPs calculation following the new formula
        total_flops = 0
        for l in range(0, L - 1):

            # FLOPs from input-output dimension computation 
            layer_flops = 2 * (self.din * self.din) + 2 * self.din

            total_flops +=  layer_flops

        return {
            'total_flops': int(total_flops),
            'total_params': None
        }
    
class CNNCalculator(FLOPCalculator):
    def __init__(self, num_cnv_layers: int = 3, num_pool_layers: int = 1, i_r: int = 10, i_c: int = 1, 
                 k_r: int = 3, k_c: int = 1, c_in: int = 1, s_r: int = 1, s_c: int = 1, N_f: int = 3, num_samples: int = 1,
                 num_classes: int = 2):
        self.num_cnv_layers = num_cnv_layers
        self.num_pool_layers = num_pool_layers
        self.i_r = i_r
        self.i_c = i_c
        self.k_r = k_r
        self.k_c = k_c
        self.p_r = k_r - 1
        self.p_c = k_c
        self.c_in = c_in
        self.s_r = s_r
        self.s_c = s_c
        self.N_f = N_f
        self.T = num_samples
        self.C = num_classes

    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        """
        Calculate FLOPs and parameters for a Multilayer Perceptron
        
        Parameters:
        - num_cnv_layers: Number of convolutional layers
        - num_pool_layers: Number of pooling layers
        """
        num_cnv_layers = self.num_cnv_layers  # Number of layers
        num_pool_layers = self.num_pool_layers  # Number of layers

        input_height = self.i_r
        input_width = self.i_c

        output_height = 0
        output_width = 0
        # Total FLOPs calculation following the new formula
        total_flops = 0
        for c in range(0, num_cnv_layers - 1):

            # FLOPs from convolutional layers
            output_height = (input_height - self.k_r + 2 * self.p_r) / self.s_r + 1
            output_width = (input_width - self.k_c + 2 * self.p_c) / self.s_c + 1

            # Create convolutional blocks with increasing channel depth
            input_height = self.i_r * (2 ** c)
            input_width = self.i_c * (2 ** c)

            layer_flops = output_height * output_width * (self.c_in * self.k_r * self.k_c + 1) * self.N_f

            total_flops +=  layer_flops

        for p in range(0, num_pool_layers):

            # FLOPs from pooling layers
            output_height = (input_height - self.k_r + 2 * self.p_r) / self.s_r + 1
            output_width = (input_width - self.k_c + 2 * self.p_c) / self.s_c + 1

            layer_flops = output_height * output_width * self.c_in

            total_flops +=  layer_flops
        
        # add final layer
        total_flops +=  2 * (output_height * output_width) + 2 * output_height

        return {
            'total_flops': int(total_flops),
            'total_params': None
        }


class KANCalculator(FLOPCalculator):
    def __init__(self, num_layers: int, grid_size: int, din: int, dout: int, k: int = 3, num_samples: int = 1,
                 num_classes: int = 2):
        self.G = grid_size
        self.din = din
        self.dout = dout
        self.k = k
        self.L = num_layers
        self.T = num_samples
        self.C = num_classes

    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        """
        Calculate FLOPs and parameters for a Kolmogorov-Arnold Network (KAN)
        
        Parameters:
        - K: B-spline degree (typically 3)
        - G: Grid size
        - L: Number of layers
        - M_l-1: Input dimension of the layer
        - M_l: Output dimension of the layer
        - M_NLF: FLOPs for non-linear function (B-spline activation)
        """
        K = self.k  # B-spline degree
        G = self.G  # Grid size
        L = self.L  # Number of layers

        # Constant for B-spline and grid computation
        M_B = 9 * K * (G + 1.5 * K) + 2 * G - 2.5 * K + 3

        # Assuming M_NLF is the FLOPs for B-spline activation function
        # This might need to be precisely defined based on the specific implementation
        M_NLF = 2  # Placeholder, adjust based on actual B-spline activation computation

        # Total FLOPs calculation following the new formula
        total_flops = 0
        for l in range(1, L):
            # FLOPs from B-spline activation
            b_spline_flops = M_NLF * self.din

            # FLOPs from input-output dimension computation with B-spline transformation
            layer_flops = (self.din * self.din) * M_B

            total_flops += b_spline_flops + layer_flops

        return {
            'total_flops': int(total_flops),
            'total_params': None
        }


class TransformerCalculator(FLOPCalculator):
    def __init__(self, context_length: int, embedding_size: int, num_heads: int, 
                 num_decoder_blocks: int, feed_forward_size: int, vocab_size: int):
        self.context_length = context_length
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_decoder_blocks = num_decoder_blocks
        self.feed_forward_size = feed_forward_size
        self.vocab_size = vocab_size

    def calculate(self, model: nn.Module, input_size: Tuple) -> Dict[str, Union[int, Dict]]:
        """
        Calculate FLOPs for a Transformer model.
        
        Parameters:
        - C: Context length
        - N_embed: Embedding size
        - N_head: Number of attention heads
        - N_decoder_blocks: Number of decoder blocks
        - FFS: Feed forward size
        """
        # Model parameters
        C = self.context_length
        N_embed = self.embedding_size
        N_head = self.num_heads
        N_decoder_blocks = self.num_decoder_blocks
        FFS = self.feed_forward_size

        # Calculate attention FLOPs (M_ATT)
        # K, Q, V positional embedding
        kqv_flops = C * N_embed * 3 * N_embed

        # Attention scores
        attention_score_flops = C * C * N_embed

        # Reduce operation
        reduce_flops = N_head * C * C * (N_embed // N_head)

        # Projection
        projection_flops = C * N_embed * N_embed

        # Total attention FLOPs (multiplied by 2 as per equation)
        M_ATT = 2 * (kqv_flops + attention_score_flops + reduce_flops + projection_flops)

        # MLP blocks FLOPs
        mlp_flops = 2 * 2 * C * N_embed * FFS

        # Total Transformer FLOPs (M_TR)
        M_TR = N_decoder_blocks * (M_ATT + mlp_flops)

        total_flops = M_TR

        # Calculate parameters

        return {
            'total_flops': int(total_flops),
            'total_params': None,
            'breakdown': {
                'attention': {
                    'kqv_embedding_flops': int(kqv_flops),
                    'attention_score_flops': int(attention_score_flops),
                    'reduce_flops': int(reduce_flops),
                    'projection_flops': int(projection_flops),
                    'total_attention_flops': int(M_ATT)
                },
                'mlp_blocks_flops': int(mlp_flops),
                'per_block_flops': int(M_ATT + mlp_flops),
            }}
