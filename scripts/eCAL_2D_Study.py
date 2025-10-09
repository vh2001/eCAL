import sys
from pathlib import Path

root = Path().resolve().parent  # go up one level from notebook’s folder
sys.path.append(str(root))

from calculators.Training import Training
from calculators.ModelFLOPS import KANCalculator, TransformerCalculator

######################################## Data Preprocessing ########################################
PREPROCESSING_TYPE = "normalization" # set the preprocessing to apply to the data options: normalization, min_max_scaling, GADF

REAL_INPUT_SIZE = 50 ## CHANGE SIZE HERE

######################################## Training ########################################
MODEL_NAME ="SimpleCNN" # set which model to use examples: KAN, resnet18, baichuan-inc/Baichuan-13B-Chat
NUM_EPOCHS = 50
BATCH_SIZE = 32
# INPUT_SIZE = (1, 3, 224, 224) # 4d input for resnet # (1,128) # batch, max_seq_length for llm
if MODEL_NAME == "SimpleCNN":
    INPUT_SIZE = (1, 1, REAL_INPUT_SIZE) # 4d input for resnet # (1,128) # batch, max_seq_length for llm
else:
    INPUT_SIZE = (1, REAL_INPUT_SIZE) # 

EVALUATION_STRATEGY = "train_test_split" # set the evaluation strategy options: cross_validation, train_test_split
K_FOLDS = 5 # Only used if EVALUATION_STRATEGY is cross_validation
SPLIT_RATIO = 0.8 # Only used if EVALUATION_STRATEGY is train_test_split



NUM_LAYERS = 2
GRID_SIZE = 5
NUM_CLASSES = 2
DIN = REAL_INPUT_SIZE
DOUT = 2

# Transformer specific parameters
CONTEXT_LENGTH = REAL_INPUT_SIZE  
EMBEDDING_SIZE = 16  
NUM_HEADS = 2       
NUM_DECODER_BLOCKS = 3  
FEED_FORWARD_SIZE = 64  

######################################## Inference ########################################
NUM_INFERENCES = 10000 # number of inferences to run


# GENERAL CONFIG
NUM_SAMPLES = 256 # number of samples that are used to calculate the energy consumption
FLOAT_PRECISION = 32 # number of bits used to represent a floating point number
#in the paper
#PROCESSOR_FLOPS_PER_SECOND = 1e12 # theoretical maximum number of floating point operations per second for the processor
#PROCESSOR_MAX_POWER = 100 # maximum power consumption of the processor in Watts

#Carolina's Mac
PROCESSOR_FLOPS_PER_SECOND = 3e12 # theoretical maximum number of floating point operations per second for the processor
PROCESSOR_MAX_POWER = 10 # maximum power consumption of the processor in Watts

# Timeseries specific
SAMPLE_SIZE = REAL_INPUT_SIZE # size of a single sample e.g. number of timesteps in a timeseries or number of pixels in an image

import calculators.ToyModels as toy_models
import pandas as pd

NUM_LAYERS = [3,4,5,6,7] # K
HIDDEN_SIZE = [5,6,7,8,9,10,11,12,13,14,15] # H
energy_consumption  ={}
model_names = ['SimpleMLP', 'SimpleCNN' ,'KAN', 'SimpleTransformer']
calculator = None
for model_name in model_names:
    curr_model = {}
    for L in NUM_LAYERS:
        for hs in HIDDEN_SIZE:
            calculator = None
            if model_name == 'SimpleMLP':
                curr_model_pt = toy_models.SimpleMLP(input_size=REAL_INPUT_SIZE, hidden_size=hs, output_size=1, num_layers=L)
            elif model_name == 'SimpleCNN':
                curr_model_pt = toy_models.SimpleCNN(input_channels=1, hidden_channels=1, output_size=1, num_layers=L)
                # INPUT_SIZE = (1, 1,10) # 

            if model_name == "KAN":
                # hs_all =[SAMPLE_SIZE] +[hs]*(L-1)
                calculator = KANCalculator(
                    num_layers=L,
                    grid_size=hs,
                    num_classes=NUM_CLASSES,
                    din=DIN,
                    dout=DOUT,
                    num_samples=SAMPLE_SIZE # for time series
                )

            elif model_name == "SimpleTransformer":
                calculator = TransformerCalculator(
                    context_length=CONTEXT_LENGTH,
                    embedding_size=hs,
                    num_heads=NUM_HEADS,
                    num_decoder_blocks=L,
                    feed_forward_size=FEED_FORWARD_SIZE,
                    vocab_size=2
                )
            if calculator is not None:
                training = Training(
                    model_name=model_name,
                    num_epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    processor_flops_per_second=PROCESSOR_FLOPS_PER_SECOND,
                    processor_max_power=PROCESSOR_MAX_POWER,
                    num_samples=NUM_SAMPLES,
                    input_size=INPUT_SIZE,
                    evaluation_strategy=EVALUATION_STRATEGY,
                    k_folds=K_FOLDS,
                    split_ratio=SPLIT_RATIO,
                    calculator=calculator
                )
            else:
                training = Training(
                model_name=curr_model_pt,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                processor_flops_per_second=PROCESSOR_FLOPS_PER_SECOND,
                processor_max_power=PROCESSOR_MAX_POWER,
                num_samples=NUM_SAMPLES,
                input_size=INPUT_SIZE,
                evaluation_strategy=EVALUATION_STRATEGY,
                k_folds=K_FOLDS,
                split_ratio=SPLIT_RATIO,
                calculator=None
                )
            if model_name == "SimpleMLP":
                model_str = "MLP"
            elif model_name == "SimpleCNN":
                model_str = "CNN"
            elif model_name == "SimpleTransformer":
                model_str = "Transformer"
            else:
                model_str = model_name

            curr_model[(L,hs)] =training.calculate_energy()['total_energy']
    energy_consumption[model_str] = curr_model


df = pd.DataFrame(energy_consumption)
df.to_csv('./results/energy_consumption_models_small.csv')
df