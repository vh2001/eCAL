import sys
from pathlib import Path

root = Path().resolve().parent  # go up one level from notebook’s folder
sys.path.append(str(root))

import pandas as pd
from calculators.Training import Training
from calculators.ModelFLOPS import MLPCalculator, CNNCalculator, KANCalculator, TransformerCalculator



######################################## Data Preprocessing ########################################
PREPROCESSING_TYPE = "normalization" # set the preprocessing to apply to the data options: normalization, min_max_scaling, GADF

REAL_INPUT_SIZE = 10 ## CHANGE SIZE HERE
INPUT_SIZE = (1, 1, REAL_INPUT_SIZE) # (channels, height, width) for CNN or (1, features) for MLP/CNN/KAN/Transformer

######################################## Training ########################################

NUM_EPOCHS = 50
BATCH_SIZE = 32

EVALUATION_STRATEGY = "train_test_split" # set the evaluation strategy options: cross_validation, train_test_split
K_FOLDS = 5 # Only used if EVALUATION_STRATEGY is cross_validation
SPLIT_RATIO = 0.8 # Only used if EVALUATION_STRATEGY is train_test_split

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
NUM_SAMPLES = 1200#256 # number of samples that are used to calculate the energy consumption
FLOAT_PRECISION = 32 # number of bits used to represent a floating point number
PROCESSOR_FLOPS_PER_SECOND = 3e12 # theoretical maximum number of floating point operations per second for the processor
PROCESSOR_MAX_POWER = 10 # maximum power consumption of the processor in Watts

# Timeseries specific
SAMPLE_SIZE = REAL_INPUT_SIZE # size of a single sample e.g. number of timesteps in a timeseries or number of pixels in an image


NUM_LAYERS = [3,4,5,6,7] # K
HIDDEN_SIZE = [10] # H
energy_consumption  ={}
model_names = ['MLP', 'CNN' ,'KAN', 'Transformer']
calculator = None
for model_name in model_names:
    curr_model = {}
    for L in NUM_LAYERS:
        for hs in HIDDEN_SIZE:
            #INPUT_SIZE = (1, 1, 10) #is this ok for all models?
            print(f"Calculating energy consumption for {model_name} with {L} layers and hidden size {hs}")
            calculator = None
            if model_name == "MLP":
                calculator = MLPCalculator(
                    num_layers=L,
                    din=REAL_INPUT_SIZE,
                    dout=DOUT,
                    num_samples=SAMPLE_SIZE,  # for time series
                    num_classes=NUM_CLASSES,
                )
            elif model_name == "CNN":
                calculator = CNNCalculator(
                    num_cnv_layers=L,
                    num_pool_layers=1,
                    i_r=10,
                    i_c=1,
                    k_r=3,
                    k_c=1,
                    c_in=1,
                    s_r = 1,
                    s_c= 1,
                    N_f=3,
                    num_samples=SAMPLE_SIZE,  # for time series
                    num_classes=NUM_CLASSES,
                )
            elif model_name == "KAN":
                hs_all =[SAMPLE_SIZE] +[hs]*L
                calculator = KANCalculator(
                    num_layers=L,
                    grid_size=hs,
                    din=10,
                    dout=DOUT,
                    k = 3,
                    num_samples=REAL_INPUT_SIZE, # for time series
                    num_classes=NUM_CLASSES
                )
            elif model_name == "Transformer":  
                calculator = TransformerCalculator(
                    context_length=CONTEXT_LENGTH,
                    embedding_size=hs,
                    num_heads=NUM_HEADS,
                    num_decoder_blocks=L,
                    feed_forward_size=FEED_FORWARD_SIZE,
                    vocab_size=CONTEXT_LENGTH
                )
           
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

            curr_model[(L,hs)] = training.calculate_energy()['training_energy']
    energy_consumption[model_name] = curr_model


df = pd.DataFrame(energy_consumption)
df.to_csv('./results/energy_consumption_models_small_theor.csv')
df