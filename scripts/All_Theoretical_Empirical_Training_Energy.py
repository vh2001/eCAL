import sys
from pathlib import Path

root = Path().resolve().parent  # go up one level from notebook’s folder
sys.path.append(str(root))

import numpy as np
import calculators.ToyModels as toy_models
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from collections import defaultdict
import pandas as pd # For reading CSV
from codecarbon import EmissionsTracker
from sklearn.metrics import r2_score
# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pickle

# set fp32 precision for better comparability
torch.set_default_dtype(torch.float32)

# --- 1. Parameters for the data and model ---
num_time_series = 1000#204
time_series_length = 10
nodes_per_layer = 10  # Hidden nodes per layer for MLP, and for internal layers of others
random_seed = 42      # for reproducibility
num_epochs_pytorch = 50 # Number of epochs for PyTorch models

# Define the range for the number of layers (hidden for MLP, dense for CNN,
# sub-layers for KAN-like, encoder layers for Transformer)
min_layers = 3
max_layers = 7
layer_range = range(min_layers, max_layers + 1)

# --- NEW: Assumed Average Power Consumption (PLACEHOLDER) ---
# This value needs to be determined empirically for your specific hardware
# (e.g., CPU, GPU, RAM) during active computation.
# For demonstration, let's assume a moderate power draw for general computation.
# A powerful GPU could be 200-400W. A CPU might be 50-150W.
assumed_average_power_watts = 0.38 # Watts.measured with tthe avg power script
print(f"ASSUMED AVERAGE POWER FOR ENERGY CALCULATION: {assumed_average_power_watts} Watts (Crucial Placeholder!)")
P_idle_GPU_watts = 0.000001   # Watts (very low idle power for M2 MacBook)
P_peak_GPU_watts = 9.73  # Watts (e.g., peak system power for M2 MacBook under heavy load)
mid_utilization_factor = 0.6 # Hypothetical mid-level utilization for 'proportional' estimate

models = ["MLP", "CNN", "KAN", "Transformer"]
runs = 10 # Number of runs to average results over

# Set device for PyTorch
device = torch.device("mps")
print(f"Using device: {device}")

# --- 2. Generate Synthetic Data ---
print(f"\nGenerating synthetic data: {num_time_series} samples, each with {time_series_length} features.")
np.random.seed(random_seed)

X_np = np.random.rand(num_time_series, time_series_length) * 10
y_np = np.sin(X_np[:, 0] * 0.5) + np.cos(X_np[:, 1] * 0.2) + np.sum(X_np[:, 2:5], axis=1) * 0.1 + np.random.randn(num_time_series) * 0.5

print(f"Shape of X: {X_np.shape}")
print(f"Shape of y: {y_np.shape}")

# --- 3. Preprocessing (Standardization) ---
scaler_X = StandardScaler()
X_scaled_np = scaler_X.fit_transform(X_np)

scaler_y = StandardScaler()
y_scaled_np = scaler_y.fit_transform(y_np.reshape(-1, 1)).flatten()

# Convert to PyTorch Tensors for PyTorch models
X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_scaled_np, dtype=torch.float32).unsqueeze(1).to(device) # unsqueeze for (N, 1) target

# Create DataLoader for PyTorch models
batch_size = 51
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- 4. PyTorch Model Definitions ---

# R^2 calculation for PyTorch models
def r2_score_pytorch(y_pred, y_true):
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
    # Add small epsilon to prevent div by zero and handle flat y_true
    return 1 - (ss_res / (ss_tot + 1e-6)) if ss_tot > 1e-6 else 0.0

def theoretical_energy_proportionality(utilization, P_idle, P_peak):
    """
    Calculates the theoretical power consumption based on utilization,
    idle power, and peak power.

    Args:
        utilization (float or np.array): The utilization level (0 to 1).
        P_idle (float): The power consumption when the system is idle.
        P_peak (float): The peak power consumption when the system is fully utilized.

    Returns:
        float or np.array: The estimated power consumption in Watts.
    """
    if isinstance(utilization, (list, np.ndarray)):
        if not (0 <= utilization).all() and (utilization <= 1).all():
            raise ValueError("Utilization must be between 0 and 1.")
    else:
        if not (0 <= utilization <= 1):
            raise ValueError("Utilization must be between 0 and 1.")


    # P(U) = P_idle + U * (P_peak - P_idle)
    power_consumption = P_idle + utilization * (P_peak - P_idle)
    return power_consumption


class SimpleMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=10, output_size=2, num_layers=3):
        super(SimpleMLP, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(num_layers - 2): # Adjusted loop for clarity
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        # Output layer - Add it to the list!
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        # Simpler forward pass that processes all layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x


# --- CNN Model (Modified to vary dense layers) ---
class SimpleCNN(nn.Module):
    """
    A simple 1D Convolutional Neural Network (CNN) for sequence or feature vector regression.
    """
    def __init__(self, output_size=2, hidden_channels=16, num_layers=3):
        super(SimpleCNN, self).__init__()
        self.layers = nn.ModuleList()
        # Input data is expected to be reshaped to have 1 channel.
        current_channels = 1  
        
        # Create convolutional blocks with increasing channel depth
        for i in range(num_layers):
            out_channels = hidden_channels * (2**i) # e.g., 16 -> 32 -> 64
            conv_block = nn.Sequential(
                nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
            self.layers.append(conv_block)
            current_channels = out_channels
            
        # Global pooling layer adapts to any input size
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(current_channels, output_size)
    
    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        # Expected input x shape: (batch_size, 1, num_features)
        for layer in self.layers:
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten the output for the linear layer
        return self.output_layer(x)
# --- Simplified KAN-like Model (Modified to vary sub-layers) ---
# This model applies an MLP with `num_hidden_layers_per_feature_fn` hidden layers
# to each input feature and sums the results.
class KANLikeRegressor(nn.Module):
    def __init__(self, input_dim, num_hidden_layers_per_feature_fn, nodes_per_layer_in_sub_fn):
        super(KANLikeRegressor, self).__init__()
        self.input_dim = input_dim
        self.feature_functions = nn.ModuleList()

        for _ in range(input_dim):
            layers_list = []
            current_in_features = 1 # Each feature function takes 1 input initially

            for i in range(num_hidden_layers_per_feature_fn):
                layers_list.append(nn.Linear(current_in_features, nodes_per_layer_in_sub_fn))
                layers_list.append(nn.ReLU())
                current_in_features = nodes_per_layer_in_sub_fn # For subsequent hidden layers

            # Final output layer for the feature function
            layers_list.append(nn.Linear(current_in_features, 1))
            self.feature_functions.append(nn.Sequential(*layers_list))

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        outputs = []
        for i in range(self.input_dim):
            # Pass each feature through its own sub-network
            output = self.feature_functions[i](x[:, i].unsqueeze(1)) # (batch_size, 1)
            outputs.append(output)
        # Sum the outputs of all feature functions
        return torch.sum(torch.cat(outputs, dim=1), dim=1, keepdim=True)



############################################################H#############################

# --- 5. PyTorch Training Function ---
def train_pytorch_model(model, train_loader, X_full, y_full, num_epochs=num_epochs_pytorch):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create a local tracker for each model trained with this function
    local_tracker = EmissionsTracker() 
    local_tracker.start()
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    local_tracker.flush() # Ensure all data is written

    emissions: float = local_tracker.stop()
    print(emissions)

    df = pd.read_csv("emissions.csv")
    latest = df.iloc[-1]
    data = {}
    if emissions:
        data = {
            "gpu_energy_Wh": (latest["gpu_energy"] / 1000) * 3600, # Ws
            "cpu_energy_Wh": (latest["cpu_energy"] / 1000) * 3600, # Ws
            "ram_energy_Wh": (latest["ram_energy"] / 1000) * 3600, # Ws
            "energy_consumed_Wh": (latest["energy_consumed"] / 1000) * 3600, # Ws
            "duration_seconds": latest["duration"], 
            "total_gpu_power_W": latest["gpu_power"],
            "idle_gpu_power_W": 0.3 * latest["gpu_power"],
            "peak_gpu_power_W": 1.2 * latest["gpu_power"],
        }

    # Evaluate R^2 and final loss on full (scaled) data
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_full)
        final_loss = criterion(y_pred_tensor, y_full).item()
        r2 = r2_score(y_full.cpu().numpy(), y_pred_tensor.cpu().numpy())

    return data, final_loss, r2


# --- Storage for all results ---
all_training_times_v = defaultdict(list)
all_training_losses_v = defaultdict(list)
all_r2_scores_v = defaultdict(list)
energy_fundamental_avg_joules_v = defaultdict(list)
energy_baseline_idle_joules_v = defaultdict(list)
energy_worst_case_peak_joules_v = defaultdict(list)
energy_proportional_mid_U_joules_v = defaultdict(list)
code_carbon_joules_v = defaultdict(list)


all_training_times = OrderedDict()
all_energy_consumed_joules = OrderedDict()
all_training_losses = OrderedDict()
all_r2_scores = OrderedDict()
energy_fundamental_avg_joules = OrderedDict()
energy_baseline_idle_joules = OrderedDict()
energy_worst_case_peak_joules = OrderedDict()
energy_proportional_mid_U_joules = OrderedDict()
code_carbon_joules = OrderedDict()

# --- 6. Loop through layers and then model types ---
print("\n--- Starting Training for all models across various layer configurations ---")

for num_layers in layer_range:
    print(f"\n===== Training Models with {num_layers} Layers =====")
    for model in models:
        hidden_layer_config = tuple([nodes_per_layer] * num_layers)
        model_name = f"{model}_{num_layers}L"
        
        for i in range(runs):
            print(f"\n  Training {model_name} with {num_layers} run no ({i}.")

            model_instance = {} 
            if (model == "MLP"): 
                model_instance = SimpleMLP(input_size=time_series_length, hidden_size=nodes_per_layer, output_size=1, num_layers=num_layers).to(device)
            elif (model == "CNN"):
                model_instance = SimpleCNN(output_size=1, hidden_channels=16, num_layers=num_layers).to(device)
            elif (model == "KAN"):
                model_instance = KANLikeRegressor(
                    input_dim=time_series_length,
                    num_hidden_layers_per_feature_fn=num_layers, # Varying sub-layers
                    nodes_per_layer_in_sub_fn=nodes_per_layer
                ).to(device)    
            elif (model == "Transformer"):
                model_instance = toy_models.Transformer(
                num_tokens=10,
                num_token_vals=20, #nodes per layer
                num_emb=10, #* 2,
                num_neurons=10,
                num_heads=2,
                num_blocks=num_layers, #L
                device=device).to(device)
            
            
            data, final_loss, model_r2 = train_pytorch_model(model_instance, train_loader, X_tensor, y_tensor, num_epochs=num_epochs_pytorch)
            fundamental_energy = assumed_average_power_watts * data["duration_seconds"] # Convert to Joules
            idle_energy = P_idle_GPU_watts * data["duration_seconds"] # Convert to Joules
            peak_energy = P_peak_GPU_watts * data["duration_seconds"] # Convert to Joules
            proportional_energy = data["total_gpu_power_W"] * data["duration_seconds"] # (P_idle_GPU_watts + mid_utilization_factor * (P_peak_GPU_watts - P_idle_GPU_watts)) * data["duration_seconds"] # Convert to Joules

            all_training_times_v[model_name].append(data["duration_seconds"])
            all_training_losses_v[model_name].append(final_loss)
            all_r2_scores_v[model_name].append(model_r2)
            energy_fundamental_avg_joules_v[model_name].append(fundamental_energy)
            energy_baseline_idle_joules_v[model_name].append(idle_energy)
            energy_worst_case_peak_joules_v[model_name].append(peak_energy)
            energy_proportional_mid_U_joules_v[model_name].append(proportional_energy)
            code_carbon_joules_v[model_name].append(data["energy_consumed_Wh"])


        all_training_times[model_name] = np.mean(all_training_times_v[model_name])
        all_training_losses[model_name] = np.mean(all_training_losses_v[model_name])
        all_r2_scores[model_name] = np.mean(all_r2_scores_v[model_name])
        energy_fundamental_avg_joules[model_name] = np.mean(energy_fundamental_avg_joules_v[model_name])
        energy_baseline_idle_joules[model_name] = np.mean(energy_baseline_idle_joules_v[model_name])
        energy_worst_case_peak_joules[model_name] = np.mean(energy_worst_case_peak_joules_v[model_name])
        energy_proportional_mid_U_joules[model_name] = np.mean(energy_proportional_mid_U_joules_v[model_name])
        code_carbon_joules[model_name] = np.mean(code_carbon_joules_v[model_name])

        print(f"    Training completed in: {all_training_times[model_name]:.4f} seconds")
        print(f"    Fundamental Physics Energy: {fundamental_energy:.4f} J")
        print(f"    Baseline Idle Energy: {idle_energy:.4f} J")
        print(f"    Worst-Case Peak Energy: {peak_energy:.4f} J")
        print(f"    Code Carbon P*t Energy: {proportional_energy:.4f} J")
        print(f"    Code Carbon E' {code_carbon_joules[model_name]:.4f} J")
        print(f"    Final Loss: {final_loss:.4f}")
        print(f"    R^2 score (on training data): {model_r2:.4f}")


# --- 10. Final Summary ---
print("\n--- FINAL SUMMARY of Training Times, Energy, and Performance ---")
# Fixed printing error in the header string: "Code Carbon" "Final Loss" -> "Code Carbon (J)", "Final Loss"
print("{:<20} {:<15} {:<20} {:<20} {:<20} {:<20} {:<15} {:<15} {:<15}".format(
    "Model", "Train Time (s)", "Baseline Idle (J)", "Emp. Avg. Energy (J)",
    "Code Carbon Pt (J)", "Code Carbon E (J)", "Worst-Case Peak (J)", "Final Loss", "R^2 Score"
))
print("-" * 150)
for model_name in all_training_times.keys():
    print("{:<20} {:<15.4f} {:<20.4f} {:<20.4f} {:<20.4f} {:<20.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
        model_name,
        all_training_times[model_name],
        energy_baseline_idle_joules[model_name],
        energy_fundamental_avg_joules[model_name],
        energy_proportional_mid_U_joules[model_name],
        code_carbon_joules[model_name],
        energy_worst_case_peak_joules[model_name],
        all_training_losses[model_name],
        all_r2_scores[model_name]
    ))


# Save all results to CSV
data_to_save = {
    'training_times': all_training_times_v,
    'training_losses': all_training_losses_v,
    'r2_scores': all_r2_scores_v,
    'energy_fundamental': energy_fundamental_avg_joules_v,
    'energy_baseline': energy_baseline_idle_joules_v,
    'energy_worst_case': energy_worst_case_peak_joules_v,
    'energy_proportional': energy_proportional_mid_U_joules_v,
    'code_carbon': code_carbon_joules_v,
}

with open('./results/my_metrics.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

with open('my_metrics.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("Data saved successfully to my_metrics.pkl")

