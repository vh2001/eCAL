import numpy as np
import time
import calculators.ToyModels as toy_models
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # For custom legend
import pandas as pd # For reading CSV
from codecarbon import EmissionsTracker
from sklearn.metrics import r2_score
# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# set fp32 precision for better comparability
torch.set_default_dtype(torch.float32)

# --- 1. Parameters for the data and model ---
num_time_series = 204
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
assumed_average_power_watts = 0.033 # Watts.measured with tthe avg power script
print(f"ASSUMED AVERAGE POWER FOR ENERGY CALCULATION: {assumed_average_power_watts} Watts (Crucial Placeholder!)")
P_idle_GPU_watts = 0.003   # Watts (e.g., system idle power for M2 MacBook) usding sudo powermetrics all
P_peak_GPU_watts = 10  # Watts (e.g., peak system power for M2 MacBook under heavy load)
mid_utilization_factor = 0.6 # Hypothetical mid-level utilization for 'proportional' estimate

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
    emissions = local_tracker.stop() # Stop the local tracker

    df = pd.read_csv("emissions.csv")
    latest = df.iloc[-1]
    data = {}
    if emissions:
        data = {
            "gpu_energy_Wh": latest["gpu_energy"] * 1000,
            "cpu_energy_Wh": latest["cpu_energy"] * 1000,
            "ram_energy_Wh": latest["ram_energy"] * 1000,
            "energy_consumed_Wh": latest["energy_consumed"] * 1000,
            "duration_seconds": latest["duration"] / 3600,
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
all_training_times = OrderedDict()
all_energy_consumed_joules = OrderedDict()
all_training_losses = OrderedDict()
all_r2_scores = OrderedDict()

# NEW: Separate OrderedDicts for each energy metric
energy_fundamental_avg_joules = OrderedDict()
energy_baseline_idle_joules = OrderedDict()
energy_worst_case_peak_joules = OrderedDict()
energy_proportional_mid_U_joules = OrderedDict()
code_carbon_joules = OrderedDict()

# --- 6. Loop through layers and then model types ---
print("\n--- Starting Training for all models across various layer configurations ---")

for num_layers in layer_range:
    print(f"\n===== Training Models with {num_layers} Layers =====")

    # --- MLP (scikit-learn) ---
    hidden_layer_config = tuple([nodes_per_layer] * num_layers)
    model_name = f"MLP_{num_layers}L"
    print(f"\n  Training {model_name} (MLP with {num_layers} hidden layers ({hidden_layer_config} nodes per layer))...")

    mlp_model = SimpleMLP(input_size=time_series_length, hidden_size=nodes_per_layer, output_size=1, num_layers=num_layers).to(device)
    data, mlp_final_loss, mlp_r2 = train_pytorch_model(mlp_model, train_loader, X_tensor, y_tensor, num_epochs=num_epochs_pytorch)
    fundamental_energy = assumed_average_power_watts * data["duration_seconds"] # Convert to Joules
    idle_energy = P_idle_GPU_watts * data["duration_seconds"] # Convert to Joules
    peak_energy = P_peak_GPU_watts * data["duration_seconds"] # Convert to Joules
    proportional_energy = (P_idle_GPU_watts + mid_utilization_factor * (P_peak_GPU_watts - P_idle_GPU_watts)) * data["duration_seconds"] # Convert to Joules

    all_training_times[model_name] = data["duration_seconds"]
    all_training_losses[model_name] = mlp_final_loss
    all_r2_scores[model_name] = mlp_r2
    energy_fundamental_avg_joules[model_name] = fundamental_energy
    energy_baseline_idle_joules[model_name] = idle_energy
    energy_worst_case_peak_joules[model_name] = peak_energy
    energy_proportional_mid_U_joules[model_name] = proportional_energy
    code_carbon_joules[model_name] = data["energy_consumed_Wh"] 

    print(f"    Training completed in: {all_training_times[model_name]:.4f} seconds")
    print(f"    Fundamental Physics Energy: {fundamental_energy:.4f} J")
    print(f"    Baseline Idle Energy: {idle_energy:.4f} J")
    print(f"    Worst-Case Peak Energy: {peak_energy:.4f} J")
    print(f"    Proportional Mid-U Energy: {proportional_energy:.4f} J")
    print(f"    Code Carbon: {code_carbon_joules[model_name]:.4f} J")
    print(f"    Final Loss: {mlp_final_loss:.4f}")
    print(f"    R^2 score (on training data): {mlp_r2:.4f}")

    # --- CNN (PyTorch) ---
    cnn_model = SimpleCNN(output_size=1, hidden_channels=16, num_layers=num_layers).to(device)
    model_name = f"CNN_{num_layers}L"
    print(f"\n  Training {model_name} (CNN with {num_layers} dense layers)...")
    print(f"    Number of parameters: {sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)}")

    data, cnn_final_loss, cnn_r2 = train_pytorch_model(cnn_model, train_loader, X_tensor, y_tensor, num_epochs=num_epochs_pytorch)
    fundamental_energy = assumed_average_power_watts * data["duration_seconds"] # Convert to Joules
    idle_energy = P_idle_GPU_watts * data["duration_seconds"] # Convert to Joules
    peak_energy = P_peak_GPU_watts * data["duration_seconds"] # Convert to Joules
    proportional_energy = (P_idle_GPU_watts + mid_utilization_factor * (P_peak_GPU_watts - P_idle_GPU_watts)) * data["duration_seconds"] # Convert to Joules

    all_training_times[model_name] = data["duration_seconds"]
    all_training_losses[model_name] = cnn_final_loss
    all_r2_scores[model_name] = cnn_r2
    energy_fundamental_avg_joules[model_name] = fundamental_energy
    energy_baseline_idle_joules[model_name] = idle_energy
    energy_worst_case_peak_joules[model_name] = peak_energy
    energy_proportional_mid_U_joules[model_name] = proportional_energy
    code_carbon_joules[model_name] = data["energy_consumed_Wh"] 

    print(f"    Training completed in: {all_training_times[model_name]:.4f} seconds")
    print(f"    Fundamental Physics Energy: {fundamental_energy:.4f} J")
    print(f"    Baseline Idle Energy: {idle_energy:.4f} J")
    print(f"    Worst-Case Peak Energy: {peak_energy:.4f} J")
    print(f"    Proportional Mid-U Energy: {proportional_energy:.4f} J")
    print(f"    Code Carbon: {code_carbon_joules[model_name]:.4f} J")
    print(f"    Final Loss: {cnn_final_loss:.4f}")
    print(f"    R^2 score (on training data): {cnn_r2:.4f}")


    # # --- KAN-like (PyTorch) ---
    kan_model = KANLikeRegressor(
        input_dim=time_series_length,
        num_hidden_layers_per_feature_fn=num_layers, # Varying sub-layers
        nodes_per_layer_in_sub_fn=nodes_per_layer
    ).to(device)
    kan_model_name = f"KAN_{num_layers}L"
    print(f"\n  Training {kan_model_name} (KAN-like with {num_layers} hidden layers per feature function)...")
    print(f"    Number of parameters: {sum(p.numel() for p in kan_model.parameters() if p.requires_grad)}")

    data, kan_final_loss, kan_r2 = train_pytorch_model(kan_model, train_loader, X_tensor, y_tensor, num_epochs=num_epochs_pytorch)

    fundamental_energy = assumed_average_power_watts * data["duration_seconds"] # Convert to Joules
    idle_energy = P_idle_GPU_watts * data["duration_seconds"] # Convert to Joules
    peak_energy = P_peak_GPU_watts * data["duration_seconds"] # Convert to Joules
    proportional_energy = (P_idle_GPU_watts + mid_utilization_factor * (P_peak_GPU_watts - P_idle_GPU_watts)) * data["duration_seconds"] # Convert to Joules

    all_training_times[kan_model_name] = data["duration_seconds"]
    all_training_losses[kan_model_name] = kan_final_loss
    all_r2_scores[kan_model_name] = kan_r2
    energy_fundamental_avg_joules[kan_model_name] = fundamental_energy
    energy_baseline_idle_joules[kan_model_name] = idle_energy
    energy_worst_case_peak_joules[kan_model_name] = peak_energy
    energy_proportional_mid_U_joules[kan_model_name] = proportional_energy
    code_carbon_joules[kan_model_name] = data['energy_consumed_Wh'] # Corrected model_name key

    print(f"    Training completed in: {data['duration_seconds']:.4f} seconds")
    print(f"    Fundamental Physics Energy: {fundamental_energy:.4f} J")
    print(f"    Baseline Idle Energy: {idle_energy:.4f} J")
    print(f"    Worst-Case Peak Energy: {peak_energy:.4f} J")
    print(f"    Proportional Mid-U Energy: {proportional_energy:.4f} J")
    print(f"    Code Carbon: {code_carbon_joules[kan_model_name]:.4f} J")
    print(f"    Final Loss: {kan_final_loss:.4f}")
    print(f"    R^2 score (on training data): {kan_r2:.4f}")

    transformer_model = toy_models.Transformer(
        num_tokens=10,
        num_token_vals=20, #nodes per layer
        num_emb=10, #* 2,
        num_neurons=10,
        num_heads=2,
        num_blocks=num_layers, #L
        device=device).to(device)

    # transformer_model = toy_models.Transformer(
    #     num_tokens=time_series_length,
    #     num_token_vals=20,
    #     num_emb=nodes_per_layer, #* 2,
    #     num_neurons=nodes_per_layer,
    #     num_heads=2,
    #     num_blocks=num_layers,
    #     device=device
    # ).to(device)
    transformer_model_name = f"Transformer_{num_layers}L"
    print(f"\n  Training {transformer_model_name} (Transformer with {num_layers} encoder layers)...")
    print(f"    Number of parameters: {sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)}")

    data, transformer_final_loss, transformer_r2 = train_pytorch_model(transformer_model, train_loader, 
                                                                       X_tensor, y_tensor, num_epochs=num_epochs_pytorch)

    fundamental_energy = assumed_average_power_watts * data["duration_seconds"] # Convert to Joules
    idle_energy = P_idle_GPU_watts * data["duration_seconds"] # Convert to Joules
    peak_energy = P_peak_GPU_watts * data["duration_seconds"] # Convert to Joules
    proportional_energy = (P_idle_GPU_watts + mid_utilization_factor * (P_peak_GPU_watts - P_idle_GPU_watts)) * data["duration_seconds"] # Convert to Joules

    all_training_times[transformer_model_name] = data["duration_seconds"]
    all_training_losses[transformer_model_name] = transformer_final_loss
    all_r2_scores[transformer_model_name] = transformer_r2
    energy_fundamental_avg_joules[transformer_model_name] = fundamental_energy
    energy_baseline_idle_joules[transformer_model_name] = idle_energy
    energy_worst_case_peak_joules[transformer_model_name] = peak_energy
    energy_proportional_mid_U_joules[transformer_model_name] = proportional_energy
    code_carbon_joules[transformer_model_name] = data['energy_consumed_Wh'] # Corrected model_name key

    print(f"    Training completed in: {data['duration_seconds']:.4f} seconds")
    print(f"    Fundamental Physics Energy: {fundamental_energy:.4f} J")
    print(f"    Baseline Idle Energy: {idle_energy:.4f} J")
    print(f"    Worst-Case Peak Energy: {peak_energy:.4f} J")
    print(f"    Proportional Mid-U Energy: {proportional_energy:.4f} J")
    print(f"    Code Carbon: {code_carbon_joules[transformer_model_name]:.4f} J")
    print(f"    Final Loss: {transformer_final_loss:.4f}")
    print(f"    R^2 score (on training data): {transformer_r2:.4f}")


# --- 10. Final Summary ---
print("\n--- FINAL SUMMARY of Training Times, Energy, and Performance ---")
# Fixed printing error in the header string: "Code Carbon" "Final Loss" -> "Code Carbon (J)", "Final Loss"
print("{:<20} {:<15} {:<20} {:<20} {:<20} {:<20} {:<15} {:<15} {:<15}".format(
    "Model", "Train Time (s)", "Baseline Idle (J)", "Emp. Avg. Energy (J)",
    "Prop. Mid-U (J)", "Code Carbon (J)", "Worst-Case Peak (J)", "Final Loss", "R^2 Score"
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


# --- NEW: Load eCAL data from CSV ---
csv_file_name = 'energy_consumption_models_small_theor.csv'
try:
    ecal_energy_df = pd.read_csv(csv_file_name)
    print(f"\nLoaded eCAL data from {csv_file_name}:\n{ecal_energy_df}")
    print(f"Detected columns in CSV: {ecal_energy_df.columns.tolist()}") # Helpful for debugging!

    # Store eCAL data in a structured dictionary for easy plotting
    ecal_energy_data = {
        'MLP': OrderedDict(),
        'CNN': OrderedDict(),
        'KAN': OrderedDict(),
        'Transformer': OrderedDict()
    }

    # CSV columns: 'Unnamed: 0' (for layers), 'Ignore', 'MLP', 'CNN', 'KAN', 'Transformer'
    for index, row in ecal_energy_df.iterrows():
        # FIX: Access the layers column using 'Unnamed: 0' as the header name
        num_layers = int(row['Unnamed: 0']) # Changed from 'Layers'
        if num_layers in layer_range: # Only consider layers we've simulated
            # Ensure these column names ('MLP', 'CNN', 'KAN', 'Transformer')
            # are also exactly as they appear in your CSV header.
            ecal_energy_data['MLP'][num_layers] = row['MLP']
            ecal_energy_data['CNN'][num_layers] = row['CNN']
            ecal_energy_data['KAN'][num_layers] = row['KAN']
            ecal_energy_data['Transformer'][num_layers] = row['Transformer']

    print("\nParsed eCAL energy data:")
    for model_type, data in ecal_energy_data.items():
        print(f"  {model_type}: {data}")

except FileNotFoundError:
    print(f"\nWARNING: CSV file '{csv_file_name}' not found. Skipping eCAL data loading and plotting.")
    ecal_energy_data = {
        'MLP': OrderedDict(), 'CNN': OrderedDict(),
        'KAN': OrderedDict(), 'Transformer': OrderedDict()
    } # Empty data if file not found


# --- NEW: Load eCAL data from CSV ---
csv_file_name = 'energy_consumption_models_small_calflops.csv'
try:
    ecal_energy_df = pd.read_csv(csv_file_name)
    print(f"\nLoaded eCAL data from {csv_file_name}:\n{ecal_energy_df}")
    print(f"Detected columns in CSV: {ecal_energy_df.columns.tolist()}") # Helpful for debugging!

    # Store eCAL data in a structured dictionary for easy plotting
    calflops_energy_data = {
        'MLP': OrderedDict(),
        'CNN': OrderedDict(),
        'KAN': OrderedDict(),
        'Transformer': OrderedDict()
    }

    # CSV columns: 'Unnamed: 0' (for layers), 'Ignore', 'MLP', 'CNN', 'KAN', 'Transformer'
    for index, row in ecal_energy_df.iterrows():
        # FIX: Access the layers column using 'Unnamed: 0' as the header name
        num_layers = int(row['Unnamed: 0']) # Changed from 'Layers'
        if num_layers in layer_range: # Only consider layers we've simulated
            # Ensure these column names ('MLP', 'CNN', 'KAN', 'Transformer')
            # are also exactly as they appear in your CSV header.
            calflops_energy_data['MLP'][num_layers] = row['MLP_practical']
            calflops_energy_data['CNN'][num_layers] = row['CNN_practical']
            calflops_energy_data['KAN'][num_layers] = row['KAN']
            calflops_energy_data['Transformer'][num_layers] = row['Transformer']

    print("\nParsed eCAL energy data:")
    for model_type, data in calflops_energy_data.items():
        print(f"  {model_type}: {data}")

except FileNotFoundError:
    print(f"\nWARNING: CSV file '{csv_file_name}' not found. Skipping eCAL data loading and plotting.")
    calflops_energy_data = {
        'MLP': OrderedDict(), 'CNN': OrderedDict(),
        'KAN': OrderedDict(), 'Transformer': OrderedDict()
    } # Empty data if file not found

# --- 11. Consolidated Plotting Results ---

# Prepare data for plotting
model_labels = list(all_training_times.keys()) # All model configurations as labels

# Energy values for the grouped bar chart
energy_metrics_data = {
    'Baseline Idle': list(energy_baseline_idle_joules.values()),
    'Fundamental Physics': list(energy_fundamental_avg_joules.values()),
    'Proportional Mid-U': list(energy_proportional_mid_U_joules.values()),
    'Code Carbon': list(code_carbon_joules.values()), # Optional: Include CodeCarbon data
    'Worst-Case Peak': list(energy_worst_case_peak_joules.values())
}
energy_metric_names = list(energy_metrics_data.keys())


# Define colors for each model type to make plots easier to read
model_type_colors = {
    'MLP': 'steelblue',
    'CNN': 'darkorange',
    'KAN': 'forestgreen',
    'Transformer': 'purple'
}

# Define colors for each energy metric
energy_metric_colors = {
    'Baseline Idle': 'tab:red',
    'Fundamental Physics': 'tab:blue',
    'Proportional Mid-U': 'tab:purple',
    'eCAL': 'tab:cyan', # New color for eCAL data
    'calflops': 'tab:brown', # New color for eCAL data
    'Code Carbon': 'tab:orange',
    'Worst-Case Peak': 'tab:green'
}

# Create custom legend handles for model types
model_type_legend_handles = [mpatches.Patch(color=color, label=model_type)
                             for model_type, color in model_type_colors.items()]
# Create custom legend handles for energy metrics
energy_metric_legend_handles = [mpatches.Patch(color=color, label=metric_name)
                                for metric_name, color in energy_metric_colors.items()]


# --- Plot 1: Training Time for All Models (Bar Plot) ---
plt.figure(figsize=(18, 9))
# Create a list of colors that corresponds to the order of `model_labels`
plot_colors_for_model_types = []
for label in model_labels:
    if label.startswith('MLP'):
        plot_colors_for_model_types.append(model_type_colors['MLP'])
    elif label.startswith('CNN'):
        plot_colors_for_model_types.append(model_type_colors['CNN'])
    elif label.startswith('KAN'):
        plot_colors_for_model_types.append(model_type_colors['KAN'])
    elif label.startswith('Transformer'):
        plot_colors_for_model_types.append(model_type_colors['Transformer'])
    else:
        plot_colors_for_model_types.append('gray') # Fallback

plt.bar(model_labels, list(all_training_times.values()), color=plot_colors_for_model_types)
plt.title('Total Training Time Across All Regression Models by Layer Count', fontsize=18)
plt.xlabel('Model Architecture and Layers', fontsize=14)
plt.ylabel('Training Time (seconds)', fontsize=14)
plt.xticks(rotation=60, ha='right', fontsize=10) # Rotate labels for readability
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(handles=model_type_legend_handles, title="Model Type", loc='upper left')
plt.tight_layout()
plt.savefig('time_bar_chart.png', dpi=300)
plt.show()


# --- Plot 2: Estimated Energy Consumption for All Models (Grouped Bar Plot) ---
fig, ax = plt.subplots(figsize=(20, 9)) # Increased figure width to accommodate more bars per group

# Add eCAL data to the energy_metrics_data for this grouped bar plot
# Only add if eCAL data was successfully loaded
if ecal_energy_data['MLP']: # Check if any eCAL data was loaded
    ecal_values_for_plot = []
    for label in model_labels:
        model_type = label.split('_')[0]
        num_layers = int(label.split('_')[1].replace('L', ''))
        if model_type in ecal_energy_data and num_layers in ecal_energy_data[model_type]:
            ecal_values_for_plot.append(ecal_energy_data[model_type][num_layers])
        else:
            ecal_values_for_plot.append(np.nan) # Use NaN for missing data to not plot a bar

# Add eCAL data to the energy_metrics_data for this grouped bar plot
# Only add if eCAL data was successfully loaded
if calflops_energy_data['MLP']: # Check if any eCAL data was loaded
    calflops_values_for_plot = []
    for label in model_labels:
        model_type = label.split('_')[0]
        num_layers = int(label.split('_')[1].replace('L', ''))
        if model_type in calflops_energy_data and num_layers in calflops_energy_data[model_type]:
            calflops_values_for_plot.append(calflops_energy_data[model_type][num_layers])
        else:
            calflops_values_for_plot.append(np.nan) # Use NaN for missing data to not plot a bar

    energy_metrics_data['eCAL'] = ecal_values_for_plot
    energy_metrics_data['calflops'] = calflops_values_for_plot
    energy_metric_names_extended = energy_metrics_data = {
        'Baseline Idle': list(energy_baseline_idle_joules.values()),
        'Fundamental Physics': list(energy_fundamental_avg_joules.values()),
        'Proportional Mid-U': list(energy_proportional_mid_U_joules.values()),
        'eCAL':   ecal_values_for_plot, # Add eCAL data
        'calflops':   calflops_values_for_plot, # Add calflops data
        'Code Carbon': list(code_carbon_joules.values()), # Optional: Include CodeCarbon data
        'Worst-Case Peak': list(energy_worst_case_peak_joules.values())
    }
else:
    energy_metric_names_extended = list(energy_metrics_data.keys()) # If no eCAL data, use original list


n_models = len(model_labels)
n_metrics_extended = len(energy_metric_names_extended)
bar_width = 0.15 # Reduced bar width for more metrics
index = np.arange(n_models)

for i, metric_name in enumerate(energy_metric_names_extended):
    # Calculate offset for grouped bars
    offset = (i - (n_metrics_extended - 1) / 2) * bar_width
    # Filter out NaN values for plotting
    bars_to_plot = [val if not np.isnan(val) else 0 for val in energy_metrics_data[metric_name]]
    ax.bar(index + offset, bars_to_plot, bar_width,
           label=metric_name, color=energy_metric_colors[metric_name])


ax.set_title(f'Estimated Energy Consumption Across All Regression Models by Layer Count\n'
             f'(Fundamental Physics: {assumed_average_power_watts}W, Idle: {P_idle_GPU_watts}W, Peak: {P_peak_GPU_watts}W, Mid-U: {mid_utilization_factor})', fontsize=16)
ax.set_xlabel('Model Architecture and Layers', fontsize=14)
ax.set_ylabel('Estimated Energy Consumption (Joules)', fontsize=14)
ax.set_xticks(index)
ax.set_xticklabels(model_labels, rotation=60, ha='right', fontsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.legend(title="Energy Metric", loc='upper left')

plt.tight_layout()
plt.savefig('energy_consumption_bar_chart.png', dpi=300)
plt.show()


# --- Plot 3: Estimated Energy Consumption (Line Plot) with eCAL data ---
plt.figure(figsize=(14, 8)) # Adjust size for line plot

# Group energy data by model type for line plotting (Simulated)
energy_by_model_type_sim = {model_type: OrderedDict() for model_type in model_type_colors.keys()}
for label, energy in energy_fundamental_avg_joules.items():
    for model_type in model_type_colors.keys():
        if label.startswith(model_type):
            num_layers_str = label.split('_')[-1].replace('L', '')
            energy_by_model_type_sim[model_type][int(num_layers_str)] = energy
            break

# Group CodeCarbon data by model type for line plotting
energy_by_model_type_codecarbon = {model_type: OrderedDict() for model_type in model_type_colors.keys()}
for label, energy in code_carbon_joules.items():
    for model_type in model_type_colors.keys():
        if label.startswith(model_type):
            num_layers_str = label.split('_')[-1].replace('L', '')
            energy_by_model_type_codecarbon[model_type][int(num_layers_str)] = energy
            break


# Plot Fundamental Physics Energy for each model type (simulated)
for model_type, energies in energy_by_model_type_sim.items():
    if energies: # Only plot if there's data for this model type
        layers = list(energies.keys())
        energy_vals = list(energies.values())
        plt.plot(layers, energy_vals,
                 marker='o', linestyle='-', linewidth=2,
                 color=model_type_colors[model_type],
                 label=f'{model_type} (Simulated Avg)')

# Plot eCAL Energy for each model type (from CSV), only if data was loaded
if ecal_energy_data['MLP']: # Check if any eCAL data was loaded
    for model_type, energies in ecal_energy_data.items():
        if energies: # Only plot if there's data for this model type
            layers = list(energies.keys())
            energy_vals = list(energies.values())
            # Use a slightly different linestyle or marker for eCAL data
            plt.plot(layers, energy_vals,
                     marker='x', linestyle='--', linewidth=1.5,
                     color=model_type_colors[model_type], # Keep model-specific color
                     label=f'eCAL_{model_type} (CSV)')
            
# Plot calfplops Energy for each model type (from CSV), only if data was loaded
if calflops_energy_data['MLP']: # Check if any eCAL data was loaded
    for model_type, energies in calflops_energy_data.items():
        if energies: # Only plot if there's data for this model type
            layers = list(energies.keys())
            energy_vals = list(energies.values())
            # Use a slightly different linestyle or marker for eCAL data
            plt.plot(layers, energy_vals,
                     marker='x', linestyle='--', linewidth=1.5,
                     color=model_type_colors[model_type], # Keep model-specific color
                     label=f'calflops_{model_type} (CSV)')

# Plot CodeCarbon Energy for each model type
for model_type, energies in energy_by_model_type_codecarbon.items():
    if energies: # Only plot if there's data for this model type
        layers = list(energies.keys())
        energy_vals = list(energies.values())
        # Use a distinct marker and linestyle for CodeCarbon data, but keep model-specific color
        plt.plot(layers, energy_vals,
                 marker='s', linestyle=':', linewidth=1.5,
                 color=model_type_colors[model_type], # Use model-specific color
                 label=f'CodeCarbon_{model_type} (Tracker)')


# Calculate overall average for theoretical reference lines (unchanged)
overall_avg_idle_energy = np.mean(list(energy_baseline_idle_joules.values()))
overall_avg_peak_energy = np.mean(list(energy_worst_case_peak_joules.values()))
overall_avg_proportional_energy = np.mean(list(energy_proportional_mid_U_joules.values()))


plt.axhline(y=overall_avg_idle_energy, color=energy_metric_colors['Baseline Idle'], linestyle='--', alpha=0.7,
            label=f'Overall Avg Baseline Idle ({overall_avg_idle_energy:.2f} J)')
plt.axhline(y=overall_avg_proportional_energy, color=energy_metric_colors['Proportional Mid-U'], linestyle=':', alpha=0.7,
            label=f'Overall Avg Proportional Mid-U ({overall_avg_proportional_energy:.2f} J)')
plt.axhline(y=overall_avg_peak_energy, color=energy_metric_colors['Worst-Case Peak'], linestyle='-.', alpha=0.7,
            label=f'Overall Avg Worst-Case Peak ({overall_avg_peak_energy:.2f} J)')


plt.title(f'Estimated Energy Consumption Trend by Model Type and Layers\n'
          f'(Simulated Avg: {assumed_average_power_watts}W, Idle: {P_idle_GPU_watts}W, Peak: {P_peak_GPU_watts}W, Mid-U: {mid_utilization_factor})', fontsize=16)
plt.xlabel('Number of Layers', fontsize=14)
plt.ylabel('Energy Consumption (Joules)', fontsize=14)
plt.xticks(list(layer_range), fontsize=10) # Ensure x-ticks align with layer numbers
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Energy Source/Context", loc='upper left', ncol=2, fontsize=10) # Increased columns for legend
plt.tight_layout()
plt.savefig('energy_consumption_line_chart.png', dpi=300)
plt.show()


# --- Plot 4: R^2 Score for All Models (Bar Plot) ---
plt.figure(figsize=(18, 9))
plt.bar(model_labels, list(all_r2_scores.values()), color=plot_colors_for_model_types)
plt.title('R^2 Score on Training Data Across All Regression Models by Layer Count', fontsize=18)
plt.xlabel('Model Architecture and Layers', fontsize=14)
plt.ylabel('R^2 Score', fontsize=14)
plt.xticks(rotation=60, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(bottom=0, top=1.05) # R^2 is typically between 0 and 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(handles=model_type_legend_handles, title="Model Type", loc='upper left')
plt.tight_layout()
plt.savefig('r_squared_bar_chart.png', dpi=300)
plt.show()