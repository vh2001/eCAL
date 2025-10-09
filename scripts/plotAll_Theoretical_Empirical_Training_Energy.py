import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # For custom legend
import pandas as pd # For reading CSV
# --- PyTorch Imports ---
import torch.nn as nn
import torch.optim as optim

import pickle

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
P_idle_GPU_watts = 0.0003   # Watts (e.g., system idle power for M2 MacBook) usding sudo powermetrics all
P_peak_GPU_watts = 9.73  # Watts (e.g., peak system power for M2 MacBook under heavy load)
mid_utilization_factor = 0.6 # Hypothetical mid-level utilization for 'proportional' estimate

models = ["MLP", "CNN", "KAN", "Transformer"]

loaded_data = {}
with open('./results/my_metrics.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Now you can access your original dictionaries
all_training_times_v = loaded_data['training_times']
all_training_losses_v = loaded_data['training_losses']
all_r2_scores_v = loaded_data['r2_scores']
energy_fundamental_avg_joules_v = loaded_data['energy_fundamental']
energy_baseline_idle_joules_v = loaded_data['energy_baseline']
energy_worst_case_peak_joules_v = loaded_data['energy_worst_case']
energy_proportional_mid_U_joules_v = loaded_data['energy_proportional']
code_carbon_joules_v = loaded_data['code_carbon']

all_training_times = OrderedDict()
all_energy_consumed_joules = OrderedDict()
all_training_losses = OrderedDict()
all_r2_scores = OrderedDict()
energy_fundamental_avg_joules = OrderedDict()
energy_baseline_idle_joules = OrderedDict()
energy_worst_case_peak_joules = OrderedDict()
energy_proportional_mid_U_joules = OrderedDict()
code_carbon_joules = OrderedDict()

all_training_times_std = OrderedDict()
all_energy_consumed_joules_std = OrderedDict()
all_training_losses_std = OrderedDict()
all_r2_scores_std = OrderedDict()
energy_fundamental_avg_joules_std = OrderedDict()
energy_baseline_idle_joules_std = OrderedDict()
energy_worst_case_peak_joules_std = OrderedDict()
energy_proportional_mid_U_joules_std = OrderedDict()
code_carbon_joules_std = OrderedDict()


for num_layers in layer_range:
    print(f"\n===== Training Models with {num_layers} Layers =====")
    for model in models:
        model_name = f"{model}_{num_layers}L"

        all_training_times[model_name] = np.mean(all_training_times_v[model_name])
        all_training_losses[model_name] = np.mean(all_training_losses_v[model_name])
        all_r2_scores[model_name] = np.mean(all_r2_scores_v[model_name])
        energy_fundamental_avg_joules[model_name] = np.mean(energy_fundamental_avg_joules_v[model_name])
        energy_baseline_idle_joules[model_name] = np.mean(energy_baseline_idle_joules_v[model_name])
        energy_worst_case_peak_joules[model_name] = np.mean(energy_worst_case_peak_joules_v[model_name])
        energy_proportional_mid_U_joules[model_name] = np.mean(energy_proportional_mid_U_joules_v[model_name])
        code_carbon_joules[model_name] = np.mean(code_carbon_joules_v[model_name])

        all_training_times_std[model_name] = np.std(all_training_times_v[model_name])
        all_training_losses_std[model_name] = np.std(all_training_losses_v[model_name])
        all_r2_scores_std[model_name] = np.std(all_r2_scores_v[model_name])
        energy_fundamental_avg_joules_std[model_name] = np.std(energy_fundamental_avg_joules_v[model_name])
        energy_baseline_idle_joules_std[model_name] = np.std(energy_baseline_idle_joules_v[model_name])
        energy_worst_case_peak_joules_std[model_name] = np.std(energy_worst_case_peak_joules_v[model_name])
        energy_proportional_mid_U_joules_std[model_name] = np.std(energy_proportional_mid_U_joules_v[model_name])
        code_carbon_joules_std[model_name] = np.std(code_carbon_joules_v[model_name])

       
print("Data loaded successfully!")
print("Example loaded data:", all_training_times)

# --- NEW: Load eCAL data from CSV ---
csv_file_name = './results/energy_consumption_models_small_theor.csv'
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
csv_file_name = './results/energy_consumption_models_small_calflops.csv'
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
            calflops_energy_data['MLP'][num_layers] = row['MLP']
            calflops_energy_data['CNN'][num_layers] = row['CNN']
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
    'Code Carbon P*t': list(energy_proportional_mid_U_joules.values()),
    'Code Carbon E': list(code_carbon_joules.values()), # Optional: Include CodeCarbon data
    'Worst-Case Peak': list(energy_worst_case_peak_joules.values())
}

energy_metrics_data_std = {
    'Baseline Idle': list(energy_baseline_idle_joules_std.values()),
    'Fundamental Physics': list(energy_fundamental_avg_joules_std.values()),
    'Code Carbon P*t': list(energy_proportional_mid_U_joules_std.values()),
    'Code Carbon E': list(code_carbon_joules_std.values()), # Optional: Include CodeCarbon data
    'Worst-Case Peak': list(energy_worst_case_peak_joules_std.values()),
    'eCAL': [0] * len(energy_worst_case_peak_joules_std),
    'calflops': [0] * len(energy_worst_case_peak_joules_std)
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
    'eCAL': 'tab:cyan', # New color for eCAL data
    'calflops': 'tab:brown', # New color for eCAL data
    'Fundamental Physics': 'tab:blue',
    'Code Carbon P*t': 'tab:purple',
    'Code Carbon E': 'tab:orange',
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

plt.bar(model_labels, list(all_training_times.values()), yerr=list(all_training_times_std.values()), color=plot_colors_for_model_types)
plt.title('Total Training Time Across All Regression Models by Layer Count', fontsize=18)
plt.xlabel('Model Architecture and Layers', fontsize=14)
plt.ylabel('Training Time (seconds)', fontsize=14)
plt.xticks(rotation=60, ha='right', fontsize=10) # Rotate labels for readability
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(handles=model_type_legend_handles, title="Model Type", loc='upper left')
plt.tight_layout()
plt.savefig('./results/time_bar_chart.png', dpi=300)


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
        'Code Carbon P*t': list(energy_proportional_mid_U_joules.values()),
        'eCAL':   ecal_values_for_plot, # Add eCAL data
        'calflops':   calflops_values_for_plot, # Add calflops data
        'Code Carbon E': list(code_carbon_joules.values()), # Optional: Include CodeCarbon data
        'Worst-Case Peak': list(energy_worst_case_peak_joules.values())
    }
else:
    energy_metric_names_extended = list(energy_metrics_data.keys()) # If no eCAL data, use original list

# Save energy metrics data for plotting

# Create a DataFrame from means
df_mean = pd.DataFrame(energy_metrics_data, index=model_labels)
df_mean.index.name = "Model"

# Create a DataFrame from std deviations
df_std = pd.DataFrame(energy_metrics_data_std, index=model_labels)
df_std.index.name = "Model"

# Combine them: e.g., one column per metric with mean ± std
df_combined = pd.DataFrame(index=model_labels)
for metric in energy_metrics_data.keys():
    df_combined[f"{metric} (mean)"] = df_mean[metric]
    df_combined[f"{metric} (std)"] = df_std[metric]

# Save to CSV
out_csv = "./results/energy_metrics_summary.csv"
df_combined.to_csv(out_csv)
print(f"Saved: {out_csv}")
print(df_combined.head())


n_models = len(model_labels)
n_metrics_extended = len(energy_metric_names_extended)
bar_width = 0.15 # Reduced bar width for more metrics
n_groups = len(model_labels)
group_spacing_factor = 1.5 # Adjust this value for more/less space
index = np.arange(n_groups) * group_spacing_factor # Creates positions [0, 1.5, 3.0, ...]


for i, metric_name in enumerate(energy_metric_names_extended):
    # Calculate offset for grouped bars
    offset = (i - (n_metrics_extended - 1) / 2) * bar_width
    # Filter out NaN values for plotting
    bars_to_plot = [val if not np.isnan(val) else 0 for val in energy_metrics_data[metric_name]]
    whiskers = [val if not np.isnan(val) else 0 for val in energy_metrics_data_std[metric_name]]
    ax.bar(index + offset, bars_to_plot, bar_width, yerr=whiskers, 
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
ax.set_yscale('log')
plt.savefig('./results/energy_consumption_bar_chart.png', dpi=300)


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
plt.axhline(y=overall_avg_proportional_energy, color=energy_metric_colors['Code Carbon P*t'], linestyle=':', alpha=0.7,
            label=f'Overall Avg Code Carbon P*t ({overall_avg_proportional_energy:.2f} J)')
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
plt.yscale('log')
plt.savefig('./results/energy_consumption_line_chart.png', dpi=300)



# --- Plot 4: R^2 Score for All Models (Bar Plot) ---
plt.figure(figsize=(18, 9))
plt.bar(model_labels, list(all_r2_scores.values()), list(all_r2_scores_std.values()), color=plot_colors_for_model_types)
plt.title('R^2 Score on Training Data Across All Regression Models by Layer Count', fontsize=18)
plt.xlabel('Model Architecture and Layers', fontsize=14)
plt.ylabel('R^2 Score', fontsize=14)
plt.xticks(rotation=60, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(bottom=0, top=1.05) # R^2 is typically between 0 and 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(handles=model_type_legend_handles, title="Model Type", loc='upper left')
plt.tight_layout()
plt.savefig('./results/r_squared_bar_chart.png', dpi=300)
