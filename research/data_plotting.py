# %%
import pickle
from matplotlib import pyplot as plt

# %%
with open("recorder_13.pkl", "rb") as f:
    recorder = pickle.load(f)
# %%
recorder.keys()
# %%
# Load the three recorder files
with open("recorder_12.pkl", "rb") as f:
    recorder1 = pickle.load(f)
with open("recorder_8.pkl", "rb") as f:
    recorder2 = pickle.load(f)
with open("recorder_3_1.pkl", "rb") as f:
    recorder3 = pickle.load(f)

# Create subplots for each metric
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(20, 10))

# Plot test loss
ax1.plot(recorder1["test_step"], recorder1["test_loss"], label="EL2N 25%")
ax1.plot(recorder2["test_step"], recorder2["test_loss"], label="Full Data")
ax1.plot(recorder3["test_step"], recorder3["test_loss"], label="Adaptive")
ax1.set_title("Test Loss")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.legend()

# Plot test accuracy
ax2.plot(recorder1["test_step"], recorder1["test_acc"], label="EL2N 25%")
ax2.plot(recorder2["test_step"], recorder2["test_acc"], label="Full Data")
ax2.plot(recorder3["test_step"], recorder3["test_acc"], label="Adaptive")
ax2.set_title("Test Accuracy")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Accuracy")
ax2.legend()

fig.suptitle("CIFAR10 - 15% noise", fontsize=16)
plt.tight_layout()
plt.show()

# %%
recorder_dict = {
    "noise_10": {
        "el2n_25": "recorder_11.pkl",
        "full_data": "recorder_7.pkl",
        "adaptive": "recorder5_1.pkl",
    },
    "noise_15": {
        "el2n_25": "recorder_12.pkl",
        "full_data": "recorder_8.pkl",
        "adaptive": "recorder_3_1.pkl",
    },
    "noise_20": {
        "el2n_25": "recorder_13.pkl",
        "full_data": "recorder_9.pkl",
        "adaptive": "recorder_4.pkl",
    },
    "noise_30": {
        "el2n_25": "recorder_14.pkl",
        "full_data": "recorder_10.pkl",
        "adaptive": "recorder_6_1.pkl",
    },
}
recorders = {}
for noise_level, files in recorder_dict.items():
    recorders[noise_level] = {}
    for key, filename in files.items():
        with open(filename, "rb") as f:
            recorders[noise_level][key] = pickle.load(f)
# %%
# Calculate the average of the last 10 training accuracy values for each noise level and algorithm
points_range = 10
results = {
    noise: {
        key: {
            "avg": sum(recorder["test_acc"][-points_range:]) / (points_range),
            "max": max(recorder["test_acc"][-points_range:]),
            "min": min(recorder["test_acc"][-points_range:]),
        }
        for key, recorder in recorders[noise].items()
    }
    for noise in recorder_dict.keys()
}

# Create a grouped histogram
labels = list(results[next(iter(results))].keys())  # Get algorithm names
averages = {label: [] for label in labels}

for noise in results:
    for label in labels:
        averages[label].append(results[noise][label])
# Create subplots for the histogram
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(recorders))  # X locations for the groups
width = 0.2  # Width of the bars

# Prepare data for plotting
averages = {label: [] for label in labels}
min_values = {label: [] for label in labels}
max_values = {label: [] for label in labels}

for noise in results:
    for label in labels:
        averages[label].append(results[noise][label]["avg"])  # Use average
        min_values[label].append(results[noise][label]["min"])  # Use min
        max_values[label].append(results[noise][label]["max"])  # Use max

colors = {
    "el2n_25": "blue",
    "full_data": "orange",
    "adaptive": "green",
}

# Plot mean with error bars for min and max
for i, label in enumerate(labels):
    # Plot average
    ax.plot(x, averages[label], "o-", label=f"{label} Avg", color=colors[label])
    # Plot min values
    ax.plot(x, min_values[label], "s--", label=f"{label} Min", color=colors[label])
    # Plot max values
    ax.plot(x, max_values[label], "d:", label=f"{label} Max", color=colors[label])

ax.set_title("Average, Min, and Max Training Accuracy by Noise Level")
ax.set_ylabel("Test Accuracy")
ax.set_xlabel("Noise Levels (%)")
ax.set_xticks(x)
ax.set_xticklabels(list(results.keys()))  # Update x-tick labels to noise levels
ax.legend()
plt.show()
# %%
recorders["noise_20"]["adaptive"].keys()
# %%
x = recorders["noise_20"]["adaptive"]["prune_steps"]
y = recorders["noise_20"]["adaptive"]["pruned_percentages"]

plt.plot(x, y)
plt.xlabel("Prune Steps")
plt.ylabel("Pruned Percentages")
plt.title("Prune Steps vs Pruned Percentages")
plt.show()


# %%
sum(y) / len(y)
# %%
