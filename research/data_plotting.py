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
with open("recorder_13.pkl", "rb") as f:
    recorder1 = pickle.load(f)
with open("recorder_9.pkl", "rb") as f:
    recorder2 = pickle.load(f)
with open("recorder_4.pkl", "rb") as f:
    recorder3 = pickle.load(f)

# Create subplots for each metric
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot training loss
ax1.plot(recorder1["train_step"], recorder1["train_loss"], label="EL2N 25%")
ax1.plot(recorder2["train_step"], recorder2["train_loss"], label="Full Data")
ax1.plot(recorder3["train_step"], recorder3["train_loss"], label="Adaptive")
ax1.set_title("Training Loss")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.legend()

# Plot training accuracy
ax2.plot(recorder1["train_step"], recorder1["train_acc"], label="EL2N 25%")
ax2.plot(recorder2["train_step"], recorder2["train_acc"], label="Full Data")
ax2.plot(recorder3["train_step"], recorder3["train_acc"], label="Adaptive")
ax2.set_title("Training Accuracy")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Accuracy")
ax2.legend()

# Plot test loss
ax3.plot(recorder1["test_step"], recorder1["test_loss"], label="EL2N 25%")
ax3.plot(recorder2["test_step"], recorder2["test_loss"], label="Full Data")
ax3.plot(recorder3["test_step"], recorder3["test_loss"], label="Adaptive")
ax3.set_title("Test Loss")
ax3.set_xlabel("Steps")
ax3.set_ylabel("Loss")
ax3.legend()

# Plot test accuracy
ax4.plot(recorder1["test_step"], recorder1["test_acc"], label="EL2N 25%")
ax4.plot(recorder2["test_step"], recorder2["test_acc"], label="Full Data")
ax4.plot(recorder3["test_step"], recorder3["test_acc"], label="Adaptive")
ax4.set_title("Test Accuracy")
ax4.set_xlabel("Steps")
ax4.set_ylabel("Accuracy")
ax4.legend()

plt.tight_layout()
plt.show()
