import read_sim_data_for_training as reader
import matplotlib.pyplot as plt

data_raw_path = "//datastore.tekniker.es/ia/data-analytics/KUBIBIT/QML/dataset/raw_dendrites_juan"
idxToSave = [1500, 4000]
exp_id = 5

X, Y, _ = reader.read_simulations_nad_get_XY(None, data_raw_path, idxToSave, exp_id, sep = ";")

fig, axes = plt.subplots(1, 2, figsize=(6,6))
# ---- Top row: op (case = 0) ----
axes[0].imshow(X[0, :, :, 0])
axes[0].set_title("X --- op (t=1500)")
axes[0].axis("off")

axes[1].imshow(Y[0, :, :, 0])
axes[1].set_title("Y --- op (t=4000)")
axes[1].axis("off")
plt.suptitle(f"Experiment ID: {exp_id}")
plt.tight_layout()
plt.show()