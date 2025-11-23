from ultralytics import YOLO
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

# === CONFIG ===
MODEL = "train20"
DATA = "data"
IMGSZ = 1280
CONF = 0.30
MODEL_PATH = f"runs/detect/{MODEL}/weights/best.pt"
TEST_FILE = Path(f"data/test.txt")
IMAGES_DIR = Path(f"data/images/test")
LABELS_DIR = Path(f"data/labels/test")
CSV_OUT = f"runs/detect/{MODEL}/test_eval_metrics_{MODEL}.csv"
PLOTS_DIR = Path(f"runs/detect/{MODEL}/test_eval_plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load trained YOLO model
model = YOLO(MODEL_PATH)

# Utility: count true colonies from YOLO label .txt
def count_objects_in_label(label_path):
    try:
        with open(label_path, "r") as f:
            return len([line for line in f if line.strip()])
    except FileNotFoundError:
        return 0

# sMAPE helper
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(
        np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8)
    )

# Collect predictions and ground truth
y_true, y_pred, stems = [], [], []
for stem in open(TEST_FILE):
    stem = stem.strip()
    img_path = IMAGES_DIR / f"{stem}.jpg"
    label_path = LABELS_DIR / f"{stem}.txt"

    true_count = count_objects_in_label(label_path)
    results = model.predict(img_path, imgsz=IMGSZ, conf=CONF, verbose=False)
    pred_count = len(results[0].boxes)

    y_true.append(true_count)
    y_pred.append(pred_count)
    stems.append(stem)  # keep track of filenames

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# === Identify Top 5 Outliers ===
errors = np.abs(y_pred - y_true)
df_outliers = pd.DataFrame({
    "image": stems,
    "true_count": y_true,
    "pred_count": y_pred,
    "error": errors
})
top5 = df_outliers.sort_values("error", ascending=False).head(5)

print("\n=== Top 5 Outliers (by absolute error) ===")
print(top5.to_string(index=False))

# === Global metrics ===
mae = mean_absolute_error(y_true, y_pred)
bias = np.mean(y_pred - y_true)
mse = mean_squared_error(y_true, y_pred)
msle = mean_squared_log_error(y_true, y_pred)
smape_val = smape(y_true, y_pred)
mean_true_all = np.mean(y_true)
mae_pct_all = 100 * mae / (mean_true_all + 1e-8)

print("=== Overall Test Metrics ===")
print(f"MAE   : {mae:.3f} ({mae_pct_all:.2f}% of mean true {mean_true_all:.1f})")
print(f"Bias  : {bias:.3f}")
print(f"MSE   : {mse:.3f}")
print(f"MSLE  : {msle:.3f}")
print(f"sMAPE : {smape_val:.2f}%\n")

# === Binned metrics ===
bins = [
    (0, 3),    # 0–2
    (3, 6),    # 3–5
    (6, 11),   # 6–10
    (11, 21),  # 11–20
    (21, 51),  # 21–50
    (50, 150),
    (150, 300),
    (300, 1e9)
]

records = []
print("=== Binned Metrics (by true colony count) ===")
for low, high in bins:
    mask = (y_true >= low) & (y_true < high)
    if np.any(mask):
        mae_bin = mean_absolute_error(y_true[mask], y_pred[mask])
        bias_bin = np.mean(y_pred[mask] - y_true[mask])
        mse_bin = mean_squared_error(y_true[mask], y_pred[mask])
        msle_bin = mean_squared_log_error(y_true[mask], y_pred[mask])
        smape_bin = smape(y_true[mask], y_pred[mask])
        mean_true = np.mean(y_true[mask])

        # percentage metrics
        mae_pct = 100 * mae_bin / (mean_true + 1e-8)

        print(f"{low:>3}–{high-1:<3} colonies "
              f"(n={mask.sum()}, mean_true={mean_true:.1f}): "
              f"MAE={mae_bin:.3f} ({mae_pct:.2f}%), "
              f"Bias={bias_bin:.3f}, "
              f"MSE={mse_bin:.3f}, MSLE={msle_bin:.3f}, "
              f"sMAPE={smape_bin:.2f}%")

        records.append({
            "bin": f"{low}-{high-1}",
            "n_samples": mask.sum(),
            "mean_true": mean_true,
            "MAE": mae_bin,
            "MAE_%": mae_pct,
            "Bias": bias_bin,
            "MSE": mse_bin,
            "MSLE": msle_bin,
            "sMAPE": smape_bin
        })

# Add overall metrics as a row 
records.append({
    "bin": "Overall",
    "n_samples": len(y_true),
    "mean_true": mean_true_all,
    "MAE": mae,
    "MAE_%": mae_pct_all,
    "Bias": bias,
    "MSE": mse,
    "MSLE": msle,
    "sMAPE": smape_val
})

# Save to CSV
df = pd.DataFrame(records)
df.to_csv(CSV_OUT, index=False)
print(f"\nSaved metrics to {CSV_OUT}")

# === PLOTS ===
# 1. Scatter plot (True vs Predicted counts)
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')  # diagonal
plt.xlabel("True Colony Count")
plt.ylabel("Predicted Colony Count")
plt.title("True vs Predicted Colony Counts")
plt.savefig(PLOTS_DIR / "scatter_true_vs_pred.png")
plt.close()

# 2. Histogram of errors
errors = y_pred - y_true
plt.figure(figsize=(6,4))
plt.hist(errors, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Prediction Error (Pred - True)")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Errors")
plt.savefig(PLOTS_DIR / "hist_errors.png")
plt.close()

# 3. Boxplot of absolute error per bin (labels include n_samples)
abs_errors = np.abs(y_pred - y_true)
bin_labels = []
bin_errors = []
for low, high in bins:
    mask = (y_true >= low) & (y_true < high)
    if np.any(mask):
        n_samples = mask.sum()
        bin_labels.append(f"{low}-{high-1} (n={n_samples})")
        bin_errors.append(abs_errors[mask])

plt.figure(figsize=(8,5))
plt.boxplot(bin_errors, labels=bin_labels, showfliers=False)
plt.xlabel("True Colony Count Range")
plt.ylabel("Absolute Error")
plt.title("Absolute Error by Colony Count Range")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "boxplot_abs_error_bins.png")
plt.close()

print(f"Saved plots to {PLOTS_DIR}")

