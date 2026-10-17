import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from gd import gradient_descent
from data import load_data, split, transform, recover_coef
from config import ALPHA_GRID, N_ITERS, TOL, DATA_PATH, RAW_ALPHA_GRID, RAW_N_ITERS, RAW_TOL

RESULT_DIR = Path("results/a")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


X, y, feature_names = load_data(DATA_PATH)

X_train, X_test, y_train, y_test = split(X, y)
X_train_scaled, params = transform(X_train, mode="standardize")
X_test_scaled, _ = transform(X_test, mode="standardize", params=params)

# Table 1: descriptive statistics on the training set
desc_rows = [
    [name, col.mean(), col.std(ddof=1), col.min(), col.max()]
    for name, col in zip(feature_names, X_train.T)
]
desc_rows.append(["Compressive Strength", y_train.mean(), y_train.std(ddof=1), y_train.min(), y_train.max()])

with open(RESULT_DIR / "descriptive_stats_train.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "mean", "std", "min", "max"])
    writer.writerows(desc_rows)
print(RESULT_DIR / "descriptive_stats_train.csv")


def evaluate(X, y, theta):
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    y_pred = X_design @ theta
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - mse / np.var(y)
    return {"mse": mse, "r2": r2}

results_by_alpha = {}
alpha_rows = []
plt.figure(figsize=(8, 5))

for alpha in ALPHA_GRID:
    res = gradient_descent(X_train_scaled, y_train, alpha=alpha, n_iters=N_ITERS, tol=TOL)
    results_by_alpha[alpha] = res
    final_loss = res["loss_history"][-1]
    res["r2"] = evaluate(X_train_scaled, y_train, res["theta"])["r2"] if np.isfinite(final_loss) else float("nan")
    if res["converged_iter"]:
        label = f"alpha={alpha} (converged at {res['converged_iter']})"
        converged_iter = res["converged_iter"]
    elif not np.isfinite(final_loss):
        label = f"alpha={alpha} (diverged)"
        converged_iter = "diverged"
    else:
        label = f"alpha={alpha} (did not converge)"
        converged_iter = "not converged"
    line, = plt.plot(res["loss_history"], label=label)
    if not np.isfinite(final_loss):
        exit_iter = next(i for i, v in enumerate(res["loss_history"]) if v > 2000)
        plt.scatter([exit_iter], [2000], color=line.get_color(), marker="x", s=80, zorder=5)
        plt.annotate(
            f"alpha={alpha} diverged\n(exceeds plot range at iter {exit_iter + 1})",
            xy=(exit_iter, 2000),
            xytext=(exit_iter + len(res["loss_history"]) * 0.05, 1500),
            arrowprops=dict(arrowstyle="->", color=line.get_color()),
            color=line.get_color(),
        )
    alpha_rows.append([alpha, converged_iter, final_loss, res["r2"]])

with open(RESULT_DIR / "alpha_convergence.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["alpha", "converged_iter", "final_loss", "train_r2"])
    writer.writerows(alpha_rows)
print(RESULT_DIR / "alpha_convergence.csv")

plt.yscale("log")
plt.ylim(50, 2000)
plt.title("Gradient Descent Loss Curves for Different Learning Rates")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss (log scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULT_DIR / "loss_curve_multi_alpha.png", dpi=150)
print(RESULT_DIR / "loss_curve_multi_alpha.png")

best_alpha = min(
    (alpha for alpha, res in results_by_alpha.items() if res["converged_iter"] and res["r2"] > 0),
    key=lambda alpha: results_by_alpha[alpha]["converged_iter"],
)
print(f"best_alpha = {best_alpha} (converged at iter {results_by_alpha[best_alpha]['converged_iter']}, train R2 = {results_by_alpha[best_alpha]['r2']:.4f})")
theta = results_by_alpha[best_alpha]["theta"]

train_metrics = evaluate(X_train_scaled, y_train, theta)
test_metrics = evaluate(X_test_scaled, y_test, theta)

with open(RESULT_DIR / "metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "mse", "r2"])
    writer.writerow(["train", train_metrics["mse"], train_metrics["r2"]])
    writer.writerow(["test", test_metrics["mse"], test_metrics["r2"]])
print(RESULT_DIR / "metrics.csv")

theta_original_scale = recover_coef(theta, params)

with open(RESULT_DIR / "coefficients.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "coefficient"])
    for name, coef in zip(["Intercept"] + feature_names, theta_original_scale):
        writer.writerow([name, coef])
print(RESULT_DIR / "coefficients.csv")

# Figures 1-2: bivariate relationships on the training set
cement = X_train[:, feature_names.index("Cement (component 1)(kg in a m^3 mixture)")]
age = X_train[:, feature_names.index("Age (day)")]

r_cement = np.corrcoef(cement, y_train)[0, 1]
r_age = np.corrcoef(age, y_train)[0, 1]
print(f"r(cement, strength) = {r_cement:.4f}")
print(f"r(age, strength)    = {r_age:.4f}")

plt.figure(figsize=(6, 4.5))
plt.scatter(cement, y_train, s=12, alpha=0.5)
plt.xlabel("Cement (kg/m^3)")
plt.ylabel("Compressive Strength (MPa)")
plt.title(f"Cement Content vs. Compressive Strength (Training Set, r={r_cement:.2f})")
plt.tight_layout()
plt.savefig(RESULT_DIR / "cement_vs_strength.png", dpi=150)
print(RESULT_DIR / "cement_vs_strength.png")

plt.figure(figsize=(6, 4.5))
plt.scatter(age, y_train, s=12, alpha=0.5)
plt.xlabel("Age (days)")
plt.ylabel("Compressive Strength (MPa)")
plt.title(f"Age vs. Compressive Strength (Training Set, r={r_age:.2f})")
plt.tight_layout()
plt.savefig(RESULT_DIR / "age_vs_strength.png", dpi=150)
print(RESULT_DIR / "age_vs_strength.png")

results_by_raw_alpha = {}
raw_alpha_rows = []
for alpha in RAW_ALPHA_GRID:
    res = gradient_descent(X_train, y_train, alpha=alpha, n_iters=RAW_N_ITERS, tol=RAW_TOL)
    final_loss = res["loss_history"][-1]
    res["r2"] = evaluate(X_train, y_train, res["theta"])["r2"] if np.isfinite(final_loss) else float("nan")
    results_by_raw_alpha[alpha] = res
    if res["converged_iter"]:
        converged_iter = res["converged_iter"]
    elif not np.isfinite(final_loss):
        converged_iter = "diverged"
    else:
        converged_iter = "not converged"
    raw_alpha_rows.append([alpha, converged_iter, final_loss, res["r2"]])

with open(RESULT_DIR / "raw_alpha_convergence.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["alpha", "converged_iter", "final_loss", "train_r2"])
    writer.writerows(raw_alpha_rows)
print(RESULT_DIR / "raw_alpha_convergence.csv")

best_raw_alpha = max(
    (alpha for alpha, res in results_by_raw_alpha.items() if res["r2"] > 0),
    key=lambda alpha: results_by_raw_alpha[alpha]["r2"],
)
print(f"best_raw_alpha = {best_raw_alpha} (selected for highest train R2 = {results_by_raw_alpha[best_raw_alpha]['r2']:.4f})")

# Figure 3: raw vs. standardized loss curves, using the R2-selected hyperparameters
res_raw = results_by_raw_alpha[best_raw_alpha]
res_std = results_by_alpha[best_alpha]

with open(RESULT_DIR / "scaling_r2_check.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["features", "alpha", "train_r2"])
    writer.writerow(["raw", best_raw_alpha, res_raw["r2"]])
    writer.writerow(["standardized", best_alpha, res_std["r2"]])
print(RESULT_DIR / "scaling_r2_check.csv")
print(f"raw train R2 = {res_raw['r2']:.4f}, standardized train R2 = {res_std['r2']:.4f}")
assert res_raw["r2"] > 0 and res_std["r2"] > 0, "both models must achieve a positive R2"

plt.figure(figsize=(7, 5))
plt.plot(res_raw["loss_history"], label=f"Raw features (alpha={best_raw_alpha})")
plt.plot(res_std["loss_history"], label=f"Standardized features (alpha={best_alpha})")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss (log scale)")
plt.title("Gradient Descent Loss: Raw vs. Standardized Features")
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "raw_vs_standardized_loss.png", dpi=150)
print(RESULT_DIR / "raw_vs_standardized_loss.png")

# Table 2: optimal solution via an external linear model tool (scikit-learn),
# raw vs. standardized-then-recovered
ols_raw = LinearRegression().fit(X_train, y_train)
theta_ols_raw = np.concatenate([[ols_raw.intercept_], ols_raw.coef_])

ols_std = LinearRegression().fit(X_train_scaled, y_train)
theta_ols_std = np.concatenate([[ols_std.intercept_], ols_std.coef_])
theta_ols_std_recovered = recover_coef(theta_ols_std, params)

with open(RESULT_DIR / "table2_ols_raw_vs_standardized.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "ols_raw", "ols_standardized_recovered"])
    for name, a, b in zip(
        ["Intercept"] + feature_names, theta_ols_raw, theta_ols_std_recovered
    ):
        writer.writerow([name, a, b])

print(RESULT_DIR / "table2_ols_raw_vs_standardized.csv")
