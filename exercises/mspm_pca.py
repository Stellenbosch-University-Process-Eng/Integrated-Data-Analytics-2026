"""Simple demonstration of multivariate statistical process monitoring (MSPM) using PCA."""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import linalg, stats
import matplotlib.pyplot as plt

# ── Data generation ────────────────────────────────────────────────────────────
N = 100  # samples per class
m = 5    # number of process variables

# NOC covariance structure
cov_noc = np.array([
    [1.0, 0.8, 0.3, 0.1, 0.1],
    [0.8, 1.0, 0.4, 0.3, 0.2],
    [0.3, 0.4, 1.0, 0.5, 0.3],
    [0.1, 0.3, 0.5, 1.0, 0.4],
    [0.1, 0.2, 0.3, 0.4, 1.0],
])
mean_noc = np.zeros(m)

data_noc      = np.random.multivariate_normal(mean_noc, cov_noc, N)
data_noc_test = np.random.multivariate_normal(mean_noc, cov_noc, N)

# Fault 1: mean shift in X1
mean_fault1    = mean_noc.copy()
mean_fault1[0] = 4
data_fault1    = np.random.multivariate_normal(mean_fault1, cov_noc, N)

# Fault 2: mean shift in X1 and X2
mean_fault2    = mean_noc.copy()
mean_fault2[0] = 4
mean_fault2[1] = 4
data_fault2    = np.random.multivariate_normal(mean_fault2, cov_noc, N)

# Fault 3: covariance shift (X1–X2 weaker, X3–X4 stronger)
cov_fault3          = cov_noc.copy()
cov_fault3[0, 1]    = cov_fault3[1, 0] = 0.1
cov_fault3[2, 3]    = cov_fault3[3, 2] = 0.8
data_fault3         = np.random.multivariate_normal(mean_noc, cov_fault3, N)

# Shared class metadata
class_labels = ["NOC Train", "NOC Test", "Fault 1", "Fault 2", "Fault 3"]
class_colors = ["blue", "green", "orange", "red", "purple"]
fault_names  = [
    "Fault 1 (Mean Shift in X1)",
    "Fault 2 (Mean Shift in X1 and X2)",
    "Fault 3 (Covariance Shift)",
]
fault_colors = class_colors[2:]   # orange, red, purple
all_raw      = [data_noc, data_noc_test, data_fault1, data_fault2, data_fault3]
var_labels   = [f"X{i + 1}" for i in range(m)]

# ── Data understanding ──────────────────────────────────────────────────────────

# ── Fig 1: raw time series ─────────────────────────────────────────────────────
plt.figure(figsize=(12, 10))
for i in range(m):
    plt.subplot(m, 1, i + 1)
    for j, (data, label, color) in enumerate(zip(all_raw, class_labels, class_colors)):
        plt.plot(
            np.arange(j * N, (j + 1) * N), data[:, i],
            "o", label=label, color=color, markersize=3,
        )
    if i == 0:
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
    if i == m - 1:
        plt.xlabel("Sample Index")
    plt.ylabel(var_labels[i])
    plt.yticks([])
plt.tight_layout()
plt.show()

# ── Fig 2 & 3: pair plots ──────────────────────────────────────────────────────
data_combined = np.vstack(all_raw)
labels        = sum([[lbl] * N for lbl in class_labels], [])
df            = pd.DataFrame(data_combined, columns=var_labels)
df["Class"]   = labels
palette_all   = dict(zip(class_labels, class_colors))

sns.pairplot(
    df,
    kind="kde",
    hue="Class",
    diag_kind="kde",
    corner=True,
    palette=palette_all,
    plot_kws={"alpha": 0.5, "linewidths": 1, "levels": 5},
)
plt.tight_layout()
plt.show()

# NOC Train vs Fault 3 only — covariance shift is subtle; easier to see in isolation
df_subset = df[df["Class"].isin(["NOC Train", "Fault 3"])]
sns.pairplot(
    df_subset,
    kind="kde",
    hue="Class",
    diag_kind="kde",
    corner=True,
    palette={"NOC Train": "blue", "Fault 3": "purple"},
    plot_kws={"alpha": 0.5, "linewidths": 1, "levels": 5},
)
plt.tight_layout()
plt.show()

# ── PCA model ─────────────────────────────────────────────────────────────────
# test_data layout: [0:N] NOC Test | [N:2N] Fault 1 | [2N:3N] Fault 2 | [3N:4N] Fault 3
training_data = data_noc
test_data     = np.vstack((data_noc_test, data_fault1, data_fault2, data_fault3))

mean_train = np.mean(training_data, axis=0)
std_train  = np.std(training_data, axis=0, ddof=1)
X_train    = (training_data - mean_train) / std_train
X_test     = (test_data     - mean_train) / std_train

S_train = np.cov(X_train, rowvar=False)
eigenvalues, eigenvectors = linalg.eigh(S_train)
sort_idx     = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[sort_idx]
eigenvectors = eigenvectors[:, sort_idx]

A = 2                     # retained principal components
P = eigenvectors[:, :A]   # loadings  [m × A]

# Scores and residuals
T_train       = X_train @ P
X_train_hat   = T_train @ P.T
X_train_tilde = X_train - X_train_hat

T_test        = X_test @ P
X_test_hat    = T_test @ P.T
X_test_tilde  = X_test - X_test_hat

# Control limits (95% and 90%)
alpha       = 0.95
T2_limit    = float(A * (N - 1) / (N - A) * stats.f.ppf(alpha, A, N - A))
T2_limit_90 = float(A * (N - 1) / (N - A) * stats.f.ppf(0.90,  A, N - A))

# SPE limit via third-moment method
theta1    = np.sum(eigenvalues[A:])
theta2    = np.sum(eigenvalues[A:] ** 2)
theta3    = np.sum(eigenvalues[A:] ** 3)
g         = theta2 ** 3 / theta3 ** 2
h         = theta2 / theta3 * (g - 1)
SPE_limit = float(theta1 * (h * stats.chi2.ppf(alpha, g)) ** (1 / h))

# Monitoring statistics
T2_train  = np.sum((T_train / np.sqrt(eigenvalues[:A])) ** 2, axis=1)
T2_test   = np.sum((T_test  / np.sqrt(eigenvalues[:A])) ** 2, axis=1)
SPE_train = np.sum(X_train_tilde ** 2, axis=1)
SPE_test  = np.sum(X_test_tilde  ** 2, axis=1)
T2_all    = np.concatenate([T2_train, T2_test])
SPE_all   = np.concatenate([SPE_train, SPE_test])

# Contribution arrays
selected_samples  = [N, 2 * N, 3 * N]   # first sample of each fault in X_test
fault_slices      = [slice(N, 2*N), slice(2*N, 3*N), slice(3*N, 4*N)]
squared_residuals = X_test_tilde ** 2
t2_contrib_all    = X_test * ((T_test / eigenvalues[:A]) @ P.T)   # [N_test × m]

# Score-space helpers
T_scores   = [T_train, T_test[:N], T_test[N:2*N], T_test[2*N:3*N], T_test[3*N:4*N]]
T1_pts     = np.linspace(T_train[:, 0].min() - 1, T_train[:, 0].max() + 1, 100)
T2_pts     = np.linspace(T_train[:, 1].min() - 1, T_train[:, 1].max() + 1, 100)
TT1, TT2   = np.meshgrid(T1_pts, T2_pts)
T2_surface = TT1 ** 2 / eigenvalues[0] + TT2 ** 2 / eigenvalues[1]


def _add_T2_contours():
    plt.contour(TT1, TT2, T2_surface, levels=[T2_limit],    colors="red",    linestyles="--")
    plt.contour(TT1, TT2, T2_surface, levels=[T2_limit_90], colors="orange", linestyles="--")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("T1")
    plt.ylabel("T2")
    plt.grid(True)


# ── PCA model estimation ────────────────────────────────────────────────────────

# ── Fig 4: covariance matrix ───────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
sns.heatmap(
    S_train,
    annot=True,
    cmap="coolwarm",
    vmin=0,
    vmax=1,
    xticklabels=var_labels,
    yticklabels=var_labels,
)
plt.title("Covariance Matrix of Training Data")
plt.tight_layout()
plt.show()

# ── Fig 5: eigenvalue matrix ───────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
sns.heatmap(
    np.diag(eigenvalues),
    annot=True,
    cmap="coolwarm",
    vmin=0,
    vmax=eigenvalues[0],
    xticklabels=[f"PC{i + 1}" for i in range(m)],
    yticklabels=[f"PC{i + 1}" for i in range(m)],
)
plt.title("Eigenvalues of Principal Components")
plt.tight_layout()
plt.show()

# ── Fig 6: loadings bar chart ──────────────────────────────────────────────────
bar_width = 0.35
x = np.arange(m)
plt.figure(figsize=(8, 6))
plt.bar(x,              P[:, 0], bar_width, label="PC1", color="blue")
plt.bar(x + bar_width,  P[:, 1], bar_width, label="PC2", color="orange")
plt.axhline(0, color="black", linewidth=0.5)
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.ylabel("Loading Value")
plt.title("Loadings of the First Two Principal Components")
plt.xticks(x + bar_width / 2, var_labels)
plt.legend()
plt.tight_layout()
plt.show()

# ── Score spaces ────────────────────────────────────────────────────────────────

# ── Fig 7: training score plot ─────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(T_train[:, 0], T_train[:, 1], label="NOC Train", color="blue", alpha=0.5)
plt.title("PCA Score Plot — Training Data")
plt.xlabel("T1")
plt.ylabel("T2")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True)
plt.legend()
plt.show()

# ── Fig 8: T² decision boundary — training only ───────────────────────────────
plt.figure(figsize=(8, 6))
_add_T2_contours()
plt.scatter(T_train[:, 0], T_train[:, 1], label="NOC Train", color="blue", alpha=0.5)
plt.title("PCA Score Plot with T² Decision Boundary — Training Data")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
plt.tight_layout()
plt.show()

# ── Fig 9: T² decision boundary — all classes ─────────────────────────────────
plt.figure(figsize=(8, 6))
_add_T2_contours()
for scores, label, color in zip(T_scores, class_labels, class_colors):
    plt.scatter(scores[:, 0], scores[:, 1], label=label, color=color, alpha=0.5)
plt.title("PCA Score Plot with T² Decision Boundary — All Classes")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
plt.tight_layout()
plt.show()

# ── Score space diagnostic ──────────────────────────────────────────────────────

# ── Fig 10: T² statistic — training ───────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(T2_train, "o", label="T²", color="blue", markersize=3)
plt.axhline(T2_limit, color="red", linestyle="--", label="95% Control Limit")
plt.title("Hotelling's T² — Training Data")
plt.xlabel("Sample Index")
plt.ylabel("T²")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ── Fig 11: T² statistic — train + test ───────────────────────────────────────
plt.figure(figsize=(8, 4))
for j, (label, color) in enumerate(zip(class_labels, class_colors)):
    start = j * N
    plt.plot(
        np.arange(start, start + N), T2_all[start:start + N],
        "o", label=label, color=color, markersize=3,
    )
plt.axhline(T2_limit, color="red", linestyle="--", label="95% Control Limit")
plt.title("Hotelling's T² — Train and Test")
plt.xlabel("Sample Index")
plt.ylabel("T²")
plt.grid(True)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
plt.tight_layout()
plt.show()

# ── Score space contributions ───────────────────────────────────────────────────

# ── Fig 12: T² contribution — single sample per fault ─────────────────────────
# Complete decomposition: contrib_j = x_j * [P @ (t/λ)]_j  →  sums to T².
plt.figure(figsize=(12, 4))
for i, sample_idx in enumerate(selected_samples):
    plt.subplot(1, 3, i + 1)
    t_sample        = T_test[sample_idx]                         # scores  [A]
    x_sample        = X_test[sample_idx]                         # std obs [m]
    t2_contrib_vars = x_sample * (P @ (t_sample / eigenvalues[:A]))
    plt.bar(np.arange(m), t2_contrib_vars, color=fault_colors[i])
    plt.axhline(0, color="black", linewidth=0.5)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.ylabel("T² Contribution")
    plt.title(f"{fault_names[i]}\n(Sample {sample_idx})")
    plt.xticks(np.arange(m), var_labels)
plt.tight_layout()
plt.show()

# ── Fig 13: T² contribution — boxplots per fault class ────────────────────────
max_t2_contrib = np.max(np.abs(t2_contrib_all[N:]))   # fault samples only
plt.figure(figsize=(12, 6))
for i, (slc, color, name) in enumerate(zip(fault_slices, fault_colors, fault_names)):
    plt.subplot(1, 3, i + 1)
    plt.boxplot(
        t2_contrib_all[slc],
        patch_artist=True,
        boxprops=dict(facecolor=color),
    )
    plt.axhline(0, color="black", linewidth=0.5)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.ylabel("T² Contribution")
    plt.title(f"{name}\n(All {N} samples)")
    plt.xticks(np.arange(1, m + 1), var_labels)
    plt.ylim(-max_t2_contrib * 1.1, max_t2_contrib * 1.1)
plt.tight_layout()
plt.show()

# ── Reconstructed time series ───────────────────────────────────────────────────

# ── Fig 14: reconstruction time series (standardised scale) ───────────────────
# Both original and reconstructed in standardised space for a fair comparison.
X_all_std = np.vstack([X_train, X_test])
X_hat_all = np.vstack([X_train_hat, X_test_hat])
plt.figure(figsize=(12, 10))
for i in range(m):
    plt.subplot(m, 1, i + 1)
    for j, color in enumerate(class_colors):
        start = j * N
        plt.plot(
            np.arange(start, start + N), X_all_std[start:start + N, i],
            "o", color=color, alpha=0.1, mec="black", markersize=3,
        )
        plt.plot(
            np.arange(start, start + N), X_hat_all[start:start + N, i],
            "x", color=color, alpha=0.8, markersize=4,
        )
    plt.ylabel(var_labels[i])
    plt.yticks([])
    if i == m - 1:
        plt.xlabel("Sample Index")
plt.tight_layout()
plt.show()

# ── SPE diagnostic ──────────────────────────────────────────────────────────────

# ── Fig 15: SPE — training ─────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(SPE_train, "o", label="SPE", color="blue", markersize=3)
plt.axhline(SPE_limit, color="red", linestyle="--", label="95% Control Limit")
plt.title("SPE — Training Data")
plt.xlabel("Sample Index")
plt.ylabel("SPE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ── Fig 16: SPE — train + test ─────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
for j, (label, color) in enumerate(zip(class_labels, class_colors)):
    start = j * N
    plt.plot(
        np.arange(start, start + N), SPE_all[start:start + N],
        "o", label=label, color=color, markersize=3,
    )
plt.axhline(SPE_limit, color="red", linestyle="--", label="95% Control Limit")
plt.title("SPE — Train and Test")
plt.xlabel("Sample Index")
plt.ylabel("SPE")
plt.grid(True)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
plt.tight_layout()
plt.show()

# ── SPE contributions ────────────────────────────────────────────────────────────

# ── Fig 17: SPE contribution — single sample per fault ────────────────────────
plt.figure(figsize=(12, 4))
for i, sample_idx in enumerate(selected_samples):
    plt.subplot(1, 3, i + 1)
    plt.bar(np.arange(m), squared_residuals[sample_idx], color=fault_colors[i])
    plt.axhline(0, color="black", linewidth=0.5)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.ylabel("Squared Residual")
    plt.title(f"{fault_names[i]}\n(Sample {sample_idx})")
    plt.xticks(np.arange(m), var_labels)
plt.tight_layout()
plt.show()

# ── Fig 18: SPE contribution — boxplots per fault class ───────────────────────
max_sq_resid = np.max(squared_residuals)
plt.figure(figsize=(12, 6))
for i, (slc, color, name) in enumerate(zip(fault_slices, fault_colors, fault_names)):
    plt.subplot(1, 3, i + 1)
    plt.boxplot(
        squared_residuals[slc],
        patch_artist=True,
        boxprops=dict(facecolor=color),
    )
    plt.axhline(0, color="black", linewidth=0.5)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.ylabel("Squared Residual")
    plt.title(f"{name}\n(All {N} samples)")
    plt.xticks(np.arange(1, m + 1), var_labels)
    plt.ylim(0, max_sq_resid * 1.1)
plt.tight_layout()
plt.show()
