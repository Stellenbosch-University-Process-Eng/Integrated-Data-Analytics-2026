# MSPM via PCA — Theory Reference
## Multivariate Statistical Process Monitoring for Industrial Processes

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{X} \in \mathbb{R}^{n \times m}$ | Autoscaled data matrix, $n$ samples, $m$ variables |
| $\mathbf{P} \in \mathbb{R}^{m \times A}$ | Loading matrix (first $A$ eigenvectors) |
| $\mathbf{T} \in \mathbb{R}^{n \times A}$ | Score matrix: $\mathbf{T} = \mathbf{X}\mathbf{P}$ |
| $\boldsymbol{\Lambda} = \mathrm{diag}(\lambda_1,\ldots,\lambda_A)$ | Diagonal matrix of retained eigenvalues |
| $A$ | Number of retained principal components |
| $m$ | Number of process variables |
| $n$ | Number of NOC training samples |
| $\alpha$ | Significance level (e.g. 0.95) |

---

## 1. Preprocessing — Autoscaling

Each variable is mean-centred and scaled to unit variance using training-set statistics:

$$\tilde{x}_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}$$

where $\bar{x}_j$ and $s_j$ are the training mean and standard deviation (ddof = 1) of variable $j$.
All subsequent operations are performed on the autoscaled matrix $\mathbf{X}$.

---

## 2. PCA Model

### 2.1 Eigendecomposition

The sample covariance matrix of the autoscaled training data is:

$$\mathbf{S} = \frac{1}{n-1}\mathbf{X}^\top\mathbf{X}$$

Eigendecomposition gives $\mathbf{S} = \mathbf{V}\boldsymbol{\Lambda}_\text{full}\mathbf{V}^\top$, where the columns of $\mathbf{V}$ are the eigenvectors (principal directions) and $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_m \geq 0$ are the eigenvalues.

### 2.2 Model Truncation

Retaining the first $A$ components gives:

$$\mathbf{P} = \mathbf{V}_{:,1:A} \in \mathbb{R}^{m \times A}$$

### 2.3 Scores and Reconstruction

$$\mathbf{T} = \mathbf{X}\mathbf{P}, \qquad \hat{\mathbf{X}} = \mathbf{T}\mathbf{P}^\top = \mathbf{X}\mathbf{P}\mathbf{P}^\top$$

The residual (part of $\mathbf{X}$ **not** captured by the model):

$$\tilde{\mathbf{X}} = \mathbf{X} - \hat{\mathbf{X}} = \mathbf{X}(\mathbf{I} - \mathbf{P}\mathbf{P}^\top)$$

---

## 3. Monitoring Statistics

### 3.1 Hotelling's $T^2$

$T^2$ measures the **distance of the score vector from the model centre**, scaled by variance:

$$T^2_i = \mathbf{t}_i^\top \boldsymbol{\Lambda}^{-1} \mathbf{t}_i = \sum_{a=1}^{A} \frac{t_{ia}^2}{\lambda_a}$$

For a new observation $\mathbf{x}_\text{new}$ (autoscaled):

$$T^2_\text{new} = \mathbf{x}_\text{new}^\top \mathbf{P}\boldsymbol{\Lambda}^{-1}\mathbf{P}^\top \mathbf{x}_\text{new}$$

$T^2$ detects **mean shifts** or other changes in the directions captured by the PCA model.

### 3.2 Squared Prediction Error (SPE)

SPE (also called $Q$) measures the **residual variance** — the part of the observation not explained by the model:

$$\text{SPE}_i = \|\tilde{\mathbf{x}}_i\|^2 = \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2 = \sum_{j=1}^{m} \tilde{x}_{ij}^2$$

SPE detects **covariance shifts** or **new fault directions** that lie outside the PCA subspace.

---

## 4. Control Limits

### 4.1 $T^2$ Control Limit (F-distribution)

Under the assumption of multivariate normality, the $T^2$ statistic follows an $F$-distribution:

$$T^2_\alpha = \frac{A(n-1)}{n-A} F_\alpha(A,\, n-A)$$

where $F_\alpha(A, n-A)$ is the $\alpha$-quantile of the $F$-distribution with $A$ and $n-A$ degrees of freedom.

### 4.2 SPE Control Limit (Third-Moment Method)

The SPE limit is estimated by matching moments of the empirical SPE distribution to a $\chi^2$ distribution (Box, 1954; Jackson & Mudholkar, 1979). Define:

$$\theta_k = \sum_{a=A+1}^{m} \lambda_a^k, \quad k = 1, 2, 3$$

Compute the effective degrees of freedom and scale:

$$g = \frac{\theta_2^3}{\theta_3^2}, \qquad h = \frac{\theta_2}{\theta_3}(g - 1)$$

Then the $\alpha$-level SPE limit is:

$$\text{SPE}_\alpha = \theta_1 \left(h\, \chi^2_\alpha(g)\right)^{1/h}$$

where $\chi^2_\alpha(g)$ is the $\alpha$-quantile of the $\chi^2$ distribution with $g$ degrees of freedom.

---

## 5. Fault Contributions

When a monitoring statistic exceeds its control limit, **contribution plots** identify which variables drive the alarm.

### 5.1 SPE Contributions

SPE decomposes exactly into per-variable squared residuals:

$$\text{SPE}_i = \sum_{j=1}^{m} \tilde{x}_{ij}^2$$

so the contribution of variable $j$ to sample $i$ is simply:

$$\text{cont}^{\text{SPE}}_{ij} = \tilde{x}_{ij}^2 \geq 0$$

These contributions are non-negative and sum to SPE.

### 5.2 $T^2$ Contributions

$T^2$ can be written as a bilinear form in variable space:

$$T^2_i = \mathbf{x}_i^\top \underbrace{\mathbf{P}\boldsymbol{\Lambda}^{-1}\mathbf{P}^\top}_{\mathbf{M}} \mathbf{x}_i = \sum_{j=1}^{m} x_{ij} \left[\mathbf{M}\mathbf{x}_i\right]_j$$

This gives the **complete $T^2$ decomposition** (Westerhuis et al., 2000):

$$\text{cont}^{T^2}_{ij} = x_{ij} \cdot \left[\mathbf{P}\boldsymbol{\Lambda}^{-1}\mathbf{P}^\top \mathbf{x}_i\right]_j = x_{ij} \cdot \left[\mathbf{P} \frac{\mathbf{t}_i}{\boldsymbol{\lambda}}\right]_j$$

where $\mathbf{t}_i / \boldsymbol{\lambda}$ denotes element-wise division of the score vector by the eigenvalues.

**Key properties:**

- Contributions sum exactly to $T^2$: $\sum_j \text{cont}^{T^2}_{ij} = T^2_i$
- Contributions can be **negative** (a variable moving *opposite* to the fault direction *reduces* $T^2$)
- High positive contributions indicate variables responsible for the alarm

**Derivation of summation property:**

$$\sum_{j=1}^{m} x_{ij} \left[\mathbf{P}\boldsymbol{\Lambda}^{-1}\mathbf{t}_i\right]_j = \mathbf{x}_i^\top \mathbf{P}\boldsymbol{\Lambda}^{-1}\mathbf{t}_i = \mathbf{t}_i^\top\boldsymbol{\Lambda}^{-1}\mathbf{t}_i = T^2_i \quad \checkmark$$

(using $\mathbf{t}_i = \mathbf{P}^\top\mathbf{x}_i$ and $\mathbf{P}^\top\mathbf{P} = \mathbf{I}$)

---

## 6. $T^2$ Decision Boundary in Score Space

The $T^2$ ellipse in the 2-D score plane ($A = 2$) is defined by the level set:

$$\frac{t_1^2}{\lambda_1} + \frac{t_2^2}{\lambda_2} = T^2_\alpha$$

This is an axis-aligned ellipse with semi-axes $\sqrt{\lambda_a\, T^2_\alpha}$ for $a = 1, 2$.

---

## 7. Fault Types Simulated

| Fault | Description | Expected detector |
|-------|-------------|-------------------|
| **NOC** | Normal operating conditions — $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\text{NOC})$ | Both in-control |
| **Fault 1** | Mean shift in $X_1$ (+4 std) | $T^2$ alarm; SPE mild |
| **Fault 2** | Mean shift in $X_1$ and $X_2$ (+4 std each) | $T^2$ alarm; SPE mild |
| **Fault 3** | Covariance shift ($X_1$–$X_2$ weaker; $X_3$–$X_4$ stronger) | SPE alarm; $T^2$ mild |

Fault 3 changes the **correlation structure** without shifting the mean, so it largely lies outside the NOC PCA subspace — detected by SPE, not by $T^2$.

---

## References

- Jackson, J.E. & Mudholkar, G.S. (1979). Control procedures for residuals associated with principal component analysis. *Technometrics*, 21(3), 341–349.
- Westerhuis, J.A., Gurden, S.P. & Smilde, A.K. (2000). Generalized contribution plots in multivariate statistical process monitoring. *Chemometrics and Intelligent Laboratory Systems*, 51(1), 95–114.
- Kourti, T. & MacGregor, J.F. (1996). Multivariate SPC methods for process and product monitoring. *Journal of Quality Technology*, 28(4), 409–428.
- Box, G.E.P. (1954). Some theorems on quadratic forms applied in the study of analysis of variance problems. *Annals of Mathematical Statistics*, 25(2), 290–302.
