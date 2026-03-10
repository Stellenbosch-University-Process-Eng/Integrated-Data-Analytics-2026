"""
Cleaned-up ARTEO-style Bayesian optimization for a CSTR.

Changes relative to the previous version:
- Grid caching has been removed.
- Diagnostic plotting has been shortened.
- Optimization logic is unchanged.

Provided by Mehmet Mercangoz (Imperial College London)

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. Problem data
# ============================================================


@dataclass(frozen=True)
class EconomicData:
    price_A: float = 5.0
    price_B: float = 19.0
    price_C: float = 7.0
    price_D: float = 2.0
    heat_price: float = 0.0


@dataclass(frozen=True)
class ReactorData:
    # Reaction 1
    k01: float = 1.0
    dH1: float = -1300.0
    Ea1: float = 21500.0

    # Reaction 2
    k02: float = 10.0
    dH2: float = -400.0
    Ea2: float = 91500.0

    # Process data
    density: float = 1.0
    heat_capacity: float = 1.0
    flow_rate: float = 5.0
    inlet_temperature: float = 140.0
    UA: float = 25.0

    # Coolant temperature range
    Tc_min: float = 150.0
    Tc_max: float = 500.0

    # Initial condition
    initial_state: tuple = (0.0147, 1.4, 674.8, 0.65, 1.5353)

    # Integration settings
    t_final: float = 1500.0
    n_time_points: int = 600

    @property
    def Tc_range(self) -> float:
        return self.Tc_max - self.Tc_min


@dataclass(frozen=True)
class BOSettings:
    temperature_limit: float = 540.0
    safety_alpha: float = 5.0
    safety_penalty_weight: float = 5e10
    exploration_bonus: float = 1000.0
    random_seed: int = 42
    gp_restarts: int = 3


ECONOMICS = EconomicData()
REACTOR = ReactorData()
BO = BOSettings()

OUTPUT_NAMES = ["Ca", "Cb", "T", "Cc", "Cd"]


# ============================================================
# 2. Reactor model and economics
# ============================================================


@dataclass
class SimulationResult:
    frac_A: float
    Tc_scaled: float
    Ca: float
    Cb: float
    T: float
    Cc: float
    Cd: float
    profit: float

    def input_array(self) -> np.ndarray:
        return np.array([self.frac_A, self.Tc_scaled], dtype=float)

    def output_array(self) -> np.ndarray:
        return np.array([self.Ca, self.Cb, self.T, self.Cc, self.Cd], dtype=float)

    def table_row(self, iteration: int) -> list[float]:
        return [
            iteration,
            self.frac_A,
            self.Tc_scaled,
            self.Ca,
            self.Cb,
            self.T,
            self.Cc,
            self.Cd,
            self.profit,
        ]


def scaled_to_coolant_temperature(
    Tc_scaled: float, reactor: ReactorData = REACTOR
) -> float:
    return reactor.Tc_min + Tc_scaled * reactor.Tc_range


def inlet_concentrations(
    frac_A: float, reactor: ReactorData = REACTOR
) -> tuple[float, float]:
    Ca_in = frac_A * reactor.flow_rate
    Cd_in = (1.0 - frac_A) * reactor.flow_rate
    return Ca_in, Cd_in


def cstr_dynamics(
    state: list[float],
    t: float,
    frac_A: float,
    Tc_scaled: float,
    reactor: ReactorData = REACTOR,
) -> list[float]:
    Ca, Cb, T, Cc, Cd = state

    F = reactor.flow_rate
    T_in = reactor.inlet_temperature
    Tc = scaled_to_coolant_temperature(Tc_scaled, reactor)
    Ca_in, Cd_in = inlet_concentrations(frac_A, reactor)

    k1 = reactor.k01 * np.exp(-reactor.Ea1 / (8.314 * (1.8 * T + 1000.0)))
    k2 = reactor.k02 * np.exp(-reactor.Ea2 / (8.314 * (2.8 * T + 1000.0)))

    R1 = k1 * Ca * Cd
    R2 = k2 * Ca

    dCadt = (F / 1000.0) * (Ca_in - Ca) - R1 - R2
    dCbdt = (F / 1000.0) * (0.0 - Cb) + R1
    dCcdt = (F / 1000.0) * (0.0 - Cc) + R2
    dCddt = (F / 1000.0) * (Cd_in - Cd) - R1

    dTdt = (
        (F * T_in - F * T) / 1000.0
        + (1.0 / (reactor.density * reactor.heat_capacity))
        * (-reactor.dH1 * R1 - reactor.dH2 * R2)
        + (reactor.UA / 1000.0 / (reactor.density * reactor.heat_capacity)) * (Tc - T)
    )

    return [dCadt, dCbdt, dTdt, dCcdt, dCddt]


def compute_profit(
    frac_A: float,
    Tc_scaled: float,
    Cb: float,
    Cc: float,
    T: float,
    reactor: ReactorData = REACTOR,
    economics: EconomicData = ECONOMICS,
) -> float:
    F = reactor.flow_rate
    Tc = scaled_to_coolant_temperature(Tc_scaled, reactor)
    Ca_in, Cd_in = inlet_concentrations(frac_A, reactor)

    profit = (
        economics.price_B * F * Cb
        + economics.price_C * F * Cc
        - economics.price_A * F * Ca_in
        - economics.price_D * F * Cd_in
        - economics.heat_price * reactor.UA * (Tc - T)
    )
    return profit


def simulate_cstr(
    frac_A: float,
    Tc_scaled: float,
    reactor: ReactorData = REACTOR,
    economics: EconomicData = ECONOMICS,
) -> SimulationResult:
    t_grid = np.linspace(0.0, reactor.t_final, reactor.n_time_points)

    solution = odeint(
        cstr_dynamics,
        reactor.initial_state,
        t_grid,
        args=(frac_A, Tc_scaled, reactor),
    )

    Ca, Cb, T, Cc, Cd = solution[-1]
    profit = compute_profit(frac_A, Tc_scaled, Cb, Cc, T, reactor, economics)

    return SimulationResult(
        frac_A=frac_A,
        Tc_scaled=Tc_scaled,
        Ca=Ca,
        Cb=Cb,
        T=T,
        Cc=Cc,
        Cd=Cd,
        profit=profit,
    )


# ============================================================
# 3. Initial seed points
# ============================================================


def generate_seed_points(reactor: ReactorData = REACTOR) -> list[tuple[float, float]]:
    frac_A_center = 0.15
    Tc_scaled_center = (350.0 - reactor.Tc_min) / reactor.Tc_range

    frac_offsets = [0.0, -0.05, 0.01]
    Tc_offsets = [0.0, -10.0 / reactor.Tc_range, 20.0 / reactor.Tc_range]

    points = []
    for df in frac_offsets:
        for dTc in Tc_offsets:
            frac_A = np.clip(frac_A_center + df, 0.0, 1.0)
            Tc_scaled = np.clip(Tc_scaled_center + dTc, 0.0, 1.0)
            points.append((frac_A, Tc_scaled))

    return points


# ============================================================
# 4. GP model set
# ============================================================


class GPModelSet:
    def __init__(self, gps, x_scaler, y_scalers):
        self.gps = gps
        self.x_scaler = x_scaler
        self.y_scalers = y_scalers

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, bo: BOSettings = BO) -> "GPModelSet":
        x_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X)

        y_scalers = [StandardScaler() for _ in range(Y.shape[1])]
        Y_scaled = np.column_stack(
            [
                scaler.fit_transform(Y[:, i].reshape(-1, 1)).ravel()
                for i, scaler in enumerate(y_scalers)
            ]
        )

        gps = []
        for i in range(Y_scaled.shape[1]):
            kernel = C(1.0, (1e-3, 1e4)) * RBF([1.0, 1.0], (1e-2, 1e3))
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=bo.gp_restarts,
                normalize_y=True,
            )
            gp.fit(X_scaled, Y_scaled[:, i])
            gps.append(gp)

        return cls(gps, x_scaler, y_scalers)

    def predict_scaled(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.atleast_2d(x)
        x_scaled = self.x_scaler.transform(x)

        means_scaled = []
        stds_scaled = []

        for gp in self.gps:
            mean_s, std_s = gp.predict(x_scaled, return_std=True)
            means_scaled.append(mean_s[0])
            stds_scaled.append(std_s[0])

        return np.array(means_scaled), np.array(stds_scaled)

    def predict_physical_means(self, x: np.ndarray) -> np.ndarray:
        means_scaled, _ = self.predict_scaled(x)
        means_physical = []

        for mean_s, scaler in zip(means_scaled, self.y_scalers):
            mean_phys = scaler.inverse_transform([[mean_s]])[0, 0]
            means_physical.append(mean_phys)

        return np.array(means_physical)

    def temperature_ucb(self, x: np.ndarray, n_sigma: float = 2.0) -> float:
        means_scaled, stds_scaled = self.predict_scaled(x)
        T_ucb_scaled = means_scaled[2] + n_sigma * stds_scaled[2]
        T_ucb = self.y_scalers[2].inverse_transform([[T_ucb_scaled]])[0, 0]
        return float(T_ucb)


# ============================================================
# 5. Acquisition function and optimizer
# ============================================================


def build_acquisition_function(
    gp_models: GPModelSet,
    reactor: ReactorData = REACTOR,
    economics: EconomicData = ECONOMICS,
    bo: BOSettings = BO,
):
    def objective(x: np.ndarray) -> float:
        predicted_outputs = gp_models.predict_physical_means(x)
        _, stds_scaled = gp_models.predict_scaled(x)

        Cb = predicted_outputs[1]
        T = predicted_outputs[2]
        Cc = predicted_outputs[3]

        predicted_profit = compute_profit(x[0], x[1], Cb, Cc, T, reactor, economics)

        T_ucb = gp_models.temperature_ucb(x, n_sigma=2.0)
        safety_penalty = (
            bo.safety_penalty_weight
            * 0.5
            * (np.tanh(bo.safety_alpha * (T_ucb - bo.temperature_limit)) + 1.0)
        )

        exploration_term = bo.exploration_bonus * np.sum(stds_scaled)

        return -predicted_profit - exploration_term + safety_penalty

    return objective


def find_next_candidate(objective, bo: BOSettings = BO) -> np.ndarray:
    result = differential_evolution(
        objective,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        seed=bo.random_seed,
        disp=False,
    )
    return result.x


# ============================================================
# 6. Grid generation for plotting
# ============================================================


def build_plotting_grids(grid_resolution: int = 60):
    """
    Compute true profit and temperature grids directly.
    No caching is used.
    """
    frac_values = np.linspace(0.0, 1.0, grid_resolution)
    Tc_scaled_values = np.linspace(0.0, 1.0, grid_resolution)

    profit_grid = np.zeros((grid_resolution, grid_resolution))
    T_true_grid = np.zeros((grid_resolution, grid_resolution))

    for i, frac_A in enumerate(frac_values):
        for j, Tc_scaled in enumerate(Tc_scaled_values):
            result = simulate_cstr(frac_A, Tc_scaled)
            profit_grid[j, i] = result.profit
            T_true_grid[j, i] = result.T

    frac_grid, Tc_scaled_grid = np.meshgrid(frac_values, Tc_scaled_values)
    return frac_grid, Tc_scaled_grid, profit_grid, T_true_grid


# ============================================================
# 7. Shortened diagnostic plot
# ============================================================


def diagnostic_plot(
    iteration: int,
    gp_models: GPModelSet,
    frac_grid: np.ndarray,
    Tc_scaled_grid: np.ndarray,
    profit_grid: np.ndarray,
    seed_points: np.ndarray,
    sampled_points_so_far: list[np.ndarray],
    T_true_grid: np.ndarray,
    bo: BOSettings = BO,
):
    """
    Shortened plot:
    - profit contour
    - seed points
    - sampled path
    - iteration numbers on sampled points
    - true temperature limit contour
    - GP-UCB temperature limit contour
    """
    grid_points = np.column_stack([frac_grid.ravel(), Tc_scaled_grid.ravel()])
    grid_points_scaled = gp_models.x_scaler.transform(grid_points)

    mean_s, std_s = gp_models.gps[2].predict(grid_points_scaled, return_std=True)
    T_ucb_scaled = mean_s + 2.0 * std_s
    T_ucb = (
        gp_models.y_scalers[2].inverse_transform(T_ucb_scaled.reshape(-1, 1)).ravel()
    )
    T_ucb_grid = T_ucb.reshape(frac_grid.shape)

    plt.figure(figsize=(7, 4.5), dpi=200)

    plt.contour(frac_grid, Tc_scaled_grid, profit_grid, levels=20, cmap="Greys")
    plt.contour(
        frac_grid,
        Tc_scaled_grid,
        T_true_grid,
        levels=[bo.temperature_limit],
        colors="blue",
        linestyles="--",
        linewidths=2,
    )
    plt.contour(
        frac_grid,
        Tc_scaled_grid,
        T_ucb_grid,
        levels=[bo.temperature_limit],
        colors="red",
        linestyles="-.",
        linewidths=2,
    )

    plt.scatter(
        seed_points[:, 0], seed_points[:, 1], color="gray", s=30, label="Seed points"
    )

    path = np.array(sampled_points_so_far)
    plt.plot(path[:, 0], path[:, 1], "ro-", label="BO path")

    # Add iteration numbers next to sampled points
    for k, (xk, yk) in enumerate(path, start=1):
        plt.text(xk, yk, str(k), fontsize=8, ha="center", va="bottom")

    plt.xlabel("frac_A")
    plt.ylabel("Tc_scaled")
    plt.title(f"Iteration {iteration}: diagnostic plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. Main optimization loop
# ============================================================


def run_arteo(n_iterations: int = 10) -> pd.DataFrame:
    frac_grid, Tc_scaled_grid, profit_grid, T_true_grid = build_plotting_grids()

    seed_points = generate_seed_points()
    seed_results = [
        simulate_cstr(frac_A, Tc_scaled) for frac_A, Tc_scaled in seed_points
    ]

    X = np.array([result.input_array() for result in seed_results])
    Y = np.array([result.output_array() for result in seed_results])

    profit_history = [result.profit for result in seed_results]
    temperature_history = [result.T for result in seed_results]

    sampled_points = []
    result_rows = []

    gp_models = GPModelSet.fit(X, Y)
    acquisition = build_acquisition_function(gp_models)

    print("\n>>> ARTEO iterations")
    print("-" * 50)

    for iteration in range(1, n_iterations + 1):
        # 1. Select next point using current GP
        x_next = find_next_candidate(acquisition)
        sampled_points.append(x_next)

        predicted_outputs = gp_models.predict_physical_means(x_next)
        predicted_profit = compute_profit(
            frac_A=x_next[0],
            Tc_scaled=x_next[1],
            Cb=predicted_outputs[1],
            Cc=predicted_outputs[3],
            T=predicted_outputs[2],
        )
        predicted_T_ucb = gp_models.temperature_ucb(x_next)

        print(f"Iteration {iteration}: x = {x_next}")
        print(f"  Predicted profit = {predicted_profit:.2f}")
        print(f"  Predicted T-UCB  = {predicted_T_ucb:.2f} K")

        # 2. Diagnostic plot before update
        diagnostic_plot(
            iteration=iteration,
            gp_models=gp_models,
            frac_grid=frac_grid,
            Tc_scaled_grid=Tc_scaled_grid,
            profit_grid=profit_grid,
            seed_points=np.array(seed_points),
            sampled_points_so_far=sampled_points,
            T_true_grid=T_true_grid,
        )

        # 3. Evaluate real system
        true_result = simulate_cstr(x_next[0], x_next[1])

        print(
            "  Realized outputs: "
            f"Ca={true_result.Ca:.3f}, "
            f"Cb={true_result.Cb:.3f}, "
            f"T={true_result.T:.1f}, "
            f"Cc={true_result.Cc:.3f}, "
            f"Cd={true_result.Cd:.3f}"
        )
        print(f"  Realized profit  = {true_result.profit:.2f}\n")

        profit_history.append(true_result.profit)
        temperature_history.append(true_result.T)
        result_rows.append(true_result.table_row(iteration))

        # 4. Update GP with new point
        X = np.vstack([X, true_result.input_array()])
        Y = np.vstack([Y, true_result.output_array()])

        gp_models = GPModelSet.fit(X, Y)
        acquisition = build_acquisition_function(gp_models)

    results_df = pd.DataFrame(
        result_rows,
        columns=["Iter", "frac_A", "Tc_scaled"] + OUTPUT_NAMES + ["Profit"],
    )

    # Final summary plot
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=200)

    ax1.plot(profit_history, "b-o", label="Realized profit")
    ax1.set_xlabel("Evaluation number")
    ax1.set_ylabel("Profit", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(temperature_history, "r-s", label="Realized temperature")
    ax2.axhline(
        BO.temperature_limit, color="gray", linestyle="--", label="Temperature limit"
    )
    ax2.set_ylabel("Temperature (K)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")

    plt.title("Realized profit and temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return results_df


# ============================================================
# 9. Main guard
# ============================================================

if __name__ == "__main__":
    results = run_arteo(n_iterations=14)
    print("\n=== Final Results ===")
    print(results.to_string(index=False))
