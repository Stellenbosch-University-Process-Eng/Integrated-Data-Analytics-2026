"""
Simple digital application example
Inspired by Pantelides et al. (2024) Process operations: from models and data to digital applications

Making use of nonisothermal CSTR model defined in nonisothermal_cstr.py

Architecture:
- Plant: generates raw plant data with disturbances, dynamics, noise, gross errors, missing data
- Data sources:
  * plant_historian: raw plant data (including equipment availability), update rate: 1 min
  * commercial_it_systems: raw material availability/costs, product demands/prices, update rate: 8 hr
- Data sinks:
  * plant_control_system: optimal setpoints, update rate: 15 min
- Digital applications:
  * steady_state_detection: checks raw plant data for steady state operation, outputs true/false
  * steady_state_data_reconciliation: reconciles raw plant data, outputs best estimate of steady state values
  * steady_state_optimization: optimizes steady state operation based on reconciled data, outputs optimal setpoints
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque
import pandas as pd

# Import reactor model components
from nonisothermal_cstr import (
    ProcessParameters,
    Disturbances,
    SteadyStateResult,
    ODEs,
    SteadyState,
    h1,
    h4,
    h7,
    phi,
    nonLinearConstraints,
    objectiveFunction,
)


# ============================================================================
# AUTOCORRELATED NOISE GENERATOR
# ============================================================================


class AutocorrelatedNoise:
    """Generates autocorrelated noise using AR(1) process"""

    def __init__(self, phi=0.9, sigma=1.0, initial_value=0.0):
        """
        Parameters:
        phi: autocorrelation coefficient (0 to 1, where 1 is perfect correlation)
        sigma: standard deviation of the white noise
        initial_value: initial value of the process
        """
        self.phi = phi
        self.sigma = sigma
        self.current_value = initial_value

    def step(self):
        """Generate next value in the autocorrelated sequence"""
        # AR(1) process: x[t] = phi * x[t-1] + epsilon[t]
        epsilon = np.random.normal(0, self.sigma)
        self.current_value = self.phi * self.current_value + epsilon
        return self.current_value

    def reset(self, initial_value=0.0):
        """Reset the process"""
        self.current_value = initial_value


# ============================================================================
# DATA SOURCES
# ============================================================================


class PlantHistorian:
    """Stores raw plant measurements with timestamps"""

    def __init__(self):
        self.data = []
        self.update_rate = 60  # seconds (1 min)

    def add_measurement(
        self, timestamp, F, V, CAs, Ts, Tjs, CA0, T0, Tj0, available=True
    ):
        """Add a new measurement to the historian"""
        self.data.append(
            {
                "timestamp": timestamp,
                "F": F,
                "V": V,
                "CAs": CAs,
                "Ts": Ts,
                "Tjs": Tjs,
                "CA0": CA0,
                "T0": T0,
                "Tj0": Tj0,
                "available": available,
            }
        )

    def query(self, start_time, end_time):
        """Query data within a time range"""
        return [d for d in self.data if start_time <= d["timestamp"] <= end_time]

    def get_recent(self, n=10):
        """Get the n most recent measurements"""
        return self.data[-n:] if len(self.data) >= n else self.data


class CommercialITSystems:
    """Stores commercial data: costs, prices, availability"""

    def __init__(self):
        self.data = []
        self.update_rate = 8 * 3600  # seconds (8 hr)

    def update(
        self, timestamp, raw_material_cost, product_price, raw_material_available=True
    ):
        """Update commercial data"""
        self.data.append(
            {
                "timestamp": timestamp,
                "raw_material_cost": raw_material_cost,
                "product_price": product_price,
                "raw_material_available": raw_material_available,
            }
        )

    def get_latest(self):
        """Get the most recent commercial data"""
        return self.data[-1] if self.data else None


# ============================================================================
# DATA SINKS
# ============================================================================


class PlantControlSystem:
    """Stores optimal setpoints for plant control"""

    def __init__(self):
        self.setpoints = []
        self.update_rate = 15 * 60  # seconds (15 min)

    def write_setpoint(self, timestamp, F_sp, V_sp, optimization_success, notes=""):
        """Write new setpoints to control system"""
        self.setpoints.append(
            {
                "timestamp": timestamp,
                "F_setpoint": F_sp,
                "V_setpoint": V_sp,
                "optimization_success": optimization_success,
                "notes": notes,
            }
        )
        print(
            f"[{timestamp}] Control System: F_sp={F_sp:.4f} L/s, V_sp={V_sp:.1f} L, Success={optimization_success}"
        )

    def get_current_setpoint(self):
        """Get the current setpoint"""
        return self.setpoints[-1] if self.setpoints else None


# ============================================================================
# PLANT SIMULATION
# ============================================================================


class PlantSimulator:
    """Simulates the actual plant with disturbances and noise"""

    def __init__(
        self, p, d, phi_disturbance=0.9, sigma_CA0=0.5, sigma_T0=2.0, sigma_Tj0=1.0
    ):
        self.p = p  # Process parameters
        self.d = d  # Nominal disturbances
        self.state = [d.CA0, d.T0, d.T0]  # [CAs, Ts, Tjs]
        self.F = 0.3  # L/s
        self.V = 400  # L
        self.time = 0

        # Initialize autocorrelated noise generators for disturbances
        self.noise_CA0 = AutocorrelatedNoise(
            phi=phi_disturbance, sigma=sigma_CA0, initial_value=0.0
        )
        self.noise_T0 = AutocorrelatedNoise(
            phi=phi_disturbance, sigma=sigma_T0, initial_value=0.0
        )
        self.noise_Tj0 = AutocorrelatedNoise(
            phi=phi_disturbance, sigma=sigma_Tj0, initial_value=0.0
        )

    def simulate_step(self, dt, add_noise=True, add_disturbance=True):
        """Simulate one time step"""
        # Apply disturbances to inputs
        d_actual = Disturbances()
        d_actual.CA0 = self.d.CA0
        d_actual.T0 = self.d.T0
        d_actual.Tj0 = self.d.Tj0
        d_actual.UA_pcp = self.d.UA_pcp

        if add_disturbance:
            # Add autocorrelated disturbances
            d_actual.CA0 += self.noise_CA0.step()
            d_actual.T0 += self.noise_T0.step()
            d_actual.Tj0 += self.noise_Tj0.step()

        # Create temporary result for ODE integration
        u_temp = SteadyStateResult(
            self.F, self.V, self.state[0], self.state[1], self.state[2]
        )

        # Integrate ODEs
        sol = solve_ivp(
            lambda t, x: ODEs(t, x, u_temp, d_actual, self.p),
            [self.time, self.time + dt],
            self.state,
            method="BDF",
        )

        self.state = sol.y[:, -1]
        self.time += dt

        # Measurements with noise
        if add_noise:
            CAs_meas = self.state[0] + np.random.normal(0, 0.05)  # 0.05 mol/L noise
            Ts_meas = self.state[1] + np.random.normal(0, 0.5)  # 0.5 K noise
            Tjs_meas = self.state[2] + np.random.normal(0, 0.3)  # 0.3 K noise
            F_meas = self.F + np.random.normal(0, 0.005)  # 0.005 L/s noise
            V_meas = self.V + np.random.normal(0, 1)  # 1 L noise
        else:
            CAs_meas = self.state[0]
            Ts_meas = self.state[1]
            Tjs_meas = self.state[2]
            F_meas = self.F
            V_meas = self.V

        return {
            "CAs": CAs_meas,
            "Ts": Ts_meas,
            "Tjs": Tjs_meas,
            "F": F_meas,
            "V": V_meas,
            "CA0": d_actual.CA0,
            "T0": d_actual.T0,
            "Tj0": d_actual.Tj0,
        }

    def set_setpoint(self, F_sp, V_sp):
        """Update plant setpoints (with first-order dynamics)"""
        tau = 300  # time constant (seconds)
        alpha = 0.1  # step size
        self.F = self.F + alpha * (F_sp - self.F)
        self.V = self.V + alpha * (V_sp - self.V)


# ============================================================================
# DIGITAL APPLICATIONS
# ============================================================================


class SteadyStateDetection:
    """Detects if the plant is at steady state"""

    def __init__(self, window_size=10, threshold=0.02):
        self.window_size = window_size
        self.threshold = threshold  # Relative threshold for steady state

    def detect(self, recent_data):
        """
        Detect steady state by checking variance of recent measurements
        Returns: (is_steady_state, variance_metrics)
        """
        if len(recent_data) < self.window_size:
            return False, {}

        # Extract recent values
        CAs_vals = [d["CAs"] for d in recent_data[-self.window_size :]]
        Ts_vals = [d["Ts"] for d in recent_data[-self.window_size :]]
        Tjs_vals = [d["Tjs"] for d in recent_data[-self.window_size :]]

        # Calculate relative standard deviations
        CAs_std = np.std(CAs_vals) / (np.mean(CAs_vals) + 1e-10)
        Ts_std = np.std(Ts_vals) / (np.mean(Ts_vals) + 1e-10)
        Tjs_std = np.std(Tjs_vals) / (np.mean(Tjs_vals) + 1e-10)

        metrics = {"CAs_std": CAs_std, "Ts_std": Ts_std, "Tjs_std": Tjs_std}

        # Check if all relative stds are below threshold
        is_steady = (
            CAs_std < self.threshold
            and Ts_std < self.threshold
            and Tjs_std < self.threshold
        )

        return is_steady, metrics


class SteadyStateDataReconciliation:
    """Reconciles noisy measurements to get best estimates"""

    def __init__(self):
        pass

    def reconcile(self, recent_data):
        """
        Simple reconciliation: moving average of recent measurements
        In practice, this would use a rigorous data reconciliation algorithm
        Returns: reconciled values dictionary
        """
        if not recent_data:
            return None

        # Simple moving average
        n = len(recent_data)
        reconciled = {
            "F": np.mean([d["F"] for d in recent_data]),
            "V": np.mean([d["V"] for d in recent_data]),
            "CAs": np.mean([d["CAs"] for d in recent_data]),
            "Ts": np.mean([d["Ts"] for d in recent_data]),
            "Tjs": np.mean([d["Tjs"] for d in recent_data]),
            "CA0": np.mean([d["CA0"] for d in recent_data]),
            "T0": np.mean([d["T0"] for d in recent_data]),
            "Tj0": np.mean([d["Tj0"] for d in recent_data]),
        }

        return reconciled


class SteadyStateOptimization:
    """Optimizes plant operation at steady state"""

    def __init__(self, p):
        self.p = p
        self.previous_setpoint = {"F": 0.3, "V": 400}

    def optimize(self, reconciled_data):
        """
        Optimize F and V to maximize profit given current conditions
        Returns: (F_opt, V_opt, success, profit, notes)
        """
        if reconciled_data is None:
            return (
                self.previous_setpoint["F"],
                self.previous_setpoint["V"],
                False,
                0,
                "No reconciled data available",
            )

        # Set up disturbances based on current conditions
        d = Disturbances()
        d.CA0 = reconciled_data["CA0"]
        d.T0 = reconciled_data["T0"]
        d.Tj0 = reconciled_data["Tj0"]
        d.UA_pcp = 0.350  # Assumed constant

        # Initial guess (current operating point or previous optimum)
        x0 = [reconciled_data["F"], reconciled_data["V"]]

        # Bounds
        bounds = [(0.05, 0.8), (100, 500)]

        # Constraints
        constraints = {
            "type": "ineq",
            "fun": lambda x: -np.array(nonLinearConstraints(x, d, self.p)),
        }

        try:
            # Run optimization
            result = minimize(
                lambda x: objectiveFunction(x, d, self.p),
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 50},
            )

            if result.success:
                F_opt = result.x[0]
                V_opt = result.x[1]
                profit = -result.fun

                # Update previous setpoint
                self.previous_setpoint = {"F": F_opt, "V": V_opt}

                return F_opt, V_opt, True, profit, "Optimization successful"
            else:
                return (
                    self.previous_setpoint["F"],
                    self.previous_setpoint["V"],
                    False,
                    0,
                    f"Optimization failed: {result.message}",
                )

        except Exception as e:
            return (
                self.previous_setpoint["F"],
                self.previous_setpoint["V"],
                False,
                0,
                f"Optimization error: {str(e)}",
            )


# ============================================================================
# MAIN SIMULATION
# ============================================================================


def run_simulation(phi_disturbance=0.9, sigma_CA0=0.5, sigma_T0=2.0, sigma_Tj0=1.0):
    """
    Run the integrated digital applications simulation

    Parameters:
    phi_disturbance: autocorrelation coefficient for disturbances (0 to 1)
    sigma_CA0: standard deviation of feed concentration disturbance (mol/L)
    sigma_T0: standard deviation of feed temperature disturbance (K)
    sigma_Tj0: standard deviation of cooling water temperature disturbance (K)
    """
    print("=" * 80)
    print("Digital Applications Simulation for Nonisothermal CSTR")
    print("=" * 80)
    print(
        f"Disturbance parameters: phi={phi_disturbance}, sigma_CA0={sigma_CA0}, sigma_T0={sigma_T0}, sigma_Tj0={sigma_Tj0}"
    )
    print("=" * 80)

    # Initialize components
    p = ProcessParameters()
    d = Disturbances()

    # Data sources and sinks
    historian = PlantHistorian()
    commercial_it = CommercialITSystems()
    control_system = PlantControlSystem()

    # Plant with autocorrelated disturbances
    plant = PlantSimulator(
        p,
        d,
        phi_disturbance=phi_disturbance,
        sigma_CA0=sigma_CA0,
        sigma_T0=sigma_T0,
        sigma_Tj0=sigma_Tj0,
    )

    # Digital applications
    ss_detection = SteadyStateDetection(window_size=10, threshold=0.02)
    ss_reconciliation = SteadyStateDataReconciliation()
    ss_optimization = SteadyStateOptimization(p)

    # Simulation parameters
    dt = 60  # 1 minute time step
    total_time = 7 * 24 * 3600  # 7 days
    current_time = 0

    # Initial commercial data
    commercial_it.update(
        timestamp=current_time,
        raw_material_cost=0.3,
        product_price=10.0,
        raw_material_available=True,
    )

    # Write initial setpoint
    control_system.write_setpoint(
        timestamp=current_time,
        F_sp=plant.F,
        V_sp=plant.V,
        optimization_success=True,
        notes="Initial setpoint",
    )

    # Storage for plotting
    time_history = []
    profit_history = []
    constraint_history = {"h1": [], "h4": [], "h7": []}
    ss_detected_history = []

    print(f"\nStarting simulation for {total_time / 3600:.1f} hours...")
    print(f"Plant historian update rate: {historian.update_rate} s")
    print(f"Control system update rate: {control_system.update_rate} s")
    print(f"Commercial IT update rate: {commercial_it.update_rate} s")
    print()

    # Main simulation loop
    step_count = 0
    while current_time < total_time:
        # Simulate plant
        measurement = plant.simulate_step(dt, add_noise=True, add_disturbance=True)

        # Update historian (every minute)
        if step_count % (historian.update_rate // dt) == 0:
            historian.add_measurement(
                timestamp=current_time,
                F=measurement["F"],
                V=measurement["V"],
                CAs=measurement["CAs"],
                Ts=measurement["Ts"],
                Tjs=measurement["Tjs"],
                CA0=measurement["CA0"],
                T0=measurement["T0"],
                Tj0=measurement["Tj0"],
                available=True,
            )

        # Update commercial IT (every 8 hours)
        if step_count % (commercial_it.update_rate // dt) == 0:
            # Simulate price fluctuations
            price_variation = np.random.normal(0, 0.5)
            commercial_it.update(
                timestamp=current_time,
                raw_material_cost=0.3 + np.random.normal(0, 0.02),
                product_price=10.0 + price_variation,
                raw_material_available=True,
            )
            print(
                f"[{current_time / 3600:.1f} hr] Commercial IT updated: Product price = {10.0 + price_variation:.2f}"
            )

        # Run digital applications and update control system (every 15 minutes)
        if step_count % (control_system.update_rate // dt) == 0:
            # Get recent data
            recent_data = historian.get_recent(n=15)  # Last 15 minutes

            # Steady state detection
            is_steady, ss_metrics = ss_detection.detect(recent_data)
            ss_detected_history.append(is_steady)

            if is_steady:
                print(
                    f"[{current_time / 3600:.1f} hr] Steady state DETECTED (CAs_std={ss_metrics['CAs_std']:.4f}, Ts_std={ss_metrics['Ts_std']:.4f})"
                )

                # Data reconciliation
                reconciled = ss_reconciliation.reconcile(recent_data)
                print(
                    f"[{current_time / 3600:.1f} hr] Data reconciled: F={reconciled['F']:.4f}, Ts={reconciled['Ts']:.2f} K"
                )

                # Optimization
                F_opt, V_opt, success, profit, notes = ss_optimization.optimize(
                    reconciled
                )

                # Write to control system
                control_system.write_setpoint(
                    timestamp=current_time,
                    F_sp=F_opt,
                    V_sp=V_opt,
                    optimization_success=success,
                    notes=notes,
                )

                # Update plant setpoints
                plant.set_setpoint(F_opt, V_opt)

                # Calculate constraints
                u_current = SteadyStateResult(
                    F_opt, V_opt, reconciled["CAs"], reconciled["Ts"], reconciled["Tjs"]
                )
                constraint_history["h1"].append(h1(u_current))
                constraint_history["h4"].append(h4(u_current, p, d))
                constraint_history["h7"].append(h7(u_current))
                profit_history.append(profit if success else np.nan)
            else:
                print(
                    f"[{current_time / 3600:.1f} hr] Steady state NOT detected (waiting...)"
                )
                # Keep previous setpoint
                prev_sp = control_system.get_current_setpoint()
                if prev_sp:
                    control_system.write_setpoint(
                        timestamp=current_time,
                        F_sp=prev_sp["F_setpoint"],
                        V_sp=prev_sp["V_setpoint"],
                        optimization_success=False,
                        notes="No steady state - keeping previous setpoint",
                    )

        time_history.append(current_time / 3600)  # Convert to hours
        current_time += dt
        step_count += 1

    print(f"\nSimulation complete!")
    print(f"Total measurements in historian: {len(historian.data)}")
    print(f"Total setpoints written: {len(control_system.setpoints)}")
    print(
        f"Steady state detected {sum(ss_detected_history)}/{len(ss_detected_history)} times"
    )

    # Plot results
    plot_results(
        historian, control_system, commercial_it, profit_history, constraint_history
    )


def plot_results(
    historian, control_system, commercial_it, profit_history, constraint_history
):
    """Plot simulation results"""
    # Extract data for plotting
    hist_times = [d["timestamp"] / 3600 for d in historian.data]
    hist_Ts = [d["Ts"] for d in historian.data]
    hist_CAs = [d["CAs"] for d in historian.data]
    hist_F = [d["F"] for d in historian.data]
    hist_CA0 = [d["CA0"] for d in historian.data]
    hist_T0 = [d["T0"] for d in historian.data]
    hist_Tj0 = [d["Tj0"] for d in historian.data]

    sp_times = [d["timestamp"] / 3600 for d in control_system.setpoints]
    sp_F = [d["F_setpoint"] for d in control_system.setpoints]
    sp_V = [d["V_setpoint"] for d in control_system.setpoints]
    sp_success = [d["optimization_success"] for d in control_system.setpoints]

    comm_times = [d["timestamp"] / 3600 for d in commercial_it.data]
    comm_rm_cost = [d["raw_material_cost"] for d in commercial_it.data]
    comm_prod_price = [d["product_price"] for d in commercial_it.data]

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))

    # Plot 1: Reactor temperature
    axes[0, 0].plot(hist_times, hist_Ts, "b-", alpha=0.7, linewidth=1)
    axes[0, 0].axhline(y=350, color="r", linestyle="--", label="Constraint (350 K)")
    axes[0, 0].set_xlabel("Time (hr)")
    axes[0, 0].set_ylabel("Reactor Temp Ts (K)")
    axes[0, 0].set_title("Reactor Temperature")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Concentration
    axes[0, 1].plot(hist_times, hist_CAs, "g-", alpha=0.7, linewidth=1)
    axes[0, 1].axhline(y=5, color="r", linestyle="--", label="Constraint (5 mol/L)")
    axes[0, 1].set_xlabel("Time (hr)")
    axes[0, 1].set_ylabel("Concentration CAs (mol/L)")
    axes[0, 1].set_title("Product Concentration")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Flow rate with setpoints
    axes[1, 0].plot(hist_times, hist_F, "b-", alpha=0.5, linewidth=1, label="Measured")
    axes[1, 0].plot(
        sp_times, sp_F, "r-", linewidth=2, marker="o", markersize=4, label="Setpoint"
    )
    axes[1, 0].set_xlabel("Time (hr)")
    axes[1, 0].set_ylabel("Flow rate F (L/s)")
    axes[1, 0].set_title("Feed Flow Rate")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Volume setpoints
    axes[1, 1].plot(sp_times, sp_V, "r-", linewidth=2, marker="o", markersize=4)
    axes[1, 1].set_xlabel("Time (hr)")
    axes[1, 1].set_ylabel("Volume V (L)")
    axes[1, 1].set_title("Reactor Volume Setpoint")
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Profit
    if profit_history:
        opt_times = [sp_times[i] for i in range(len(profit_history))]
        axes[2, 0].plot(
            opt_times, profit_history, "g-", linewidth=2, marker="s", markersize=4
        )
        axes[2, 0].set_xlabel("Time (hr)")
        axes[2, 0].set_ylabel("Profit")
        axes[2, 0].set_title("Optimized Profit")
        axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Constraint violations
    if constraint_history["h1"]:
        opt_times = [sp_times[i] for i in range(len(constraint_history["h1"]))]
        axes[2, 1].plot(
            opt_times,
            constraint_history["h1"],
            "r-",
            linewidth=2,
            marker="o",
            label="h1 (Temp)",
        )
        axes[2, 1].plot(
            opt_times,
            constraint_history["h4"],
            "b-",
            linewidth=2,
            marker="s",
            label="h4 (Cooling)",
        )
        axes[2, 1].plot(
            opt_times,
            constraint_history["h7"],
            "m-",
            linewidth=2,
            marker="^",
            label="h7 (Conc)",
        )
        axes[2, 1].axhline(y=0, color="k", linestyle="--", linewidth=1)
        axes[2, 1].set_xlabel("Time (hr)")
        axes[2, 1].set_ylabel("Constraint Value (≤0 feasible)")
        axes[2, 1].set_title("Constraint Status")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

    # Plot 7: Feed concentration disturbance (CA0)
    axes[0, 2].plot(hist_times, hist_CA0, "purple", alpha=0.7, linewidth=1.5)
    axes[0, 2].axhline(
        y=20, color="k", linestyle="--", linewidth=1, label="Nominal (20 mol/L)"
    )
    axes[0, 2].set_xlabel("Time (hr)")
    axes[0, 2].set_ylabel("CA0 (mol/L)")
    axes[0, 2].set_title("Feed Concentration Disturbance")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 8: Feed temperature disturbance (T0)
    axes[1, 2].plot(hist_times, hist_T0, "orange", alpha=0.7, linewidth=1.5)
    axes[1, 2].axhline(
        y=300, color="k", linestyle="--", linewidth=1, label="Nominal (300 K)"
    )
    axes[1, 2].set_xlabel("Time (hr)")
    axes[1, 2].set_ylabel("T0 (K)")
    axes[1, 2].set_title("Feed Temperature Disturbance")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Plot 9: Cooling water temperature disturbance (Tj0)
    axes[2, 2].plot(hist_times, hist_Tj0, "cyan", alpha=0.7, linewidth=1.5)
    axes[2, 2].axhline(
        y=283, color="k", linestyle="--", linewidth=1, label="Nominal (283 K)"
    )
    axes[2, 2].set_xlabel("Time (hr)")
    axes[2, 2].set_ylabel("Tj0 (K)")
    axes[2, 2].set_title("Cooling Water Temperature Disturbance")
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    # Plot 10: Raw material cost
    axes[3, 0].plot(
        comm_times,
        comm_rm_cost,
        "brown",
        linewidth=2,
        marker="o",
        markersize=6,
        linestyle="-",
    )
    axes[3, 0].set_xlabel("Time (hr)")
    axes[3, 0].set_ylabel("Cost ($/unit)")
    axes[3, 0].set_title("Raw Material Cost")
    axes[3, 0].grid(True, alpha=0.3)

    # Plot 11: Product price
    axes[3, 1].plot(
        comm_times,
        comm_prod_price,
        "darkgreen",
        linewidth=2,
        marker="s",
        markersize=6,
        linestyle="-",
    )
    axes[3, 1].set_xlabel("Time (hr)")
    axes[3, 1].set_ylabel("Price ($/unit)")
    axes[3, 1].set_title("Product Price")
    axes[3, 1].grid(True, alpha=0.3)

    # Plot 12: Optimization success rate
    success_count = []
    window = 10
    for i in range(len(sp_success)):
        start_idx = max(0, i - window + 1)
        success_rate = (
            sum(sp_success[start_idx : i + 1])
            / len(sp_success[start_idx : i + 1])
            * 100
        )
        success_count.append(success_rate)

    axes[3, 2].plot(sp_times, success_count, "darkblue", linewidth=2)
    axes[3, 2].set_xlabel("Time (hr)")
    axes[3, 2].set_ylabel("Success Rate (%)")
    axes[3, 2].set_title(f"Optimization Success Rate (window={window})")
    axes[3, 2].set_ylim([0, 105])
    axes[3, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("digital_applications_simulation.png", dpi=300, bbox_inches="tight")
    print("\nPlot saved to: digital_applications_simulation.png")

    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    run_simulation()
