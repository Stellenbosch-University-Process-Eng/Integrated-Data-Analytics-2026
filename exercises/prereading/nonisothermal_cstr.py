"""
Ordinary differential equations for a nonisothermal continuous stirred-tank reactor (CSTR)
Python implementation converted from MATLAB code.
Case study from Romagnoli and Plazoglu (2020) Introduction to Process Control, Chapter 20.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class ProcessParameters:
    """Process parameters structure"""

    def __init__(self):
        self.dH_pcp = 5  # K.L/mol, heat of reaction divided by molar heat capacity
        self.E_R = 6000  # K, activation energy divided by universal gas constant
        self.Fj = 1.04  # L/s, cooling water flowrate
        self.k0 = 2.7e5  # 1/s, pre-exponential rate coefficient
        self.Vj = 10  # L, cooling jacket volume


class Disturbances:
    """Nominal values for process disturbances"""

    def __init__(self):
        self.Tj0 = 283  # K, cooling water temperature
        self.T0 = 300  # K, feed stream temperature
        self.CA0 = 20  # mol/L feed stream concentration
        self.UA_pcp = 0.350  # K/s, jacket overall heat transfer coefficient and area


class SteadyStateResult:
    """Structure to hold steady state results"""

    def __init__(self, F, V, CAs, Ts, Tjs):
        self.F = F  # L/s, feed flowrate
        self.V = V  # L, reactor volume
        self.CAs = CAs  # mol/L, steady state concentration
        self.Ts = Ts  # K, steady state reactor temperature
        self.Tjs = Tjs  # K, steady state jacket temperature


def ODEs(t, x, u, d, p):
    """
    Ordinary differential equations for nonisothermal CSTR

    Parameters:
    t: time (not used, but required by solve_ivp)
    x: state vector [CAs, Ts, Tjs]
    u: SteadyStateResult object with F and V
    d: Disturbances object
    p: ProcessParameters object

    Returns:
    dxdt: derivative vector
    """
    CAs = x[0]
    Ts = x[1]
    Tjs = x[2]

    # Reaction rate
    rxn = p.k0 * CAs * np.exp(-p.E_R / Ts)

    # ODEs
    dCAs_dt = u.F / u.V * (d.CA0 - CAs) - rxn
    dTs_dt = u.F / u.V * (d.T0 - Ts) + p.dH_pcp * rxn - d.UA_pcp / u.V * (Ts - Tjs)
    dTjs_dt = p.Fj / p.Vj * (d.Tj0 - Tjs) + d.UA_pcp / p.Vj * (Ts - Tjs)

    return [dCAs_dt, dTs_dt, dTjs_dt]


def SteadyState(F, V, d, p):
    """
    Find steady state values of CAs, Ts, Tjs

    Parameters:
    F: feed flowrate (L/s)
    V: reactor volume (L)
    d: Disturbances object
    p: ProcessParameters object

    Returns:
    u: SteadyStateResult object
    """
    # Create temporary result object for integration
    u_temp = SteadyStateResult(F, V, d.CA0, d.T0, d.T0)

    # Initial conditions
    x0 = [d.CA0, d.T0, d.T0]

    # Solve ODEs to steady state
    sol = solve_ivp(
        lambda t, x: ODEs(t, x, u_temp, d, p),
        [0, 1e4],
        x0,
        method="BDF",
        dense_output=True,
    )

    # Extract steady state values
    CAs = sol.y[0, -1]
    Ts = sol.y[1, -1]
    Tjs = sol.y[2, -1]

    return SteadyStateResult(F, V, CAs, Ts, Tjs)


def h1(u):
    """h1: maximum reactor temperature constraint"""
    return u.Ts - 350  # K


def h4(u, p, d):
    """h4: maximum cooling rate constraint"""
    return p.Fj * (u.Tjs - d.Tj0) - 20.1


def h7(u):
    """h7: maximum reactant concentration in product stream constraint"""
    return u.CAs - 5


def phi(u, d, p):
    """Profit function"""
    return (
        10 * u.F * (d.CA0 - u.CAs) - 0.3 * u.F * d.CA0 - 0.01 * p.Fj * (d.Tj0 - u.Tjs)
    )


def nonLinearConstraints(x, d, p):
    """
    The checks for constraint violation

    Parameters:
    x: decision variables [F, V]
    d: Disturbances object
    p: ProcessParameters object

    Returns:
    constraints: list of constraint values (must be <= 0 for scipy.optimize.minimize)
    """
    F = x[0]
    V = x[1]
    u = SteadyState(F, V, d, p)

    # All constraints must be <= 0
    c = [h1(u), h4(u, p, d), h7(u)]

    return c


def objectiveFunction(x, d, p):
    """
    Calculates the objective function to minimize (-profit)

    Parameters:
    x: decision variables [F, V]
    d: Disturbances object
    p: ProcessParameters object

    Returns:
    J: negative profit (to minimize)
    """
    F = x[0]
    V = x[1]
    u = SteadyState(F, V, d, p)
    J = -phi(u, d, p)

    return J


def main():
    """Main simulation and optimization"""
    # Initialize parameters and disturbances
    p = ProcessParameters()
    d = Disturbances()

    # Define bounds
    Vmin = 100  # L, h2: minimum reactor volume
    Vmax = 500  # L, h3: maximum reactor volume
    Fmin = 0.05  # L/s, h5: minimum feed flowrate
    Fmax = 0.8  # L/s, h6: maximum feed flowrate
    N = 20  # Number of gridpoints to evaluate is N^2

    # Create grid
    F_grid = np.linspace(Fmin, Fmax, N)
    V_grid = np.linspace(Vmin, Vmax, N)
    F, V = np.meshgrid(F_grid, V_grid)

    # Initialize arrays for constraints and profit
    H1 = np.zeros((N, N))
    H4 = np.zeros((N, N))
    H7 = np.zeros((N, N))
    Phi = np.zeros((N, N))

    # Loop over every gridpoint: For visualization
    print("Evaluating grid points...")
    for i in range(N):
        for j in range(N):
            u = SteadyState(F[i, j], V[i, j], d, p)
            # Calculate constraint functions at each grid point
            H1[i, j] = h1(u)
            H4[i, j] = h4(u, p, d)
            H7[i, j] = h7(u)
            # Calculate profit at each grid point
            Phi[i, j] = phi(u, d, p)

    print("Grid evaluation complete.")

    # Find optimal F and V
    Fguess = 0.3  # initial guess
    Vguess = 400  # initial guess
    x0 = [Fguess, Vguess]

    # Bounds
    bounds = [(Fmin, Fmax), (Vmin, Vmax)]

    # Constraints for scipy.optimize.minimize
    # For minimize with method='SLSQP', constraints should be dictionaries
    constraints = {
        "type": "ineq",
        "fun": lambda x: -np.array(nonLinearConstraints(x, d, p)),
    }

    print("Running optimization...")
    result = minimize(
        lambda x: objectiveFunction(x, d, p),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    x_opt = result.x
    print(f"Optimization complete.")
    print(f"Optimal F: {x_opt[0]:.4f} L/s")
    print(f"Optimal V: {x_opt[1]:.2f} L")
    print(f"Maximum profit: {-result.fun:.4f}")

    # Show constraints and objective function with optimization result
    plt.figure(figsize=(12, 8))

    # Contour plot for profit
    contourf = plt.contourf(F, V, Phi, levels=20, cmap="viridis")

    # Constraint contours
    h1_line = plt.contour(F, V, H1, [0], colors="k", linewidths=2, linestyles="solid")
    h1_neg = plt.contour(F, V, H1, [-1], colors="k", linewidths=2, linestyles="dashed")

    h4_line = plt.contour(F, V, H4, [0], colors="r", linewidths=2, linestyles="solid")
    h4_neg = plt.contour(F, V, H4, [-1], colors="r", linewidths=2, linestyles="dashed")

    h7_line = plt.contour(
        F, V, H7, [0], colors="magenta", linewidths=2, linestyles="solid"
    )
    h7_neg = plt.contour(
        F, V, H7, [-1], colors="magenta", linewidths=2, linestyles="dashed"
    )

    # Colorbar
    cbar = plt.colorbar(contourf)
    cbar.set_label("Phi (Profit)", fontsize=12)

    # Plot optimum
    plt.plot(
        x_opt[0], x_opt[1], "wx", markersize=12, markeredgewidth=3, label="Optimum"
    )

    # Labels and legend
    plt.xlabel("F (L/s)", fontsize=12)
    plt.ylabel("V (L)", fontsize=12)
    plt.title("CSTR Optimization with Constraints", fontsize=14)

    # Create custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="k", linewidth=2, label="h1 constraint"),
        Line2D([0], [0], color="k", linewidth=2, linestyle="--", label="h1 decreasing"),
        Line2D([0], [0], color="r", linewidth=2, label="h4 constraint"),
        Line2D([0], [0], color="r", linewidth=2, linestyle="--", label="h4 decreasing"),
        Line2D([0], [0], color="magenta", linewidth=2, label="h7 constraint"),
        Line2D(
            [0],
            [0],
            color="magenta",
            linewidth=2,
            linestyle="--",
            label="h7 decreasing",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markersize=10,
            markeredgewidth=2,
            linestyle="None",
            label="Optimum",
        ),
    ]
    plt.legend(handles=legend_elements, loc="best", fontsize=10)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    output_file = "cstr_optimization.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")

    # Try to show if interactive backend is available
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    main()
