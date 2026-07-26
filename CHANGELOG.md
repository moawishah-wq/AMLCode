"""Numbers, tables and figure for "Exact uncertainty subspaces for affine
differential inclusions".

For a segment disturbance D(s) = [-eps*b, eps*b] and z(0) = 0, Eq. (4) gives

    h_R(T)(u) = eps * int_0^T |<u, Phi(T,s) b>| ds,

so the reachable set follows from a scalar quadrature per direction.

    python reproduce.py [--out DIR] [--dirs N] [--nodes N]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 0.10          # disturbance half-width
T = 3.0             # terminal time
N_TIME = 6001       # time quadrature nodes
N_DIR = 6000        # directions sampled on the circle
CHUNK = 512         # directions per block; see support_function()

angles = np.linspace(0, 2 * np.pi, N_DIR, endpoint=False)
d_angle = 2 * np.pi / N_DIR
times = np.linspace(0, T, N_TIME)


def set_resolution(n_dir=N_DIR, n_time=N_TIME):
    """Rebuild the angular and time grids (used by the CLI and by the tests)."""
    global N_DIR, N_TIME, angles, d_angle, times
    N_DIR, N_TIME = n_dir, n_time
    angles = np.linspace(0, 2 * np.pi, N_DIR, endpoint=False)
    d_angle = 2 * np.pi / N_DIR
    times = np.linspace(0, T, N_TIME)


def support_function(propagated_b, directions):
    """h(u) = eps * int |<u, Phi(T,s) b>| ds for each row u of `directions`.

    Blocked: the full integrand would be (N_TIME x N_DIR), a few hundred MB.
    """
    supports = np.empty(len(directions))
    for start in range(0, len(directions), CHUNK):
        block = directions[start:start + CHUNK]
        integrand = np.abs(propagated_b @ block.T)          # (N_TIME, <=CHUNK)
        supports[start:start + CHUNK] = EPS * np.trapezoid(integrand, times, axis=0)
    return supports


def boundary_and_area(support):
    """Boundary and area of a planar convex body from its support function.

    Boundary: the envelope x(u) = h(u) u + h'(u) u_perp.
    Area:     (1/2) int (h^2 - h'^2) dtheta.
    Both differentiate h by a central difference, so both assume h is smooth.
    See README (limitations) for what happens when it is not.
    """
    dh = (np.roll(support, -1) - np.roll(support, 1)) / (2 * d_angle)
    x = support * np.cos(angles) - dh * np.sin(angles)
    y = support * np.sin(angles) + dh * np.cos(angles)
    area = 0.5 * np.sum(support ** 2 - dh ** 2) * d_angle
    return x, y, area


def uncertainty_dimension(A, b):
    """dim W = rank[b, Ab, ..., A^(m-1) b] (Corollary 6; Cayley-Hamilton truncates)."""
    m = len(A)
    columns, w = [b], b.copy()
    for _ in range(m - 1):
        w = A @ w
        columns.append(w)
    return int(np.linalg.matrix_rank(np.column_stack(columns)))


def damped_oscillator():
    """x'' + 2*d*w*x' + w^2*x = f(t), |f| <= eps, as z' = Az + f(t) b.

    The force enters the acceleration coordinate only, so D is the segment
    {0} x [-eps, eps], not a disc: V = span{b} is one-dimensional.
    """
    omega, damping = 2.0, 0.15
    A = np.array([[0.0, 1.0], [-omega ** 2, -2 * damping * omega]])
    b = np.array([0.0, 1.0])

    # Phi(T,s) = exp(A(T-s)); underdamped closed form, sampled at tau = T-s.
    decay = damping * omega
    freq = np.sqrt(omega ** 2 - decay ** 2)
    envelope = np.exp(-decay * times)
    sin, cos = np.sin(freq * times), np.cos(freq * times)

    Phi = np.empty((N_TIME, 2, 2))
    Phi[:, 0, 0] = envelope * (cos + decay * sin / freq)
    Phi[:, 0, 1] = envelope * sin / freq
    Phi[:, 1, 0] = -envelope * omega ** 2 * sin / freq
    Phi[:, 1, 1] = envelope * (cos - decay * sin / freq)

    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    support = support_function(Phi @ b, directions)
    x, y, area = boundary_and_area(support)

    enclosing = support.max()
    flow_norm = EPS * np.trapezoid([np.linalg.norm(P, 2) for P in Phi], times)
    norm_A = np.linalg.norm(A, 2)
    gronwall = EPS * np.expm1(norm_A * T) / norm_A

    return {
        "support_min": support.min(),
        "support_max": support.max(),
        "anisotropy": support.max() / support.min(),
        "exact_area": area,
        "smallest_ball_radius": enclosing,
        "smallest_ball_area": np.pi * enclosing ** 2,
        "smallest_ball_ratio": np.pi * enclosing ** 2 / area,
        "flow_norm_ball_radius": flow_norm,
        "flow_norm_ball_area": np.pi * flow_norm ** 2,
        "flow_norm_ball_ratio": np.pi * flow_norm ** 2 / area,
        "gronwall_radius": gronwall,
        "gronwall_area": np.pi * gronwall ** 2,
        "gronwall_ratio": np.pi * gronwall ** 2 / area,
        "norm_A": norm_A,
        "dim_W": uncertainty_dimension(A, b),
    }, (x, y)


def consensus_network():
    """z' = -Lambda z + f(t) e on a three-node path graph, |f| <= eps.

    e = (1,-1,0)/sqrt(2), so e _|_ 1 and 1'Lambda = 0: the average is conserved and
    W(T) = 1^perp. The support is therefore sampled inside that plane, not the sphere.
    """
    laplacian = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    A = -laplacian
    edge = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    ones = np.ones(3)

    # exp(A*tau) e via eigh (Lambda is symmetric). Conservation then holds to
    # rounding rather than quadrature error; see the corresponding check in
    # test_reproduce.py.
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    coefficients = eigenvectors.T @ edge
    propagated_e = (np.exp(times[:, None] * eigenvalues) * coefficients) @ eigenvectors.T

    q1 = edge
    q2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)        # o.n. basis of 1^perp with q1
    plane = np.cos(angles)[:, None] * q1 + np.sin(angles)[:, None] * q2

    support = support_function(propagated_e, plane)
    x, y, _ = boundary_and_area(support)

    special = support_function(propagated_e, np.vstack((q1, q2, ones)))
    dim_W = uncertainty_dimension(A, edge)

    return {
        "half_width_q1": special[0],
        "half_width_q2": special[1],
        "support_along_ones": special[2],
        "max_support_radius": support.max(),
        "diameter": 2 * support.max(),
        "dim_W": dim_W,
        "dim_N": 3 - dim_W,
        "null_direction": (ones / np.sqrt(3)).tolist(),
    }, (x, y)


def switching_disturbance(t):
    """A = diag(-1,-2); D(s) is a segment along e1 for s < 1, along e2 after.

    A is diagonal, so Phi preserves the axes and Eq. (4) integrates in closed form:
    no quadrature, hence no discretisation error.
    """
    reach_e1 = EPS * (np.exp(-(t - min(t, 1.0))) - np.exp(-t)) if t > 0 else 0.0
    reach_e2 = EPS * (1 - np.exp(-2 * (t - 1))) / 2 if t > 1 else 0.0

    radius = float(np.hypot(reach_e1, reach_e2))
    dim_W = int(reach_e1 > 1e-14) + int(reach_e2 > 1e-14)
    return {
        "half_width_e1": reach_e1,
        "half_width_e2": reach_e2,
        "dim_W": dim_W,
        "dim_N": 2 - dim_W,
        "exact_area": 4 * reach_e1 * reach_e2,      # zero while the set is a segment
        "smallest_ball_radius": radius,
        "smallest_ball_area": np.pi * radius ** 2,
        "fabricated_width": 2 * radius,
    }


def write_tables(oscillator, consensus, switching, out):
    with open(out / "table_1.csv", "w", newline="") as handle:
        table = csv.writer(handle)
        table.writerow(["certificate", "radius_or_extent", "area", "relative_area"])
        table.writerow(["Gronwall ball", oscillator["gronwall_radius"],
                        oscillator["gronwall_area"], oscillator["gronwall_ratio"]])
        table.writerow(["Flow-norm ball", oscillator["flow_norm_ball_radius"],
                        oscillator["flow_norm_ball_area"], oscillator["flow_norm_ball_ratio"]])
        table.writerow(["Smallest enclosing ball", oscillator["smallest_ball_radius"],
                        oscillator["smallest_ball_area"], oscillator["smallest_ball_ratio"]])
        table.writerow(["Exact set",
                        f"[{oscillator['support_min']:.6g}, {oscillator['support_max']:.6g}]",
                        oscillator["exact_area"], 1.0])

    with open(out / "table_2.csv", "w", newline="") as handle:
        table = csv.writer(handle)
        table.writerow(["example", "m", "dim_V", "dim_W", "dim_N", "null_direction",
                        "exact_ambient_measure", "ball_ambient_measure", "fabricated_width"])
        table.writerow(["Oscillator, T=3", 2, 1, oscillator["dim_W"], 0, "none",
                        oscillator["exact_area"], oscillator["smallest_ball_area"], ""])
        table.writerow(["Consensus, T=3", 3, 1, consensus["dim_W"], consensus["dim_N"],
                        "ones/sqrt(3)", "0 in R^3", ">0", consensus["diameter"]])
        for label, case in switching.items():
            table.writerow([f"Switching, {label}", 2, 1, case["dim_W"], case["dim_N"],
                            "e2" if case["dim_N"] else "none",
                            case["exact_area"] if case["dim_W"] == 2 else "0 in R^2",
                            case["smallest_ball_area"],
                            case["fabricated_width"] if case["dim_N"] else ""])


def draw_figure(oscillator, oscillator_boundary, consensus, consensus_boundary, out):
    circle = np.linspace(0, 2 * np.pi, 1000)
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.15))

    r_flow = oscillator["flow_norm_ball_radius"]
    r_ball = oscillator["smallest_ball_radius"]
    x, y = oscillator_boundary
    left.grid(True, lw=0.3, color="0.85")
    left.plot(r_flow * np.cos(circle), r_flow * np.sin(circle), ls=(0, (1, 2.5)), lw=1.3,
              color="0.45",
              label=f"flow-norm ball (6), area {oscillator['flow_norm_ball_area']:.3f}")
    left.plot(r_ball * np.cos(circle), r_ball * np.sin(circle), ls=(0, (5, 2.5)), lw=1.3,
              color="tab:blue",
              label=f"smallest enclosing ball, area {oscillator['smallest_ball_area']:.3f}")
    left.fill(x, y, color="crimson", alpha=0.16)
    left.plot(np.append(x, x[0]), np.append(y, y[0]), color="crimson", lw=1.7,
              label=f"exact set (3), area {oscillator['exact_area']:.3f}")
    left.plot(0, 0, "k+", ms=7, mew=1.2)
    left.annotate(r"$\bar{z}(T)$", (0.018, 0.02), fontsize=8.5)
    left.set_xlabel(r"$x$", fontsize=10)
    left.set_ylabel(r"$\dot{x}$", fontsize=10)
    left.set_title(r"(a) oscillator: $\dim W(T)=2$, no null direction", fontsize=9, pad=6)
    left.set_aspect("equal")
    left.set_xlim(-0.38, 0.38)
    left.set_ylim(-0.38, 0.38)
    left.tick_params(labelsize=7.5)
    left.legend(fontsize=6.8, loc="lower left", framealpha=0.92, borderpad=0.4)

    r_consensus = consensus["max_support_radius"]
    x, y = consensus_boundary
    right.grid(True, lw=0.3, color="0.85")
    right.plot(r_consensus * np.cos(circle), r_consensus * np.sin(circle),
               ls=(0, (5, 2.5)), lw=1.3, color="tab:blue",
               label=f"smallest enclosing ball, $r={r_consensus:.4f}$")
    right.fill(x, y, color="crimson", alpha=0.16)
    right.plot(np.append(x, x[0]), np.append(y, y[0]), color="crimson", lw=1.7,
               label=r"exact set (3), in the plane $\mathbf{1}^{\perp}$")
    right.plot(0, 0, "k+", ms=7, mew=1.2)
    right.annotate(r"$\mathbf{1}/\sqrt{3}$ points out of" "\n" r"this plane: width $0$",
                   xy=(0, 0), xytext=(-0.068, 0.045), fontsize=7.5,
                   arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    right.set_xlabel(r"$q_1=e$", fontsize=10)
    right.set_ylabel(r"$q_2$", fontsize=10)
    right.set_title(r"(b) consensus: $\dim W(T)=2$, $N(T)=\mathrm{span}\{\mathbf{1}\}$",
                    fontsize=9, pad=6)
    right.set_aspect("equal")
    right.set_xlim(-0.075, 0.075)
    right.set_ylim(-0.075, 0.075)
    right.tick_params(labelsize=7.5)
    right.legend(fontsize=6.8, loc="lower left", framealpha=0.92, borderpad=0.4)

    figure.tight_layout(pad=0.4, w_pad=1.6)
    figure.savefig(out / "figure_1.pdf")
    figure.savefig(out / "figure_1.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("outputs"),
                        help="output directory (default: outputs)")
    parser.add_argument("--dirs", type=int, default=N_DIR, help="directions sampled")
    parser.add_argument("--nodes", type=int, default=N_TIME, help="time quadrature nodes")
    args = parser.parse_args()

    set_resolution(args.dirs, args.nodes)
    args.out.mkdir(parents=True, exist_ok=True)

    oscillator, oscillator_boundary = damped_oscillator()
    consensus, consensus_boundary = consensus_network()
    switching = {f"t={t:g}": switching_disturbance(t) for t in (0.5, 2.0)}

    results = {"oscillator": oscillator, "consensus": consensus, "switching": switching}
    with open(args.out / "numerical_results.json", "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=float)

    write_tables(oscillator, consensus, switching, args.out)
    draw_figure(oscillator, oscillator_boundary, consensus, consensus_boundary, args.out)

    print(json.dumps(results, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {args.out}/: numerical_results.json, table_1.csv, table_2.csv, "
          "figure_1.pdf/.png")


if __name__ == "__main__":
    main()
