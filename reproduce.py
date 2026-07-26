"""Verification suite for the manuscript reproduction bundle.

Run with either::

    pytest
    python test_reproduce.py

The suite always generates a fresh set of outputs in a temporary directory before
checking them. It therefore cannot pass merely because an old outputs/ directory is
present, and it never silently skips the manuscript-agreement checks.

Tolerances. Time quadrature is trapezoidal on 6001 nodes over [0, 3] (step 5e-4).
The integrand is smooth apart from isolated zeros, so the error is about 1e-6.
Angular sampling at 6000 directions misses a support extremum by about 3e-7
relative. The area formula differentiates h and loses an order. Tolerances below
are deliberately looser than that numerical noise, so a pass tests the model and
the reported rounded values rather than incidental floating-point details. The
switching example is closed form and receives tighter checks.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import reproduce
from reproduce import EPS, T, boundary_and_area, support_function, uncertainty_dimension

ROOT = Path(__file__).resolve().parent
EXPECTED_OUTPUTS = {
    "numerical_results.json",
    "table_1.csv",
    "table_2.csv",
    "figure_1.pdf",
    "figure_1.png",
}


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the public CLI once and return its fresh temporary output directory."""
    out = tmp_path_factory.mktemp("reproduction_outputs")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "reproduce.py"), "--out", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote" in completed.stdout
    return out


@pytest.fixture(scope="session")
def results(output_dir: Path) -> dict:
    return json.loads((output_dir / "numerical_results.json").read_text(encoding="utf-8"))


# --- closed-form checks of the numerical machinery --------------------------


def test_area_of_a_disc():
    disc = np.full(len(reproduce.angles), 0.37)
    _, _, area = boundary_and_area(disc)
    assert abs(area - np.pi * 0.37**2) < 1e-9


def test_area_of_a_segment_is_only_first_order():
    # h(u) = a|<u,e1>| has kinks, so the central difference for h' is wrong
    # there and the area comes out at about 4e-5 instead of 0. Widths stay exact.
    segment = 0.2 * np.abs(np.cos(reproduce.angles))
    _, _, area = boundary_and_area(segment)
    assert abs(area) < 1e-4
    quarter = len(reproduce.angles) // 4
    assert abs(segment[quarter] + segment[3 * quarter]) < 1e-15


def test_identity_flow():
    # A = 0, so R(T) = [-eps*T, eps*T] b exactly.
    flow = np.tile(np.array([0.0, 1.0]), (reproduce.N_TIME, 1))
    assert abs(support_function(flow, np.array([[0.0, 1.0]]))[0] - EPS * T) < 1e-12
    assert abs(support_function(flow, np.array([[1.0, 0.0]]))[0]) < 1e-15


def test_eigenvector_disturbance_does_not_spread():
    # Guards uncertainty_dimension() against always returning full rank.
    A = np.diag([-1.0, -2.0])
    assert uncertainty_dimension(A, np.array([1.0, 0.0])) == 1
    assert uncertainty_dimension(A, np.array([1.0, 1.0])) == 2


# --- fresh-output and metadata checks ---------------------------------------


def test_all_declared_outputs_are_created(output_dir: Path):
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    assert EXPECTED_OUTPUTS <= actual
    for name in EXPECTED_OUTPUTS:
        assert (output_dir / name).stat().st_size > 0


def test_table_2_uses_dimension_neutral_measure_labels(output_dir: Path):
    with (output_dir / "table_2.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert "exact_ambient_measure" in reader.fieldnames
        assert "ball_ambient_measure" in reader.fieldnames
        assert "exact_area" not in reader.fieldnames
        assert "ball_area" not in reader.fieldnames
        rows = list(reader)
    consensus = next(row for row in rows if row["example"] == "Consensus, T=3")
    assert consensus["exact_ambient_measure"] == "0 in R^3"
    assert consensus["ball_ambient_measure"] == ">0"


def test_release_date_metadata():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert "date-released: 2026-07-26" in citation
    assert zenodo["publication_date"] == "2026-07-26"
    assert "given-names: Muhammad" in citation
    assert "family-names: Hagui" in citation
    assert "Woodbridge Senior High School, Woodbridge, VA, USA" in citation
    creators = {creator["name"]: creator["affiliation"] for creator in zenodo["creators"]}
    assert creators["Hagui, Muhammad"] == "Woodbridge Senior High School, Woodbridge, VA, USA"
    assert "Mahapatra, Rahul" not in creators
    assert zenodo["version"] == "1.0.0"


# --- agreement with every numerical quantity reported in the examples -------


@pytest.mark.parametrize(
    "key, expected, tol",
    [
        ("support_min", 0.065, 5e-4),
        ("support_max", 0.124, 5e-4),
        ("anisotropy", 1.90, 5e-3),
        ("exact_area", 0.0250, 5e-5),
        ("smallest_ball_radius", 0.124, 5e-4),
        ("smallest_ball_area", 0.0480, 1e-4),
        ("smallest_ball_ratio", 1.92, 5e-3),
        ("flow_norm_ball_radius", 0.325, 5e-4),
        ("flow_norm_ball_area", 0.332, 5e-4),
        ("flow_norm_ball_ratio", 13.3, 5e-2),
        ("gronwall_radius", 4.64e3, 5.0),
        ("gronwall_area", 6.76e7, 5.0e4),
        ("gronwall_ratio", 2.7e9, 5.0e6),
        ("norm_A", 4.05, 5e-3),
    ],
)
def test_oscillator_reported_quantities(results, key, expected, tol):
    assert abs(results["oscillator"][key] - expected) < tol


def test_subspace_dimensions(results):
    assert results["oscillator"]["dim_W"] == 2
    assert results["consensus"]["dim_W"] == 2
    assert results["consensus"]["dim_N"] == 1


def test_conserved_direction(results):
    consensus = results["consensus"]
    null = np.array(consensus["null_direction"])
    assert np.linalg.norm(null - np.ones(3) / np.sqrt(3)) < 1e-12
    # Conservation is exact in theory, so hold it to rounding, not quadrature.
    assert abs(consensus["support_along_ones"]) < 1e-14


@pytest.mark.parametrize(
    "key, expected, tol",
    [
        ("half_width_q1", 0.0488, 5e-5),
        ("half_width_q2", 0.0267, 5e-5),
        ("max_support_radius", 0.0556, 5e-5),
        ("diameter", 0.111, 5e-4),
    ],
)
def test_consensus_reported_quantities(results, key, expected, tol):
    assert abs(results["consensus"][key] - expected) < tol


def test_switching_reported_quantities(results):
    early = results["switching"]["t=0.5"]
    late = results["switching"]["t=2"]

    assert early["dim_W"] == 1 and early["dim_N"] == 1
    assert late["dim_W"] == 2 and late["dim_N"] == 0

    assert abs(early["half_width_e1"] - 0.0394) < 6e-5
    assert early["half_width_e2"] == 0.0
    assert early["exact_area"] == 0.0
    assert abs(early["fabricated_width"] - 0.0787) < 5e-5

    assert abs(late["half_width_e1"] - 0.0233) < 5e-5
    assert abs(late["half_width_e2"] - 0.0432) < 5e-5
    assert abs(late["exact_area"] - 0.00402) < 5e-6
    assert abs(late["smallest_ball_area"] - 0.00757) < 5e-6


def test_csv_values_match_json(results, output_dir: Path):
    """The exported tables must agree with the full-precision JSON source."""
    with (output_dir / "table_1.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["certificate"]: row for row in csv.DictReader(handle)}
    assert float(rows["Gronwall ball"]["area"]) == results["oscillator"]["gronwall_area"]
    assert float(rows["Flow-norm ball"]["relative_area"]) == results["oscillator"][
        "flow_norm_ball_ratio"
    ]
    assert float(rows["Smallest enclosing ball"]["area"]) == results["oscillator"][
        "smallest_ball_area"
    ]

    with (output_dir / "table_2.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["example"]: row for row in csv.DictReader(handle)}
    assert float(rows["Oscillator, T=3"]["exact_ambient_measure"]) == results["oscillator"][
        "exact_area"
    ]
    assert float(rows["Switching, t=2"]["ball_ambient_measure"]) == results["switching"][
        "t=2"
    ]["smallest_ball_area"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
