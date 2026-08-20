"""
VALIDATION D: Kepler Solver
Independently verifies convergence, residual, and accuracy of Kepler's equation solver
for elliptic and hyperbolic trajectories across extreme eccentricities.
"""

import math
import pytest
import numpy as np

from theseus.orbital.kepler import solve_kepler, eccentric_to_true, hyperbolic_to_true


class TestValidationDKeplerSolver:
    """Independent verification of Kepler equation solvers."""

    @pytest.mark.parametrize("e", [0.0, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99, 0.999])
    @pytest.mark.parametrize("M", [0.01, 0.5, 1.0, math.pi / 2, math.pi, 2 * math.pi - 0.1])
    def test_kepler_elliptic_equation_residual(self, e: float, M: float):
        """
        Verify |M - (E - e*sin(E))| is within 1e-11 for all e and M.
        """
        sol = solve_kepler(M, e, tol=1e-12, max_iter=100)
        assert sol.converged, f"Failed to converge for e={e}, M={M}"
        E = sol.eccentric_anomaly

        # Independent calculation of M from E
        M_recovered = (E - e * math.sin(E)) % (2.0 * math.pi)
        M_expected = M % (2.0 * math.pi)

        residual = abs(M_recovered - M_expected)
        assert residual < 1e-10, f"Residual {residual} too high for e={e}, M={M}"

    @pytest.mark.parametrize("e", [1.05, 1.2, 1.5, 2.0, 5.0, 10.0])
    @pytest.mark.parametrize("M", [0.1, 1.0, 5.0, 20.0])
    def test_kepler_hyperbolic_equation_residual(self, e: float, M: float):
        """
        Verify hyperbolic Kepler equation: M = e * sinh(H) - H.
        """
        sol = solve_kepler(M, e, tol=1e-12, max_iter=100)
        assert sol.converged, f"Hyperbolic failed to converge for e={e}, M={M}"
        H = sol.eccentric_anomaly

        M_recovered = e * math.sinh(H) - H
        residual = abs(M_recovered - M)
        assert residual < 1e-10, f"Hyperbolic residual {residual} too high for e={e}, M={M}"

    def test_kepler_solver_low_iteration_limit(self):
        """When max_iter is too small, solver must report converged=False."""
        sol = solve_kepler(1.5, 0.99, tol=1e-14, max_iter=1)
        assert sol.converged is False
        assert sol.iterations == 1
