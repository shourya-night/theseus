"""
Finite-state validation at the conjunction-analysis boundary.

Why this module exists
----------------------
Every stage of the Phase 9 pipeline -- screening, TCA refinement, collision
geometry, accounting -- is built out of arithmetic that propagates NaN and
infinity silently rather than failing.  Feed a NaN position into the screen and

    distances[i]  = |r_a - r_b|          -> nan
    bounds[i]     = (d0 + d1 - V h) / 2  -> nan
    bounds < threshold                   -> False

so the interval is *rejected as provably clear*.  Feed it into the TCA search
and ``f(t) = r_rel . v_rel`` is NaN at every sample, so ``f[i] < 0 and
f[i+1] >= 0`` is never true, no bracket is found, and the analysis returns zero
events with an internally consistent, entirely truthful-looking trace.

The result is indistinguishable from a genuinely clear pass.  That is the same
failure class as the fabricated Phase 10 risk output: **an absence of valid
information rendered as a valid negative answer.**  A corrupted or diverged
trajectory must never be reportable as "no conjunction".

What this module guarantees
---------------------------
Every position and velocity vector that enters the screening/TCA pipeline is
checked for finiteness at the moment it is produced, and a non-finite component
raises :class:`NonFiniteStateError` naming

    * which object the state belongs to,
    * whether it was a position or a velocity,
    * the time at which it was evaluated,
    * which component(s) were non-finite, and what they were.

The check is at the *evaluation point*: it covers every state the pipeline
actually consumes.  It cannot say anything about times the pipeline never
evaluates -- a trajectory function that returns NaN only on a set of times no
stage samples is not detected, because no stage ever used a bad number.  That
is the correct scope for input validation and is stated here so it is not
mistaken for a stronger claim.

Cost
----
One :func:`numpy.isfinite` over a 3-vector per state evaluation.  The guard is
idempotent: re-wrapping an already-guarded function for the same object and
quantity returns the same object, so a function that passes through several
stages is validated once, not once per stage.

Error, not INDETERMINATE
------------------------
This layer raises.  A non-finite state is a defect in the caller's input, not
an inconclusive analysis: unlike "the window contained no closest approach",
there is no meaningful analysis to report and no way to bound what the answer
would have been.  Callers that must degrade rather than fail -- the HTTP API --
catch :class:`NonFiniteStateError` and render its :meth:`~NonFiniteStateError.
to_dict` diagnostic, which is why every field a caller might want to surface is
carried on the exception rather than only formatted into its message.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Callable, Optional, Sequence

import numpy as np


#: Largest 1-D state for which the scalar finiteness path is used.  A state
#: vector is 3 long; the margin covers a 6-element Cartesian state without
#: reaching a size where the vectorised check would be cheaper.
_SCALAR_PATH_MAX_SIZE = 8


QUANTITY_POSITION = "position"
QUANTITY_VELOCITY = "velocity"

#: Component labels for a 3-vector state, used in diagnostics.
COMPONENT_NAMES = ("x", "y", "z")

#: Marker attribute recording what a guarded callable was wrapped for.
_GUARD_ATTR = "_orbitx_state_guard"


#: Quantities that are scalars rather than 3-vectors, so component names do
#: not apply to them.
_SCALAR_QUANTITIES = frozenset({"time"})


def _component_label(index: int, quantity: str = QUANTITY_POSITION) -> str:
    """Human-readable name for component *index* of a state vector."""
    if quantity in _SCALAR_QUANTITIES:
        return "value"
    if 0 <= index < len(COMPONENT_NAMES):
        return COMPONENT_NAMES[index]
    return f"[{index}]"


def _describe(value: float) -> str:
    """Render a float for a diagnostic, keeping nan/inf legible and exact."""
    return repr(float(value))


class NonFiniteStateError(ValueError):
    """
    A position or velocity entering the conjunction pipeline was not finite.

    Subclasses :class:`ValueError` so that callers already handling invalid
    numerical input keep working, while callers that want the structured
    diagnostic can catch this type specifically.

    Attributes
    ----------
    object_id : str
        Identifier of the object whose state was invalid.
    quantity : str
        ``"position"`` or ``"velocity"``.
    time_s : float | None
        Time at which the state was evaluated (s), when known.
    values : tuple[float, ...]
        The offending vector, verbatim.
    invalid_indices : tuple[int, ...]
        Indices of the non-finite components.
    invalid_components : tuple[str, ...]
        Names of those components (``x``, ``y``, ``z``).
    """

    def __init__(
        self,
        *,
        object_id: str,
        quantity: str,
        values: Sequence[float],
        invalid_indices: Sequence[int],
        time_s: Optional[float] = None,
    ) -> None:
        self.object_id = str(object_id)
        self.quantity = str(quantity)
        self.time_s = None if time_s is None else float(time_s)
        self.values = tuple(float(v) for v in values)
        self.invalid_indices = tuple(int(i) for i in invalid_indices)
        self.invalid_components = tuple(
            _component_label(i, self.quantity) for i in self.invalid_indices
        )
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        where = (
            "" if self.time_s is None
            else f" at t = {self.time_s:.6f} s"
        )
        bad = ", ".join(
            f"{_component_label(i, self.quantity)} = {_describe(self.values[i])}"
            for i in self.invalid_indices
        )
        full = ", ".join(
            f"{_component_label(i, self.quantity)}={_describe(v)}"
            for i, v in enumerate(self.values)
        )
        return (
            f"Object {self.object_id!r} {self.quantity}{where} is not finite: "
            f"{bad} (full vector: {full}). "
            "Conjunction analysis cannot proceed on a non-finite state, and no "
            "result -- including 'no conjunction found' -- may be reported for "
            "it. Check the trajectory source for divergence or corruption."
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Structured diagnostic, safe to serialise.

        Component values are rendered as strings because ``nan`` and ``inf``
        have no valid JSON literal; emitting them as bare tokens produces a
        document that strict parsers reject.
        """
        return {
            "error": "NON_FINITE_STATE",
            "analysis_status": "INVALID_STATE",
            "object_id": self.object_id,
            "quantity": self.quantity,
            "time_s": self.time_s,
            "invalid_components": list(self.invalid_components),
            "invalid_component_indices": list(self.invalid_indices),
            "values": [_describe(v) for v in self.values],
            "message": str(self),
        }


def validate_state_vector(
    value: Any,
    *,
    object_id: str,
    quantity: str,
    time_s: Optional[float] = None,
) -> np.ndarray:
    """
    Check one state vector for finiteness and return it as a float64 array.

    Parameters
    ----------
    value : array-like
        The state vector produced by a trajectory function.
    object_id : str
        Which object this state belongs to, for the diagnostic.
    quantity : str
        ``"position"`` or ``"velocity"``.
    time_s : float, optional
        Evaluation time (s), for the diagnostic.

    Returns
    -------
    np.ndarray
        ``value`` as float64.  For an input that is already a float64 array
        this is the same object, so read-only and memoised arrays keep their
        identity and no copy is made.

    Raises
    ------
    NonFiniteStateError
        If any component is NaN, +inf or -inf.

    Notes
    -----
    Shape is deliberately not constrained here.  This module's responsibility
    is finiteness; the surrounding stages already assume 3-vectors and would
    fail loudly on anything else.

    This runs once per trajectory evaluation, so the common case -- a finite
    3-vector -- takes the scalar path: ``ndarray.tolist()`` followed by
    :func:`math.isfinite` costs about 0.1 us, against about 1.0 us for
    ``np.isfinite(arr).all()``, whose ufunc dispatch dominates at this size.
    The two agree exactly; only the cost differs.  Anything larger, or not
    1-D, falls back to the vectorised check.
    """
    arr = np.asarray(value, dtype=np.float64)

    if arr.ndim == 1 and arr.size <= _SCALAR_PATH_MAX_SIZE:
        flat = arr.tolist()
        for component in flat:
            if not math.isfinite(component):
                raise NonFiniteStateError(
                    object_id=object_id,
                    quantity=quantity,
                    values=flat,
                    invalid_indices=[
                        i for i, x in enumerate(flat) if not math.isfinite(x)
                    ],
                    time_s=time_s,
                )
        return arr

    finite = np.isfinite(arr)
    if not finite.all():
        flat = arr.ravel()
        raise NonFiniteStateError(
            object_id=object_id,
            quantity=quantity,
            values=flat.tolist(),
            invalid_indices=np.flatnonzero(~finite.ravel()).tolist(),
            time_s=time_s,
        )
    return arr


def guard_state_function(
    fn: Callable[[float], np.ndarray],
    *,
    object_id: str,
    quantity: str,
) -> Callable[[float], np.ndarray]:
    """
    Wrap a trajectory function so every state it returns is validated.

    Idempotent: a function already guarded for the same ``(object_id,
    quantity)`` is returned unchanged, so passing a guarded function down
    through screening and TCA costs one check per evaluation, not one per
    stage.

    The wrapper returns exactly what :func:`validate_state_vector` returns, so
    for the float64 arrays every trajectory source in this engine produces, the
    guarded function hands back the identical object the unguarded one did.
    """
    if getattr(fn, _GUARD_ATTR, None) == (object_id, quantity):
        return fn

    @functools.wraps(fn)
    def guarded(t: float) -> np.ndarray:
        return validate_state_vector(
            fn(t), object_id=object_id, quantity=quantity, time_s=t
        )

    setattr(guarded, _GUARD_ATTR, (object_id, quantity))
    setattr(guarded, "_orbitx_unguarded", fn)
    return guarded


def guard_position_functions(
    pos_fn_a: Callable[[float], np.ndarray],
    pos_fn_b: Callable[[float], np.ndarray],
    *,
    object_a_id: str = "A",
    object_b_id: str = "B",
) -> tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    """Guard a pair of position functions.  See :func:`guard_state_function`."""
    return (
        guard_state_function(pos_fn_a, object_id=object_a_id, quantity=QUANTITY_POSITION),
        guard_state_function(pos_fn_b, object_id=object_b_id, quantity=QUANTITY_POSITION),
    )


def guard_velocity_functions(
    vel_fn_a: Optional[Callable[[float], np.ndarray]],
    vel_fn_b: Optional[Callable[[float], np.ndarray]],
    *,
    object_a_id: str = "A",
    object_b_id: str = "B",
) -> tuple[Optional[Callable[[float], np.ndarray]],
           Optional[Callable[[float], np.ndarray]]]:
    """
    Guard a pair of velocity functions, passing ``None`` through unchanged.

    ``None`` means "no velocity function supplied", which the screen handles
    with a documented fallback; it is not an invalid state.
    """
    return (
        None if vel_fn_a is None else guard_state_function(
            vel_fn_a, object_id=object_a_id, quantity=QUANTITY_VELOCITY),
        None if vel_fn_b is None else guard_state_function(
            vel_fn_b, object_id=object_b_id, quantity=QUANTITY_VELOCITY),
    )


def validate_state_samples(
    array: Any,
    *,
    object_id: str,
    quantity: str,
    times: Optional[Any] = None,
) -> np.ndarray:
    """
    Check a whole ``(N, 3)`` block of sampled states for finiteness.

    Used by the array-based screening entry point, where states arrive already
    tabulated rather than as callables.  The diagnostic names the first
    offending sample, its time when ``times`` is supplied, and the components
    that were non-finite in it.

    Returns
    -------
    np.ndarray
        ``array`` as float64.

    Raises
    ------
    NonFiniteStateError
        If any sample contains a non-finite component.
    """
    arr = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.all():
        return arr

    bad_rows = np.flatnonzero(~finite.all(axis=-1)) if arr.ndim > 1 else np.array([0])
    row = int(bad_rows[0])
    sample = arr[row] if arr.ndim > 1 else arr
    row_finite = np.isfinite(sample)

    t_bad: Optional[float] = None
    if times is not None:
        t_arr = np.asarray(times, dtype=np.float64)
        if row < len(t_arr) and np.isfinite(t_arr[row]):
            t_bad = float(t_arr[row])

    raise NonFiniteStateError(
        object_id=f"{object_id} (sample {row} of {len(arr)})",
        quantity=quantity,
        values=np.ravel(sample).tolist(),
        invalid_indices=np.flatnonzero(~np.ravel(row_finite)).tolist(),
        time_s=t_bad,
    )


def validate_time_samples(times: Any, *, name: str = "times") -> np.ndarray:
    """
    Check a time array for finiteness.

    A non-finite time is as corrupting as a non-finite state -- it silently
    poisons every interval bound computed from it -- and is reported the same
    way rather than being allowed through.
    """
    arr = np.asarray(times, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.all():
        bad = np.flatnonzero(~finite)
        raise NonFiniteStateError(
            object_id=f"{name} (sample {int(bad[0])} of {len(arr)})",
            quantity="time",
            values=[float(arr[int(bad[0])])],
            invalid_indices=[0],
            time_s=None,
        )
    return arr
