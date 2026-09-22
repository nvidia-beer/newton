# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Central tire inflation for the vehicle examples: the air amount in the ANCF tire cavities.

The solver models each tire as sealed gas, ``p = K_gas / V(x)`` (see ``SolverANCFShell.set_cavity``);
the CTIS sets ``K_gas`` through a nominal pressure (the pressure at the built volume) that ramps
toward a setpoint at the vehicle's fill rate. Two arrangements, chosen by the vehicle asset
(``customData:ctisCircuit``, see ``newton_tire_tool.vehicle``):

* :class:`CtisPerTire` — one valve per tire (the super jeep's system): every tire holds its own
  setpoint and its own air; a squeezed tire stiffens on its own.
* :class:`CtisCircuit` — one air line joins all tires (SHERP, US 2017/0240008 A1: the frame's
  hollow side members and cross bars are the line, exhaust gas fills it, one valve bleeds it).
  One setpoint for the circuit; the solver equalises ``p = Σ K_gas / Σ V`` every iteration, so a
  tire pushed up by a rock hands its air to the other three (the patent's "pneumocirculating
  suspension" — the pneumatic part of the air spring is spread over 4 cavities).

Nothing here knows which vehicle it serves.
"""

from __future__ import annotations


class CtisPerTire:
    """Per-tire valves: independent setpoints, ramped at the envelope's rate."""

    def __init__(self, solver, n_tires: int, pressure: float, build_pressure: float, envelope) -> None:
        self._solver = solver
        self._n = n_tires
        self._build = float(build_pressure)
        self._unit, self._per_unit, self._lo, self._hi, self._rate_per_s, self._presets, self._title = envelope
        self.setpoint_all = float(pressure)  # "all tires" slider position [Pa]
        self.targets = [float(pressure)] * n_tires  # [Pa]
        self.currents = [float(pressure)] * n_tires  # [Pa] nominal air amount now in each tire
        # Live gas state per tire, refreshed by read_live() on the example's diagnostics period
        # (one host copy of two small arrays; never per frame).
        self.live_p = [float(pressure)] * n_tires  # [Pa] absolute cavity pressure K_gas / V
        self.live_v_ratio = [1.0] * n_tires  # V / V_ref
        self._apply()

    def read_live(self) -> None:
        """Host copy of the solver's cavity pressure and volume (call on the diagnostics period)."""
        p = self._solver.cav_p.numpy()
        v = self._solver.cav_V.numpy()
        v_ref = self._solver.cavity_volume_ref  # property
        self.live_p = [float(p[e]) for e in range(self._n)]
        self.live_v_ratio = [float(v[e]) / v_ref for e in range(self._n)]

    def _gui_live(self, ui, wheel_order) -> None:
        unit, per = self._unit, self._per_unit
        ui.text(f"live gauge [{unit}]  (cavity K_gas / V, refreshed with the diagnostics)")
        for label, e in wheel_order:
            ui.text(
                f"  {label} {(self.live_p[e] - self._build) / per:6.2f} {unit}"
                f"   ({self.live_p[e]:7.0f} Pa abs   V/V_ref {self.live_v_ratio[e]:.3f})"
            )

    def _apply(self) -> None:
        self._solver.set_cavity(self.currents, [self._build] * self._n)

    def step(self, dt: float) -> None:
        """Ramp every tire toward its setpoint at the fill rate; nothing to do at the setpoints."""
        if self.currents == self.targets:
            return
        rate = self._rate_per_s * self._per_unit * dt  # [Pa per call]
        for e in range(self._n):
            d = self.targets[e] - self.currents[e]
            if d != 0.0:
                self.currents[e] += min(abs(d), rate) * (1.0 if d > 0.0 else -1.0)
        self._apply()

    def set_all(self, pressure: float) -> None:
        self.setpoint_all = float(pressure)
        self.targets = [self.setpoint_all] * self._n

    def gui(self, ui, wheel_order) -> None:
        unit, per = self._unit, self._per_unit
        ui.separator()
        ui.text(f"CTIS setpoint [{unit}]   {self._title} {self._lo:g} - {self._hi:g}")
        changed, val = ui.slider_float("all tires", self.setpoint_all / per, self._lo, self._hi)
        if changed:
            self.set_all(float(val) * per)
        for i, (level, name) in enumerate(self._presets):
            if i:
                ui.same_line()
            if ui.button(f"{level:g} {name}"):
                self.set_all(level * per)
        self._gui_tires(ui, wheel_order)
        # The shell carries the gauge value (nominal - build); set_cavity holds absolute.
        ui.text(f"setpoint now [{unit}]")
        for label, e in wheel_order:
            gauge = self.currents[e] - self._build
            ui.text(f"  {label} {gauge / per:6.2f} {unit}   ({self.currents[e]:7.0f} Pa abs)")
        self._gui_live(ui, wheel_order)

    def _gui_tires(self, ui, wheel_order) -> None:
        for label, e in wheel_order:
            changed, val = ui.slider_float(label, self.targets[e] / self._per_unit, self._lo, self._hi)
            if changed:
                self.targets[e] = float(val) * self._per_unit


class CtisCircuit(CtisPerTire):
    """One air line for all tires: one setpoint, and the solver equalises the pressure."""

    def __init__(self, solver, n_tires: int, pressure: float, build_pressure: float, envelope) -> None:
        solver.set_cavity_circuit(True)  # before the solver's graph capture: it picks the gas-law kernel
        super().__init__(solver, n_tires, pressure, build_pressure, envelope)

    def gui(self, ui, wheel_order) -> None:
        unit, per = self._unit, self._per_unit
        ui.separator()
        ui.text(f"CTIS setpoint [{unit}]   {self._title} {self._lo:g} - {self._hi:g}   (one air line, 4 tires)")
        changed, val = ui.slider_float("circuit", self.setpoint_all / per, self._lo, self._hi)
        if changed:
            self.set_all(float(val) * per)
        for i, (level, name) in enumerate(self._presets):
            if i:
                ui.same_line()
            if ui.button(f"{level:g} {name}"):
                self.set_all(level * per)
        gauge = self.currents[0] - self._build
        ui.text(f"  circuit setpoint now {gauge / per:6.2f} {unit}   ({self.currents[0]:7.0f} Pa abs nominal)")
        # One line -> one live pressure; the per-wheel V/V_ref shows who is squeezed and who took the air.
        self._gui_live(ui, wheel_order)
