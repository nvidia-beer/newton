# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservation checks for the tire-to-spindle interface, independent of terrain tuning."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.ancf_shell.coupling import InterfaceCouplerGS as ANCFCoupler
from newton._src.solvers.ancf_shell.kernels_contact import apply_ground_contact
from newton._src.solvers.ancf_shell.kernels_coupling import (
    _adaptive_interface_step,
    _advance_response_probe,
    _aitken_coefficient,
    _choose_response_refresh,
    _integrate_interface_joints,
    _make_response_inverse,
    _newton_interface_step,
    _predict_interface_velocity,
    _probe_guess,
    _relax_coordinates,
    _response_column,
    _response_correction,
    _response_secant_apply,
    _response_secant_curvature,
    _response_secant_products,
    _save_interface_increment,
    _update_response_secant,
)
from newton._src.solvers.ancf_shell.model_ancf_shell import ANCFShellModel
from newton._src.solvers.ancf_shell.solver_ancf_shell_rigid import (
    _WRENCH_STRIDE,
    SolverANCFShellRigid,
    _accum_impulse_wheel,
    _accum_wheel_impulses,
    _transfer_wheel_wrenches,
)
from newton._src.solvers.ancf_shell.terrain_scm import TerrainSCM
from newton.examples.ancf._vehicle_kernels import prescribe_beads, stage_interface_kinematics
from newton.solvers import InterfaceCouplerGS, SolverMuJoCo
from newton.tests.test_ancf_shell_formulation import ElementFixture, rotation


@wp.kernel
def _two_rotor_response(
    guess: wp.array[float], response: wp.array[float], previous: float, dt: float, motor: float, hub: float, tire: float
):
    i = wp.tid()
    reaction = -tire * (guess[i] - previous) / dt
    response[i] = previous + dt * (motor + reaction) / hub


class TestANCFCoupling(unittest.TestCase):
    def test_coupler_import_compatibility(self):
        self.assertIs(InterfaceCouplerGS, ANCFCoupler)

    def test_wrench_resolves_small_momentum_changes_during_motion(self):
        """Small acceleration must remain accurate on top of steady wheel motion."""
        rng = np.random.default_rng(26)
        # Reference uses the exact float32 inputs, with independent float64 arithmetic.
        # Test one node at a time to separate cancellation from summation error.
        for device in ("cpu", *wp.get_cuda_devices()):
            for _ in range(8):
                old = rng.uniform(-1.0, 1.0, (4, 3)).astype(np.float32)
                current = (old + rng.uniform(-2e-5, 2e-5, (4, 3))).astype(np.float32)
                mass = rng.uniform(0.1, 3.0, 6).astype(np.float32)
                force = rng.uniform(-1e-3, 1e-3, 6).astype(np.float32)
                hub = rng.uniform(-0.5, 0.5, 3).astype(np.float32)
                dt = np.float32(1.0 / 600.0)
                x, d, v, vd = current.astype(float)
                x0, d0, v0, vd0 = old.astype(float)
                mx, md = mass[:3].astype(float), mass[3:].astype(float)
                f, fd = force[:3].astype(float), force[3:].astype(float)
                angular = np.cross(x - hub, mx * v) + np.cross(d, md * vd)
                angular0 = np.cross(x0 - hub, mx * v0) + np.cross(d0, md * vd0)
                expected = np.concatenate(
                    (
                        np.cross(x - hub, f) + np.cross(d, fd) - (angular - angular0) / dt,
                        f - (mx * v - mx * v0) / dt,
                    )
                )
                state = [wp.array([a], dtype=wp.vec3, device=device) for a in (*current, *old)]
                position = wp.array([[[hub[2], hub[0], hub[1]]]], dtype=wp.vec3, device=device)
                staging = wp.zeros(1, dtype=wp.spatial_vector, device=device)
                wp.launch(
                    _accum_impulse_wheel,
                    dim=1,
                    inputs=[
                        wp.array(force, device=device),
                        *state,
                        wp.array(mass, device=device),
                        position,
                        0,
                        0,
                        0.0,
                        0,
                        float(dt),
                        staging,
                    ],
                    device=device,
                )
                with self.subTest(device=device):
                    np.testing.assert_allclose(staging.numpy()[0], expected, rtol=2e-5, atol=2e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "Batched wrench graph requires CUDA")
    def test_batched_wrench_preserves_momentum_and_accumulates_shared_spindles(self):
        """Independent momentum balance, padding, translated hubs and graph reuse."""
        rng = np.random.default_rng(418)
        device = "cuda:0"
        maps = np.array([[0, 0], [1, 1], [0, 0], [0, 2]], dtype=np.int32)
        offsets = np.array([[0.0, 5.0], [0.5, -2.0], [-0.25, 0.0], [1.0, 3.0]], dtype=np.float32)
        xpos = rng.uniform(-2, 2, (2, 3, 3)).astype(np.float32)
        xpos += np.array([128, -64, 32], dtype=np.float32)

        def array(data, dtype=float):
            return wp.array(data, dtype=dtype, device=device)

        for n, torque_alpha in ((31, 0.0), (240, 0.35), (511, 1.0)):
            with self.subTest(nodes=n, torque_alpha=torque_alpha):
                hub = xpos[maps[:, 0], maps[:, 1]][:, [1, 2, 0]].copy()
                hub[:, 0] += offsets[:, 0]
                old = rng.uniform(-1, 1, (4, 4 * n, 3)).astype(np.float32)
                old[0] += np.repeat(hub, n, axis=0)
                current = (old + rng.uniform(-0.002, 0.002, old.shape)).astype(np.float32)
                mass = rng.uniform(0.01, 0.3, (4 * n, 6)).astype(np.float32)
                forces = rng.uniform(-2, 2, (4 * n, 6)).astype(np.float32)
                force_device = array(forces.reshape(-1))
                staging = wp.full(
                    4 * _WRENCH_STRIDE,
                    wp.spatial_vector(17, 17, 17, 17, 17, 17),
                    dtype=wp.spatial_vector,
                    device=device,
                )
                initial = rng.normal(size=(2, 3, 6)).astype(np.float32)
                applied = array(initial, wp.spatial_vector)
                dt = np.float32(1 / 600)
                inputs = [
                    force_device,
                    *[array(a, wp.vec3) for a in (*current, *old)],
                    array(mass.reshape(-1)),
                    array(xpos, wp.vec3),
                    array(maps, wp.vec2i),
                    array(offsets, wp.vec2),
                    n,
                    torque_alpha,
                    applied,
                    float(dt),
                    staging,
                ]

                def apply(staging=staging, n=n, inputs=inputs, dt=dt, torque_alpha=torque_alpha, applied=applied):
                    staging.zero_()
                    wp.launch(
                        _accum_wheel_impulses,
                        dim=(4, ((n + 255) // 256) * 256),
                        block_dim=256,
                        inputs=[*inputs[:14], float(dt), staging],
                        device=device,
                    )
                    wp.launch(
                        _transfer_wheel_wrenches,
                        dim=4,
                        inputs=[staging, inputs[11], inputs[12], torque_alpha, applied],
                        device=device,
                    )

                apply()
                applied.assign(initial)
                with wp.ScopedCapture(device=device) as capture:
                    apply()
                expected_applied = initial.astype(float)
                for scale in (1.0, -0.5, 2.0):
                    force_np = (forces * scale).astype(np.float32)
                    force_device.assign(force_np.reshape(-1))
                    wp.capture_launch(capture.graph)
                    x, d, v, vd = current.astype(float)
                    x0, d0, v0, vd0 = old.astype(float)
                    mx, md = mass[:, :3].astype(float), mass[:, 3:].astype(float)
                    arm, arm0 = x - np.repeat(hub, n, axis=0), x0 - np.repeat(hub, n, axis=0)
                    angular = np.cross(arm, mx * v) + np.cross(d, md * vd)
                    angular0 = np.cross(arm0, mx * v0) + np.cross(d0, md * vd0)
                    f, fd = force_np[:, :3].astype(float), force_np[:, 3:].astype(float)
                    torque = np.cross(arm, f) + np.cross(d, fd) - (angular - angular0) / dt
                    force = f - (mx * v - mx * v0) / dt
                    expected = np.concatenate((torque, force), axis=1).reshape(4, n, 6).sum(axis=1)
                    for w, (world, spindle) in enumerate(maps):
                        mapped = expected[w, [5, 3, 4, 2, 0, 1]].copy()
                        mapped[2] += offsets[w, 1]
                        mapped[3:] *= torque_alpha
                        expected_applied[world, spindle] += mapped
                    np.testing.assert_allclose(staging.numpy()[::_WRENCH_STRIDE], expected, rtol=3e-5, atol=3e-5)
                    np.testing.assert_allclose(applied.numpy(), expected_applied, rtol=3e-5, atol=5e-5)

    def test_rigid_contact_majorizer_preserves_force_and_dissipation(self):
        device = "cpu"
        reference = wp.array(
            [[-0.2, 0.0, -0.2], [0.2, 0.0, -0.2], [0.2, 0.0, 0.2], [-0.2, 0.0, 0.2]], dtype=wp.vec3, device=device
        )
        nodes = wp.array([[0, 1, 2, 3]], dtype=int, device=device)
        x = wp.array([[0.0, -0.002, 0.0]], dtype=wp.vec3, device=device)
        v = wp.array([[0.3, -0.1, -0.4]], dtype=wp.vec3, device=device)
        field = np.array([[-0.1, 0.1], [-0.1, 0.1]], dtype=np.float32)
        outputs = []
        for rigid in (False, True):
            terrain = TerrainSCM(
                field,
                1.0,
                1.0,
                reference,
                nodes,
                n_envs=1,
                friction_angle_deg=np.rad2deg(np.arctan(0.9)),
                janosi_k=0.0,
                rigid=rigid,
                mu_rigid=0.9,
                device=device,
            )
            # One queried point; reference mesh only supplies the node area.
            terrain.n_nodes = 1
            terrain.node_f = wp.zeros(1, dtype=wp.vec3, device=device)
            force, diagonal = wp.zeros(6, dtype=float, device=device), wp.zeros(6, dtype=float, device=device)
            terrain.apply_contact(x, v, 100000.0, 10.0, 0.01, 100.0, force, diagonal)
            outputs.append((force.numpy(), diagonal.numpy()))
        np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=1e-6, atol=1e-6)
        self.assertTrue((outputs[1][1] >= outputs[0][1] - 1e-4).all())
        normal = np.array([0.0, 1.0, -0.1])
        normal /= np.linalg.norm(normal)
        velocity = v.numpy()[0]
        tangent_velocity = velocity - normal * (normal @ velocity)
        self.assertLess(float(outputs[1][0][:3] @ tangent_velocity), 0.0, "Friction must dissipate energy")

    def test_flat_contact_friction_does_not_stiffen_normal_solve(self):
        device = "cpu"
        reference = wp.array(
            [[-0.2, 0.0, -0.2], [0.2, 0.0, -0.2], [0.2, 0.0, 0.2], [-0.2, 0.0, 0.2]],
            dtype=wp.vec3,
            device=device,
        )
        terrain = TerrainSCM(
            np.zeros((2, 2), dtype=np.float32),
            1.0,
            1.0,
            reference,
            wp.array([[0, 1, 2, 3]], dtype=int, device=device),
            n_envs=1,
            rigid=True,
            device=device,
        )
        x = wp.array([[0.0, -0.002, 0.0]] * 4, dtype=wp.vec3, device=device)
        v = wp.array([[0.3, -0.1, -0.4]] * 4, dtype=wp.vec3, device=device)
        force = wp.zeros(24, dtype=float, device=device)
        tangent = wp.zeros_like(force)
        terrain.apply_contact(x, v, 100000.0, 10.0, 0.01, 100.0, force, tangent)
        np.testing.assert_allclose(force.numpy().reshape(4, 6)[:, 1], 201.0, rtol=1e-6)
        np.testing.assert_allclose(tangent.numpy().reshape(4, 6)[:, 1], 101000.0, rtol=1e-6)
        plane_force, plane_tangent = wp.zeros_like(force), wp.zeros_like(force)
        wp.launch(
            apply_ground_contact,
            dim=4,
            inputs=[x, v, 0.0, 100000.0, 10.0, 0.9, 0.01, 100.0, plane_force, plane_tangent],
            device=device,
        )
        np.testing.assert_allclose(plane_force.numpy(), force.numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(plane_tangent.numpy(), tangent.numpy(), rtol=1e-6, atol=1e-6)

    def test_unconverged_added_inertia_does_not_amplify_predictor_error(self):
        """An exhausted trial budget must not feed its error into the next predictor."""
        for device in ("cpu", *wp.get_cuda_devices()):
            with self.subTest(device=device):
                velocity = wp.array([3.0], dtype=float, device=device)
                previous = wp.zeros_like(velocity)
                guess, response = wp.zeros_like(velocity), wp.zeros_like(velocity)
                residual, weight = wp.zeros_like(velocity), wp.zeros_like(velocity)
                increment = wp.zeros_like(velocity)
                converged = wp.zeros(1, dtype=int, device=device)
                active, iterations = wp.zeros_like(converged), wp.zeros_like(converged)
                totals = wp.zeros(2, dtype=int, device=device)
                norms = wp.zeros(2, dtype=float, device=device)
                dt, torque, hub, tire = 0.01, 1.0, 1.0, 4.0
                for step in range(12):
                    start = float(velocity.numpy()[0])
                    wp.copy(previous, velocity)
                    wp.launch(
                        _predict_interface_velocity,
                        dim=1,
                        inputs=[velocity, increment, converged],
                        device=device,
                    )
                    for iteration in range(2):
                        wp.copy(guess, velocity)
                        wp.launch(
                            _two_rotor_response,
                            dim=1,
                            inputs=[guess, response, start, dt, torque, hub, tire],
                            device=device,
                        )
                        wp.launch(
                            _adaptive_interface_step,
                            dim=1,
                            inputs=[
                                guess,
                                response,
                                residual,
                                weight,
                                velocity,
                                norms,
                                active,
                                converged,
                                iterations,
                                iteration,
                                2,
                                1e-4,
                                1e-3,
                                0.1,
                                True,
                            ],
                            device=device,
                        )
                    wp.copy(velocity, response)
                    wp.launch(
                        _save_interface_increment,
                        dim=1,
                        inputs=[velocity, previous, increment, iterations, totals],
                        device=device,
                    )
                    # Two trials, weight 0.1: dw = (1 - 0.1*It/Ih)*tau*dt/Ih.
                    # Extrapolating this unconverged increment adds a -2*dw_previous
                    # term: an unstable error recurrence for these inertias.
                    expected = 3.0 + (step + 1) * (1 - 0.1 * tire / hub) * torque * dt / hub
                    self.assertAlmostEqual(float(velocity.numpy()[0]), expected, delta=1e-5)
                    self.assertEqual(int(converged.numpy()[0]), 0)

    @unittest.skipUnless(wp.is_cuda_available(), "MuJoCo-Warp requires CUDA")
    def test_conditional_repeated_equality_force_matches_cpu(self):
        import mujoco
        import mujoco_warp as mjw

        xml = """<mujoco><option gravity="0 0 -9.81"><flag contact="disable"/></option>
        <worldbody><body name="hinged" pos="0 0 1"><freejoint/><geom type="box" size=".2 .1 .1" mass="2"/></body>
        <body name="welded" pos="1 0 1"><freejoint/><geom type="box" size=".1 .2 .1" mass="1"/></body></worldbody>
        <equality><connect body1="hinged" anchor="0 .1 .1"/><weld body1="welded"/></equality></mujoco>"""
        cpu_model = mujoco.MjModel.from_xml_string(xml)
        cpu_data = mujoco.MjData(cpu_model)
        cpu_data.qvel[:] = np.linspace(-0.05, 0.05, cpu_model.nv)
        mujoco.mj_forward(cpu_model, cpu_data)
        mujoco.mj_rnePostConstraint(cpu_model, cpu_data)
        with wp.ScopedDevice("cuda:0"):
            model = mjw.put_model(cpu_model)
            data = mjw.put_data(cpu_model, cpu_data)
            mjw.rne_postconstraint(model, data)
            active = wp.ones(1, dtype=int)
            with wp.ScopedCapture() as capture:
                wp.capture_if(active, lambda: mjw.rne_postconstraint(model, data))
            for _ in range(3):
                wp.capture_launch(capture.graph)
                for name in ("cacc", "cfrc_ext", "cfrc_int"):
                    np.testing.assert_allclose(
                        getattr(data, name).numpy()[0], getattr(cpu_data, name), atol=5e-4, rtol=5e-4
                    )

    @unittest.skipUnless(wp.is_cuda_available(), "MuJoCo-Warp requires CUDA")
    def test_conditional_tendon_steps_match_fresh_workspaces(self):
        import mujoco
        import mujoco_warp as mjw

        xml = """<mujoco><option timestep="0.001666666667" integrator="implicitfast"><flag contact="disable"/></option>
        <worldbody><site name="anchor" pos="0 0 2"/>
        <body name="bob" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="2"/><site name="tip" pos=".1 0 0"/></body>
        </worldbody><equality><connect body1="bob" anchor="0 .1 .1"/></equality>
        <tendon><spatial name="spring" stiffness="50" damping="1" armature=".1" springlength=".5">
        <site site="anchor"/><site site="tip"/></spatial></tendon><actuator><motor tendon="spring" gear=".1"/></actuator></mujoco>"""
        cpu_model = mujoco.MjModel.from_xml_string(xml)
        cpu_data = mujoco.MjData(cpu_model)
        cpu_data.qvel[:] = np.linspace(-0.05, 0.05, cpu_model.nv)
        mujoco.mj_forward(cpu_model, cpu_data)
        with wp.ScopedDevice("cuda:0"):
            model = mjw.put_model(cpu_model)
            captured = mjw.put_data(cpu_model, cpu_data)
            fresh = mjw.put_data(cpu_model, cpu_data)
            # Warm the complete integrator, which owns additional scratch,
            # before entering a conditional CUDA graph. Advance both references equally.
            for data in (captured, fresh):
                mjw.step(model, data)
            active = wp.ones(1, dtype=int)
            with wp.ScopedCapture() as capture:
                wp.capture_if(active, lambda: mjw.step(model, captured))
            for step in range(24):
                for data in (captured, fresh):
                    data.ctrl.fill_(float(np.sin(step)))
                fresh._scratch_arrays.clear()
                wp.capture_launch(capture.graph)
                mjw.step(model, fresh)
                for name in ("qpos", "qvel", "qacc", "qfrc_passive", "qfrc_actuator"):
                    np.testing.assert_allclose(
                        getattr(captured, name).numpy(),
                        getattr(fresh, name).numpy(),
                        atol=1e-6,
                        rtol=1e-6,
                        err_msg=name,
                    )

    @unittest.skipUnless(wp.is_cuda_available(), "MuJoCo-Warp requires CUDA")
    def test_mujoco_workspace_reuse_matches_fresh_contact_solves(self):
        """Reused collision/integration scratch must not change an impacting, sliding box."""
        builder = newton.ModelBuilder()
        body = builder.add_link(xform=wp.transform(wp.vec3(0, 0, 0.19), wp.quat_identity()))
        joint = builder.add_joint_free(body)
        builder.add_articulation([joint])
        builder.add_shape_box(body, hx=0.2, hy=0.15, hz=0.2)
        builder.add_shape_box(-1, hx=2, hy=2, hz=0.1, xform=wp.transform(wp.vec3(0, 0, -0.1), wp.quat_identity()))
        builder.joint_qd[0] = 0.3
        builder.joint_qd[5] = 0.5
        model = builder.finalize(device="cuda:0")
        pairs = []
        for _ in range(2):
            solver = SolverMuJoCo(model, use_mujoco_cpu=False, update_data_interval=1, integrator="implicitfast")
            state, output = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            pairs.append([solver, state, output])
        control = model.control()
        for _ in range(24):
            for index, (solver, state, output) in enumerate(pairs):
                if index == 1:
                    # Match fresh wp.empty allocations at the original call sites.
                    for name in ("_scratch_arrays", "_collision_contexts"):
                        if hasattr(solver.mjw_data, name):
                            delattr(solver.mjw_data, name)
                solver.step(state, output, control, None, 1 / 600)
                pairs[index] = [solver, output, state]
        np.testing.assert_allclose(pairs[0][1].joint_q.numpy(), pairs[1][1].joint_q.numpy(), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(pairs[0][1].joint_qd.numpy(), pairs[1][1].joint_qd.numpy(), atol=1e-6, rtol=1e-6)
        self.assertTrue(pairs[0][0].mjw_data._scratch_arrays)

    def test_reset_refreshes_coupling_snapshots(self):
        device = "cpu"
        soft = SimpleNamespace(
            **{
                name: wp.zeros(2, dtype=wp.vec3, device=device)
                for name in ("node_x", "node_xd", "node_xdd", "node_D", "node_Dd", "node_Ddd")
            }
        )
        for name in ("global_f_int", "global_f_int0", "global_f_ext", "global_f_ext0"):
            setattr(soft, name, wp.zeros(12, dtype=float, device=device))
        state = SimpleNamespace(
            body_q=wp.zeros(1, dtype=wp.transform, device=device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=device),
            joint_q=wp.zeros(1, dtype=float, device=device),
            joint_qd=wp.zeros(1, dtype=float, device=device),
        )
        coupler = InterfaceCouplerGS(soft)
        coupler.allocate(state)
        soft.node_x.fill_(wp.vec3(5.0))
        soft.global_f_ext.fill_(3.0)
        state.joint_q.fill_(2.0)
        coupler._velocity_increment.fill_(5.0)
        coupler.interface_totals.fill_(42)
        coupler._interface_converged.fill_(1)
        coupler.interface_iterations.fill_(6)
        coupler.reset(state)
        np.testing.assert_array_equal(coupler._velocity_increment.numpy(), 0.0)
        np.testing.assert_array_equal(coupler._interface_converged.numpy(), 0)
        np.testing.assert_array_equal(coupler.interface_totals.numpy(), 0)
        np.testing.assert_array_equal(coupler.interface_iterations.numpy(), 0)
        for index in range(2):
            soft.node_x.zero_()
            soft.global_f_ext.zero_()
            state.joint_q.zero_()
            coupler._rigid_cur = index
            coupler._unpack_ancf(index)
            coupler._restore_rigid(state)
            coupler._restore_extra()
            np.testing.assert_array_equal(soft.node_x.numpy(), 5.0)
            np.testing.assert_array_equal(soft.global_f_ext.numpy(), 3.0)
            np.testing.assert_array_equal(state.joint_q.numpy(), 2.0)

    def test_trial_free_pose_uses_com_velocity_far_from_origin(self):
        for device in ("cpu", *wp.get_cuda_devices()):
            with self.subTest(device=device):
                builder = newton.ModelBuilder(gravity=0)
                com = np.array([0.17, -0.03, 0.12])
                for position in ((0.0, 0.0, 0.0), (100.0, -80.0, 5.0)):
                    body = builder.add_link(
                        xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
                        mass=2.0,
                        com=wp.vec3(*com),
                        inertia=wp.mat33(np.eye(3)),
                    )
                    joint = builder.add_joint_free(body)
                    builder.add_articulation([joint])
                model = builder.finalize(device=device)
                velocity = np.tile([0.2, -0.1, 0.05, 0.3, -0.4, 0.8], 2).astype(np.float32)
                qd = wp.array(velocity, dtype=float, device=device)
                output = wp.zeros_like(model.joint_q)
                qd_output = wp.zeros_like(qd)
                dt = 1 / 600
                wp.launch(
                    _integrate_interface_joints,
                    dim=2,
                    inputs=[
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_X_c,
                        model.body_com,
                        model.joint_q,
                        qd,
                        wp.zeros_like(qd),
                        dt,
                        output,
                        qd_output,
                    ],
                    device=device,
                )
                initial = model.joint_q.numpy().reshape(2, 7)
                final = output.numpy().reshape(2, 7)
                for i in range(2):
                    quat = final[i, 3:]
                    rotated_com = com + 2 * np.cross(quat[:3], np.cross(quat[:3], com) + quat[3] * com)
                    np.testing.assert_allclose(
                        final[i, :3] + rotated_com, initial[i, :3] + com + dt * velocity[:3], atol=1e-5, rtol=0
                    )
                np.testing.assert_array_equal(qd_output.numpy(), velocity)

    def test_response_reuse_requires_adaptive_coupling(self):
        with self.assertRaisesRegex(ValueError, "requires adaptive"):
            InterfaceCouplerGS(None, reuse_response=True)

    def test_response_inverse_pivoting_and_singular_fallback(self):
        block = np.array([[0.0, 2.0, 0.1], [1.0, 0.4, 0.0], [0.0, 0.1, 1.5]], dtype=np.float64)
        for size in (3, 32, 50, 64):
            matrix = np.eye(size, dtype=np.float64)
            matrix[:3, :3] = block
            for device in ("cpu", *wp.get_cuda_devices()):
                with self.subTest(device=device, size=size):
                    source = wp.array(matrix, dtype=wp.float64, device=device)
                    inverse = wp.zeros_like(source)
                    ready, age, refreshed, count = [wp.zeros(1, dtype=int, device=device) for _ in range(4)]
                    kernel = _make_response_inverse(size)
                    wp.launch(kernel, dim=1, inputs=[source, inverse, ready, age, refreshed, count], device=device)
                    np.testing.assert_allclose(inverse.numpy(), np.linalg.inv(matrix), rtol=1e-10, atol=1e-10)
                    self.assertEqual(int(ready.numpy()[0]), 1)
                    # Singular responses disable reuse without replacing a valid stored matrix.
                    accepted = inverse.numpy()
                    source.zero_()
                    wp.launch(kernel, dim=1, inputs=[source, inverse, ready, age, refreshed, count], device=device)
                    self.assertEqual(int(ready.numpy()[0]), 0)
                    np.testing.assert_array_equal(inverse.numpy(), accepted)

    def test_response_refresh_budget_includes_invalid_estimates(self):
        for device in ("cpu", *wp.get_cuda_devices()):
            with self.subTest(device=device):
                guess = wp.zeros(2, dtype=float, device=device)
                response = wp.full(2, 0.2, dtype=float, device=device)
                previous = wp.full(2, 0.1, dtype=float, device=device)
                ready, age, refreshed, active, iteration, count = [
                    wp.zeros(1, dtype=int, device=device) for _ in range(6)
                ]
                totals = wp.array([200, 800], dtype=int, device=device)

                arguments = [
                    guess,
                    response,
                    previous,
                    ready,
                    age,
                    totals,
                    refreshed,
                    active,
                    iteration,
                    1e-3,
                    1e-3,
                    200,
                    500,
                    50,
                    count,
                ]

                def choose(arguments=arguments, active=active, device=device):
                    wp.launch(_choose_response_refresh, dim=1, inputs=arguments, device=device)
                    return int(active.numpy()[0])

                self.assertEqual(choose(), 1, "The first estimate needs no cooldown")
                count.fill_(2)
                age.zero_()
                for _ in range(49):
                    self.assertEqual(choose(), 0, "An invalid estimate must use Aitken until the cooldown expires")
                self.assertEqual(choose(), 1)
                iteration.fill_(1)
                refreshed.fill_(1)
                self.assertEqual(choose(), 0, "At most one refresh per substep")
                refreshed.zero_()
                ready.fill_(1)
                self.assertEqual(choose(), 1, "Stalled valid estimates can refresh after the cooldown")

    @unittest.skipUnless(wp.is_cuda_available(), "Conditional graphs require CUDA")
    def test_response_probe_loop_visits_each_column_once(self):
        device = "cuda:0"
        n = 3
        base = wp.array([1.0, 2.0, 3.0], dtype=float, device=device)
        trial, reference, response = [wp.zeros_like(base) for _ in range(3)]
        matrix = wp.zeros((n, n), dtype=wp.float64, device=device)
        column, active = [wp.zeros(1, dtype=int, device=device) for _ in range(2)]
        wp.launch(_two_rotor_response, dim=n, inputs=[base, reference, 3.0, 0.01, 1.0, 1.0, 4.0], device=device)

        def probe():
            wp.launch(_probe_guess, dim=n, inputs=[base, trial, column, 0.05], device=device)
            wp.launch(_two_rotor_response, dim=n, inputs=[trial, response, 3.0, 0.01, 1.0, 1.0, 4.0], device=device)
            wp.launch(_response_column, dim=n, inputs=[reference, response, matrix, column, 0.05], device=device)
            wp.launch(_advance_response_probe, dim=1, inputs=[column, active, n], device=device)

        with wp.ScopedCapture(device=device) as capture:
            column.zero_()
            active.fill_(1)
            wp.capture_while(active, probe)
        for _ in range(2):
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(matrix.numpy(), 5.0 * np.eye(n), atol=2e-5, rtol=0)
            self.assertEqual(int(column.numpy()[0]), n)
            self.assertEqual(int(active.numpy()[0]), 0)
            np.testing.assert_array_equal(base.numpy(), [1.0, 2.0, 3.0])

    def test_response_update_satisfies_measured_secant(self):
        jacobian = np.array([[2.0, 0.4], [0.05, 3.0]], dtype=np.float32)
        step = np.array([0.05, -0.02], dtype=np.float32)
        previous_residual = np.array([0.2, -0.1], dtype=np.float32)
        residual = previous_residual - jacobian @ step
        for device in ("cpu", *wp.get_cuda_devices()):
            with self.subTest(device=device):
                guess = wp.array(step, device=device)
                response = wp.array(step + residual, device=device)
                previous_guess = wp.zeros(2, dtype=float, device=device)
                history = wp.array(previous_residual, device=device)
                inverse = wp.array(0.3 * np.eye(2), dtype=wp.float64, device=device)
                ready = wp.ones(1, dtype=int, device=device)
                refresh = wp.zeros_like(ready)
                hy, sh = [wp.zeros(2, dtype=wp.float64, device=device) for _ in range(2)]
                wp.launch(
                    _update_response_secant,
                    dim=1,
                    inputs=[
                        guess,
                        response,
                        previous_guess,
                        history,
                        inverse,
                        ready,
                        refresh,
                        hy,
                        sh,
                        wp.ones(1, dtype=int, device=device),
                    ],
                    device=device,
                )
                np.testing.assert_allclose(inverse.numpy() @ (previous_residual - residual), step, atol=1e-7, rtol=1e-5)
                self.assertEqual(int(ready.numpy()[0]), 1)

    def test_parallel_response_matches_serial_update_and_fallback(self):
        rng = np.random.default_rng(824)
        for n in (50, 64):
            base = rng.normal(0, 0.01, n).astype(np.float32)
            step = rng.normal(0, 0.05, n).astype(np.float32)
            history = rng.normal(0, 0.1, n).astype(np.float32)
            jacobian = np.diag(np.linspace(2.0, 3.0, n)).astype(np.float32)
            initial_inverse = 0.3 * np.eye(n) + rng.normal(0, 0.01 / n, (n, n))
            for mode in ("update", "startup", "refresh", "tiny", "negative", "invalid", "aitken"):
                delta = step * (1e-5 if mode == "tiny" else 1.0)
                residual = history - (jacobian @ delta) * (-1.0 if mode == "negative" else 1.0)
                for device in ("cpu", *wp.get_cuda_devices()):
                    with self.subTest(size=n, mode=mode, device=device):
                        outputs = []
                        for parallel in (False, True):
                            guess = wp.array(base + delta, dtype=float, device=device)
                            response = wp.array(base + delta + residual, dtype=float, device=device)
                            previous_guess = wp.array(base, dtype=float, device=device)
                            previous = wp.array(history, dtype=float, device=device)
                            inverse = wp.array(
                                20.0 * np.eye(n) if mode == "invalid" else initial_inverse,
                                dtype=wp.float64,
                                device=device,
                            )
                            ready = wp.full(1, int(mode != "aitken"), dtype=int, device=device)
                            refresh = wp.full(1, int(mode == "refresh"), dtype=int, device=device)
                            iteration = wp.full(1, int(mode != "startup"), dtype=int, device=device)
                            hy, sh = [wp.zeros(n, dtype=wp.float64, device=device) for _ in range(2)]
                            correction = wp.zeros(n if parallel else 0, dtype=wp.float64, device=device)
                            if parallel:
                                curvature = wp.zeros(1, dtype=wp.float64, device=device)
                                wp.launch(
                                    _response_secant_products,
                                    dim=n,
                                    inputs=[
                                        guess,
                                        response,
                                        previous_guess,
                                        previous,
                                        inverse,
                                        ready,
                                        refresh,
                                        hy,
                                        sh,
                                        iteration,
                                    ],
                                    device=device,
                                )
                                wp.launch(
                                    _response_secant_curvature,
                                    dim=1,
                                    inputs=[
                                        guess,
                                        response,
                                        previous_guess,
                                        previous,
                                        ready,
                                        refresh,
                                        hy,
                                        iteration,
                                        curvature,
                                    ],
                                    device=device,
                                )
                                wp.launch(
                                    _response_secant_apply,
                                    dim=n,
                                    inputs=[guess, previous_guess, inverse, ready, hy, sh, curvature],
                                    device=device,
                                )
                                wp.launch(
                                    _response_correction,
                                    dim=n,
                                    inputs=[guess, response, inverse, ready, correction],
                                    device=device,
                                )
                            else:
                                wp.launch(
                                    _update_response_secant,
                                    dim=1,
                                    inputs=[
                                        guess,
                                        response,
                                        previous_guess,
                                        previous,
                                        inverse,
                                        ready,
                                        refresh,
                                        hy,
                                        sh,
                                        iteration,
                                    ],
                                    device=device,
                                )
                            weight = wp.full(1, 0.7, dtype=float, device=device)
                            corrected = wp.zeros(n, dtype=float, device=device)
                            norm = wp.zeros(6, dtype=float, device=device)
                            active, converged, used = [wp.zeros(1, dtype=int, device=device) for _ in range(3)]
                            wp.launch(
                                _newton_interface_step,
                                dim=1,
                                inputs=[
                                    guess,
                                    response,
                                    previous,
                                    weight,
                                    inverse,
                                    correction,
                                    ready,
                                    corrected,
                                    norm,
                                    active,
                                    converged,
                                    used,
                                    iteration,
                                    6,
                                    1e-3,
                                    1e-3,
                                    0.5,
                                    True,
                                ],
                                device=device,
                            )
                            outputs.append(
                                [
                                    a.numpy()
                                    for a in (
                                        inverse,
                                        ready,
                                        previous_guess,
                                        previous,
                                        corrected,
                                        norm,
                                        active,
                                        converged,
                                        used,
                                        weight,
                                    )
                                ]
                            )
                        for actual, expected in zip(outputs[1], outputs[0], strict=True):
                            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    @unittest.skipUnless(wp.is_cuda_available(), "MuJoCo-Warp requires CUDA")
    def test_trial_kinematics_preserves_spindle_origin_velocity(self):
        builder = newton.ModelBuilder(gravity=0)
        for i in range(2):
            body = builder.add_link(
                xform=wp.transform(
                    wp.vec3(2.0 * i, -0.3 * i, 1.0), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, 3)), 0.4 + i)
                ),
                mass=2.0,
                com=wp.vec3(0.13, -0.04, 0.02),
                inertia=wp.mat33(np.eye(3) * 0.2),
            )
            builder.add_articulation([builder.add_joint_free(body)])
        builder.joint_qd[:] = [0.3, -0.2, 0.1, 0.4, -0.5, 0.6] * 2
        model = builder.finalize(device="cuda:0")
        state, output = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        rigid = SolverMuJoCo(model, use_mujoco_cpu=False, update_data_interval=1, integrator="implicitfast")
        rigid.step_kinematics(state, output, model.control(), None, 1 / 600)
        mapping = rigid.mjc_body_to_newton.numpy()[0]
        indices = np.array([np.flatnonzero(mapping == i)[0] for i in range(2)], dtype=np.int32)
        roots = rigid.mjw_model.body_rootid.numpy()[indices]

        def origin_data():
            position = rigid.xpos.numpy()[0, indices]
            twist = rigid.cvel.numpy()[0, indices]
            center = rigid.mjw_data.subtree_com.numpy()[0, roots]
            return position, twist[:, :3], twist[:, 3:] + np.cross(twist[:, :3], position - center)

        expected = origin_data()
        # The cheap path may use the previous t_n COM as its spatial-velocity reference.
        rigid.mjw_data.subtree_com.assign(
            rigid.mjw_data.subtree_com.numpy() + np.array([2.0, -1.0, 0.5], dtype=np.float32)
        )
        wp.launch(
            stage_interface_kinematics,
            dim=2,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                wp.array([0, 1], dtype=int, device="cuda:0"),
                wp.array(indices, dtype=int, device="cuda:0"),
                rigid.xpos,
                rigid.xquat,
                rigid.cvel,
                rigid.mjw_data.subtree_com,
                rigid.mjw_model.body_rootid,
            ],
            device="cuda:0",
        )
        for actual, reference in zip(origin_data(), expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=5e-6, atol=5e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "The coupled solvers require CUDA")
    def test_reused_response_rotor_acceleration_braking_and_energy(self):
        self._check_rotor(True, reuse_response=True)

    @unittest.skipUnless(wp.is_cuda_available(), "The coupled solvers require CUDA")
    def test_rotor_acceleration_braking_and_energy(self):
        self._check_rotor(False)

    @unittest.skipUnless(wp.is_cuda_available(), "The coupled solvers require CUDA")
    def test_adaptive_rotor_acceleration_braking_energy_and_iteration_savings(self):
        self._check_rotor(True)

    @unittest.skipUnless(wp.is_cuda_available(), "The coupled solvers require CUDA")
    def test_condensed_response_rotor_acceleration_braking_and_energy(self):
        self._check_rotor(True, reuse_response=True, condensed=True)

    @unittest.skipUnless(wp.is_cuda_available(), "Conditional fallback requires CUDA")
    def test_invalid_condensed_response_preserves_rotor_work_on_fallback(self):
        self._check_rotor(True, reuse_response=True, bad_linearization=True)

    def _check_rotor(self, recycle, reuse_response=False, condensed=False, bad_linearization=False):
        """MuJoCo and a pinned shell share a torque, with tire inertia four times the hub inertia."""
        f = ElementFixture([rotation(0)], "cuda:0")
        a = ANCFShellModel(4, 1, f.x0, f.d0, f.nodes, f.h, f.material, f.zeros((1, 5)), f.cos, f.sin, device="cuda:0")
        m = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0).finalize(device="cuda:0")
        soft = SolverANCFShellRigid(m, a, n_tires=1, torque_alpha=1, ground_z=-10, nr_max_iter=2, pcg_max_iter=10)
        mass = soft.lumped_mass.numpy().reshape(4, 6)
        it = float(
            np.sum(
                mass[:, 0] * (f.rest[:, 1] ** 2 + f.rest[:, 2] ** 2)
                + mass[:, 3] * (f.directors[:, 1] ** 2 + f.directors[:, 2] ** 2)
            )
        )
        ih = it / 4
        dt = 1 / 600
        b = newton.ModelBuilder(gravity=0)
        link = b.add_link(mass=1, com=wp.vec3(0), inertia=wp.mat33(np.eye(3) * ih))
        joint = b.add_joint_revolute(-1, link, axis=(0, 1, 0))
        b.add_articulation([joint])
        b.joint_qd[0] = 3.0
        model = b.finalize(device="cuda:0")
        s0 = model.state()
        out = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        rigid = SolverMuJoCo(model, use_mujoco_cpu=False, update_data_interval=1, integrator="implicitfast")
        sp = int(np.flatnonzero(rigid.mjc_body_to_newton.numpy()[0] == link)[0])
        ids = wp.array([sp], dtype=int, device="cuda:0")
        beads = wp.array(np.arange(4), dtype=int, device="cuda:0")
        soft.set_dirichlet_nodes(np.arange(4))
        soft.setup_wheel(0, sp, np.arange(4), tare_fz=0, world_idx=0)

        def prescribe():
            wp.launch(
                prescribe_beads,
                dim=4,
                inputs=[
                    rigid.xpos,
                    rigid.xquat,
                    rigid.cvel,
                    rigid.mjw_data.subtree_com,
                    rigid.mjw_model.body_rootid,
                    ids,
                    beads,
                    f.x0,
                    f.d0,
                    soft.node_x,
                    soft.node_xd,
                    soft.node_xdd,
                    soft.node_D,
                    soft.node_Dd,
                    soft.node_Ddd,
                    4,
                    4,
                    1 / dt,
                    0.0,
                ],
                device="cuda:0",
            )

        rigid.step_kinematics(s0, out, control, None, dt)
        prescribe()
        soft.capture_graph(dt)
        # capture_graph changes state: prescribe again and clear force history for the torque-free initial ring.
        rigid.step_kinematics(s0, out, control, None, dt)
        prescribe()
        soft.global_f_ext.zero_()
        soft.global_f_ext0.zero_()
        soft.global_f_int.zero_()
        soft.global_f_int0.zero_()
        # Allocate MuJoCo workspace outside conditional capture without advancing s0.
        rigid.step_dynamics(out)
        rigid.step_kinematics(s0, out, control, None, dt)
        c = InterfaceCouplerGS(soft, n_iters=6, acceleration=True, adaptive=recycle, reuse_response=reuse_response)
        c.allocate(s0, model=model, interface_body_indices=[link] if condensed else None)
        if bad_linearization:
            c._shell_response = SimpleNamespace(refresh=lambda state, dt, inverse, ready: ready.zero_())
        if reuse_response:
            # This contact-free fixture can identify its added inertia immediately.
            c._response_warmup_steps = 0

        def accumulate():
            soft.accumulate_wheel_wrenches(rigid.xfrc_applied, rigid.xpos)

        graph = None
        if recycle:
            with wp.ScopedCapture(device="cuda:0") as capture:
                for _ in range(2):
                    c.substep(
                        s0,
                        out,
                        control,
                        dt,
                        rigid.step_kinematics,
                        prescribe,
                        accumulate,
                        rigid.step_dynamics,
                        ancf_step_fn=lambda: soft.step(None, None, None, None, dt),
                    )
            graph = capture.graph

        for sign in [1, -1]:
            torque = sign * 4 * (ih + it)
            control.joint_f.fill_(torque)
            start = float(s0.joint_qd.numpy()[0])
            work = 0
            stride = 2 if graph else 1
            for _ in range(12 // stride):
                prev = float(s0.joint_qd.numpy()[0])
                if graph:
                    wp.capture_launch(graph)
                else:
                    c.substep(s0, out, control, dt, rigid.step_kinematics, prescribe, accumulate, rigid.step_dynamics)
                speed = float(s0.joint_qd.numpy()[0])
                work += torque * (prev + speed) / 2 * dt * stride
            end = float(s0.joint_qd.numpy()[0])
            expected = start + sign * 4 * dt * 12
            self.assertLess(abs(end - expected), 2e-4)
            self.assertLess(abs(0.5 * (ih + it) * (end**2 - start**2) - work), abs(work) * 0.002)
        if recycle:
            substeps, passes = c.interface_totals.numpy()
            self.assertEqual(substeps, 24)
            self.assertLess(passes / substeps, 4.0, "Smooth rotor motion should need fewer interface trials")
            if reuse_response:
                if condensed:
                    self.assertEqual(int(c.interface_probe_count.numpy()[0]), 0)
                    self.assertGreater(int(c.interface_linearization_count.numpy()[0]), 0)
                else:
                    self.assertGreater(int(c.interface_probe_count.numpy()[0]), 0)
                if bad_linearization:
                    self.assertGreater(int(c.interface_linearization_count.numpy()[0]), 0)
                self.assertLess(passes / substeps, 2.0)
                c.reset(s0)
                self.assertEqual(int(c._response_ready.numpy()[0]), 0)
                self.assertEqual(int(c.interface_probe_count.numpy()[0]), 0)
                self.assertEqual(int(c.interface_linearization_count.numpy()[0]), 0)
                np.testing.assert_array_equal(c._previous_corrected.numpy(), s0.joint_qd.numpy())

    @unittest.skipUnless(wp.is_cuda_available(), "The shell solvers require CUDA")
    def test_free_fall_does_not_invent_hub_support(self):
        """Gravity and momentum cancel, including HHT startup and force history."""
        fixture = ElementFixture([rotation(0)], "cuda:0")
        shell = ANCFShellModel(
            4,
            1,
            fixture.x0,
            fixture.d0,
            fixture.nodes,
            fixture.h,
            fixture.material,
            fixture.zeros((1, 5)),
            fixture.cos,
            fixture.sin,
            device="cuda:0",
        )
        model = newton.ModelBuilder(up_axis=newton.Axis.Y).finalize(device="cuda:0")
        soft = SolverANCFShellRigid(
            model, shell, n_tires=1, torque_alpha=1, ground_z=-10, nr_max_iter=2, pcg_max_iter=10
        )
        soft.setup_wheel(0, 0, np.array([], dtype=np.int32), tare_fz=0, world_idx=0)
        dt = 1 / 600
        soft.capture_graph(dt)
        positions = wp.zeros((1, 1), dtype=wp.vec3, device="cuda:0")
        applied = wp.zeros((1, 1), dtype=wp.spatial_vector, device="cuda:0")
        for _ in range(5):
            soft.begin_coupling_step(dt)
            soft.graph_step()
            applied.zero_()
            soft.accumulate_wheel_wrenches(applied, positions)
            np.testing.assert_allclose(applied.numpy()[0, 0, :3], 0, atol=0.003, rtol=0)

    def test_accelerated_coupling_solves_added_inertia(self):
        for device in ("cpu", *wp.get_cuda_devices()):
            for motor in (1.5, -1.5):
                with self.subTest(device=device, motor=motor):
                    previous, dt, hub, tire = 3.0, 1.0 / 600, 0.1, 0.4
                    guess = wp.full(4, previous, dtype=float, device=device)
                    response, history = wp.zeros_like(guess), wp.zeros_like(guess)
                    weight = wp.ones(1, dtype=float, device=device)
                    residual = wp.zeros(3, dtype=float, device=device)
                    for iteration in range(3):
                        wp.launch(
                            _two_rotor_response,
                            dim=4,
                            inputs=[guess, response, previous, dt, motor, hub, tire],
                            device=device,
                        )
                        wp.launch(
                            _aitken_coefficient,
                            dim=1,
                            inputs=[guess, response, history, weight, residual, iteration, 0.1],
                            device=device,
                        )
                        wp.launch(_relax_coordinates, dim=4, inputs=[guess, response, weight, guess], device=device)
                    expected = previous + motor * dt / (hub + tire)
                    np.testing.assert_allclose(guess.numpy(), expected, atol=1e-6, rtol=0)
                    self.assertLess(float(residual.numpy()[-1]), 1e-5)

    def wrench(self, acceleration, angular_acceleration, omega=0.0):
        device = "cpu"
        radius = 0.5
        x = np.array([[0, radius, 0], [0, 0, radius], [0, -radius, 0], [0, 0, -radius]], dtype=np.float32)
        directors = x / radius
        alpha = np.array([angular_acceleration, 0, 0])
        angular = np.array([omega, 0, 0])
        acc = acceleration + np.cross(alpha, x) + np.cross(angular, np.cross(angular, x))
        dacc = np.cross(alpha, directors) + np.cross(angular, np.cross(angular, directors))
        mass, director_mass = 2.0, 0.03

        def array(data, dtype=float):
            return wp.array(data, dtype=dtype, device=device)

        staging = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        solver = SimpleNamespace(
            ancf=SimpleNamespace(n_nodes=4),
            n_tires=1,
            torque_alpha=1.0,
            global_f_ext=wp.zeros(24, dtype=float, device=device),
            node_x=array(x, wp.vec3),
            node_D=array(directors, wp.vec3),
            node_xdd=array(acc, wp.vec3),
            node_Ddd=array(dacc, wp.vec3),
            lumped_mass_tiled=array(np.tile([mass] * 3 + [director_mass] * 3, 4)),
            _xfrc_stg_per_tire=[staging],
            _world_idx_per_tire=[0],
            _spindle_mj_arr=[0],
            _lateral_offset_per_tire=[0.0],
            _tare_fz_per_tire=[0.0],
        )
        xpos = wp.zeros((1, 1), dtype=wp.vec3, device=device)
        applied = wp.zeros((1, 1), dtype=wp.spatial_vector, device=device)
        SolverANCFShellRigid.accumulate_wheel_wrenches(solver, applied, xpos, device=device)
        return applied.numpy()[0, 0], 4 * mass, 4 * (mass * radius**2 + director_mass)

    def test_translation_inertia_reaches_spindle(self):
        acceleration = np.array([1.0, -2.0, 3.0])
        wrench, mass, _ = self.wrench(acceleration, 0.0)
        np.testing.assert_allclose(wrench[:3], -mass * acceleration[[2, 0, 1]], atol=1e-6)
        np.testing.assert_allclose(wrench[3:], 0.0, atol=1e-6)

    def test_acceleration_braking_and_director_inertia(self):
        for alpha in (4.0, -4.0):
            with self.subTest(alpha=alpha):
                wrench, _, inertia = self.wrench(np.zeros(3), alpha, omega=3.0)
                np.testing.assert_allclose(wrench[:3], 0.0, atol=1e-6)
                np.testing.assert_allclose(wrench[3:], [0, -inertia * alpha, 0], atol=1e-6)
                # Midpoint interface work equals the opposite tire kinetic-energy change.
                dt, previous_speed = 0.01, 3.0 - alpha * 0.01 / 2
                next_speed = previous_speed + alpha * dt
                transferred = -wrench[4] * (previous_speed + next_speed) / 2 * dt
                energy_change = 0.5 * inertia * (next_speed**2 - previous_speed**2)
                self.assertAlmostEqual(transferred, energy_change, places=6)


if __name__ == "__main__":
    unittest.main()
