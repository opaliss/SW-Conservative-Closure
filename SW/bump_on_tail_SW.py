"""
Bump-on-tail instability module

closure options:by truncation, mass, momentum, energy, and L2!

Last modified: April 18th, 2024 [Opal Issan]
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.implicit_midpoint import implicit_midpoint_solver
import numpy as np
from operators.operators import RHS, solve_poisson_equation_two_stream, integral_I0, J_matrix_inv, J_matrix
import scipy
from operators.closure import closure_momentum, closure_energy, closure_mass


def dydt(y, t):
    dydt_ = np.zeros(len(y), dtype="complex128")

    state_e1 = np.zeros((Nv, Nx_total), dtype="complex128")
    state_e2 = np.zeros((Nv, Nx_total), dtype="complex128")
    state_i = np.zeros((Nv, Nx_total), dtype="complex128")

    for jj in range(Nv):
        state_e1[jj, :] = y[jj * Nx_total: (jj + 1) * Nx_total]
        state_e2[jj, :] = y[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total]
        state_i[jj, :] = y[2 * Nv * Nx_total + jj * Nx_total: 2 * Nv * Nx_total + (jj + 1) * Nx_total]

    E = solve_poisson_equation_two_stream(state_e1=state_e1, state_e2=state_e2, state_i=state_i,
                                          alpha_e1=alpha_e1, alpha_e2=alpha_e2, alpha_i=alpha_i, Nx=Nx, L=L, Nv=Nv)
    if closure == "energy":
        # energy closure
        closure_e1 = closure_energy(state=state_e1, alpha_s=alpha_e1, u_s=u_e1, Nv=Nv, E=E, J_inv=J_inv, q_s=q_e1, m_s=m_e1, Nx_total=Nx_total, Nx=Nx)
        closure_e2 = closure_energy(state=state_e2, alpha_s=alpha_e2, u_s=u_e2, Nv=Nv, E=E, J_inv=J_inv, q_s=q_e2, m_s=m_e2,  Nx_total=Nx_total, Nx=Nx)
        closure_i = closure_energy(state=state_i, alpha_s=alpha_i, u_s=u_i, Nv=Nv, E=E, J_inv=J_inv, q_s=q_i, m_s=m_i, Nx_total=Nx_total, Nx=Nx)

    if closure == "momentum":
        # momentum closure
        closure_e1 = closure_momentum(state=state_e1, alpha_s=alpha_e1, u_s=u_e1, Nv=Nv)
        closure_e2 = closure_momentum(state=state_e2, alpha_s=alpha_e2, u_s=u_e2, Nv=Nv)
        closure_i = closure_momentum(state=state_i, alpha_s=alpha_i, u_s=u_i, Nv=Nv)

    elif closure == "mass":
        # mass closure
        closure_e1 = closure_mass(state=state_e1, E=E, Nx=Nx)
        closure_e2 = closure_mass(state=state_e2, E=E, Nx=Nx)
        closure_i = closure_mass(state=state_i, E=E, Nx=Nx)

    elif closure == "truncation":
        # truncation closure
        closure_e1 = np.zeros(Nx_total)
        closure_e2 = np.zeros(Nx_total)
        closure_i = np.zeros(Nx_total)

    for jj in range(Nv):
        # electron 1 evolution
        dydt_[jj * Nx_total: (jj + 1) * Nx_total] = RHS(state=state_e1, n=jj, Nv=Nv, alpha_s=alpha_e1, q_s=q_e1,
                                                        Nx=Nx, m_s=m_e1, E=E, u_s=u_e1, L=L, closure=closure_e1)

        # enforce that the coefficients live in the reals
        dydt_[jj * Nx_total: (jj + 1) * Nx_total][:Nx] = np.flip(
            np.conjugate(dydt_[jj * Nx_total: (jj + 1) * Nx_total][Nx + 1:]))
        dydt_[jj * Nx_total: (jj + 1) * Nx_total][Nx] = dydt_[jj * Nx_total: (jj + 1) * Nx_total][Nx].real

        # electron 2 evolution
        dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total] = RHS(state=state_e2, n=jj,
                                                                                        Nv=Nv, alpha_s=alpha_e2,
                                                                                        q_s=q_e2, Nx=Nx, m_s=m_e2, E=E,
                                                                                        u_s=u_e2, L=L,
                                                                                        closure=closure_e2)
        # enforce that the coefficients live in the reals
        dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][:Nx] = \
            np.flip(np.conjugate(dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][Nx + 1:]))
        dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][Nx] = dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][Nx].real

        # ion evolution
        dydt_[2 * Nv * Nx_total + jj * Nx_total: 2 * Nv * Nx_total + (jj + 1) * Nx_total] = RHS(state=state_i, n=jj,
                                                                                                Nv=Nv, alpha_s=alpha_i,
                                                                                                q_s=q_i, Nx=Nx, m_s=m_i,
                                                                                                E=E,
                                                                                                u_s=u_i, L=L,
                                                                                                closure=closure_i)
        # enforce that the coefficients live in the reals
        dydt_[2 * Nv * Nx_total + jj * Nx_total: 2 * Nv * Nx_total + (jj + 1) * Nx_total][:Nx] = \
            np.flip(np.conjugate(dydt_[2 * Nv * Nx_total + jj * Nx_total: 2 * Nv * Nx_total + (jj + 1) * Nx_total][Nx + 1:]))
        dydt_[2 * Nv * Nx_total + jj * Nx_total: 2 * Nv * Nx_total + (jj + 1) * Nx_total][Nx] = dydt_[2 * Nv * Nx_total + jj * Nx_total: 2 * Nv * Nx_total + (jj + 1) * Nx_total][Nx].real

    # mass (odd)
    dydt_[-1] = -L * np.sqrt(Nv / 2) * integral_I0(n=Nv - 1) * np.flip(E).T @ (
            (q_e1 / m_e1) * closure_e1 + (q_e2 / m_e2) * closure_e2 + (q_i / m_i) * closure_i)

    # mass (even)
    dydt_[-2] = -L * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * np.flip(E).T @ (
            (q_e1 / m_e1) * state_e1[-1, :] + (q_e2 / m_e2) * state_e2[-1, :] + (q_i / m_i) * state_i[-1, :])

    # momentum (odd)
    dydt_[-3] = -L * integral_I0(n=Nv - 1) * np.flip(E).T @ (np.sqrt(Nv / 2) * (
            u_e1 * q_e1 * closure_e1 + u_e2 * q_e2 * closure_e2 + u_i * q_i * closure_i)
            + Nv * (alpha_e1 * q_e1 * state_e1[-1, :] + alpha_e2 * q_e2 * state_e2[-1, :] + alpha_i * q_i * state_i[-1, :]))

    # momentum (even)
    dydt_[-4] = -L * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * np.flip(E).T @ (
            u_e1 * q_e1 * state_e1[-1, :] + u_e2 * q_e2 * state_e2[-1, :] + u_i * q_i * state_i[-1, :] +
            np.sqrt(2 * Nv) * (q_e1 * alpha_e1 * closure_e1 + q_e2 * alpha_e2 * closure_e2 + q_i * alpha_i * closure_i))

    # energy (odd)
    dydt_[-5] = -L * integral_I0(n=Nv - 1) * np.flip(E).T @ (np.sqrt(Nv / 2) * (
            q_e1 * (0.5 * ((2 * Nv + 1) * (alpha_e1 ** 2) + u_e1 ** 2) * closure_e1 + q_e1 / m_e1 * J_inv @ scipy.signal.convolve(in1=closure_e1, in2=E, mode="same")) +
            q_e2 * (0.5 * ((2 * Nv + 1) * (alpha_e2 ** 2) + u_e2 ** 2) * closure_e2 + q_e2 / m_e2 * J_inv @ scipy.signal.convolve(in1=closure_e2, in2=E, mode="same")) +
            q_i * (0.5 * ((2 * Nv + 1) * (alpha_i ** 2) + u_i ** 2) * closure_i + q_i / m_i * J_inv @ scipy.signal.convolve(in1=closure_i, in2=E, mode="same"))) +
            + Nv * (u_e1 * q_e1 * alpha_e1 * state_e1[-1, :] + u_e2 * q_e2 * alpha_e2 * state_e2[-1,:] + u_i * q_i * alpha_i * state_i[-1, :]))

    # energy (even)
    dydt_[-6] = -L * integral_I0(n=Nv - 2) * np.flip(E).T @ (np.sqrt((Nv - 1) / 2) * (
            q_e1 * (0.5 * ((2 * Nv - 1) * (alpha_e1 ** 2) + u_e1 ** 2) * state_e1[-1, :] + q_e1 / m_e1 * J_inv @ scipy.signal.convolve(in1=state_e1[-1, :], in2=E, mode="same"))
            + q_e2 * (0.5 * ((2 * Nv - 1) * (alpha_e2 ** 2) + u_e2 ** 2) * state_e2[-1, :] + q_e2 / m_e2 * J_inv @ scipy.signal.convolve(in1=state_e2[-1, :], in2=E, mode="same"))
            + q_i * (0.5 * ((2 * Nv - 1) * (alpha_i ** 2) + u_i ** 2) * state_i[-1, :] + q_i / m_i * J_inv @ scipy.signal.convolve(in1=state_i[-1, :], in2=E, mode="same")))
            + np.sqrt(Nv * (Nv - 1)) * (u_e1 * q_e1 * alpha_e1 * closure_e1 + u_e2 * q_e2 * alpha_e2 * closure_e2 + u_i * q_i * alpha_i * closure_i))

    # L2 norm (even or odd)
    E_conv = scipy.linalg.convolution_matrix(a=E, n=Nx_total, mode='same')
    dydt_[-7] = - np.sqrt(2 * Nv) * L * (np.flip(state_e1[-1, :]).T @ (q_e1/m_e1 * E_conv) @ closure_e1
                    + np.flip(state_e2[-1, :]).T @ (q_e2/m_e2 * E_conv) @ closure_e2
                    + np.flip(state_i[-1, :]).T @ (q_i/m_i * E_conv) @ closure_i)
    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of Fourier spectral terms in x
    Nx = 20
    Nx_total = 2 * Nx + 1
    # number of Hermite spectral terms in v
    Nv = 6
    # Velocity scaling of electron and ion
    alpha_e1 = 1
    alpha_e2 = np.sqrt(1 / 2)
    alpha_i = np.sqrt(1 / 1836)
    # perturbation magnitude
    epsilon = 0.03
    # x grid is from 0 to L
    L = 20 * np.pi / 3
    # final time
    T = 10
    # time stepping
    dt = 0.01
    # time vector
    t_vec = np.linspace(0, T, int(T / dt) + 1)
    # velocity scaling
    u_e1 = -1/2
    u_e2 = 9/2
    u_i = -1/2
    # mass normalized
    m_e1 = 1
    m_e2 = 1
    m_i = 1836
    # charge normalized
    q_e1 = -1
    q_e2 = -1
    q_i = 1
    # scaling of bulk and bump
    delta_e1 = 9 / 10
    delta_e2 = 1 / 10
    # closure type
    closure = "L2"

    # inverse J
    J_inv = J_matrix_inv(Nx=Nx, L=L)
    J = J_matrix(Nx=Nx, L=L)
    # x direction
    x_project = np.linspace(0, L, int(1e5))

    # initial condition of the first expansion coefficient
    C_0e1 = np.zeros(Nx_total, dtype="complex128")
    C_0e2 = np.zeros(Nx_total, dtype="complex128")

    C_0e1[Nx] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * delta_e1 / alpha_e1
    C_0e1[Nx - 1] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * delta_e1 * 0.5 * epsilon / alpha_e1
    C_0e1[Nx + 1] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * delta_e1 * 0.5 * epsilon / alpha_e1

    C_0e2[Nx] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * delta_e2 / alpha_e2
    C_0e2[Nx - 1] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * delta_e2 * 0.5 * epsilon / alpha_e2
    C_0e2[Nx + 1] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * delta_e2 * 0.5 * epsilon / alpha_e2

    # initialize states (electrons and ions)
    states_e1 = np.zeros((Nv, Nx_total), dtype="complex128")
    states_e2 = np.zeros((Nv, Nx_total), dtype="complex128")
    states_i = np.zeros((Nv, Nx_total), dtype="complex128")

    # initialize the expansion coefficients
    states_e1[0, :] = C_0e1
    states_e2[0, :] = C_0e2
    states_i[0, Nx] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) / alpha_i

    # initial condition of the semi-discretized ODE
    y0 = np.append(states_e1.flatten("C"), states_e2.flatten("C"))
    y0 = np.append(y0, states_i.flatten("C"))
    y0 = np.append(y0, np.zeros(7))

    # set up implicit midpoint
    sol_midpoint_u, t_vec = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=dydt,
                                                     nonlinear_solver_type="newton_krylov",
                                                     r_tol=1e-10, a_tol=1e-10, max_iter=100)

    # save results
    np.save("../data/SW/bump_on_tail/sol_midpoint_u_" + str(Nv) + "_" + str(closure) + "_closure", sol_midpoint_u)
    np.save("../data/SW/bump_on_tail/sol_midpoint_t_" + str(Nv) + "_" + str(closure) + "_closure", t_vec)
