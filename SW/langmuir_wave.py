"""
Langmuir wave module

closure options: by truncation, momentum, and energy!

Last modified: May 2nd, 2024 [Opal Issan]
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.implicit_midpoint import implicit_midpoint_solver
import numpy as np
from operators.operators import RHS, solve_poisson_equation, integral_I0, J_matrix_inv
import scipy
from operators.closure import closure_momentum, closure_energy, closure_mass


def dydt(y, t):
    dydt_ = np.zeros(len(y), dtype="complex128")

    state_e = np.zeros((Nv, Nx_total), dtype="complex128")
    state_i = np.zeros((Nv, Nx_total), dtype="complex128")

    for jj in range(Nv):
        state_e[jj, :] = y[jj * Nx_total: (jj + 1) * Nx_total]
        state_i[jj, :] = y[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total]

    E = solve_poisson_equation(state_e=state_e, state_i=state_i, alpha_e=alpha_e, alpha_i=alpha_i, Nx=Nx, L=L, Nv=Nv)

    if closure == "energy":
        # energy closure
        closure_e = closure_energy(state=state_e, alpha_s=alpha_e, u_s=u_e, Nv=Nv, E=E, J_inv=J_inv, q_s=q_e, m_s=m_e, Nx=Nx, Nx_total=Nx_total)
        closure_i = closure_energy(state=state_i, alpha_s=alpha_i, u_s=u_i, Nv=Nv, E=E, J_inv=J_inv, q_s=q_i, m_s=m_i, Nx=Nx, Nx_total=Nx_total)

    elif closure == "momentum":
        # momentum closure
        closure_e = closure_momentum(state=state_e, alpha_s=alpha_e, u_s=u_e, Nv=Nv)
        closure_i = closure_momentum(state=state_i, alpha_s=alpha_i, u_s=u_i, Nv=Nv)

    elif closure == "mass":
        #  mass closure
        closure_e = closure_mass(state=state_e, E=E, Nx=Nx)
        closure_i = closure_mass(state=state_i,  E=E, Nx=Nx)

    elif closure == "truncation":
        #  mass closure
        closure_e = 0 * closure_momentum(state=state_e, alpha_s=alpha_e, u_s=u_e, Nv=Nv)
        closure_i = 0 * closure_momentum(state=state_i, alpha_s=alpha_i, u_s=u_i, Nv=Nv)

    for jj in range(Nv):
        # electron evolution
        dydt_[jj * Nx_total: (jj + 1) * Nx_total] = RHS(state=state_e, n=jj, Nv=Nv, alpha_s=alpha_e, q_s=q_e,
                                                        Nx=Nx, m_s=m_e, E=E, u_s=u_e, L=L, closure=closure_e)

        # enforce that the coefficients live in the reals
        dydt_[jj * Nx_total: (jj + 1) * Nx_total][:Nx] = np.flip(
            np.conjugate(dydt_[jj * Nx_total: (jj + 1) * Nx_total][Nx + 1:]))

        # ion evolution
        dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total] = RHS(state=state_i, n=jj,
                                                                                        Nv=Nv, alpha_s=alpha_i,
                                                                                        q_s=q_i, Nx=Nx, m_s=m_i, E=E,
                                                                                        u_s=u_i, L=L,
                                                                                        closure=closure_i)
        # enforce that the coefficients live in the reals
        dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][:Nx] = \
            np.flip(np.conjugate(dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][Nx+1:]))

    # mass (odd)
    dydt_[-1] = -L * np.sqrt(Nv / 2) * integral_I0(n=Nv - 1) * np.flip(E).T @ ((q_e / m_e) * closure_e + (q_i / m_i) * closure_i)

    # mass (even)
    dydt_[-2] = -L * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * np.flip(E).T @ ((q_e / m_e) * state_e[-1, :] + (q_i / m_i) * state_i[-1, :])

    # momentum (odd)
    dydt_[-3] = -L * integral_I0(n=Nv - 1) * np.flip(E).T @ (np.sqrt(Nv / 2) * ( u_e * q_e * closure_e + u_i * q_i * closure_i) +
            Nv * (alpha_e * q_e * state_e[-1, :] + alpha_i * q_i * state_i[-1, :]))

    # momentum (even)
    dydt_[-4] = -L * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * np.flip(E).T @ (u_e * q_e * state_e[-1, :] + u_i * q_i * state_i[-1, :] +
          np.sqrt(2 * Nv) * (q_e * alpha_e * closure_e + q_i * alpha_i * closure_i))


    # energy (odd)
    dydt_[-5] = -L * integral_I0(n=Nv - 1) * np.flip(E).T @ (np.sqrt(Nv / 2) * (
            q_e * (0.5 * ((2 * Nv + 1) * (alpha_e ** 2) + u_e ** 2) * closure_e + q_e/m_e * J_inv @ scipy.signal.convolve(in1=closure_e, in2=E, mode="same")) +
            q_i * (0.5 * ((2 * Nv + 1) * (alpha_i ** 2) + u_i ** 2) * closure_i + q_i/m_i * J_inv @ scipy.signal.convolve(in1=closure_i, in2=E, mode="same"))) +
            + Nv * (u_e * q_e * alpha_e * state_e[-1, :] + u_i * q_i * alpha_i * state_i[-1, :]))

    # energy (even)
    dydt_[-6] = -L * integral_I0(n=Nv - 2) * np.flip(E).T @ (np.sqrt((Nv - 1) / 2) * (
            q_e * (0.5 * ((2 * Nv - 1) * (alpha_e ** 2) + u_e ** 2) * state_e[-1, :] + q_e/m_e * J_inv @ scipy.signal.convolve(in1=state_e[-1, :], in2=E, mode="same"))
            + q_i * (0.5 * ((2 * Nv - 1) * (alpha_i ** 2) + u_i ** 2) * state_i[-1, :] + q_i/m_i * J_inv @ scipy.signal.convolve(in1=state_i[-1, :], in2=E, mode="same")))
            + np.sqrt(Nv * (Nv - 1)) * (u_e * q_e * alpha_e * closure_e + u_i * q_i * alpha_i * closure_i))

    # L2 norm
    E_conv = scipy.linalg.convolution_matrix(a=E, n=Nx_total, mode='same')
    dydt_[-7] = - np.sqrt(2 * Nv) * L * (np.flip(state_e[-1, :]).T @ (q_e/m_e * E_conv) @ closure_e
                    + np.flip(state_i[-1, :]).T @ (q_i/m_i * E_conv) @ closure_i)

    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of Fourier spectral terms in x
    Nx = 50
    Nx_total = 2 * Nx + 1
    # number of Hermite spectral terms in v
    Nv = 1001
    # Velocity scaling of electron and ion
    alpha_e = 0.1
    alpha_i = np.sqrt(1 / 1836)
    # perturbation magnitude
    epsilon = 0.01
    # x grid is from 0 to L
    L = 2 * np.pi
    # final time
    T = 50
    # time stepping
    dt = 0.1
    # time vector
    t_vec = np.linspace(0, T, int(T / dt) + 1)
    # velocity scaling
    u_e = 1
    u_i = 1
    # mass normalized
    m_e = 1
    m_i = 1836
    # charge normalized
    q_e = -1
    q_i = 1
    # closure
    closure = "truncation"

    # inverse J
    J_inv = J_matrix_inv(Nx=Nx, L=L)
    # x direction
    x_project = np.linspace(0, L, int(1e5))

    # initial condition of the first expansion coefficient
    C_0e = np.zeros(Nx_total, dtype="complex128")
    C_0i = np.zeros(Nx_total, dtype="complex128")

    C_0e[Nx] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) / alpha_e
    C_0e[Nx - 1] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * 0.5 * epsilon / alpha_e
    C_0e[Nx + 1] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * 0.5 * epsilon / alpha_e

    C_0i[Nx] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) / alpha_i

    # initialize states (electrons and ions)
    states_e = np.zeros((Nv, Nx_total), dtype="complex128")
    states_i = np.zeros((Nv, Nx_total), dtype="complex128")

    # initialize the expansion coefficients
    states_e[0, :] = C_0e
    states_i[0, :] = C_0i

    # initial condition of the semi-discretized ODE
    y0 = np.append(states_e.flatten("C"), states_i.flatten("C"))
    y0 = np.append(y0, np.zeros(7))

    # set up implicit midpoint
    sol_midpoint_u, t_vec = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=dydt,
                                                     nonlinear_solver_type="newton_krylov",
                                                     r_tol=1e-12, a_tol=1e-12, max_iter=100)

    # save results
    np.save("../data/SW/langmuir/sol_midpoint_u_" + str(Nv) + "_" + str(closure) + "_closure", sol_midpoint_u)
    np.save("../data/SW/langmuir/sol_midpoint_t_" + str(Nv) + "_" + str(closure) + "_closure", t_vec)
