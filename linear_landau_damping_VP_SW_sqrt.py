from implicit_midpoint import implicit_midpoint_solver
import numpy as np
from operators import RHS, solve_poisson_equation, J_matrix


def dydt(y, t):
    dydt_ = np.zeros(len(y), dtype="complex128")

    Nx_total = 2*Nx + 1
    state_e = np.zeros((Nv, Nx_total), dtype="complex128")
    state_i = np.zeros((Nv, Nx_total), dtype="complex128")
    # fix ions
    state_i[0, Nx] = np.sqrt(1 / alpha_i)

    for jj in range(Nv):
        state_e[jj, :] = y[jj * (2 * Nx + 1): (jj + 1) * (2 * Nx + 1)]

    E = solve_poisson_equation(state_e=state_e,
                               state_i=state_i,
                               alpha_e=alpha_e,
                               alpha_i=alpha_i,
                               Nx=Nx,
                               L=L, Nv=Nv, solver="SWSR")

    for jj in range(Nv):
        dydt_[jj * (2 * Nx + 1): (jj + 1) * (2 * Nx + 1)] = RHS(state=state_e, n=jj, Nv=Nv,
                                                                alpha_s=alpha_e, q_s=q_e,
                                                                Nx=Nx, m_s=m_e, E=E,
                                                                u_s=u_e, L=L, solver="SWSR")

        dydt_[jj * Nx_total: (jj + 1) * Nx_total][:Nx] = np.flip(
            np.conjugate(dydt_[jj * Nx_total: (jj + 1) * Nx_total][Nx + 1:]))


    D = J_matrix(Nx=Nx, L=L)
    dydt_[-1] = -L * (alpha_e ** 2) * np.sqrt((Nv - 1) / 2) * (Nv / 2) * (
                -m_e * (alpha_e ** 2) * np.flip(np.conjugate(state_e[-2, :])).T @ D @ state_e[-1, :] + q_e * np.flip(np.conjugate(E)).T @ (
                    state_e[-2, :] * state_e[-1, :])) \
                -L * (alpha_i ** 2) * np.sqrt((Nv - 1) / 2) * (Nv / 2) * (
                            -m_i * (alpha_i ** 2) * np.flip(np.conjugate(state_i[-2, :])).T @ D @ state_i[-1, :] + q_i * np.flip(np.conjugate(E)).T @ (
                                state_i[-2, :] * state_i[-1, :]))

    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of mesh points in x
    Nx = 50
    # number of spectral expansions
    Nv = 101
    # epsilon displacement in initial electron distribution
    epsilon = 1e-2
    # velocity scaling of electron and ion
    alpha_e = np.sqrt(2)
    alpha_i = np.sqrt(2 / 1863)
    # x grid is from 0 to L
    L = 2 * np.pi
    # time stepping
    dt = 1e-2
    # final time (nondimensional)
    T = 10.
    t_vec = np.linspace(0, T, int(T / dt) + 1)
    # velocity scaling
    u_e = 0
    u_i = 0
    # mass normalized
    m_e = 1
    m_i = 1863
    # charge normalized
    q_e = -1
    q_i = 1

    # x direction
    x = np.linspace(0, L, int(1e5) + 1)

    # initial condition of the first expansion coefficient
    C_0e = np.zeros(2 * Nx + 1, dtype="complex128")

    # project the electron species onto fourier space
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        C_0e[ii] = np.trapz(y=np.sqrt((1 + epsilon*np.cos(x)) / alpha_e) * np.exp(-2 * np.pi * 1j * kk * x / L), x=x,
                            dx=x[1] - x[0]) / L

    # initialize states (electrons and ions)
    states_e = np.zeros((Nv, Nx * 2 + 1), dtype="complex128")

    # initialize the expansion coefficients
    states_e[0, :] = C_0e

    # initial condition of the semi-discretized ODE
    y0 = states_e.flatten("C")
    y0 = np.append(y0, 0)

    # set up implicit midpoint
    sol_midpoint_u = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=dydt, nonlinear_solver_type="newton_krylov",
                                              r_tol=1e-10, a_tol=1e-16, max_iter=100)

    # save results
    np.save("data/SW_sqrt/linear_landau/poisson/sol_midpoint_u_101", sol_midpoint_u)
    np.save("data/SW_sqrt/linear_landau/poisson/sol_midpoint_t_101", t_vec)
