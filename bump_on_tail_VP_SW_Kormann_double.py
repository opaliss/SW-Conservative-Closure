from implicit_midpoint import implicit_midpoint_solver
import numpy as np
from operators import RHS, solve_poisson_equation_two_stream, integral_I0, J_matrix_inv


def dydt(y, t):
    dydt_ = np.zeros(len(y), dtype="complex128")

    Nx_total = 2 * Nx + 1

    state_e1 = np.zeros((Nv, Nx_total), dtype="complex128")
    state_e2 = np.zeros((Nv, Nx_total), dtype="complex128")
    state_i = np.zeros((Nv, Nx_total), dtype="complex128")
    state_i[0, Nx] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) / alpha_i

    for jj in range(Nv):
        state_e1[jj, :] = y[jj * Nx_total: (jj + 1) * Nx_total]
        state_e2[jj, :] = y[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total]

    E = solve_poisson_equation_two_stream(state_e1=state_e1,
                                          state_e2=state_e2,
                                          state_i=state_i,
                                          alpha_e1=alpha_e1,
                                          alpha_i=alpha_i,
                                          alpha_e2=alpha_e2,
                                          Nx=Nx,
                                          L=L,
                                          u_e1=u_e1,
                                          u_e2=u_e2,
                                          u_i=u_i, Nv=Nv, solver="SW")

    for jj in range(Nv):
        dydt_[jj * (2 * Nx + 1): (jj + 1) * (2 * Nx + 1)] = RHS(state=state_e1, n=jj, Nv=Nv,
                                                                alpha_s=alpha_e1, q_s=q_e1,
                                                                Nx=Nx, m_s=m_e1, E=E,
                                                                u_s=u_e1, L=L)
        # enforce that the coefficients live in the reals
        dydt_[jj * Nx_total: (jj + 1) * Nx_total][:Nx] = np.flip(
            np.conjugate(dydt_[jj * Nx_total: (jj + 1) * Nx_total][Nx + 1:]))

        dydt_[Nv * (2 * Nx + 1) + jj * (2 * Nx + 1): Nv * (2 * Nx + 1) + (jj + 1) * (2 * Nx + 1)] = RHS(state=state_e2,
                                                                                                        n=jj, Nv=Nv,
                                                                                                        alpha_s=alpha_e2,
                                                                                                        q_s=q_e2,
                                                                                                        Nx=Nx, m_s=m_e2,
                                                                                                        E=E,
                                                                                                        u_s=u_e2, L=L)
        # enforce that the coefficients live in the reals
        dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][:Nx] = \
            np.flip(np.conjugate(dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][Nx + 1:]))

        # TODO: quick change to the drift rate in Kormann et al.

        # mass (even)
        dydt_[-1] = -L * (q_e1 / m_e1) * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2, Nv=Nv) * np.conjugate(E).T @ state_e1[-1, :] \
                    -L * (q_e2 / m_e2) * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2, Nv=Nv) * np.conjugate(E).T @ state_e2[-1, :] \
                    -L * (q_i / m_i) * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2, Nv=Nv) * np.conjugate(E).T @ state_i[-1, :]

        # momentum (odd)
        dydt_[-2] = -L * (Nv - 1) * integral_I0(n=Nv - 1, Nv=Nv) * np.conjugate(E).T @ (
                    alpha_e1 * q_e1 * state_e1[-1, :] +
                    alpha_e2 * q_e2 * state_e2[-1, :] +
                    alpha_i * q_i * state_i[-1, :])

        # momentum (even)
        dydt_[-3] = -L * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2, Nv=Nv) * np.conjugate(E).T @ (
                    u_e1 * q_e1 * state_e1[-1, :] +
                    u_e2 * q_e2 * state_e2[-1, :] +
                    u_i * q_i * state_i[-1, :])

        # energy (odd)
        dydt_[-4] = -L * (Nv - 1) * integral_I0(n=Nv - 1, Nv=Nv) * np.conjugate(E).T @ (
                    u_e1 * q_e1 * state_e1[-1, :] +
                    u_e2 * q_e2 * state_e2[-1, :] +
                    u_i * q_i * state_i[-1, :])

        # energy (even)
        dydt_[-5] = -L * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2, Nv=Nv) * np.conjugate(E).T @ (
                q_e1 * ((2 * Nv - 1) * (alpha_e1 ** 2) + u_e1 ** 2) * state_e1[-1, :]
                + q_e2 * ((2 * Nv - 1) * (alpha_e2 ** 2) + u_e2 ** 2) * state_e2[-1, :]
                + q_i * ((2 * Nv - 1) * (alpha_i ** 2) + u_i ** 2) * state_i[-1, :])

    return dydt_


if __name__ == '__main__':
    # TODO: parameterization following Kormann et al. 2021
    # set up configuration parameters
    # number of mesh points in x
    Nx = 30
    # number of spectral expansions
    Nv = 80
    # Velocity scaling of electron and ion
    alpha_e1 = 1
    alpha_e2 = 1 / np.sqrt(2)
    alpha_i = np.sqrt(1 / 1863)
    # perturbation magnitude
    epsilon = 0.03
    # x grid is from 0 to L
    L = 20 * np.pi / 3
    # final time
    T = 20.
    dt = 1e-2
    t_vec = np.linspace(0, T, int(T/dt) + 1)
    # velocity scaling
    u_e1 = 0
    u_e2 = 0
    u_i = 0
    # mass normalized
    m_e1 = 1
    m_e2 = 1
    m_i = 1863
    # charge normalized
    q_e1 = -1
    q_e2 = -1
    q_i = 1
    # scaling of bulk and bump
    delta_e1 = 9 / 10
    delta_e2 = 1 / 10

    # initialize the expansion coefficients
    states_e1 = np.load("data/SW/bump_on_tail/e1_initial.npy")
    states_e2 = np.load("data/SW/bump_on_tail/e2_initial.npy")

    # initial condition of the semi-discretized ODE
    y0 = np.append(states_e1.flatten("C"), states_e2.flatten("C"))
    y0 = np.append(y0, np.zeros(5))

    # set up implicit midpoint
    sol_midpoint_u = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=dydt, nonlinear_solver_type="newton_krylov",
                                              r_tol=1e-12, a_tol=1e-15, max_iter=100)

    # save results
    np.save("data/SW/bump_on_tail/poisson/sol_midpoint_u_80_kormann", sol_midpoint_u)
    np.save("data/SW/bump_on_tail/poisson/sol_midpoint_t_80_kormann", t_vec)
