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

        # mass (even)
        dydt_[-1] = -L * (q_e1 / m_e1) * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2, Nv=Nv) * np.flip(
            np.conjugate(E).T) @ state_e1[-1, :] \
                    - L * (q_e2 / m_e2) * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2, Nv=Nv) * np.flip(
            np.conjugate(E).T) @ state_e2[-1, :] \
                    - L * (q_i / m_i) * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2, Nv=Nv) * np.flip(
            np.conjugate(E).T) @ state_i[-1, :]

        # momentum (odd)
        dydt_[-2] = -L * (Nv - 1) * integral_I0(n=Nv - 1, Nv=Nv) * np.flip(np.conjugate(E).T) @ (
                    alpha_e1 * q_e1 * state_e1[-1, :] +
                    alpha_e2 * q_e2 * state_e2[-1, :] +
                    alpha_i * q_i * state_i[-1, :])

        # momentum (even)
        dydt_[-3] = -L * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2, Nv=Nv) * np.flip(np.conjugate(E).T) @ (
                    u_e1 * q_e1 * state_e1[-1, :] +
                    u_e2 * q_e2 * state_e2[-1, :] +
                    u_i * q_i * state_i[-1, :]) \
                    - (L/2) / m_e1 * integral_I0(n=Nv - 1, Nv=Nv) * np.sqrt((Nv - 1) / 2) * np.flip(np.conjugate(E).T) \
                    @ J_matrix_inv(Nx=Nx, L=L) @ np.convolve(E, state_e1[-1, :], mode="same") \
                    - (L / 2) / m_e2 * integral_I0(n=Nv - 1, Nv=Nv) * np.sqrt((Nv - 1) / 2) * np.flip(np.conjugate(E).T) \
                    @ J_matrix_inv(Nx=Nx, L=L) @ np.convolve(E, state_e2[-1, :], mode="same") \
                    - (L / 2) / m_i * integral_I0(n=Nv - 1, Nv=Nv) * np.sqrt((Nv - 1) / 2) * np.flip(np.conjugate(E).T) \
                    @ J_matrix_inv(Nx=Nx, L=L) @ np.convolve(E, state_i[-1, :], mode="same")

        # energy (odd)
        dydt_[-4] = -L * (Nv - 1) * integral_I0(n=Nv - 1, Nv=Nv) * np.flip(np.conjugate(E).T) @ (
                    u_e1 * q_e1 * state_e1[-1, :] +
                    u_e2 * q_e2 * state_e2[-1, :] +
                    u_i * q_i * state_i[-1, :])
        # energy (even)
        dydt_[-5] = -L * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2, Nv=Nv) * np.flip(np.conjugate(E).T) @ (
                q_e1 * ((2 * Nv - 1) * (alpha_e1 ** 2) + u_e1 ** 2) * state_e1[-1, :]
                + q_e2 * ((2 * Nv - 1) * (alpha_e2 ** 2) + u_e2 ** 2) * state_e2[-1, :]
                + q_i * ((2 * Nv - 1) * (alpha_i ** 2) + u_i ** 2) * state_i[-1, :])

    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of mesh points in x
    Nx = 50
    # number of spectral expansions
    Nv = 100
    # Velocity scaling of electron and ion
    alpha_e1 = 1
    alpha_e2 = np.sqrt(1 / 2)
    alpha_i = np.sqrt(1 / 1863)
    # perturbation magnitude
    epsilon = 0.03
    # x grid is from 0 to L
    L = 2 * np.pi / 0.3
    # final time
    T = 20.
    # time stepping
    dt = 0.1
    # time vector
    t_vec = np.linspace(0, T, int(T / dt) + 1)
    # velocity scaling
    u_e1 = 0
    u_e2 = 4.5
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

    # x direction
    x_project = np.linspace(0, L, int(1e5))

    # initial condition of the first expansion coefficient
    C_0e1 = np.zeros(2 * Nx + 1, dtype="complex128")
    C_0e2 = np.zeros(2 * Nx + 1, dtype="complex128")

    # initialize the expansion coefficients
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        C_0e1[ii] = np.trapz(y=delta_e1 * (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * (
                1 + epsilon * np.cos(0.3 * x_project)) / alpha_e1 * np.exp(-2 * np.pi * 1j * kk * x_project / L),
                             x=x_project,
                             dx=x_project[1] - x_project[0]) / L
        C_0e2[ii] = np.trapz(y=delta_e2 * (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * (
                1 + epsilon * np.cos(0.3 * x_project)) / alpha_e2 * np.exp(-2 * np.pi * 1j * kk * x_project / L),
                             x=x_project,
                             dx=x_project[1] - x_project[0]) / L

    # initialize states (electrons and ions)
    states_e1 = np.zeros((Nv, Nx * 2 + 1), dtype="complex128")
    states_e2 = np.zeros((Nv, Nx * 2 + 1), dtype="complex128")

    # initialize the expansion coefficients
    states_e1[0, :] = C_0e1
    states_e2[0, :] = C_0e2

    # initial condition of the semi-discretized ODE
    y0 = np.append(states_e1.flatten("C"), states_e2.flatten("C"))
    y0 = np.append(y0, np.zeros(5))

    # set up implicit midpoint
    sol_midpoint_u = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=dydt, nonlinear_solver_type="newton_krylov",
                                              r_tol=1e-10, a_tol=1e-15, max_iter=100)

    # save results
    np.save("data/SW/bump_on_tail/poisson/sol_midpoint_u_64", sol_midpoint_u)
    np.save("data/SW/bump_on_tail/poisson/sol_midpoint_t_64", t_vec)
