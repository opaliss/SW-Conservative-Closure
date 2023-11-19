from implicit_midpoint import implicit_midpoint_solver
import numpy as np
from operators import RHS, solve_poisson_equation_two_stream


def dydt(y, t):
    dydt_ = np.zeros(len(y), dtype="complex128")

    Nx_total = 2 * Nx + 1

    state_e1 = np.zeros((Nv, Nx_total), dtype="complex128")
    state_e2 = np.zeros((Nv, Nx_total), dtype="complex128")
    state_i = np.zeros((Nv, Nx_total), dtype="complex128")
    state_i[0, Nx] = 1 / alpha_i


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
                                          u_i=u_i, Nv=Nv, solver="AW")

    for jj in range(Nv):
        dydt_[jj * (2 * Nx + 1): (jj + 1) * (2 * Nx + 1)] = RHS(state=state_e1, n=jj, Nv=Nv,
                                                                alpha_s=alpha_e1, q_s=q_e1,
                                                                Nx=Nx, m_s=m_e1, E=E,
                                                                u_s=u_e1, L=L, solver="AW", nu=nu)
        # enforce that the coefficients live in the reals
        #dydt_[jj * Nx_total: (jj + 1) * Nx_total][:Nx] = np.flip(np.conjugate(dydt_[jj * Nx_total: (jj + 1) * Nx_total][Nx+1:]))

        dydt_[Nv * (2 * Nx + 1) + jj * (2 * Nx + 1): Nv * (2 * Nx + 1) + (jj + 1) * (2 * Nx + 1)] = RHS(state=state_e2,
                                                                                                        n=jj, Nv=Nv,
                                                                                                        alpha_s=alpha_e2,
                                                                                                        q_s=q_e2,
                                                                                                        Nx=Nx, m_s=m_e2,
                                                                                                        E=E,
                                                                                                        u_s=u_e2, L=L, solver="AW", nu=nu)
        # enforce that the coefficients live in the reals
        #dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][:Nx] = \
        #    np.flip(np.conjugate(dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][Nx+1:]))

    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of mesh points in x
    Nx = 20
    # number of spectral expansions
    Nv = 10
    # Velocity scaling of electron and ion
    alpha_e1 = 0.5
    alpha_e2 = 0.5
    alpha_i = np.sqrt(2 / 1863)
    # perturbation magnitude
    epsilon = 1e-3
    # x grid is from 0 to L
    L = 2 * np.pi
    # final time
    T = 35
    # time stepping
    dt = 0.05
    # time vector
    t_vec = np.linspace(0, T, int(T / dt) + 1)
    # velocity scaling
    u_e1 = -1
    u_e2 = 1
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
    delta_e1 = 0.5
    delta_e2 = 0.5
    # collisions
    nu = 5

    # x direction
    x = np.linspace(0, L, int(1e5)+1)

    # initial condition of the first expansion coefficient
    C_0e1 = np.zeros(2 * Nx + 1, dtype="complex128")
    C_0e2 = np.zeros(2 * Nx + 1, dtype="complex128")
    C_0e1[Nx] = delta_e1 / alpha_e1
    C_0e2[Nx] = delta_e2 / alpha_e2
    C_0e1[Nx - 1] = epsilon * 0.5 * delta_e1 / alpha_e1
    C_0e1[Nx + 1] = epsilon * 0.5 * delta_e1 / alpha_e1
    C_0e2[Nx - 1] = epsilon * 0.5 * delta_e2 / alpha_e2
    C_0e2[Nx + 1] = epsilon * 0.5 * delta_e2 / alpha_e2

    # initialize states (electrons and ions)
    states_e1 = np.zeros((Nv, Nx * 2 + 1), dtype="complex128")
    states_e2 = np.zeros((Nv, Nx * 2 + 1), dtype="complex128")

    # initialize the expansion coefficients
    states_e1[0, :] = C_0e1
    states_e2[0, :] = C_0e2

    # initial condition of the semi-discretized ODE
    y0 = np.append(states_e1.flatten("C"), states_e2.flatten("C"))

    # set up implicit midpoint
    sol_midpoint_u = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=dydt, nonlinear_solver_type="newton_krylov",
                                              r_tol=1e-8, a_tol=1e-15, max_iter=30)

    # save results
    np.save("data/AW/two_stream/poisson/sol_midpoint_u_10", sol_midpoint_u)
    np.save("data/AW/two_stream/poisson/sol_midpoint_t_10", t_vec)
