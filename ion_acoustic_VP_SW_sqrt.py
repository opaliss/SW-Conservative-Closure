from implicit_midpoint import implicit_midpoint_solver
import numpy as np
from operators import RHS, solve_poisson_equation, ampere_equation_RHS


def dydt(y, t):
    dydt_ = np.zeros(len(y), dtype="complex128")

    Nx_total = 2 * Nx + 1

    state_e = np.zeros((Nv, Nx_total), dtype="complex128")
    state_i = np.zeros((Nv, Nx_total), dtype="complex128")

    for jj in range(0, Nv):
        state_e[jj, :] = y[jj * Nx_total: (jj + 1) * Nx_total]
        state_i[jj, :] = y[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total]

    E = solve_poisson_equation(state_e=state_e,
                               state_i=state_i,
                               alpha_e=alpha_e,
                               alpha_i=alpha_i,
                               Nx=Nx,
                               L=L,
                               u_e=u_e,
                               u_i=u_i, Nv=Nv)


    for jj in range(Nv):
        dydt_[jj * Nx_total: (jj + 1) * Nx_total] = RHS(state=state_e,
                                                                n=jj,
                                                                Nv=Nv,
                                                                alpha_s=alpha_e,
                                                                q_s=q_e,
                                                                Nx=Nx,
                                                                m_s=m_e,
                                                                E=E,
                                                                u_s=u_e,
                                                                L=L)

        dydt_[jj * Nx_total: (jj + 1) * Nx_total][:Nx] = np.flip(np.conjugate(dydt_[jj * Nx_total: (jj + 1) * Nx_total][Nx+1:]))


        dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) *Nx_total] = RHS(state=state_i,
                                                                                                        n=jj,
                                                                                                        Nv=Nv,
                                                                                                        alpha_s=alpha_i,
                                                                                                        q_s=q_i,
                                                                                                        Nx=Nx,
                                                                                                        m_s=m_i,
                                                                                                        E=E,
                                                                                                        u_s=u_i,
                                                                                                        L=L)

        dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][:Nx] = np.flip(np.conjugate(dydt_[Nv * Nx_total + jj * Nx_total: Nv * Nx_total + (jj + 1) * Nx_total][Nx+1:]))


    print(t)
    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of mesh points in x
    Nx = 20
    # number of spectral expansions
    Nv = 10
    # epsilon displacement in initial electron distribution
    epsilon = 1e-4
    # velocity scaling of electron and ion
    alpha_e = np.sqrt(2)
    alpha_i = np.sqrt(2 / 50)
    # x grid is from 0 to L
    L = 10 * np.pi
    # spacial spacing dx = x[i+1] - x[i]
    dx = L / (Nx - 1)
    # time stepping
    dt = 1e-1
    # final time (non-dimensional)
    T = 20.
    t_vec = np.linspace(0, T, int(T / dt) + 1)
    # velocity scaling
    u_e = 2
    u_i = 0
    # mass normalized
    m_e = 1
    m_i = 25
    # charge normalized
    q_e = -1
    q_i = 1

    # x direction
    x = np.linspace(0, L, int(1e5) + 1)

    # initial condition of the first expansion coefficient
    C_0e = np.zeros(2 * Nx + 1, dtype="complex128")

    # project the electron species onto fourier space
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        C_0e[ii] = np.trapz(
            y=np.sqrt((1 + epsilon * np.cos(x/5)) / alpha_e) * np.exp(-2 * np.pi * 1j * kk * x / L), x=x,
            dx=x[1] - x[0]) / L

    # initialize states (electrons and ions)
    states_e = np.zeros((Nv, Nx * 2 + 1), dtype="complex128")
    states_i = np.zeros((Nv, Nx * 2 + 1), dtype="complex128")

    # initialize the expansion coefficients
    states_e[0, :] = C_0e
    states_i[0, Nx] = np.sqrt(1 / alpha_i)

    # initial condition of the semi-discretized ODE
    y0 = np.append(states_e.flatten("C"), states_i.flatten("C"))

    # set up implicit midpoint
    sol_midpoint_u = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=dydt,
                                              nonlinear_solver_type="newton",
                                              r_tol=1e-16, a_tol=1e-16, max_iter=100)

    # save results
    np.save("data/SW_sqrt/ion_acoustic/poisson/sol_midpoint_u_10_newton", sol_midpoint_u)
    np.save("data/SW_sqrt/ion_acoustic/poisson/sol_midpoint_t_10_newton", t_vec)
