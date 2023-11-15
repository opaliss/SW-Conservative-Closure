import numpy as np
import scipy.optimize


def implicit_midpoint_formulation(y_new, y_old, dt, t_old, rhs):
    """return the nonlinear equation for implicit midpoint.

    :param y_new: y_{n+1}
    :param y_old: y_{n}
    :param dt: time step t_{n+1} - t_{n}
    :param t_old: t_{n}
    :param rhs: a function of the rhs of the dynamical system dy/dt = rhs(y, t)
    :return: y_{n+1} - y_{n} -dt*rhs(y=(y_{n}+y_{n+1})/2, t=t_{n} + dt/2)
    """
    return y_new - y_old - dt * rhs(y=0.5 * (y_old + y_new), t=t_old + dt / 2)


def implicit_midpoint_solver(t_vec, y0, rhs, nonlinear_solver_type="anderson", r_tol=1e-8, a_tol=1e-15, max_iter=100):
    """Solve the system

        dy/dt = rhs(y),    y(0) = y0,

    via the implicit midpoint method.

    The nonlinear equation at each time step is solved using Anderson acceleration.

    Parameters
    ----------
    :param max_iter: maximum iterations of nonlinear solver
    :param a_tol: absolute tolerance nonlinear solver
    :param r_tol: relative tolerance nonlinear solver
    :param nonlinear_solver_type: type of nonlinear solver, options: "anderson", "newton_kyrlov", and "newton".
    :param t_vec: array with timesteps
    :param y0 : initial condition.
    :param rhs : function of the right-hand-side, i.e. dy/dt = rhs(y, t)

    Returns
    -------
    u : (Nx, Nt) ndarray
        Solution to the ODE at time t_vec; that is, y[:,j] is the
        computed solution corresponding to time t[j].

    """
    # store the dimensions of the problem.
    Nt = len(t_vec)
    Nx = len(y0)

    # array of time steps.
    dt = t_vec[1:] - t_vec[:-1]

    # initialize the solution matrix
    y_sol = np.zeros((Nx, Nt), dtype="complex128")
    y_sol[:, 0] = y0

    # for-loop each time-step
    for tt in range(1, Nt):
        print("\n time = ", t_vec[tt])
        if nonlinear_solver_type == "anderson":
            y_sol[:, tt] = scipy.optimize.anderson(F=lambda y: implicit_midpoint_formulation(y_new=y,
                                                                                             y_old=y_sol[:, tt - 1],
                                                                                             rhs=rhs,
                                                                                             dt=dt[tt - 1],
                                                                                             t_old=t_vec[tt - 1]),
                                                   xin=y_sol[:, tt - 1],
                                                   maxiter=max_iter,
                                                   f_tol=a_tol,
                                                   f_rtol=r_tol,
                                                   verbose=True)
        elif nonlinear_solver_type == "newton":
            y_sol[:, tt] = scipy.optimize.newton(func=lambda y: implicit_midpoint_formulation(y_new=y,
                                                                                              y_old=y_sol[:, tt - 1],
                                                                                              rhs=rhs,
                                                                                              dt=dt[tt - 1],
                                                                                              t_old=t_vec[tt - 1]),
                                                 x0=y_sol[:, tt - 1],
                                                 maxiter=max_iter,
                                                 tol=a_tol,
                                                 rtol=r_tol)

        elif nonlinear_solver_type == "newton_krylov":
            y_sol[:, tt] = scipy.optimize.newton_krylov(F=lambda y: implicit_midpoint_formulation(y_new=y,
                                                                                                  y_old=y_sol[:, tt - 1],
                                                                                                  rhs=rhs,
                                                                                                  dt=dt[tt - 1],
                                                                                                  t_old=t_vec[tt - 1]),
                                                        xin=y_sol[:, tt - 1],
                                                        maxiter=100,
                                                        f_tol=a_tol,
                                                        f_rtol=r_tol,
                                                        verbose=True)
    return y_sol