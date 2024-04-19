"""
Operators of the SW square-root Hermite-Fourier Expansion
"""
import numpy as np
import scipy.special


def integral_I0(n):
    """
    :param n: int, the order of the integral
    :return: the integral I0_{n}
    """
    if n < 0:
        return 0
    elif n == 0:
        return np.sqrt(2) * (np.pi ** (1 / 4))
    elif n % 2 == 1:
        return 0
    else:
        term = np.zeros(n+10)
        term[0] = np.sqrt(2) * (np.pi ** (1 / 4))
        for m in range(2, n+10):
            term[m] = np.sqrt((m - 1) / m) * term[m - 2]
        return term[n]


def integral_I1(n, u_s, alpha_s):
    """
    :param n: int, order of the integral
    :param u_s: float, the velocity shifting of species s
    :param alpha_s: float, the velocity scaling of species s
    :return: the integral I1_{n}
    """
    if n % 2 == 0:
        return u_s * integral_I0(n=n)
    else:
        return alpha_s * np.sqrt(2) * np.sqrt(n) * integral_I0(n=n - 1)


def integral_I2(n, u_s, alpha_s):
    """integral I2 in SW formulation

    :param n: int, order of the integral
    :param u_s: float, the velocity shifting of species s
    :param alpha_s: float, the velocity scaling of species s
    :return: the integral I2_{n}
    """
    if n % 2 == 0:
        return (alpha_s ** 2) * (0.5 * np.sqrt((n + 1) * (n + 2)) * integral_I0(n=n + 2) + (
                (2 * n + 1) / 2 + (u_s / alpha_s) ** 2) * integral_I0(n=n) + 0.5 * np.sqrt(n * (n - 1)) *
                                 integral_I0(n=n - 2))
    else:
        return 2 * u_s * integral_I1(n=n, u_s=u_s, alpha_s=alpha_s)


def fft_(coeff, Nx, x, L):
    """evaluate the fourier expansion given the fourier coefficients.

    :param coeff: vector of all fourier coefficients
    :param Nx: number of fourier modes (total 2Nx+1)
    :param x: spatial domain
    :param L: length of spatial domain
    :return: fourier expansion
    """
    sol = np.zeros(2 * Nx + 1, dtype="complex128")
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        sol += coeff[ii] * np.exp(1j * kk * x * 2 * np.pi / L)
    return sol.real


def psi_ln_sw(xi, n, alpha_s, u_s, v):
    """
    :param alpha_s: float, velocity scaling parameter
    :param u_s, float, velocity shifting parameter
    :param v: float or array, the velocity coordinate sampled at version points
    :param xi: float or array, xi^{s} scaled velocity, i.e. xi = (v - u^{s})/alpha^{s}
    :param n: int, order of polynomial
    :return: float or array,  asymmetrically weighted (AW) hermite polynomial of degree n evaluated at xi
    """
    if n == 0:
        return np.exp(-(xi ** 2) / 2) / np.sqrt(np.sqrt(np.pi))
    if n == 1:
        return np.exp(-(xi ** 2) / 2) * (2 * xi) / np.sqrt(2 * np.sqrt(np.pi))
    else:
        psi = np.zeros((n + 1, len(xi)))
        psi[0, :] = np.exp(-(xi ** 2) / 2) / np.sqrt(np.sqrt(np.pi))
        psi[1, :] = np.exp(-(xi ** 2) / 2) * (2 * xi) / np.sqrt(2 * np.sqrt(np.pi))
        for jj in range(1, n):
            factor = - alpha_s * np.sqrt((jj + 1) / 2)
            psi[jj + 1, :] = (alpha_s * np.sqrt(jj / 2) * psi[jj - 1, :] + u_s * psi[jj, :] - v * psi[jj, :]) / factor
    return psi[n, :]


def J_matrix(Nx, L):
    """return J matrix for linear term

    :param Nx: number of spatial spectral terms
    :param L: the length of the spatial domain
    :return: J matrix (anti-symmetric + diagonal)
    """
    J = np.zeros(((2 * Nx + 1), (2 * Nx + 1)), dtype="complex128")
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        J[ii, ii] = (2 * np.pi * 1j * kk) / L
    return J


def J_matrix_inv(Nx, L):
    """return J matrix inverse for drift term

    :param Nx: number of spatial spectral terms
    :param L: the length of the spatial domain
    :return: J matrix (anti-symmetric + diagonal)
    """
    J = np.zeros(((2 * Nx + 1), (2 * Nx + 1)), dtype="complex128")
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        if kk != 0:
            J[ii, ii] = L / (2 * np.pi * 1j * kk)
    return J


def linear(state, n, u_s, alpha_s, Nv, closure):
    """

    :param state: state of all coefficients, dimensions =  Nv x (2*Nx+1)
    :param n: the current state velocity index
    :param u_s: velocity shifting parameter
    :param alpha_s: velocity scaling parameter
    :param Nv: the number of Hermite spectral terms in velocity
    :return: L_{1}(x, t=t*) for C^{s}_{n}
    """
    # the first term in the expansion
    if n == 0:
        term1 = 0 * state[n, :]
        term2 = alpha_s * np.sqrt((n + 1) / 2) * state[n + 1, :]
        term3 = u_s * state[n, :]
        return term1, term2, term3

    # the last term in the expansion
    elif n == Nv - 1:
        term1 = alpha_s * np.sqrt(n / 2) * state[n - 1, :]
        term2 = alpha_s * np.sqrt((n + 1) / 2) * closure
        term3 = u_s * state[n, :]
        return term1, term2, term3

    # all other terms
    else:
        term1 = alpha_s * np.sqrt(n / 2) * state[n - 1, :]
        term2 = alpha_s * np.sqrt((n + 1) / 2) * state[n + 1, :]
        term3 = u_s * state[n, :]
        return term1, term2, term3


def nonlinear_SW(state, n, E, Nv, q_s, m_s, alpha_s, closure):
    """

    :param state: state of all coefficients, dimensions = Nv x (2*Nx+1)
    :param E: electric field coefficients, dimensions = (2*Nk+1)
    :param Nv: the number of Hermite spectral terms in velocity
    :param q_s: charge of species s
    :param m_s: mass of species s
    :param alpha_s: velocity scaling parameter
    :return: N(x, t=t*) for C^{s}_{n}
    """
    if n == 0:
        return (q_s / (m_s * alpha_s)) * np.sqrt((n + 1) / 2) * scipy.signal.convolve(in1=state[n + 1, :], in2=E,
                                                                                      mode="same")
    elif n == Nv - 1:
        return (q_s / (m_s * alpha_s)) * (np.sqrt((n + 1) / 2) * scipy.signal.convolve(in1=closure, in2=E, mode="same")
                                          - np.sqrt(n / 2) * scipy.signal.convolve(in1=state[n - 1, :], in2=E, mode="same"))
    else:
        return (q_s / (m_s * alpha_s)) * (np.sqrt((n + 1) / 2) * scipy.signal.convolve(in1=state[n + 1, :],
                                                                                       in2=E, mode="same")
                                          - np.sqrt(n / 2) * scipy.signal.convolve(in1=state[n - 1, :],
                                                                                   in2=E, mode="same"))


def linear_2_SW(state_e, state_i, alpha_e, alpha_i, Nx, Nv, q_e=-1, q_i=1):
    """
    :param q_i: ion charge (normalized)
    :param q_e: electron charge (normalized)
    :param Nv: number of spectral Hermite modes
    :param Nx: number of spectral fourier modes (total 2Nx+1)
    :param state_e: a matrix of electron states at time t=t*
    :param state_i: a matrix of ion states at time t=t*
    :param alpha_e: the velocity scaling of electrons
    :param alpha_i: the velocity scaling of ions
    :return: L_{2}(x, t=t*)
    """
    term1 = np.zeros(2 * Nx + 1, dtype="complex128")
    term2 = np.zeros(2 * Nx + 1, dtype="complex128")
    for m in range(Nv):
        term1 += alpha_e * state_e[m, :] * integral_I0(n=m)
        term2 += alpha_i * state_i[m, :] * integral_I0(n=m)
    return q_e * term1 + q_i * term2


def linear_2_two_stream_SW(state_e1, state_e2, state_i, alpha_e1, alpha_e2, alpha_i, Nx, Nv, q_e1=-1, q_e2=-1, q_i=1):
    """
    :param q_i: ion charge
    :param q_e2: electron 2 charge
    :param q_e1: electron 1 charge
    :param state_e1: a matrix of electron coefficients (species 1) at time t=t*
    :param state_e2: a matrix of electron coefficients (species 2) at time t=t*
    :param state_i: a matrix of ion states at time t=t*
    :param Nv: number of spectral Hermite modes
    :param Nx: number of spectral fourier modes (total 2Nx+1)
    :param alpha_e1: the velocity scaling of electrons (species 1)
    :param alpha_e2: the velocity scaling of electrons (species 2)
    :param alpha_i: the velocity scaling of ions
    :return: L_{2}(x, t=t*).
    """
    term1 = np.zeros(2 * Nx + 1, dtype="complex128")
    term2 = np.zeros(2 * Nx + 1, dtype="complex128")
    term3 = np.zeros(2 * Nx + 1, dtype="complex128")
    for m in range(Nv):
        term1 += alpha_e1 * state_e1[m, :] * integral_I0(n=m)
        term2 += alpha_e2 * state_e2[m, :] * integral_I0(n=m)
        term3 += alpha_i * state_i[m, :] * integral_I0(n=m)
    return q_e1 * term1 + q_e2 * term2 + q_i * term3


def RHS(state, n, q_s, m_s, L, u_s, alpha_s, E, Nv, Nx, closure):
    """
    :param state: state of all coefficients, dimensions = Nv x (2*Nx+1)
    :param n: the current state velocity index
    :param q_s: charge of species s
    :param m_s: mass of species s
    :param L: spatial length L
    :param u_s: velocity space shifting parameter
    :param alpha_s: velocity scaling parameter
    :param E: electric field coefficients, dimensions = (2*Nk+1)
    :param Nv: the number of Hermite spectral terms in velocity
    :param Nx: the number of Fourier spectral terms is 2*Nx+1
    :return: dC_{n}/dt
    """
    term1, term2, term3 = linear(state=state, n=n, u_s=u_s, alpha_s=alpha_s, Nv=Nv, closure=closure)
    J = J_matrix(Nx=Nx, L=L)
    return -J @ (term1 + term2 + term3) - nonlinear_SW(state=state, n=n, E=E, Nv=Nv, q_s=q_s, m_s=m_s, alpha_s=alpha_s,
                                                       closure=closure)


def solve_poisson_equation_two_stream(state_e1, state_e2, state_i, alpha_e1, alpha_e2, alpha_i, Nx, Nv, L):
    """
    :param L: spatial length
    :param Nv: number of velocity Hermite modes
    :param state_e1: a matrix of electron coefficients (species 1) at time t=t*
    :param state_e2: a matrix of electron coefficients (species 2) at time t=t*
    :param state_i: a matrix of ion states at time t=t*
    :param alpha_e1: the velocity scaling of electrons (species 1)
    :param alpha_e2: the velocity scaling of electrons (species 2)
    :param alpha_i: the velocity scaling of ions
    :param Nx: number of spectral terms in space (2*Nx+1) in total
    :return: E(x, t=t*)
    """
    rhs = linear_2_two_stream_SW(state_e1=state_e1, state_e2=state_e2, state_i=state_i,
                                 alpha_e1=alpha_e1, alpha_e2=alpha_e2, alpha_i=alpha_i,
                                 Nx=Nx, Nv=Nv)

    #print("mean density", np.abs(rhs[Nx]))

    E = np.zeros(2*Nx+1, dtype="complex128")

    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        if kk != 0:
            if kk > 0:
                E[ii] = np.conjugate(E[Nx - kk])
            else:
                E[ii] = (L / (2 * np.pi * kk * 1j)) * rhs[ii]
    return E


def solve_poisson_equation(state_e, state_i, alpha_e, alpha_i, Nx, Nv, L):
    """
    :param L: spatial length
    :param Nv: number of velocity Hermite modes
    :param state_e: a matrix of electron coefficients at time t=t*
    :param state_i: a matrix of ion states at time t=t*
    :param alpha_e: the velocity scaling of electrons
    :param alpha_i: the velocity scaling of ions
    :param Nx: number of spectral terms in space (2*Nx+1) in total
    :return: E(x, t=t*)
    """
    rhs = linear_2_SW(state_e=state_e, state_i=state_i, alpha_e=alpha_e, alpha_i=alpha_i, Nx=Nx, Nv=Nv)
    E = np.zeros(2*Nx+1, dtype="complex128")

    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        if kk != 0:
            if kk > 0:
                E[ii] = np.conjugate(E[Nx - kk])
            else:
                E[ii] = L / (2 * np.pi * kk * 1j) * rhs[ii]
    return E


def mass(state, Nv, Nx):
    """mass of the particular state

    :param state: ndarray, electron or ion state
    :param Nv: int, number of velocity Hermite spectral terms
    :return: mass for the state
    """
    res = 0
    for m in range(Nv):
        res += integral_I0(n=m) * state[m, Nx]
    return res


def momentum(state, u_s, alpha_s, Nv, Nx):
    """momentum of the particular state

    :param state: ndarray, electron or ion state
    :param Nv: int, number of velocity Hermite spectral terms
    :param u_s: float, the velocity shifting parameter of species s
    :param alpha_s: float, the velocity scaling parameter of species s
    :return: momentum for the state
    """
    res = 0
    for m in range(Nv):
        res += integral_I1(n=m, u_s=u_s, alpha_s=alpha_s) * state[m, Nx]
    return res


def energy_k(state, u_s, alpha_s, Nv, Nx):
    """kinetic energy of the particular state

    :param state: ndarray, electron or ion state
    :param Nv: int, number of velocity Hermite spectral terms
    :param u_s: float, the velocity shifting parameter of species s
    :param alpha_s: float, the velocity scaling parameter of species s
    :return: kinetic energy for the state
    """
    res = 0
    for m in range(Nv):
        res += integral_I2(n=m, u_s=u_s, alpha_s=alpha_s) * state[m, Nx]
    return res


def L2(state, Nv):
    res = 0
    for m in range(Nv):
        res += np.flip(state[m, :]).T @ state[m, :]
    return res


def total_mass(state, alpha_s, L, Nv, Nx):
    """total mass of single electron and ion setup

    :param state: ndarray, species s state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param Nv: int, the number of velocity spectral terms
    :return: total mass of single electron and ion setup
    """
    return mass(state=state, Nv=Nv, Nx=Nx) * L * alpha_s


def total_momentum(state, alpha_s, L, Nv, m_s, u_s, Nx):
    """total momentum of single electron and ion setup

    :param state: ndarray, species s state
    :param alpha_s: float, velocity scaling of species s
    :param L: float, spatial length
    :param Nv: int, the number of velocity spectral terms
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total momentum of single electron and ion setup
    """
    return momentum(state=state, Nv=Nv, alpha_s=alpha_s, u_s=u_s, Nx=Nx) * L * alpha_s * m_s


def total_energy_k(state, alpha_s, L, Nv, m_s, u_s, Nx):
    """total kinetic energy of single electron and ion setup

    :param state: ndarray, species s  state
    :param alpha_s: float, velocity scaling of species s
    :param L: float, spatial length
    :param Nv: int, the number of velocity spectral terms
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total kinetic energy of single electron and ion setup
    """
    return 0.5 * energy_k(state=state, Nv=Nv, alpha_s=alpha_s, u_s=u_s, Nx=Nx) * L * alpha_s * m_s


def total_L2_norm(state,  L, Nv, alpha_s):
    """total kinetic energy of single electron and ion setup

    :param state: ndarray, species s  state
    :param alpha_s: float, velocity scaling of species s
    :param L: float, spatial length
    :param Nv: int, the number of velocity spectral terms
    :return: total kinetic energy of single electron and ion setup
    """
    return L * alpha_s * L2(state=state, Nv=Nv)