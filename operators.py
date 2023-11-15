"""
Operators of the SW square-root Hermite-Fourier Expansion
"""
import numpy as np
import scipy.special


def fft_(coeff, Nx, x, L):
    sol = np.zeros(2 * Nx + 1, dtype="complex128")
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        sol += coeff[ii] * np.exp(1j * kk * x * 2 * np.pi / L)
    return sol.real


def psi_ln_sw(xi, n):
    """
    :param xi: xi^{s} scaled velocity, i.e. xi = (v - u^{s})/alpha^{s}
    :param n: order of polynomial
    :return: symmetrically weighted hermite polynomial of degree n evaluated at xi
    """
    # hermite polynomial of degree n
    hermite = scipy.special.hermite(n=n)
    # (pi *2^n n!)^{-1/2}
    factor = (np.sqrt(np.pi) * (2 ** n) * scipy.special.factorial(n=n)) ** (-1 / 2)
    return factor * hermite(xi) * np.exp(0.5 * (-xi ** 2))


def J_matrix(Nx, L):
    """return J matrix for linear term

    :param Nx: number of spatial spectral terms
    :param L: the length of the spatial domain
    :return: J matrix (anti-symmetric + diagonal)
    """
    J = np.zeros(((2 * Nx + 1), (2 * Nx + 1)), dtype="complex128")
    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        J[ii, ii] = 2 * np.pi * 1j * kk / L
    return J


def linear(state, n, u_s, alpha_s, Nv):
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
        term2 = 0 * state[n, :]
        term3 = u_s * state[n, :]
        return term1, term2, term3

    # all other terms
    else:
        term1 = alpha_s * np.sqrt(n / 2) * state[n - 1, :]
        term2 = alpha_s * np.sqrt((n + 1) / 2) * state[n + 1, :]
        term3 = u_s * state[n, :]
        return term1, term2, term3


def nonlinear(state, n, E, Nv, q_s, m_s, alpha_s):
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
        return (q_s / (m_s * alpha_s)) * np.sqrt((n + 1) / 2) * np.convolve(a=state[n + 1, :], v=E, mode="same")
    elif n == Nv - 1:
        return (q_s / (m_s * alpha_s)) * -np.sqrt(n / 2) * np.convolve(a=state[n - 1, :], v=E, mode="same")
    else:
        return (q_s / (m_s * alpha_s)) * (np.sqrt((n + 1) / 2) * np.convolve(a=state[n + 1, :], v=E, mode="same")
                                          - np.sqrt(n / 2) * np.convolve(a=state[n - 1, :], v=E, mode="same"))


def density_convolve(state):
    """ state of all coefficients, dimensions = Nv x (2*Nx+1)

    :param state:
    :return:
    """
    Nv, Nx_total = np.shape(state)
    y = np.zeros(Nx_total, dtype="complex128")
    for ii in range(Nv):
        y += np.convolve(a=state[ii, :], v=state[ii, :], mode="same")
    return y


def linear_2(state_e, state_i, alpha_e, alpha_i, q_e=-1, q_i=1):
    """
    :param state_e: a matrix of electron states at time t=t*
    :param state_i: a matrix of ion states at time t=t*
    :param alpha_e: the velocity scaling of electrons
    :param alpha_i: the velocity scaling of ions
    :return: L_{2}(x, t=t*)
    """
    term1 = alpha_e * density_convolve(state=state_e)
    term2 = alpha_i * density_convolve(state=state_i)
    return q_e * term1 + q_i * term2


def linear_2_two_stream(state_e1, state_e2, state_i, alpha_e1, alpha_e2, alpha_i, q_e1=-1, q_e2=-1, q_i=1):
    """
    :param state_e1: a matrix of electron coefficients (species 1) at time t=t*
    :param state_e2: a matrix of electron coefficients (species 2) at time t=t*
    :param state_i: a matrix of ion states at time t=t*
    :param alpha_e1: the velocity scaling of electrons (species 1)
    :param alpha_e2: the velocity scaling of electrons (species 2)
    :param alpha_i: the velocity scaling of ions
    :return: L_{2}(x, t=t*).
    """
    term1 = alpha_e1 * density_convolve(state=state_e1)
    term2 = alpha_e2 * density_convolve(state=state_e2)
    term3 = alpha_i * density_convolve(state=state_i)
    return q_e1 * term1 + q_e2 * term2 + q_i * term3


def RHS(state, n, q_s, m_s, L, u_s, alpha_s, E, Nv, Nx):
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
    term1, term2, term3 = linear(state=state, n=n, u_s=u_s, alpha_s=alpha_s, Nv=Nv)
    J = J_matrix(Nx=Nx, L=L)
    return -J @ (term1 + term2 + term3) - nonlinear(state=state, n=n, E=E, Nv=Nv, q_s=q_s, m_s=m_s, alpha_s=alpha_s)


def solve_poisson_equation_two_stream(state_e1, state_e2, state_i, alpha_e1, alpha_e2, alpha_i, Nx, L, u_e1, u_e2, u_i):
    """
    :param state_e1: a matrix of electron coefficients (species 1) at time t=t*
    :param state_e2: a matrix of electron coefficients (species 2) at time t=t*
    :param state_i: a matrix of ion states at time t=t*
    :param alpha_e1: the velocity scaling of electrons (species 1)
    :param alpha_e2: the velocity scaling of electrons (species 2)
    :param alpha_i: the velocity scaling of ions
    :param Nx: number of spectral terms in space (2*Nx+1) in total
    :return: E(x, t=t*)
    """
    rhs = linear_2_two_stream(state_e1=state_e1, state_e2=state_e2, state_i=state_i, alpha_e1=alpha_e1,
                              alpha_e2=alpha_e2, alpha_i=alpha_i)

    E = np.zeros(len(rhs), dtype="complex128")

    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        if kk != 0:
            if kk > 0:
                E[ii] = np.conjugate(E[Nx - kk])
            else:
                E[ii] = L / (2 * np.pi * kk * 1j) * rhs[ii]

    J_current = ampere_equation_RHS_two_stream(state_e1, state_e2, state_i, alpha_e1, alpha_e2, alpha_i, Nx,
                                               u_e1=u_e1, u_e2=u_e2, u_i=u_i, q_e1=-1, q_e2=-1, q_i=1)
    print("charge neutrality = ", rhs[Nx].real)
    return E


def solve_poisson_equation(state_e, state_i, alpha_e, alpha_i, Nx, L, u_e, u_i):
    """
    :param state_e: a matrix of electron coefficients at time t=t*
    :param state_i: a matrix of ion states at time t=t*
    :param alpha_e: the velocity scaling of electrons
    :param alpha_i: the velocity scaling of ions
    :param Nx: number of spectral terms in space (2*Nx+1) in total
    :return: E(x, t=t*)
    """
    rhs = linear_2(state_e=state_e, state_i=state_i, alpha_e=alpha_e, alpha_i=alpha_i)
    E = np.zeros(len(rhs), dtype="complex128")

    for ii, kk in enumerate(range(-Nx, Nx + 1)):
        if kk != 0:
            if kk > 0:
                E[ii] = np.conjugate(E[Nx - kk])
            else:
                E[ii] = L / (2 * np.pi * kk * 1j) * rhs[ii]

    J_current = ampere_equation_RHS(state_e=state_e, state_i=state_i,
                                    alpha_e=alpha_e, alpha_i=alpha_i,
                                    Nx=Nx, u_e=u_e, u_i=u_i)

    print("charge neutrality = ", rhs[Nx].real)
    return E


def ampere_term(state):
    """
    :param state: coefficient of species s
    :return: quadratic term
    """
    Nv, Nx = np.shape(state)
    res = np.zeros(Nx, dtype="complex128")
    for n in range(1, Nv):
        res += np.sqrt(n) * np.convolve(a=state[n, :], v=state[n - 1, :], mode="same")
    return res


def ampere_equation_RHS(state_e, state_i, alpha_e, alpha_i, u_e, u_i, Nx, q_e=-1, q_i=1):
    """
    :param q_i: ion normalized charge (+1)
    :param q_e: electron normalized charge (-1)
    :param L: length of spatial dimension
    :param u_i: ion velocity shift
    :param u_e: electron velocity shift
    :param state_e: a matrix of electron coefficients at time t=t*
    :param state_i: a matrix of ion coefficients at time t=t*
    :param alpha_e: the velocity scaling of electrons
    :param alpha_i: the velocity scaling of ions
    :param dx: spatial spacing dx = x_{i+1} - x_{i} (we assume uniform spacing)
    :return: d/dt E(x, t=t*)
    """
    # first term
    term1 = alpha_e * u_e * density_convolve(state=state_e)
    term2 = alpha_i * u_i * density_convolve(state=state_i)
    # second term
    term3 = np.sqrt(2) * (alpha_e ** 2) * ampere_term(state=state_e)
    term4 = np.sqrt(2) * (alpha_i ** 2) * ampere_term(state=state_i)
    J_current = q_e * (term1 + term3) + q_i * (term2 + term4)

    print("mean current = ", J_current[Nx].real)
    J_current[Nx] = 0
    return -J_current


def ampere_equation_RHS_two_stream(state_e1, state_e2, state_i, alpha_e1, alpha_e2, alpha_i, Nx,
                                   u_e1, u_e2, u_i, q_e1=-1, q_e2=-1, q_i=1):
    """
    :param Nx: number of spatial terms
    :param q_i: ion normalized charge (+1)
    :param q_e1: electron (species 1) normalized charge (-1)
    :param q_e2: electron (species 2) normalized charge (-1)
    :param u_i: ion velocity shift
    :param u_e1: electron (species 1) velocity shift
    :param u_e2: electron (species 2) velocity shift
    :param state_e2: a matrix of electron (species 2) coefficients at time t=t*
    :param state_e1: a matrix of electron (species 1) coefficients at time t=t*
    :param state_i: a matrix of ion states at time t=t*
    :param alpha_e1: the velocity scaling of electrons (species 1)
    :param alpha_e2: the velocity scaling of electrons (species 2)
    :param alpha_i: the velocity scaling of ions
    :return: d/dt E(x, t=t*)
    """
    # first term
    term1 = alpha_e1 * u_e1 * density_convolve(state=state_e1)
    term2 = alpha_e2 * u_e2 * density_convolve(state=state_e2)
    term3 = alpha_i * u_i * density_convolve(state=state_i)
    # second term
    term4 = np.sqrt(2) * (alpha_e1 ** 2) * ampere_term(state=state_e1)
    term5 = np.sqrt(2) * (alpha_e2 ** 2) * ampere_term(state=state_e2)
    term6 = np.sqrt(2) * (alpha_i ** 2) * ampere_term(state=state_i)

    J = q_e1 * (term1 + term4) + q_e2 * (term2 + term5) + q_i * (term3 + term6)

    print("mean current = ", J[Nx].real)
    J[Nx] = 0
    return -J
