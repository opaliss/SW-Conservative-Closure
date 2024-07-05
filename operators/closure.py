"""module for closure to improve conservation properties

"""
import numpy as np
import scipy


def closure_momentum(state, Nv, u_s, alpha_s):
    """

    :param state: state of the species s
    :param Nv: number of Hermite modes
    :param u_s: velocity shifting parameter of species s
    :param alpha_s: velocity scaling parameter of species s
    :return: C_{Nv}^{s} = F(C_{Nv-1}^{s})
    """
    # even Hermite modes
    if Nv % 2 == 0:
        return -u_s * state[-1, :] / (alpha_s * np.sqrt(2 * Nv))

    # odd Hermite modes
    elif Nv % 2 == 1:
        if u_s != 0:
            return - alpha_s * Nv * state[-1, :] / (u_s * np.sqrt(Nv / 2))
        else:
            return 0 * state[-1, :]


def closure_energy(state, Nv, u_s, alpha_s, J_inv, E, q_s, m_s, Nx_total, Nx):
    """

    :param state: state of species s
    :param Nv: number of Hermite modes
    :param u_s: velocity shifting parameter of species s
    :param alpha_s: velocity scaling parameter of species s
    :param J_inv: derivative inverse matrix
    :param E: electric field vector
    :param Nx_total: total number of Fourier modes
    :param q_s: charge of species s
    :param m_s: mass of species s
    :return: C_{Nv}^{s} = F(C_{Nv-1}^{s})
    """
    # construct the Toeplitz matrix representing one-dimensional convolution
    E_conv = scipy.linalg.convolution_matrix(a=E, n=Nx_total, mode='same')

    # even Hermite modes
    if Nv % 2 == 0:
        if u_s != 0:
            gamma = 0.5 * ((2 * Nv - 1) * (alpha_s ** 2) + u_s ** 2)
            matrix = gamma * np.identity(Nx_total) + q_s / m_s * J_inv @ E_conv
            closure = - matrix / (np.sqrt(2 * Nv) * u_s * alpha_s) @ state[-1, :]
            # enforce real closures
            closure[:Nx] = np.flip(np.conjugate(closure[Nx + 1:]))
            closure[Nx] = closure[Nx].real
            return closure
        else:
            return 0 * state[-1, :]

    # odd Hermite modes
    elif Nv % 2 == 1:
        eta = 0.5 * ((2 * Nv + 1) * (alpha_s ** 2) + u_s ** 2)
        matrix_inv = np.linalg.inv(eta * np.identity(Nx_total) + q_s / m_s * J_inv @ E_conv)
        closure = - u_s * alpha_s * np.sqrt(2 * Nv) * matrix_inv @ state[-1, :]
        # enforce real closures
        closure[Nx] = closure[Nx].real
        closure[:Nx] = np.flip(np.conjugate(closure[Nx + 1:]))
        return closure



def closure_energy_local(state, Nv, u_s, alpha_s, Nx):
    """

    :param state: state of species s
    :param Nv: number of Hermite modes
    :param u_s: velocity shifting parameter of species s
    :param alpha_s: velocity scaling parameter of species s
    :param J_inv: derivative inverse matrix
    :param E: electric field vector
    :param Nx_total: total number of Fourier modes
    :param q_s: charge of species s
    :param m_s: mass of species s
    :return: C_{Nv}^{s} = F(C_{Nv-1}^{s})
    """

    # even Hermite modes
    if Nv % 2 == 0:
        if u_s != 0:
            gamma = 0.5 * ((2 * Nv - 1) * (alpha_s ** 2) + u_s ** 2)
            closure = - gamma / (np.sqrt(2 * Nv) * u_s * alpha_s) * state[-1, :]
            # enforce real closures
            closure[:Nx] = np.flip(np.conjugate(closure[Nx + 1:]))
            closure[Nx] = closure[Nx].real
            return closure
        else:
            return 0 * state[-1, :]

    # odd Hermite modes
    elif Nv % 2 == 1:
        eta = 0.5 * ((2 * Nv + 1) * (alpha_s ** 2) + u_s ** 2)
        closure = - u_s * alpha_s * np.sqrt(2 * Nv) / eta * state[-1, :]
        # enforce real closures
        closure[Nx] = closure[Nx].real
        closure[:Nx] = np.flip(np.conjugate(closure[Nx + 1:]))
        return closure
