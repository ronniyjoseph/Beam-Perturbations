from scipy.constants import c
from scipy import signal
from generaltools import symlog_bounds
from radiotelescope import beam_width
from matplotlib import pyplot

from generaltools import colorbar
from generaltools import from_eta_to_k_par
from generaltools import from_u_to_k_perp
from generaltools import from_jansky_to_milikelvin
import numpy
import powerbox

import matplotlib.colors as colors


def sky_covariance(u, v, nu):
    gamma = 0.0
    nn1, nn2 = numpy.meshgrid(nu, nu)

    width_1_tile = beam_width(nn1)
    width_2_tile = beam_width(nn2)

    Sigma = width_1_tile**2*width_2_tile**2/(width_1_tile**2 + width_2_tile**2)

    mu_2_r = moment_returner(2, S_high = 1)

    sky_covariance = (nn1*nn2/numpy.min(nu)**2)**-gamma * mu_2_r *Sigma**2 *numpy.exp(-2*numpy.pi**2*(u**2 + v**2)*(nn1 - nn2)**2/numpy.min(nu)**2*Sigma)

    return sky_covariance




def beam_covariance(u, v, nu, dx =1):
    x_offsets = numpy.array([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5, -1.5,
                         -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5], dtype=numpy.float32) * dx

    y_offsets = numpy.array([1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5,
                         -0.5, -0.5, -1.5, -1.5, -1.5, -1.5], dtype=numpy.float32) * dx

    #x_offsets = x_offsets[:2]
    #y_offsets = y_offsets[:2]

    nn1, nn2, xx = numpy.meshgrid(nu, nu, x_offsets)
    nn1, nn2, yy = numpy.meshgrid(nu, nu, y_offsets)

    mu_1_r = moment_returner(1, S_high = 1)
    mu_2_r = moment_returner(2, S_high = 1)

    mu_1_m = moment_returner(1, S_low = 1)
    mu_2_m = moment_returner(2, S_low = 1)


    width_1_tile = beam_width(nn1)
    width_2_tile = beam_width(nn2)
    width_1_dipole = beam_width(nn1, diameter=1)
    width_2_dipole = beam_width(nn2, diameter=1)

    sigma_A = (width_1_tile * width_2_tile * width_1_dipole * width_2_dipole) ** 2 / (
                width_2_tile ** 2 * width_1_dipole ** 2 * width_2_dipole ** 2 +
                width_1_tile ** 2 * width_1_dipole ** 2 * width_2_dipole ** 2 +
                width_1_tile ** 2 * width_2_tile ** 2 * width_1_dipole ** 2 +
                width_1_tile ** 2 * width_2_tile ** 2 * width_2_dipole ** 2)

    sigma_B = (width_1_tile*width_2_tile*width_2_dipole)**2/(2*width_2_tile**2*width_2_dipole**2 + width_1_tile**2*width_2_dipole**2 +
                                                        width_1_tile**2*width_2_tile**2)

    sigma_C = (width_1_tile*width_2_tile*width_1_dipole)**2/(width_2_tile**2*width_1_dipole**2 + 2*width_2_tile**2*width_1_dipole**2 +
                                                        width_1_tile**2*width_2_tile**2)


    sigma_D1 = width_1_tile**2*width_1_dipole**2/(width_1_tile**2 + width_1_dipole**2)
    sigma_D2 = width_2_tile**2*width_2_dipole**2/(width_2_tile**2 + width_2_dipole**2)


    A = (mu_2_m + mu_2_r)*numpy.sum(2 * numpy.pi * sigma_A / len(y_offsets) ** 3 * numpy.exp(-2 * numpy.pi ** 2 * sigma_A * (
        (u / nu[0] + xx /c) ** 2 + (v / nu[0] + yy /c)**2)*(nn1 - nn2)**2.), axis=-1)

    B = -2*numpy.pi*mu_2_r/len(y_offsets)**2*numpy.sum(sigma_B*numpy.exp(-2 * numpy.pi ** 2 * sigma_B * (
            (u*(nn1 - nn2) / nu[0] + xx/c*nn2)**2 + (v*(nn1 - nn2)**2 / nu[0] + yy/c*nn2)**2)), axis = -1)

    print(sigma_A.shape)
    C = -2*numpy.pi*mu_2_r/len(y_offsets)**2*numpy.sum(sigma_C*numpy.exp(-2 * numpy.pi ** 2 * sigma_C * (
            (u*(nn1 - nn2) / nu[0] + xx/c*nn2)**2 + (v*(nn1 - nn2)**2 / nu[0] + yy/c*nn1)**2)), axis = -1)

    D = (mu_1_m**2 + 2*mu_1_m*mu_1_r + mu_1_r**2) *2*numpy.pi* numpy.sum(sigma_D1*sigma_D2/len(x_offsets)**3*\
        numpy.exp(-2*numpy.pi**2*sigma_D1*((u*nn1/nu[0] - xx/c*nn1)**2 + (v*nn1/nu[0] - yy/c*nn1)**2)) *\
        numpy.exp(-2*numpy.pi**2*sigma_D2*((u*nn2/nu[0] - xx/c*nn2)**2 + (v*nn2/nu[0] - yy/c*nn2)**2)), axis = -1)

    E = (mu_1_m**2 + 2*mu_1_m*mu_1_r + mu_1_r**2) *2*numpy.pi* numpy.sum(sigma_D1*sigma_D2/len(x_offsets)**4*\
        numpy.exp(-2*numpy.pi**2*sigma_D1*((u*nn1/nu[0] - xx/c*nn1)**2 +(v*nn1/nu[0] - yy/c*nn1)**2)), axis = -1)*\
        numpy.sum(numpy.exp(-2*numpy.pi**2*sigma_D2*((u * nn2 / nu[0] - xx / c * nn2) ** 2 +
                                                                  (v * nn2 / nu[0] - yy / c * nn2) ** 2)), axis=-1)

    return A #(A + B + C + D + E)


def moment_returner(n_order, k1=4100, gamma1=1.59, k2=4100, gamma2=2.5, S_low=400e-3, S_mid=1, S_high=5.):
    moment = k1/(n_order + 1 - gamma1)*(S_mid**(n_order + 1 - gamma1)) - S_low**(n_order + 1 - gamma1) + \
    k2 / (n_order + 1 - gamma2) * (S_high ** (n_order + 1 - gamma2)) - S_mid ** (n_order + 1 - gamma2)

    return moment


def dft_matrix(nu):
    dft = numpy.exp(-2 * numpy.pi * 1j / len(nu)) ** numpy.arange(0, len(nu), 1)
    dftmatrix = numpy.vander(dft, increasing=True)/numpy.sqrt(len(nu))

    eta = numpy.arange(0, len(nu), 1)/(nu.max() - nu.min())

    return dftmatrix, eta


def blackman_harris_taper(frequency_range):
    window = signal.blackmanharris(len(frequency_range))

    return window


def calculate_beam_PS():
    u = numpy.linspace(200, 200, 20)
    nu = numpy.linspace(135, 165, 50)*1e6

    uu, vv = numpy.meshgrid(u, u)
    variance_cube = numpy.zeros((len(u), len(u), len(nu)))

    window_function = blackman_harris_taper(nu)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)

    dftmatrix, eta = dft_matrix(nu)

    print("calculating all variances for all uv-cells")
    for i in range(len(u)):
        nu_cov = beam_covariance(uu[i, j], vv[i, j], nu)
        tapered_cov = nu_cov*taper1*taper2

        eta_cov = numpy.dot(numpy.dot(dftmatrix.conj().T, tapered_cov), dftmatrix)

        variance_cube[i, j, :] = numpy.diag(numpy.real(eta_cov))

    print("Taking circular average")
    # Take circular average
    #print(variance_cube)
    #PS, u_bins = powerbox.tools.angular_average_nd(variance_cube, coords=[u, u, eta], bins=len(u) / 2, n=2)

    pyplot.imshow(variance_cube[:, 10, :].T)
    pyplot.show()

    return


def calculate_beam_2DPS():
    u = numpy.logspace(0, 2.1, 100)
    nu = numpy.linspace(135, 165, 100) * 1e6

    window_function = blackman_harris_taper(nu)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)

    dftmatrix, eta = dft_matrix(nu)

    variance = numpy.zeros((len(u), len(nu)))

    #figure = pyplot.figure(figsize=(23,4))


    for i in range(len(u)):
        nu_cov = beam_covariance(u[i], 0, nu)
        tapered_cov = nu_cov * taper1 * taper2
        eta_cov = numpy.dot(numpy.dot(dftmatrix.conj().T, tapered_cov), dftmatrix)
        variance[i, :] = numpy.diag(numpy.real(eta_cov))


        #axes_label = r"$\nu$ [MHz]"
        #axes = figure.add_subplot(1, 4, i+1)
        #plot = axes.pcolor(nu/1e6, nu/1e6, numpy.real(nu_cov))
        #if i == 0:
        #    axes.set_ylabel((axes_label))
        #cax = colorbar(plot)
        #axes.set_xlabel(axes_label)
    plot_PS(u, eta[:int(len(eta) / 2)], variance[:, :int(len(eta) / 2)])
    #pyplot.show()

    return


def calculate_sky_PS():
    u = numpy.logspace(-1, 2.5, 200)
    nu = numpy.linspace(135, 165, 200)*1e6

    window_function = blackman_harris_taper(nu)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)

    dftmatrix, eta = dft_matrix(nu)

    variance = numpy.zeros((len(u), len(nu)))
    #figure = pyplot.figure(figsize=(23,4))
    #axes = figure.add_subplot((111))
    for i in range(len(u)):
        nu_cov = sky_covariance(u[i], 0, nu)
        tapered_cov = nu_cov * taper1 * taper2
        eta_cov = numpy.dot(numpy.dot(dftmatrix.conj().T, tapered_cov), dftmatrix)
        variance[i, :] = numpy.diag(numpy.real(eta_cov))

        #axes_label = r"$\nu$ [MHz]"
        #axes = figure.add_subplot(1, 4, i + 1)
        #plot = axes.pcolor(nu/1e6, nu/1e6,  numpy.real(nu_cov))
        #if i == 0:
        #    axes.set_ylabel((axes_label))
        #cax = colorbar(plot)
        #axes.set_xlabel(axes_label)


    plot_PS(u, eta[:int(len(eta)/2)], nu, variance[:, :int(len(eta)/2)], cosmological=True)

    return



def plot_PS(u_bins, eta_bins, nu, PS, cosmological= False):
    figure = pyplot.figure()
    axes = figure.add_subplot(111)

    if cosmological:
        central_frequency = nu[int(len(nu)/2)]
        x_values = from_u_to_k_perp(u_bins, central_frequency)
        y_values = from_eta_to_k_par(eta_bins, central_frequency)
        z_values = from_jansky_to_milikelvin(PS, nu)

        x_label = r"$k_{\perp}$ [Mpc$^{-1}$]"
        y_label = r"$k_{\parallel}$ [Mpc$^{-1}$]"
        z_label = r"Variance [mK$^2$ Mpc$^3$ ]"


        axes.set_xlim(5e-3, 2e-1)
        axes.set_ylim(9e-3, 1)
    else:
        x_values = u_bins
        y_values = eta_bins
        z_values = PS

        x_label = r"|u|"
        y_label = r"$\eta$ [MHz$^{-1}$]"
        z_label = r"Variance [Jy$^2$ Hz$^2$]"

        axes.set_xlim(xmin = 1)
        axes.set_ylim(eta_bins[1], eta_bins.max())
    if PS.min() < 0:
        symlog_min, symlog_max, symlog_threshold, symlog_scale = symlog_bounds(numpy.real(z_values))
        norm = colors.SymLogNorm(linthresh=10**-5, linscale=symlog_scale, vmin=symlog_min, vmax=symlog_max)
    else:
        symlog_min, symlog_max, symlog_threshold, symlog_scale = symlog_bounds(numpy.real(z_values))
        norm = colors.LogNorm(vmin=symlog_min, vmax=symlog_max)


    psplot = axes.pcolor(x_values, y_values, z_values.T, norm=norm)

    cax = colorbar(psplot)

    axes.set_xscale('log')
    axes.set_yscale('log')

    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    cax.set_label(z_label)

    #pyplot.show()

    return

def test_dft_on_signal():
    #make a sinusoidal signal
    time = numpy.linspace(0, 100, 101)

    f1 = 1/5
    f2 = 1/4
    sample_rate = 1/(time.max() - time.min())

    signal = numpy.sin(2*numpy.pi*f1*time) + numpy.sin(2*numpy.pi*f2*time)
    taper = blackman_harris_taper(time)
    dftmatrix = dft_matrix(time)

    frequencies = numpy.arange(0, len(time), 1)*sample_rate
    ft_signal = numpy.dot(dftmatrix, taper*signal)

    inverse_ft_signal = numpy.dot(dftmatrix.conj().T, ft_signal)

    fig = pyplot.figure()
    axes1 = fig.add_subplot(131)
    axes2 = fig.add_subplot(132)
    axes3 = fig.add_subplot(133)

    axes1.plot(time, taper*signal)
    axes2.plot(frequencies[:int(len(frequencies)/2)], numpy.abs(ft_signal[:int(len(frequencies)/2)]))
    axes3.plot(inverse_ft_signal)

    pyplot.show()

    return





if __name__ == "__main__":
    #calculate_sky_PS()
    calculate_beam_2DPS()
    #nu = numpy.linspace(135, 165, 200)*1e6
    #print(from_jansky_to_milikelvin(1, nu))
    pyplot.show()
