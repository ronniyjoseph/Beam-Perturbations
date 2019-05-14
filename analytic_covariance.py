from scipy.constants import c
from radiotelescope import beam_width
from matplotlib import pyplot
from generaltools import colorbar
import numpy
import powerbox


def sky_covariance(u, v, nu):
    gamma = 0.8
    nn1, nn2 = numpy.meshgrid(nu, nu)

    width_1_tile = beam_width(nn1)
    width_2_tile = beam_width(nn2)

    Sigma = width_1_tile**2*width_2_tile**2/(width_1_tile**2 + width_2_tile**2)

    mu_2_r = moment_returner(2, S_high = 1)

    sky_covariance = (nn1*nn2)**-gamma * mu_2_r *Sigma**2 *numpy.exp(-2*numpy.pi**2*(u**2 + v**2)*(nn1 - nn2)**2*Sigma)

    return sky_covariance





def beam_covariance(u, v, nu, dx =1):
    x_offsets = numpy.array([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5, -1.5,
                         -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5], dtype=numpy.float32) * dx

    y_offsets = numpy.array([1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5,
                         -0.5, -0.5, -1.5, -1.5, -1.5, -1.5], dtype=numpy.float32) * dx

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

    sigma_null = width_1_tile**2*width_2_tile**2/(width_1_tile**2 + width_2_tile**2)

    sigma_A = (width_1_tile * width_2_tile * width_1_dipole * width_2_dipole) ** 2 / (
                width_2_tile ** 2 * width_1_dipole ** 2 * width_2_dipole ** 2 +
                width_1_tile ** 2 * width_1_dipole ** 2 * width_2_dipole ** 2 +
                width_1_tile ** 2 * width_2_tile ** 2 * width_1_dipole ** 2 +
                width_1_tile ** 2 * width_2_tile ** 2 * width_2_dipole ** 2)

    sigma_B = (width_1_tile*width_2_tile*width_2_dipole)**2/(2*width_2_tile**2*width_2_dipole**2 + width_1_tile**2*width_2_dipole**2 +
                                                        width_1_tile**2*width_2_tile**2)

    sigma_C = (width_1_tile*width_2_tile*width_1_dipole)**2/(2*width_1_tile**2*width_1_dipole**2 + width_2_tile**2*width_1_dipole**2 +
                                                        width_1_tile**2*width_2_tile**2)


    sigma_D = width_1_tile**2*width_1_dipole**2/(width_1_tile**2 + width_1_dipole**2)
    sigma_D_prime = width_2_tile**2*width_2_dipole**2/(width_2_tile**2 + width_2_dipole**2)

    #null = 2*numpy.pi*mu_2*sigma_null*numpy.exp(-2*numpy.pi**2*(u**2+v**2)*((nn1-nn2)/nu[0])**2*sigma_null)

    A = (mu_2_m + mu_2_r)*numpy.sum(2 * numpy.pi * sigma_A / len(y_offsets) ** 2 * numpy.exp(-2 * numpy.pi ** 2 * sigma_A * (
                (u * (nn1 - nn2) / nu[0] + xx /c* (nn1 - nn2)) ** 2 + (v * (nn1 - nn2) / nu[0] + yy /c*(nn1 - nn2))**2))
                                                                         , axis=-1)

    B = mu_2_r*numpy.sum(2*numpy.pi*sigma_B/len(y_offsets)**2*numpy.exp(-2 * numpy.pi ** 2 * sigma_B * (
            (u * (nn1 - nn2) / nu[0] + xx/c*nn2)**2 + (v * (nn1 - nn2) / nu[0] + yy/c*nn2)**2)), axis = -1)

    C = mu_2_r*numpy.sum(-2*numpy.pi*sigma_B/len(y_offsets)**2*numpy.exp(-2 * numpy.pi ** 2 * sigma_B * (
            (u * (nn1 - nn2) / nu[0] + xx/c*nn2)**2 + (v * (nn1 - nn2) / nu[0] + yy/c*nn1)**2)), axis = -1)


    D = (mu_1_m**2 + mu_1_m*mu_1_r + mu_1_r**2) * numpy.sum(2*numpy.pi*sigma_D*sigma_D_prime/len(x_offsets)**3*\
        numpy.exp(-2*numpy.pi**2*sigma_D*((u*nn1/nu[0] - xx/c*nn1)**2 + (v*nn1/nu[0] - yy/c*nn1)**2)) *\
        numpy.exp(-2*numpy.pi**2*sigma_D_prime*((u*nn2/nu[0] - xx/c*nn2)**2 + (v*nn2/nu[0] - yy/c*nn2)**2)), axis = -1)

    E = (mu_1_m**2 + mu_1_m*mu_1_r + mu_1_r**2) * numpy.sum(2*numpy.pi*sigma_D*sigma_D_prime/len(x_offsets)**3*\
        numpy.exp(-2*numpy.pi**2*sigma_D*((u*nn1/nu[0] - xx/c*nn1)**2 +
                                                    (v*nn1/nu[0] - yy/c*nn1)**2)), axis = -1)*\
        numpy.sum(numpy.exp(-2*numpy.pi**2*sigma_D_prime*((u * nn2 / nu[0] - xx / c * nn2) ** 2 +
                                                                  (v * nn2 / nu[0] - yy / c * nn2) ** 2)), axis=-1)

    return (A + B + C + D + E)


def moment_returner(n_order, k1=4100, gamma1=1.59, k2=4100, gamma2=2.5, S_low=400e-3, S_mid=1, S_high=5.):
    moment = k1/(n_order + 1 - gamma1)*(S_mid**(n_order + 1 - gamma1)) - S_low**(n_order + 1 - gamma1) + \
    k2 / (n_order + 1 - gamma2) * (S_high ** (n_order + 1 - gamma2)) - S_mid ** (n_order + 1 - gamma2)

    return moment


def dft_matrix(nu):
    dft = numpy.exp(-2 * numpy.pi * 1j / len(nu)) ** numpy.arange(0, len(nu), 1)
    dftmatrix = numpy.vander(dft, increasing=True)/numpy.sqrt(len(nu))

    return dftmatrix


def calculate_PS():

    nu = numpy.linspace(135, 165, 5)*1e6

    u = numpy.linspace(-200, 200, 100)
    uu, vv = numpy.meshgrid(u, u)

    variance_cube = numpy.zeros((len(u), len(u), len(nu)), dtype=complex)
    dftmatrix = dft_matrix(nu)

    print("calculating all variances for all uv-cells")
    for i in range(len(u)):
        for j in range(len(u)):
            nu_cov = sky_covariance(uu[i, j], vv[i, j], nu)
            eta_cov = numpy.dot(numpy.dot(dftmatrix.T.conj(), nu_cov), dftmatrix)

            #etanu, eta = powerbox.dft.fft(matrix, L=numpy.max(nu) - numpy.min(nu), axes=(0,))
            #etaeta, etaprime = powerbox.dft.fft(matrix, L=numpy.max(nu) - numpy.min(nu), axes=(1,))

            variance_cube[i, j, :] = numpy.diag(eta_cov)

    print("Taking circular average")
    # Take circular average
    PS, bins = powerbox.tools.angular_average_nd(numpy.real(variance_cube), coords=[u, u, nu], bins=len(u) / 2, n=2)

    figure = pyplot.figure()
    axes = figure.add_subplot(111)
    plot = axes.pcolor(bins, nu/1e6, PS.T)
    colorbar(plot)
    pyplot.show()

    return


def calculate_sky_PS():

    nu = numpy.linspace(135, 165, 3)*1e6

    u = numpy.linspace(0, 200, 10)

    variance_cube = numpy.zeros((len(u), len(nu)), dtype=complex)
    dftmatrix = dft_matrix(nu)


    nu_cov = sky_covariance(u[2], 0, nu)

    #what does the covariance look like
    #print(numpy.diag(nu_cov))
    #pyplot.plot(nu, numpy.diag(nu_cov))
    #pyplot.show()

    #pyplot.pcolor(nu, nu, nu_cov)
    #pyplot.show()

    print(dftmatrix*numpy.sqrt(len(nu)))
    pyplot.imshow(numpy.imag(numpy.dot(dftmatrix.conj().T, dftmatrix)))
    pyplot.show()

    eta_cov = numpy.dot(numpy.dot(dftmatrix.conj().T, nu_cov), dftmatrix)

            #etanu, eta = powerbox.dft.fft(matrix, L=numpy.max(nu) - numpy.min(nu), axes=(0,))
            #etaeta, etaprime = powerbox.dft.fft(matrix, L=numpy.max(nu) - numpy.min(nu), axes=(1,))

    #print(numpy.diag(numpy.real(eta_cov)))
    #pyplot.loglog(nu, numpy.diag(numpy.abs(eta_cov)))
    #pyplot.show()

    #pyplot.imshow(numpy.log10(numpy.real(eta_cov)))
    #pyplot.show()


    return



if __name__ == "__main__":
    calculate_sky_PS()

