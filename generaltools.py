import numpy
from scipy.constants import c
from scipy.constants import parsec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    return fig.colorbar(mappable, cax=cax)

def symlog_bounds(data):
    data_min = numpy.nanmin(data)
    data_max = numpy.nanmax(data)

    if data_min == 0:
        indices = numpy.where(data > 0)[0]
        if len(indices) == 0:
            lower_bound = -0.1
        else:
            lower_bound = numpy.nanmin(data[indices])
    else:
        lower_bound = numpy.abs(data_min)/data_min*numpy.abs(data_min)

    if data_max == 0:
        indices = numpy.where(data < 0)[0]
        if len(indices) == 0:
            upper_bound = 0.1
        else:
            upper_bound = numpy.nanmax(data[indices])
    else:
        upper_bound = numpy.abs(data_max)/data_max*numpy.abs(data_max)

    ### Figure out what the lintresh is (has to be linear)
    threshold = 1e-5# 1e-4*min(numpy.abs(lower_bound), numpy.abs(upper_bound))
    #### Figure out the linscale parameter (has to be in log)
    scale = numpy.log10(upper_bound - lower_bound)/6

    return lower_bound, upper_bound, threshold, scale


def from_lm_to_theta_phi(ll, mm):
    theta = numpy.arcsin(numpy.sqrt(ll ** 2. + mm ** 2.))
    phi = numpy.arctan(mm / ll)

    #phi is undefined for theta = 0, correct
    index = numpy.where(theta == 0)
    phi[index] = 0
    return theta, phi


def from_eta_to_k_par(eta, nu_observed, H0 = 70.4, nu_emission = 1.42e9):
    # following Morales 2004

    z = redshift(nu_observed, nu_emission)
    hubble_distance = c/H0 *1e-3 #[Mpc]

    E = E_function(z)
    k_par = eta*2*numpy.pi*nu_emission*E/(hubble_distance*(1+z)**2)

    return k_par

def from_u_to_k_perp(u, frequency):
    #following Morales 2004
    distance = comoving_distance(frequency)
    print(distance)
    k_perp = 2*numpy.pi*u/distance

    return k_perp


def comoving_distance(nu_observed, H0 = 70.4):

    hubble_distance = c/H0  *1e-3
    z = redshift(nu_observed)
    z_integration = numpy.linspace(0,z,100)
    E = E_function(z_integration)

    d = hubble_distance*numpy.trapz(1/E, z_integration)

    return d


def E_function(z, Omega_M = 0.27, Omega_k = 0, Omega_Lambda = 0.73 ):
    E = numpy.sqrt(Omega_M*(1+z)**3 + Omega_k*(1+z)**2 + Omega_Lambda)

    return E

def redshift(nu_observed, nu_emission = 1.42e9):

    z = (nu_emission - nu_observed)/nu_observed

    return z

def visibility_to_temperature(measurements_jansky, nu_emission = 1.4, H0 = 70.4):
    #following morales & wyithe 2010
    G = H0
    return
