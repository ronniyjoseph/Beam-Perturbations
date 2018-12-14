import numpy
import powerbox
import time
import numexpr

from matplotlib import pyplot
from matplotlib.widgets import Slider
from scipy.constants import c as light_speed
from scipy import interpolate

import sys
sys.path.append('../../../redundant_calibration/code/SCAR')
from RadioTelescope import antenna_gain_creator
from RadioTelescope import baseline_converter
from RadioTelescope import xyz_position_creator
from SkyModel import flux_list_to_sky_image
from SkyModel import flux_distribution
from SkyModel import uv_list_to_baseline_measurements


""""
TODO - curved sky
- multi frequency optimisation (What's the best way to optimise this, i should definitely do the calculation per 
frequency)
- optimise calculating the beam. (there are three unique ways into destroy the beam, the other 16 dipoles are rotations

"""


def main():
    nu_low = 150e6
    bandwidth = 40e6  # MHz
    nu_resolution = 1e6 # MHz
    number_channels = bandwidth / nu_resolution
    frequency_range = numpy.linspace(nu_low, nu_low + bandwidth, number_channels)

    # create array
    sky_param = ['point', 200, 0.05, 0.]
    noise_param = [False, 20e3, 40e3, 120]
    beam_param = ['gaussian', 0.25, 0.25]
    # telescope_param = ["hex", 14., 0, 0]
    telescope_param = ["linear", 1e2, 5, 0]

    covariance = visibility_beam_covariance(telescope_param, frequency_range, sky_param)
    pyplot.imshow(covariance)
    pyplot.show()

    return


def visibility_beam_covariance(telescope_param, frequency_range, sky_param, sky_seed = 0):
    xyz_positions = xyz_position_creator(telescope_param)
    gain_table = antenna_gain_creator(xyz_positions, frequency_range)
    baseline_table = baseline_converter(xyz_positions, gain_table, frequency_range)


    if sky_param[0] == 'random':
        all_flux, all_l, all_m = flux_distribution(['random', sky_seed])
    elif sky_param[0] == 'point':
        all_flux, all_l, all_m = flux_distribution(['single', sky_param[1],
                                                    sky_param[2], sky_param[3]])
    point_source_list = numpy.stack((all_flux, all_l, all_m), axis=1)

    print("Creating the sky\n")
    sky_cube, l_coordinates, m_coordinates = flux_list_to_sky_image(point_source_list, baseline_table)
    ll, mm, ff = numpy.meshgrid(l_coordinates, m_coordinates, frequency_range)
    tt, pp, = lm_to_theta_phi(ll, mm)

    print("Creating the idealised MWA beam\n")
    ideal_beam = mwa_tile_beam(tt, pp, frequency=ff)

    baseline_index = 0
    baseline_selection = numpy.array([baseline_table[baseline_index]])
    visibility_realisations = numpy.zeros((frequency_range.shape[0], 16), dtype=complex)

    print("Iterating of 16 realisations of a perturbed MWA beam")
    for faulty_dipole in range(16):
        dipole_weights = numpy.zeros(16) + 1
        dipole_weights[faulty_dipole] = 0
        perturbed_beam = mwa_tile_beam(tt, pp, weights=dipole_weights, frequency=ff)


        print("Extracting visibilities per frequency channel\n")
        for frequency_index in range(len(frequency_range)):
            visibility_realisations[frequency_index, faulty_dipole] = visibility_extractor(
                baseline_selection[:, :, frequency_index], sky_cube[:, :, frequency_index],
                ideal_beam[:, :, frequency_index], perturbed_beam[:, :, frequency_index])


    print("Calculating the covariance matrix for a single baseline over the frequency range")

    visibility_covariance = numpy.cov(visibility_realisations)

    return visibility_realisations


def visibility_extractor(baseline_table, sky_cube, antenna1_response, antenna2_response):
    apparent_sky = sky_cube * antenna1_response * numpy.conj(antenna2_response)
    padding_factor = 3

    padded_sky = numpy.pad(apparent_sky, padding_factor * len(apparent_sky), mode="constant")
    shifted_image = numpy.fft.ifftshift(padded_sky, axes=(0, 1))
    visibility_grid, uv_coordinates = powerbox.dft.fft(shifted_image, L=2 * (2 * padding_factor + 1), axes=(0, 1))

    measured_visibilities = uv_list_to_baseline_measurements(baseline_table, visibility_grid, uv_coordinates)

    return measured_visibilities


def uv_list_to_baseline_measurements(baseline_table, visibility_grid, uv_grid):

    u_bin_centers = uv_grid[0]
    v_bin_centers = uv_grid[1]

    # now we have the bin edges we can start binning our baseline table
    # Create an empty array to store our baseline measurements in
    visibility_data = visibility_grid

    real_component = interpolate.RegularGridInterpolator((u_bin_centers, v_bin_centers), numpy.real(visibility_data))
    imag_component = interpolate.RegularGridInterpolator((u_bin_centers, v_bin_centers), numpy.imag(visibility_data))
    visibilities = real_component(baseline_table[:, 2:4]) + \
                   1j * imag_component(baseline_table[:, 2:4])

    return visibilities

def mwa_tile_beam(theta, phi, target_theta=0, target_phi=0, frequency=150e6, weights=1, dipole_type='cross',
                  gaussian_width=30 / 180 * numpy.pi):


    dipole_sep = 1.1  # meters
    x_offsets = numpy.array([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5, -1.5,
                             -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5], dtype=numpy.float32) * dipole_sep

    y_offsets = numpy.array([1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5,
                             -0.5, -0.5, -1.5, -1.5, -1.5, -1.5], dtype=numpy.float32) * dipole_sep
    z_offsets = numpy.zeros(x_offsets.shape)

    weights += numpy.zeros(x_offsets.shape)

    if dipole_type == 'cross':
        dipole_jones_matrix = cross_dipole(theta)
    elif dipole_type == 'gaussian':
        # print(theta_width)
        dipole_jones_matrix = gaussian_response(theta, gaussian_width)
    else:
        print("Wrong dipole_type: select cross or gaussian")

    ground_plane_field = electric_field_ground_plane(theta, frequency)
    array_factor = get_array_factor(x_offsets, y_offsets, z_offsets, weights, theta, phi, target_theta, target_phi,
                                    frequency)

    tile_response = array_factor * ground_plane_field * dipole_jones_matrix
    tile_response[numpy.isnan(tile_response)] = 0

    beam_normalisation = numpy.add(numpy.zeros(tile_response.shape), numpy.amax(tile_response, axis=(0, 1)))
    normalised_response = tile_response / beam_normalisation

    return normalised_response


def get_array_factor(x, y, z, weights, theta, phi, theta_pointing=0, phi_pointing=0, frequency=150e6):

    wavelength = light_speed / frequency
    number_dipoles = len(x)

    k_x = (2. * numpy.pi / wavelength) * numpy.sin(theta) * numpy.sin(phi)
    k_y = (2. * numpy.pi / wavelength) * numpy.sin(theta) * numpy.cos(phi)
    k_z = (2. * numpy.pi / wavelength) * numpy.cos(theta)

    k_x0 = (2. * numpy.pi / wavelength) * numpy.sin(theta_pointing) * numpy.sin(phi_pointing)
    k_y0 = (2. * numpy.pi / wavelength) * numpy.sin(theta_pointing) * numpy.cos(phi_pointing)
    k_z0 = (2. * numpy.pi / wavelength) * numpy.cos(theta_pointing)
    array_factor_map = numpy.zeros(theta.shape, dtype=complex)

    for i in range(number_dipoles):
        complex_exponent = 1.j * ((k_x - k_x0) * x[i] + (k_y - k_y0) * y[i] + (k_z - k_z0) * z[i])

        # !This step takes a long time, look into optimisation through vectorisation/clever numpy usage
        dipole_factor = weights[i]*numpy.exp(complex_exponent)

        array_factor_map += dipole_factor

    #filter all NaN
    array_factor_map[numpy.isnan(array_factor_map)] = 0
    array_factor_map = array_factor_map/numpy.sum(weights)

    return array_factor_map


def cross_dipole(theta):
    response = numpy.cos(theta)
    return response


def gaussian_response(theta,target_theta = 0, theta_width = 30./180*numpy.pi):
    response = 1./numpy.sqrt(2.*numpy.pi*theta_width**2.)*numpy.exp(-0.5*(theta-target_theta)**2./theta_width**2.)
    return response


def electric_field_ground_plane(theta, frequency=150e6 , height= 0.3):
    wavelength = light_speed/frequency
    ground_plane_electric_field = numpy.sin(2.*numpy.pi*height/wavelength*numpy.cos(theta))
    return ground_plane_electric_field


def lm_to_theta_phi(ll, mm):
    theta = numpy.arcsin(numpy.sqrt(ll ** 2. + mm ** 2.))
    phi = numpy.arctan(mm / ll)

    #phi is undefined for theta = 0, correct
    index = numpy.where(theta == 0)
    phi[index] = 0
    return theta, phi


def interactive_frequency_plotter(data):
    idx = 0
    # figure axis setup
    fig, ax = pyplot.subplots()
    fig.subplots_adjust(bottom=0.15)

    # display initial image
    im_h = ax.imshow(data[:, :, idx], cmap='cubehelix', interpolation='nearest')
    fig.colorbar(im_h)

    # setup a slider axis and the Slider
    ax_depth = pyplot.axes([0.23, 0.02, 0.56, 0.04])
    slider_depth = Slider(ax_depth, 'depth', 0, data.shape[2] - 1, valinit=idx)



    # update the figure with a change on the slider
    def update_depth(val):
        idx = int(round(slider_depth.val))
        im_h.set_data(data[:, :, idx])

    slider_depth.on_changed(update_depth)
    pyplot.show()

if __name__ == "__main__":
    main()
