import numpy
import powerbox
from matplotlib import pyplot
from scipy.constants import c as light_speed
import sys
sys.path.append('../../../redundant_calibration/code/SCAR')
from RadioTelescope import antenna_gain_creator
from RadioTelescope import baseline_converter
from RadioTelescope import xyz_position_creator
from SkyModel import flux_list_to_sky_image
from SkyModel import flux_distribution
from SkyModel import uv_list_to_baseline_measurements

def main():
    nu_low = 140e6
    bandwidth = 32e6  # MHz
    nu_resolution = 40e3  # MHz
    number_channels = bandwidth / nu_resolution
    frequency_range = numpy.linspace(nu_low, nu_low + bandwidth, number_channels)

    # create array
    sky_param = ['background', 200, 0.05, 0.]
    noise_param = [False, 20e3, 40e3, 120]
    beam_param = ['gaussian', 0.25, 0.25]
    telescope_param = ["hex", 14., 0, 0]

    # Create Telescope
    xyz_positions = xyz_position_creator(telescope_param)
    gain_table = antenna_gain_creator(xyz_positions, frequency_range)
    baseline_table = baseline_converter(xyz_positions, gain_table, frequency_range)

    visibilities = numpy.zeros((frequency_range.shape[0], 16), dtype=complex)
    for off_dipole in range(16):
        visibilities[:, off_dipole] = numerical_visibilities(numpy.array([baseline_table[0, :]]), frequency_range, off_dipole, 0)
        print("Running visibility calculation for", off_dipole)
    covariance = numpy.cov(visibilities)


    pyplot.pcolor(frequency_range, frequency_range, numpy.real(covariance))
    pyplot.show()
    return


def numerical_visibilities(baseline_table, frequencies, faulty_dipole, seed):
    numpy.random.seed(seed)

    n_measurements = baseline_table.shape[0]
    n_frequencies = frequencies.shape[0]

    # Select the sky model
    all_flux, all_l, all_m = flux_distribution(['random', seed])
    point_source_list = numpy.stack((all_flux, all_l, all_m), axis=1)
    point_source_list = numpy.array([[200,0.1,0.1]])
    # Calculate the ideal measured amplitudes for these sources at different
    # frequencies
    sky_cube, l_coordinates, m_coordinates = flux_list_to_sky_image(point_source_list, baseline_table)
    delta_l = numpy.diff(l_coordinates)

    # convert l and m coordinates into theta,phi in which the beam is defined
    # note this doesn't take into account the curvature of the sky
    ll, mm, ff = numpy.meshgrid(l_coordinates, m_coordinates, frequencies)
    tt, pp, = lm_to_theta_phi(ll, mm)

    beam1_image = mwa_tile_beam(tt, pp, frequency=ff)

    dipole_weights = numpy.zeros(16) + 1
    dipole_weights[faulty_dipole] = 0
    beam2_image = mwa_tile_beam(tt, pp, weights=dipole_weights, frequency=ff)

    beam1_normed = beam1_image / numpy.max(numpy.abs(beam1_image))
    beam2_normed = beam2_image / numpy.max(numpy.abs(beam2_image))

    #figure = pyplot.figure()
    #model_beamplot = figure.add_subplot(131)
    #off_beamplot = figure.add_subplot(132)
    #diff_beamplot = figure.add_subplot(133)

    #model_beamplot.imshow(numpy.abs(beam1_normed[:,:,0]))
    #off_beamplot.imshow(numpy.abs(beam2_normed[:,:,0]))
    #diff_beamplot.imshow(numpy.abs(beam1_normed[:,:,0]) - numpy.abs(beam2_normed[:,:,0]))

    #pyplot.show()
    apparent_sky = sky_cube * beam1_normed * beam2_normed

    ##shift the zero point of the array to [0,0]
    shifted_image = numpy.fft.ifftshift(apparent_sky, axes=(0, 1))
    visibility_grid, uv_coordinates = powerbox.dft.fft(shifted_image, L=2., axes=(0, 1))
    normalised_visibilities = visibility_grid/delta_l[0]**2.

    model_visibilities = uv_list_to_baseline_measurements(baseline_table, normalised_visibilities, uv_coordinates)

    return model_visibilities


def mwa_tile_beam(theta, phi, target_theta=0, target_phi=0, frequency=150e6, weights=1, dipole_type='cross',
                  theta_width=30 / 180 * numpy.pi):
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
        print(theta_width)
        dipole_jones_matrix = gaussian_response(theta, theta_width)
    else:
        print("Wrong dipole_type: select cross or gaussian")

    ground_plane_field = electric_field_ground_plane(theta, frequency)

    array_factor = get_array_factor(x_offsets, y_offsets, z_offsets, weights, theta, phi, target_theta, target_phi,
                                    frequency)

    tile_response = numpy.zeros(dipole_jones_matrix.shape, dtype=complex)
    tile_response = array_factor * ground_plane_field * dipole_jones_matrix

    tile_response[numpy.isnan(tile_response)] = 0
    return tile_response


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
        array_factor_map += weights[i] * numpy.exp(
            1.j * ((k_x - k_x0) * x[i] + (k_y - k_y0) * y[i] + (k_z - k_z0) * z[i]))

    return array_factor_map / sum(weights)


def cross_dipole(theta):
    response = numpy.cos(theta)
    return response

def electric_field_ground_plane(theta, frequency=150e6 , height= 0.3):
    wavelength = light_speed/frequency
    ground_plane_electric_field = numpy.sin(2.*numpy.pi*height/wavelength*numpy.cos(theta))
    return ground_plane_electric_field

def lm_to_theta_phi(ll, mm):
    theta = numpy.arcsin(numpy.sqrt(ll ** 2. + mm ** 2.))
    phi = numpy.arctan(mm / ll)
    return theta, phi


if __name__ == "__main__":
    main()
