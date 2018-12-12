import numpy
import powerbox
import time
import numexpr

from matplotlib import pyplot
from matplotlib.widgets import Slider
from scipy.constants import c as light_speed

import sys
sys.path.append('../../../redundant_calibration/code/SCAR')
from RadioTelescope import antenna_gain_creator
from RadioTelescope import baseline_converter
from RadioTelescope import xyz_position_creator
from SkyModel import flux_list_to_sky_image
from SkyModel import flux_distribution
from SkyModel import uv_list_to_baseline_measurements
import time
import numexpr
from matplotlib.widgets import Slider


"""
Calculate the model beam, the perturbed beam, and generate visibilities
To do optimise beam calculation for a large number of frequency channels einsum!?!?
Central pixel beam image is zero because phi becomes ill-defined from l,m to theta, phi transform
"""





""""
TODO - curved sky
- multi frequency optimisation

"""


def main():
    nu_low = 150e6
    bandwidth = 40e6  # MHz
<<<<<<< HEAD
    nu_resolution = 1e6 # MHz
=======
    nu_resolution = 200e3 # MHz
>>>>>>> e84f040d61eeadbf8809932b5ad9d04e8d7b9453
    number_channels = bandwidth / nu_resolution
    frequency_range = numpy.linspace(nu_low, nu_low + bandwidth, number_channels)

    # create array
    sky_param = ['point', 200, 0.05, 0.]
    noise_param = [False, 20e3, 40e3, 120]
    beam_param = ['gaussian', 0.25, 0.25]
<<<<<<< HEAD
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
=======
    telescope_param = ["hex", 14., 0, 0]
    telescope_param = ["linear", 14., 5, 0]

    # Create Telescope
    xyz_positions = xyz_position_creator(telescope_param)
    gain_table = antenna_gain_creator(xyz_positions, frequency_range)
    baseline_table = baseline_converter(xyz_positions, gain_table, frequency_range)

    baseline = 0
    ideal_visibility = numerical_visibilities(numpy.array([baseline_table[baseline, :]]), frequency_range, faulty_dipole = None)

    ideal_covariance = numpy.outer(ideal_visibility, ideal_visibility)
    print(ideal_covariance.shape)
    # print(ideal_visibility .shape)
    # print(frequency_range.shape)
    # pyplot.plot(frequency_range, numpy.imag(ideal_visibility [0,:]), "+")
    # pyplot.show()


    visibilities = numpy.zeros((frequency_range.shape[0], 16), dtype=complex)
    for off_dipole in range(16):
        visibilities[:, off_dipole] = numerical_visibilities(numpy.array([baseline_table[baseline, :]]), frequency_range, off_dipole, 0)
        print("Running visibility calculation for", off_dipole)
    pert_covariance = numpy.cov(visibilities)

    fig = pyplot.figure()
    ideal_plot = fig.add_subplot(131)
    pert_plot = fig.add_subplot(132)
    diff_plot = fig.add_subplot(133)

    ideal_plot.pcolor(frequency_range, frequency_range, numpy.abs(ideal_covariance))
    pert_plot.pcolor(frequency_range, frequency_range, numpy.abs(pert_covariance))
    pert_plot.pcolor(frequency_range, frequency_range, numpy.abs(pert_covariance) - numpy.abs(ideal_covariance) )

    pyplot.show()


    return


def numerical_visibilities(baseline_table, frequencies, faulty_dipole, sky_seed = 0):
    numpy.random.seed(sky_seed)
>>>>>>> e84f040d61eeadbf8809932b5ad9d04e8d7b9453


<<<<<<< HEAD
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
    interactive_frequency_plotter(ideal_beam)

    baseline_index = 0
    baseline_selection = numpy.array([baseline_table[baseline_index]])
    visibility_realisations = numpy.zeros((frequency_range.shape[0], 16), dtype=complex)

    print("Iterating of 16 realisations of a perturbed MWA beam\n")
    for faulty_dipole in range(16):
        dipole_weights = numpy.zeros(16) + 1
        dipole_weights[faulty_dipole] = 0
        perturbed_beam = mwa_tile_beam(tt, pp, weights=dipole_weights, frequency=ff)
=======
    # Select the sky model
    # all_flux, all_l, all_m = flux_distribution(['random', sky_seed])
    # point_source_list = numpy.stack((all_flux, all_l, all_m), axis=1)

    point_source_list = numpy.array([[200 , 0.2, 0.0]])


    # Calculate the ideal measured amplitudes for these sources at different
    # frequencies

    t0 = time.time()

    sky_cube, l_coordinates, m_coordinates = flux_list_to_sky_image(point_source_list, baseline_table)

    t1 = time.time()
    time_sky = t1-t0
    # print(f"Generating the Sky Image cubes takes {time_sky}\n")
    # print("The size of the cube is", sky_cube.shape)

    delta_l = numpy.diff(l_coordinates)

    # convert l and m coordinates into theta,phi in which the beam is defined
    # note this doesn't take into account the curvature of the sky
    t0 = time.time()

    ll, mm, ff = numpy.meshgrid(l_coordinates, m_coordinates, frequencies)
    tt, pp, = lm_to_theta_phi(ll, mm)

    t1 = time.time()
    time_grid = t1-t0
    # print("Generating the l, m, frequency grid takes", time_grid)


    t0 = time.time()

    beam1_image = mwa_tile_beam(tt, pp, frequency=ff)



    t1 = time.time()
    time_beam_mwa = f"Generating the beam images takes {t1-t0}s"
    # print(time_beam_mwa)
    t0 = time.time()
    if faulty_dipole is None:
        beam2_image = beam1_image
    else:

        dipole_weights = numpy.zeros(16) + 1
        dipole_weights[faulty_dipole] = 0
        beam2_image = mwa_tile_beam(tt, pp, weights=dipole_weights, frequency=ff)
    t1 = time.time()
    time_beam_perturbed = f"Generating the perturbed beam images takes {t1-t0}\n"


    # interactive_frequency_plotter(numpy.real(beam1_image*numpy.conj(beam2_image)))

    # beam_product = beam1_image*numpy.conj(beam2_image)
    # pyplot.plot(l_coordinates,beam_product[int(len(l_coordinates)/2),:,0])
    # pyplot.show()
>>>>>>> e84f040d61eeadbf8809932b5ad9d04e8d7b9453


        visibility_realisations[:, faulty_dipole] = visibility_extractor(baseline_selection, sky_cube, ideal_beam,
                                                                         perturbed_beam)

<<<<<<< HEAD
    print("Calculating the covariance matrix for a single baseline over the frequency range")
    visibility_covariance = numpy.cov(visibility_realisations)

    return visibility_covariance
=======
    #pyplot.show()

    apparent_sky = sky_cube * beam1_image*numpy.conj(beam2_image)

    ##shift the zero point of the array to [0,0]

    t0 = time.time()
    shifted_image = numpy.fft.ifftshift(sky_cube,  axes=(0, 1))

    visibility_grid, uv_coordinates = powerbox.dft.fft(shifted_image, L=2.,  axes=(0, 1))
    normalised_visibilities = visibility_grid
    # interactive_frequency_plotter(numpy.abs(normalised_visibilities))


    t1 = time.time()
    time_fft = t1-t0
    # print("Doing the FFT takes", time_fft)

>>>>>>> e84f040d61eeadbf8809932b5ad9d04e8d7b9453

def visibility_extractor(baseline_table, sky_cube, antenna1_response, antenna2_response):
    apparent_sky = sky_cube * antenna1_response*numpy.conj(antenna2_response)
    shifted_image = numpy.fft.ifftshift(apparent_sky,  axes=(0, 1))

    visibility_grid, uv_coordinates = powerbox.dft.fft(shifted_image, L=2.,  axes=(0, 1))
    measured_visibilities = uv_list_to_baseline_measurements(baseline_table, visibility_grid, uv_coordinates)

    return measured_visibilities

def mwa_tile_beam(theta, phi, target_theta=0, target_phi=0, frequency=150e6, weights=1, dipole_type='cross',
                  gaussian_width=30 / 180 * numpy.pi):
    dipole_sep = 1.1  # meters
    x_offsets = numpy.array([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5, -1.5,
                             -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5], dtype=numpy.float32) * dipole_sep
    y_offsets = numpy.array([1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5,
                             -0.5, -0.5, -1.5, -1.5, -1.5, -1.5], dtype=numpy.float32) * dipole_sep
    z_offsets = numpy.zeros(x_offsets.shape)

    weights += numpy.zeros(x_offsets.shape)

    t0 = time.time()
    if dipole_type == 'cross':
        dipole_jones_matrix = cross_dipole(theta)
    elif dipole_type == 'gaussian':
        # print(theta_width)
<<<<<<< HEAD
        dipole_jones_matrix = gaussian_response(theta, gaussian_width)
=======
        dipole_jones_matrix = gaussian_response(theta, theta_width)
>>>>>>> e84f040d61eeadbf8809932b5ad9d04e8d7b9453
    else:
        print("Wrong dipole_type: select cross or gaussian")
    t1 = time.time()
    time_dipole = t1-t0
    # print("Doing the dipole beam takes", time_dipole)


    t0 = time.time()
    ground_plane_field = electric_field_ground_plane(theta, frequency)
<<<<<<< HEAD
=======
    t1 = time.time()
    time_ground = t1-t0
    # print("Doing the ground screen beam takes", time_ground)

    t0 = time.time()
>>>>>>> e84f040d61eeadbf8809932b5ad9d04e8d7b9453
    array_factor = get_array_factor(x_offsets, y_offsets, z_offsets, weights, theta, phi, target_theta, target_phi,
                                    frequency)
    t1 = time.time()
    time_AF = t1-t0

    # t0 = time.time()
    # array_factor_ES = get_array_factor_ES(x_offsets, y_offsets, z_offsets, weights, theta, phi, target_theta, target_phi,
    #                                 frequency)
    # t1 = time.time()
    # time_AF_ES = t1-t0

    # # print(numpy.mean(array_factor-array_factor_ES))
    # print("Doing the array factor in ES mode takes", time_AF_ES)
    # print("Doing the array factor takes", time_AF)
    # figure = pyplot.figure()
    # model_beamplot = figure.add_subplot(131)
    # off_beamplot = figure.add_subplot(132)
    # diff_beamplot = figure.add_subplot(133)
    #
    # model_beamplot.imshow(numpy.abs(array_factor[:,:,0]))
    # off_beamplot.imshow(numpy.abs(array_factor_ES[:,:,0]))
    # diff_beamplot.imshow(numpy.abs(array_factor[:,:,0]) - numpy.abs(array_factor_ES[:,:,0]))
    # pyplot.show()

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
<<<<<<< HEAD
        complex_exponent = 1.j * ((k_x - k_x0) * x[i] + (k_y - k_y0) * y[i] + (k_z - k_z0) * z[i])

        # !This step takes a long time, look into optimisation through vectorisation/clever numpy usage
        dipole_factor = weights[i]*numpy.exp(complex_exponent)

        array_factor_map += dipole_factor
=======
        t0 = time.time()

        direction_cosine = 1.j * ((k_x - k_x0) * x[i] + (k_y - k_y0) * y[i] + (k_z - k_z0) * z[i])
        dipole_factor = weights[i]*numexpr.evaluate('exp(direction_cosine)')

        t0 = time.time()

        array_factor_map += dipole_factor
        t1 = time.time()
        time_AF = t1-t0
        #print("Summing the direction cosines  takes", time_AF)

    array_factor_map[numpy.isnan(array_factor_map)] = 0
>>>>>>> e84f040d61eeadbf8809932b5ad9d04e8d7b9453

    #filter all NaN
    array_factor_map[numpy.isnan(array_factor_map)] = 0
    array_factor_map = array_factor_map/numpy.sum(weights)

    return array_factor_map

def get_array_factor_ES(x, y, z, weights, theta, phi, theta_pointing=0, phi_pointing=0, frequency=150e6):
    wavelength = light_speed / frequency
    number_dipoles = len(x)
    k_x = (2. * numpy.pi / wavelength) * numpy.sin(theta) * numpy.sin(phi)
    k_y = (2. * numpy.pi / wavelength) * numpy.sin(theta) * numpy.cos(phi)
    k_z = (2. * numpy.pi / wavelength) * numpy.cos(theta)

    k_x0 = (2. * numpy.pi / wavelength) * numpy.sin(theta_pointing) * numpy.sin(phi_pointing)
    k_y0 = (2. * numpy.pi / wavelength) * numpy.sin(theta_pointing) * numpy.cos(phi_pointing)
    k_z0 = (2. * numpy.pi / wavelength) * numpy.cos(theta_pointing)
    array_factor_map = numpy.zeros(theta.shape, dtype=complex)


    kxx = numpy.einsum("ijk,l -> ijkl", k_x - k_x0, x)
    kyy = numpy.einsum("ijk,l -> ijkl", k_y - k_y0, y)
    kzz = numpy.einsum("ijk,l -> ijkl", k_z - k_z0, z)

    dipole_exponential = numexpr.evaluate('exp(kxx + kyy + kzz)')
    array_factor_MD  = numpy.einsum("ijkl,l -> ijkl",dipole_exponential, weights)
    array_factor_map = numpy.sum(array_factor_MD, axis=-1)

    array_factor_map[numpy.isnan(array_factor_map)] = 0
    return array_factor_map / sum(weights)


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
<<<<<<< HEAD
    index = numpy.where(theta == 0)
    phi[index] = 0
=======
    index = numpy.where(theta[:,:,0] == 0)
    phi[index,:] = 0
>>>>>>> e84f040d61eeadbf8809932b5ad9d04e8d7b9453
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

<<<<<<< HEAD
=======

>>>>>>> e84f040d61eeadbf8809932b5ad9d04e8d7b9453
if __name__ == "__main__":
    main()
