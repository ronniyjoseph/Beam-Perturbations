import numpy
import powerbox
import sys
import time

import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors as colors

from scipy.constants import c
from scipy import interpolate

from functools import partial
import multiprocessing

from generaltools import colorbar
from generaltools import symlog_bounds

from skymodel import SkyRealisation
from radiotelescope import RadioTelescope
from radiotelescope import ideal_gaussian_beam
from radiotelescope import broken_gaussian_beam

from powerspectrum import get_power_spectrum


def main(verbose=True):
    path = "./HexCoords_Luke.txt"
    frequency_range = numpy.linspace(135, 165, 2) * 1e6
    faulty_dipole = 6
    faulty_tile = 81
    sky_param = "random"
    mode = "serial"
    processes = 2
    calibrate = True
    beam_type = "gaussian"
    plot_file_name = "Compare_new_code_Long_Gain_Corrected_6_Test.pdf"

    telescope = RadioTelescope(load=True, path=path, verbose=verbose)
    source_population = SkyRealisation(sky_type=sky_param, verbose=verbose)



    ####################################################################################################################
    if verbose:
        print("Generating visibility measurements for each frequency")
    ideal_measured_visibilities, broken_measured_visibilities = get_observations(source_population, telescope,
                                                                                 faulty_dipole, faulty_tile,
                                                                                 frequency_range, beam_type, calibrate,
                                                                                 compute_mode= mode,
                                                                                 processes=processes, verbose=verbose)

    ####################################################################################################################
    get_power_spectrum(frequency_range, telescope, ideal_measured_visibilities, broken_measured_visibilities,
                       faulty_tile, plot_file_name, verbose)
    return


def get_observations(source_population, radio_telescope, faulty_dipole, faulty_tile, frequency_range, beam_type,
                     calibrate, compute_mode=False, processes=4, verbose=False):
    if compute_mode == "parallel":
        baseline_table = radio_telescope.baseline_table

        # Determine maximum resolution
        max_frequency = numpy.max(frequency_range)
        max_u = numpy.max(numpy.abs(baseline_table.u(max_frequency)))
        max_v = numpy.max(numpy.abs(baseline_table.v(max_frequency)))
        max_b = max(max_u, max_v)
        # sky_resolutions
        min_l = 1. / (2 * max_b)

        pool = multiprocessing.Pool(processes=processes)
        iterator = partial(get_observation_single_channel, source_population, baseline_table, min_l, faulty_dipole,
                           faulty_tile, beam_type, frequency_range, calibrate)
        ideal_observations_list, broken_observations_list = zip(*pool.map(iterator, range(len(frequency_range))))

        ideal_observations = numpy.moveaxis(numpy.array(ideal_observations_list), 0, -1)
        broken_observations = numpy.moveaxis(numpy.array(broken_observations_list), 0, -1)
    elif compute_mode == "serial":
        get_observations_all_channels_serial(source_population, radio_telescope, faulty_dipole, faulty_tile, frequency_range,
                                      beam_type, calibrate, verbose)

    return ideal_observations, broken_observations


def get_observations_all_channels_serial(source_population, radio_telescope, faulty_dipole, faulty_tile, frequency_range,
                                  beam_type,
                                  calibrate, verbose):
    # Create Sky
    ######################################################################
    sky_cube, l_coordinates = source_population.create_sky_image(frequency_channels=frequency_range,
                                                                 radiotelescope = radio_telescope)
    ll, mm, ff = numpy.meshgrid(l_coordinates, l_coordinates, frequency_range)
    ############################################################################
    print(f"The size of the sky is {sky_cube.shape}")
    baseline_table = radio_telescope.baseline_table
    if verbose:
        print("Calculating beam patterns")
    # Create Beam
    #############################################################################
    if beam_type == "MWA":
        tt, pp, = lm_to_theta_phi(ll, mm)
        ideal_beam = ideal_mwa_beam_loader(tt, pp, ff, load)
        broken_beam = broken_mwa_beam_loader(tt, pp, ff, faulty_dipole, load)

    elif beam_type == "gaussian":
        ideal_beam = ideal_gaussian_beam(ll, mm, ff)
        broken_beam = broken_gaussian_beam(faulty_dipole, ideal_beam, ll, mm, ff)
    else:
        raise ValueError("The only valid option for the beam are 'MWA' or 'gaussian'")
    ##################################################################

    if verbose:
        print("Generating observed visibilities")
    # Create visibilities
    ##############################################################################
    ideal_measured_visibilities = numpy.zeros((baseline_table.number_of_baselines, len(frequency_range)), dtype=complex)
    broken_measured_visibilities = ideal_measured_visibilities.copy()

    ##### Select perfect baselines #####
    perfect_baseline_indices = numpy.where((baseline_table.antenna_id1 != faulty_tile) &
                                           (baseline_table.antenna_id2 != faulty_tile))[0]
    broken_baseline_indices = numpy.where((baseline_table.antenna_id1 == faulty_tile) |
                                          (baseline_table.antenna_id2 == faulty_tile))[0]

    print(f"there are broken {len(broken_baseline_indices)} baselines")

    if verbose:
        print(" Generating Visbilities for each frequencies")
    for frequency_index in range(len(frequency_range)):
        # Sample all baselines, and get the relevant uv_grid coordinates
        ideal_measured_visibilities[..., frequency_index] = visibility_extractor(
            baseline_table, sky_cube[..., frequency_index],frequency_range[frequency_index], ideal_beam[..., frequency_index],
            ideal_beam[..., frequency_index])

        if calibrate:
            correction = 16 / 15
        else:
            correction = 1

        # Copy good baselines to broken table
        broken_measured_visibilities[perfect_baseline_indices, frequency_index] = ideal_measured_visibilities[
            perfect_baseline_indices, frequency_index]

        broken_measured_visibilities[broken_baseline_indices, frequency_index] = visibility_extractor(
            baseline_table[broken_baseline_indices, :, frequency_index], sky_cube[..., frequency_index],
            ideal_beam[..., frequency_index], broken_beam[..., frequency_index]) * correction

    return ideal_measured_visibilities, broken_measured_visibilities


def get_observation_single_channel(source_population, baseline_table, min_l, faulty_dipole, faulty_tile, beam_type,
                                   frequency_range, calibrate, frequency_index, ):
    sky_image, l_coordinates = source_population.create_sky_image(
        frequency_channels=frequency_range[frequency_index], resolution=min_l, oversampling=1)
    ll, mm = numpy.meshgrid(l_coordinates, l_coordinates)

    # Create Beam
    #############################################################################
    if beam_type == "MWA":
        tt, pp, = lm_to_theta_phi(ll, mm)
        ideal_beam = ideal_mwa_beam_loader(tt, pp, frequency_range[frequency_index])
        broken_beam = broken_mwa_beam_loader(tt, pp, frequency_range[frequency_index], faulty_dipole)

    elif beam_type == "gaussian":
        ideal_beam = ideal_gaussian_beam(ll, mm, frequency_range[frequency_index])
        broken_beam = broken_gaussian_beam(faulty_dipole, ideal_beam, ll, mm, frequency_range[frequency_index])
    else:
        raise ValueError("The only valid option for the beam are 'MWA' or 'gaussian'")

    if calibrate:
        correction = 16 / 15
    else:
        correction = 1

    ##Determine the indices of the broken baselines and calulcate the visibility measurements
    ##################################################################

    broken_baseline_indices = numpy.where((baseline_table.antenna_id1 == faulty_tile) |
                                          (baseline_table.antenna_id2 == faulty_tile))[0]

    ideal_measured_visibilities = visibility_extractor(baseline_table, sky_image, frequency_range[frequency_index],
                                                       ideal_beam, ideal_beam)

    broken_measured_visibilities = ideal_measured_visibilities.copy()
    broken_measured_visibilities[broken_baseline_indices] = visibility_extractor(
        baseline_table.sub_table(broken_baseline_indices), sky_image, frequency_range[frequency_index],
        ideal_beam, broken_beam) * correction

    return ideal_measured_visibilities, broken_measured_visibilities


def visibility_extractor(baseline_table_object, sky_image, frequency, antenna1_response,
                         antenna2_response, padding_factor=3):
    apparent_sky = sky_image * antenna1_response * numpy.conj(antenna2_response)

    padded_sky = numpy.pad(apparent_sky, padding_factor * apparent_sky.shape[0], mode="constant")
    shifted_image = numpy.fft.ifftshift(padded_sky, axes=(0, 1))
    visibility_grid, uv_coordinates = powerbox.dft.fft(shifted_image, L=2 * (2 * padding_factor + 1), axes=(0, 1))

    measured_visibilities = uv_list_to_baseline_measurements(baseline_table_object, frequency, visibility_grid,
                                                             uv_coordinates)

    return measured_visibilities


def uv_list_to_baseline_measurements(baseline_table_object, frequency, visibility_grid, uv_grid):
    u_bin_centers = uv_grid[0]
    v_bin_centers = uv_grid[1]

    print(numpy.min(u_bin_centers), numpy.max(u_bin_centers), numpy.min(v_bin_centers), numpy.max(v_bin_centers))

    baseline_coordinates = numpy.array([baseline_table_object.u(frequency), baseline_table_object.v(frequency)])

    print(numpy.min(baseline_table_object.u(frequency)), numpy.max(baseline_table_object.u(frequency)),
          numpy.min(baseline_table_object.v(frequency)), numpy.max(baseline_table_object.v(frequency)))

    # now we have the bin edges we can start binning our baseline table
    # Create an empty array to store our baseline measurements in
    visibility_data = visibility_grid

    real_component = interpolate.RegularGridInterpolator([u_bin_centers, v_bin_centers], numpy.real(visibility_data))
    imag_component = interpolate.RegularGridInterpolator([u_bin_centers, v_bin_centers], numpy.imag(visibility_data))

    visibilities = real_component(baseline_coordinates.T) + 1j * imag_component(baseline_coordinates.T)

    print(frequency.shape)
    return visibilities


def regrid_visibilities(measured_visibilities, baseline_u, baseline_v, u_grid):
    u_shifts = numpy.diff(u_grid) / 2.

    u_bin_edges = numpy.concatenate((numpy.array([u_grid[0] - u_shifts[0]]), u_grid[1:] - u_shifts,
                                     numpy.array([u_grid[-1] + u_shifts[-1]])))
    weights_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u, baseline_v, bins=(u_bin_edges, u_bin_edges))

    real_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u, baseline_v, bins=(u_bin_edges, u_bin_edges),
                                                     weights=numpy.real(measured_visibilities))

    imag_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u, baseline_v, bins=(u_bin_edges, u_bin_edges),
                                                     weights=numpy.imag(measured_visibilities))

    regridded_visibilities = real_regrid + 1j * imag_regrid
    normed_regridded_visibilities = numpy.nan_to_num(regridded_visibilities / weights_regrid)

    return normed_regridded_visibilities, weights_regrid


def regrid_visibilities_gaussian(measured_visibilities, baseline_table, u_grid, frequency):
    u_shifts = numpy.diff(u_grid) / 2.

    u_bin_edges = numpy.concatenate((numpy.array([u_grid[0] - u_shifts[0]]), u_grid[1:] - u_shifts,
                                     numpy.array([u_grid[-1] + u_shifts[-1]])))

    regridded_visibilities = numpy.zeros((len(u_grid), len(u_grid)), dtype=complex)
    regridded_weights = numpy.zeros((len(u_grid), len(u_grid)))

    grid_indices_u = numpy.digitize(baseline_table.u(frequency), bins=u_bin_edges)
    grid_indices_v = numpy.digitize(baseline_table.v(frequency), bins=u_bin_edges)

    kernel = gaussian_gridding_kernel(u_grid, u_grid, frequency)
    kernel_threshold = 1e-3
    kernel_indices = numpy.where(kernel < kernel_threshold)

    for baseline_index in range(baseline_table.number_of_baselines):
        # Find the index of this baseline in the new grid
        index_u = grid_indices_u[baseline_index]
        index_v = grid_indices_v[baseline_index]

        # check whether the b

    return


def gaussian_gridding_kernel(u_coordinates, v_coordinates, nu, diameter=4, epsilon=1):
    sigma = beam_width(nu, diameter, epsilon)

    beam_attenuation = numpy.sqrt(2 * numpy.pi * sigma ** 2) * \
                       numpy.exp(-(u_coordinates ** 2. + v_coordinates ** 2.)(2 * numpy.pi ** 2 * sigma ** 2))

    return beam_attenuation


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print("Total time is", end - start)
