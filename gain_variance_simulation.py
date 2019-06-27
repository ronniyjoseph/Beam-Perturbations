import sys
import os
import numpy
import powerbox
from matplotlib import pyplot

from generaltools import visibility_extractor

from radiotelescope import RadioTelescope
from skymodel import SkyRealisation
from radiotelescope import ideal_gaussian_beam

from Single_Dipole_PS_Impact_OO import uv_list_to_baseline_measurements

from analytic_covariance import sky_covariance
from analytic_covariance import beam_covariance
import multiprocessing
from functools import partial
from numba import prange, njit, float32, complex64, void

import time


def simulate_gain_variances(create_signal=True, compute_FIM=True, plot_variance=True):
    frequency_range = numpy.linspace(135, 165, 50) * 1e6
    path = "Data/MWA_All_Coordinates_Cath.txt"
    output_path = "/data/rjoseph/Hybrid_Calibration/Tile_Pertubation/Simulation_Output/"
    project_path = "Numba_Gain_Variance_Simulation"

    sky_param = "random"
    n_realisations = 100

    tile_id1 = 36
    tile_id2 = 1036
    tile_id3 = 81

    tile_name1 = "Core"
    tile_name2 = "Hex"
    tile_name3 = "Outer"

    telescope = RadioTelescope(load=True, path=path)
    baseline_table = telescope.baseline_table

    if not os.path.exists(output_path + project_path + "/"):
        print
        ""
        print
        "!!!Warning: Creating output folder at output destination!"
        os.makedirs(output_path + project_path + "/" + "Simulated_Visibilities")
        os.makedirs(output_path + project_path + "/" + "FIM_realisations")

    if create_signal:
        print("Creating Signal Realisations")
        for i in range(n_realisations):
            print(f"Realisation {i}")
            signal_time0 = time.perf_counter()
            source_population = SkyRealisation(sky_type=sky_param, seed=i)
            signal = get_observations(source_population, baseline_table, frequency_range, interpolation='numba')
            numpy.save(output_path + project_path + "/" + "Simulated_Visibilities/" + f"visibility_realisation_{i}",
                       signal)
            signal_time1 = time.perf_counter()
            print(f"Realisation {i} Time = {signal_time1 - signal_time0} \n")
    if compute_FIM:
        covariance_matrices = compute_frequency_covariance(baseline_table, frequency_range)

        for i in range(n_realisations):
            print(f"Realisation {i}")

            signal = numpy.load(
                output_path + project_path + "/" + "Simulated_Visibilities/" + f"visibility_realisation_{i}.npy")

            FIM = get_FIM(signal, telescope, baseline_table, frequency_range, covariance_matrices)

            numpy.save(output_path + project_path + "/" + "FIM_realisations/" + f"fim_realisation_{i}", FIM)

    if plot_variance:
        print("Plotting variances")

        antenna_id = telescope.antenna_positions.antenna_ids
        n_antennas = len(antenna_id)

        tile1_index = numpy.where(antenna_id == tile_id1)[0]
        tile2_index = numpy.where(antenna_id == tile_id2)[0]
        tile3_index = numpy.where(antenna_id == tile_id3)[0]

        figure = pyplot.figure(figsize=(18, 5))
        tile1_plot = figure.add_subplot(131)
        tile2_plot = figure.add_subplot(132)
        tile3_plot = figure.add_subplot(133)


        for i in range(100):
            FIM = numpy.load(output_path + project_path + "/" + "FIM_realisations/" + f"fim_realisation_{i}.npy")

            covariance = numpy.zeros((n_antennas, n_antennas, len(frequency_range)))
            for i in range(len(frequency_range)):
                covariance[..., i] = numpy.linalg.pinv(FIM[..., i])




            tile1_plot.plot(frequency_range / 1e6, covariance[tile1_index, tile1_index, :].flatten(), 'k', alpha=0.1)
            tile2_plot.plot(frequency_range / 1e6, covariance[tile2_index, tile2_index, :].flatten(), 'k', alpha=0.1)
            tile3_plot.plot(frequency_range / 1e6, covariance[tile3_index, tile3_index, :].flatten(), 'k', alpha=0.1)

        tile1_plot.set_title(tile_name1)
        tile2_plot.set_title(tile_name2)
        tile3_plot.set_title(tile_name3)

        pyplot.show()









        indices = numpy.array([tile1_index, tile2_index, tile3_index])
        tile_names = [tile_name1, tile_name2, tile_name3]

        #        figure = pyplot.figure(figsize=(18, 5))
        #        tile1_plot = figure.add_subplot(131)
        #        tile2_plot = figure.add_subplot(132)
        #        tile3_plot = figure.add_subplot(133)

        figure, axes = pyplot.subplots(3, 3, figsize=(18, 5))

        for i in range(33):
            FIM = numpy.load(output_path + project_path + "/" + "FIM_realisations/" + f"fim_realisation_{i}.npy")

            covariance = numpy.zeros((n_antennas, n_antennas, len(frequency_range)))

            for j in range(len(frequency_range)):
                covariance[..., i] = numpy.linalg.pinv(FIM[..., i])

            for k in range(3):
                for l in range(3):
                    print(covariance[indices[k], indices[l], :].flatten())
                    axes[k, l].plot(frequency_range / 1e6, covariance[indices[k], indices[l], :].flatten(), 'k',
                                    alpha=0.1)
                    axes[k, l].set_title(tile_names[k] + ", " + tile_names[l])







    print("Finished")
    return


def compute_frequency_covariance(baseline_table, frequency_range):
    covariance_matrices = numpy.zeros((len(frequency_range), len(frequency_range), baseline_table.number_of_baselines))
    print("Pre-Computing all covariance matrices")

    # Compute the covariances for each baseline
    for l in range(baseline_table.number_of_baselines):
        u = baseline_table.u(frequency=frequency_range[0])[l]
        v = baseline_table.v(frequency=frequency_range[0])[l]

        covariance_matrices[..., l] = sky_covariance(u, v, frequency_range) + beam_covariance(u, v, frequency_range)

    return covariance_matrices


def get_FIM(signal, telescope, baseline_table, frequency_range, covariance_matrices):
    antenna_id = telescope.antenna_positions.antenna_ids
    n_antennas = len(antenna_id)

    FIM = numpy.zeros((n_antennas, n_antennas, len(frequency_range)))

    for j in range(n_antennas):
        for k in range(n_antennas):
            # find all baselines in which these antennas participate
            baseline_indices = numpy.where(((baseline_table.antenna_id1 == antenna_id[j]) |
                                            (baseline_table.antenna_id2 == antenna_id[j])) &
                                           ((baseline_table.antenna_id1 == antenna_id[k]) |
                                            (baseline_table.antenna_id2 == antenna_id[k])))[0]

            for l in range(len(baseline_indices)):
                FIM[j, k, :] += 2 * numpy.real(
                    numpy.abs(signal[baseline_indices[l], :]) ** 2 / numpy.diag(
                        covariance_matrices[..., baseline_indices[l]]))

    return FIM


def get_observations(source_population=None, baseline_table=None, frequency_range=None,
                     beam_type='gaussian', interpolation='spline', processes=4):
    if interpolation == 'spline':
        oversampling = 2
    elif interpolation == 'linear':
        oversampling = 2

    if interpolation == 'spline':
        sky_cube, l_coordinates = source_population.create_sky_image(baseline_table=baseline_table,
                                                                     frequency_channels=frequency_range,
                                                                     oversampling=oversampling)
        pool = multiprocessing.Pool(processes=processes)
        iterator = partial(get_observations_single, sky_cube, l_coordinates, baseline_table, frequency_range)
        observations = numpy.array(pool.map(iterator, range(len(frequency_range)))).T
    elif interpolation == 'linear':
        sky_cube, l_coordinates = source_population.create_sky_image(baseline_table=baseline_table,
                                                                     frequency_channels=frequency_range,
                                                                     oversampling=oversampling)
        observations = get_observations_serial(sky_cube, l_coordinates, baseline_table, frequency_range)
    elif interpolation == 'analytic':
        observations = get_observations_analytic(source_population, baseline_table, frequency_range)
    elif interpolation == 'numba':
        observations = get_observations_numba(source_population, baseline_table, frequency_range)

    return observations


def get_observations_serial(sky_cube, l_coordinates, baseline_table, frequency_range):
    observations = numpy.zeros((baseline_table.number_of_baselines, len(frequency_range)), dtype=complex)

    for frequency_index in range(len(frequency_range)):
        ll, mm = numpy.meshgrid(l_coordinates, l_coordinates)

        # Create Beam
        #############################################################################
        antenna_response1 = ideal_gaussian_beam(ll, mm, frequency_range[frequency_index])
        antenna_response2 = antenna_response1.copy()

        observations[..., frequency_index] = visibility_extractor(baseline_table,
                                                                  sky_cube[..., frequency_index],
                                                                  frequency_range[frequency_index],
                                                                  antenna_response1,
                                                                  antenna_response2, interpolation='linear')
    return observations


def get_observations_single(sky_cube, l_coordinates, baseline_table, frequency_range, frequency_index):
    observations = numpy.zeros((baseline_table.number_of_baselines), dtype=complex)
    ll, mm = numpy.meshgrid(l_coordinates, l_coordinates)

    # Create Beam
    #############################################################################
    antenna_response1 = ideal_gaussian_beam(ll, mm, frequency_range[frequency_index])
    antenna_response2 = antenna_response1.copy()

    observations = visibility_extractor(baseline_table,
                                                              sky_cube[..., frequency_index],
                                                              frequency_range[frequency_index],
                                                              antenna_response1,
                                                              antenna_response2, interpolation='spline')
    return observations


def get_observations_analytic(source_population, baseline_table, frequency_range):
    observations = numpy.zeros((baseline_table.number_of_baselines, len(frequency_range)), dtype=complex)
    ff = numpy.tile(frequency_range, (baseline_table.number_of_baselines, 1))

    for source_index in range(len(source_population.fluxes)):
        source_flux = source_population.fluxes[source_index]
        source_l = source_population.l_coordinates[source_index]
        source_m = source_population.m_coordinates[source_index]

        antenna_response = ideal_gaussian_beam(source_l, source_m, ff)

        observations += source_flux*antenna_response*numpy.conj(antenna_response)*numpy.exp(
            -2*numpy.pi*1j*(source_l*baseline_table.u(frequency_range) + source_m*baseline_table.v(frequency_range)))

    return observations


def get_observations_numba(source_population, baseline_table, frequency_range):
    observations = numpy.zeros((baseline_table.number_of_baselines, len(frequency_range)), dtype=complex)

    #pre-compute all apparent fluxes at all frequencies
    apparent_flux = apparent_fluxes_numba(source_population, frequency_range)
    numbafied_for_loop(observations, apparent_flux, source_population.l_coordinates,
                                      source_population.m_coordinates, baseline_table.u(frequency_range),
                                      baseline_table.v(frequency_range))

    return observations

@njit(parallel = True)
def numbafied_for_loop(observations, fluxes, l_source, m_source, u_baselines, v_baselines ):
    for source_index in prange(len(fluxes)):
        for baseline_index in range(u_baselines.shape[0]):
            for frequency_index in range(u_baselines.shape[1]):
                kernel = numpy.exp(-2j*numpy.pi*(u_baselines[baseline_index, frequency_index]*l_source[source_index] +
                                                 v_baselines[baseline_index, frequency_index]*m_source[source_index]))
                observations[baseline_index, frequency_index] += fluxes[source_index, frequency_index]*kernel



def apparent_fluxes_numba(source_population, frequency_range):
    ff = numpy.tile(frequency_range, (len(source_population.fluxes), 1))
    ss = numpy.tile(source_population.fluxes, (len(frequency_range), 1 ))
    ll = numpy.tile(source_population.l_coordinates, (len(frequency_range), 1 ))
    mm = numpy.tile(source_population.m_coordinates, (len(frequency_range), 1 ))

    antenna_response = ideal_gaussian_beam(ll.T, mm.T, ff)

    apparent_fluxes = antenna_response*numpy.conj(antenna_response)*ss.T
    return apparent_fluxes.astype(complex)




def get_observations_dumb(source_population, baseline_table, frequency_range):
    observations = numpy.zeros((baseline_table.number_of_baselines, len(frequency_range)), dtype=complex)
    for source_index in range(len(source_population.fluxes)):

        source_flux = source_population.fluxes[source_index]
        source_l = source_population.l_coordinates[source_index]
        source_m = source_population.m_coordinates[source_index]

        for frequency_index in range(len(frequency_range)):
            u = baseline_table.u(frequency_range[frequency_index])
            v = baseline_table.v(frequency_range[frequency_index])

            for baseline_index in range(baseline_table.number_of_baselines):

                beam_1 = ideal_gaussian_beam(source_l, source_m, frequency_range[frequency_index])
                beam_2 = numpy.conj(beam_1)
                kernel = numpy.exp(-2*numpy.pi*1j*(source_l*u[baseline_index] + source_m*v[baseline_index]))
                observations[baseline_index, frequency_index] += source_flux*beam_1*beam_2*kernel

    return observations




if __name__ == "__main__":
    simulate_gain_variances(create_signal=False, compute_FIM=False, plot_variance=True)
