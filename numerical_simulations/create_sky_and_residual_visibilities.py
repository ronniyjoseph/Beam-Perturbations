import os
import numpy
from matplotlib import pyplot


from radiotelescope import RadioTelescope
from skymodel import SkyRealisation
from gain_variance_simulation import get_observations

from analytic_covariance import dft_matrix
from analytic_covariance import blackman_harris_taper
from analytic_covariance import sky_covariance
from analytic_covariance import beam_covariance
from analytic_covariance import plot_PS

from generaltools import from_eta_to_k_par




import time

def main():
    output_path = "/data/rjoseph/Hybrid_Calibration/Tile_Pertubation/Simulation_Output/"
    project_path = "sky_and_residual_simulation"
    n_realisations = 10000
    frequency_range= numpy.linspace(135,165, 100)*1e6
    shape = ['linear', 400, 20, 'log']
    load = False
    create_signal = False
    compute_ratio = False
    compute_covariance = False
    serial = True
    plot_covariance = True
    plot_model_signal = False
    telescope = RadioTelescope(load=load, shape =shape)
    baseline_table = telescope.baseline_table


    if not os.path.exists(output_path + project_path + "/"):
        print("Creating Project folder at output destination!")
        os.makedirs(output_path + project_path)

    if create_signal:
        create_model_and_residuals(baseline_table, frequency_range, n_realisations, output_path + project_path)

    if plot_model_signal:
        pass



    if compute_ratio:

        ratio_full = numpy.load(output_path + project_path + "/" + "Simulated_Visibilities/" + f"residual_model_ratios_full.npy")

        maximum_baseline = numpy.min(baseline_table.u_coordinates)
        max_index = numpy.where(numpy.abs(baseline_table.u_coordinates - maximum_baseline) ==
                                numpy.min(numpy.abs(baseline_table.u_coordinates - maximum_baseline)))[0][0]
        half_index = numpy.where(numpy.abs(baseline_table.u_coordinates - maximum_baseline/2) ==
                                numpy.min(numpy.abs(baseline_table.u_coordinates - maximum_baseline/2)))[0][0]
        min_index = numpy.where(numpy.abs(baseline_table.u_coordinates + 7) ==
                                numpy.min(numpy.abs(baseline_table.u_coordinates + 7)))[0][0]

        indices = [min_index, half_index, max_index]



        ######### Compute variances as approximation ##########
        model_variance = sky_covariance(0,0, frequency_range, S_low = 1, S_high = 10)
        residual_variance = sky_covariance(0,0, frequency_range, S_high = 1)

        figure, axes  = pyplot.subplots(2, 3,  figsize =(15,5), subplot_kw=dict(xlabel = r"$\nu$ [MHz]"))

        for i in range(3):
            axes[0, i].plot(frequency_range / 1e6, numpy.abs(1 - ratio_full[indices[i], :, ::1]), color='k', alpha=0.01)
            axes[0, i].plot(frequency_range / 1e6, numpy.diag(residual_variance)/numpy.diag(model_variance), color='C0')

            axes[1, i].pcolormesh(frequency_range/1e6, frequency_range/1e6,
                                  numpy.log10(numpy.abs(numpy.cov(1 - ratio_full[indices[i], :, :]))))
            axes[0, i].set_yscale("symlog")
            axes[0, i].set_title(f"u={int(numpy.abs(baseline_table.u_coordinates[indices[i]]))}")

        axes[0, 0].set_ylabel(r"$\delta$g")
        axes[1, 0].set_ylabel(r"$\nu$ [MHz]")

        pyplot.show()

    if compute_covariance:
        if serial:
            compute_frequency_frequency_covariance_serial(baseline_table, frequency_range, output_path + project_path,
                                                          n_realisations)
        else:
            pass

    if plot_covariance:
        #gain_covariance_impact(baseline_table, frequency_range, output_path + project_path)
        residual_PS_error(baseline_table, frequency_range, output_path + project_path)


    return


def PS(taper1, taper2, covariance, dft_matrix):

    tapered_cov = covariance * taper1 * taper2
    eta_cov = numpy.dot(numpy.dot(dft_matrix.conj().T, tapered_cov), dft_matrix)
    variance = numpy.diag(numpy.real(eta_cov))
    return variance


def gain_covariance_impact(baseline_table, frequency_range, path):
    frequency_frequency_covariance = numpy.load(path + "/Simulated_Covariance/" +
                                                f"/frequency_frequency_covariance_9999.npy")

    gain_covariance = numpy.sum(frequency_frequency_covariance, axis=0) / baseline_table.number_of_baselines/226
    pyplot.imshow(numpy.abs(gain_covariance))
    pyplot.colorbar()
    pyplot.show()

    window_function = blackman_harris_taper(frequency_range)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)

    dftmatrix, eta = dft_matrix(frequency_range)
    tapered_cov = gain_covariance * taper1 * taper2
    eta_cov = numpy.dot(numpy.dot(dftmatrix.conj().T, tapered_cov), dftmatrix)
    gain_variance = numpy.diag(numpy.real(eta_cov))

    k_parallel = from_eta_to_k_par(eta, frequency_range)
    pyplot.semilogx(k_parallel[:int(len(eta) / 2)], gain_variance[:int(len(eta) / 2)])
    pyplot.show()

    return



def create_model_and_residuals(baseline_table, frequency_range, n_realisations, path ):
    print("Creating Signal Realisations")
    if not os.path.exists(path + "/" + "Simulated_Visibilities"):
        print("Creating realisation folder in Project path")
        os.makedirs(path + "/" + "Simulated_Visibilities")

    for i in range(n_realisations):
        print(f"Realisation {i}")
        signal_time0 = time.perf_counter()
        source_population = SkyRealisation(sky_type='random', flux_high=10, gamma2=4, seed=i)

        residual_indices = numpy.where(source_population.fluxes < 1)
        model_indices = numpy.where(source_population.fluxes >= 1)

        residual_sky = source_population.select_sources(residual_indices)
        modelled_sky = source_population.select_sources(model_indices)

        model_signal = get_observations(modelled_sky, baseline_table, frequency_range, interpolation='numba')
        residual_signal = get_observations(residual_sky, baseline_table, frequency_range, interpolation='numba')

        numpy.save(path +  "/" + "Simulated_Visibilities/" + f"model_realisation_{i}", model_signal)
        numpy.save(path + "/" + "Simulated_Visibilities/" + f"residual_realisation_{i}", residual_signal)
        signal_time1 = time.perf_counter()

        print(f"Realisation {i} Time = {signal_time1 - signal_time0} \n")
    return


def compute_frequency_frequency_covariance_single(path, n_realisations, frequency_range,  baseline_index):
    noise_signal_ratios = numpy.zeros((len(frequency_range), n_realisations), dtype=complex)
    for i in range(n_realisations):
        model = numpy.load(path + f"model_realisation_{i}.npy")
        residual = numpy.load(path + f"residual_realisation_{i}.npy")
        noise_signal_ratios[:, i] = residual[baseline_index, :] / model[baseline_index, :]
    frequency_covariance = numpy.cov(noise_signal_ratios)

    return frequency_covariance


def compute_frequency_frequency_covariance_serial(baseline_table, frequency_range, path, n_realisations):
    if not os.path.exists(path + "/" + "Simulated_Covariance"):
        print("Creating realisation folder in Project path")
        os.makedirs(path + "/" + "Simulated_Covariance")

    baseline_frequency_covariance = numpy.zeros((baseline_table.number_of_baselines, len(frequency_range),
                                                 len(frequency_range)), dtype=complex)

    for j in range(baseline_table.number_of_baselines):
        noise_signal_ratio = numpy.zeros((len(frequency_range), n_realisations), dtype = complex)

        if not j%100 and j != 0 or j == 1:
            print(f"Estimated time to finish re-processing = {delta_t*(baseline_table.number_of_baselines - j)}")

        t0 = time.perf_counter()
        for i in range(n_realisations):
            model_signal = numpy.load(path + f"model_realisation_{i}.npy")
            residual_signal = numpy.load(path + f"residual_realisation_{i}.npy")
            noise_signal_ratio[:, i] = residual_signal[j, :] / model_signal[j, :]

        baseline_frequency_covariance[j, ...] = numpy.cov(noise_signal_ratio)
        t1 = time.perf_counter()
        delta_t = t1 - t0

    numpy.save(path + f"frequency_frequency_covariance", baseline_frequency_covariance)

    return baseline_frequency_covariance


def compute_residual_to_model_ratio_serial(baseline_table, frequency_range, path, n_realisations):
    print("Computing Ratios")
    residual_model_ratios = numpy.zeros((baseline_table.number_of_baselines, len(frequency_range), n_realisations),
                                        dtype = complex)
    full_ratios = residual_model_ratios.copy()
    for i in range(n_realisations):
        model_signal = numpy.load(path +  "/" + "Simulated_Visibilities/" + f"model_realisation_{i}.npy")
        residual_signal = numpy.load(path +  "/" + "Simulated_Visibilities/" + f"residual_realisation_{i}.npy")
        residual_model_ratios[:, :, i] = 1 -  residual_signal / model_signal
        full_ratios[:, :, i] = model_signal / (model_signal + residual_signal)

    print("Saving Ratios")
    numpy.save(path +  "/" + "Simulated_Visibilities/" + f"residual_model_ratios_taylor", residual_model_ratios)
    numpy.save(path +  "/" + "Simulated_Visibilities/" + f"residual_model_ratios_full", full_ratios)

    return residual_model_ratios


def plot_model_serial(baseline_table, frequency_range, path, n_realisations):
    print("Plotting Model Data")
    figure, axes = pyplot.subplots(2, 1, figsize = (12, 5))

    for i in range(n_realisations):
        model_signal = numpy.load(path +  "/" + "Simulated_Visibilities/" + f"model_realisation_{i}.npy")
        residual_signal = numpy.load(path +  "/" + "Simulated_Visibilities/" + f"residual_realisation_{i}.npy")

        axes[0].plot()

    print("Saving Ratios")

    return



if __name__ == "__main__":
    main()