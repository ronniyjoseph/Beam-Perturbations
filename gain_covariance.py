import numpy
import matplotlib
from matplotlib import pyplot
from radiotelescope import RadioTelescope
from radiotelescope import beam_width
from radiotelescope import ideal_gaussian_beam
from generaltools import visibility_extractor

from analytic_covariance import sky_covariance
from analytic_covariance import beam_covariance
from analytic_covariance import moment_returner
from analytic_covariance import dft_matrix
from analytic_covariance import blackman_harris_taper
from analytic_covariance import plot_PS
from analytic_covariance import calculate_total_2DPS


def gain_variance(nu, path):
    # Step 1 Load the MWA tile positions
    # Step 2 Calculate the baselines
    # Step 3 Pick a tile (core, redundant, outrigger)
    # Step 4 Calculate the Covariances for those baselines at all frequencies
    # Step 4 Calculate the mean signals for those baselines
    # Step 5 Calculate the gain variance by summing by the rations per frequency
    # DFT that gain covariance matrix (off-diagonals == 0)
    # What is the frequency structure
    tile_id = [31, 1036, 81]  # 81 #1036

    mwa_telescope = RadioTelescope(load=True, path=path, frequency_channels=nu)
    average_sky_brightness = moment_returner(n_order=1, S_low= 1, S_mid=1, S_high= 5)
    print(average_sky_brightness)
    #print("brightness", average_sky_brightness)

    ratios = numpy.zeros((3, len(mwa_telescope.antenna_positions.antenna_ids), len(nu)))
    variance = numpy.zeros((3, len(nu)))

    for k in range(len(tile_id)):
    # select baseline table indices in which tile_id is present
        baseline_indices = numpy.where((mwa_telescope.baseline_table.antenna_id1 == tile_id[k]) |
                                       (mwa_telescope.baseline_table.antenna_id2 == tile_id[k]))[0]

        baseline_table_selection = mwa_telescope.baseline_table.sub_table((baseline_indices))

        for i in range(len(baseline_indices)):

            u = baseline_table_selection.u(frequency=nu[0])[i]
            v = baseline_table_selection.v(frequency=nu[0])[i]

            data_covariance = sky_covariance(u, v, nu) + beam_covariance(u, v, nu)
            sigma = beam_width(frequency=nu)/numpy.sqrt(2)
            #pyplot.loglog(nu / 1e6, numpy.diag(data_covariance))
            #pyplot.show()

            signal = 2*numpy.pi*sigma**2*average_sky_brightness
            #print("signal", average_sky_brightness)
            #print("noise", numpy.diag(data_covariance))

        variance[k, :] = numpy.sqrt(1/(2*numpy.real(numpy.sum(ratios[k,...], axis=0))))


    alt_variance = numpy.diag(data_covariance)/(2*signal**2*127*(nu/nu[0])**(-2*0.5))
    #pyplot.plot(nu/1e6, alt_variance, label = "analytic")
    #pyplot.legend()
    #pyplot.show()
    #print(variance[0,...] - alt_variance)
    return alt_variance


def calculate_residual_2DPS(u, nu, plot = False, save = False, plot_name = "total_ps.pdf", path  =""):
    window_function = blackman_harris_taper(nu)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)

    dftmatrix, eta = dft_matrix(nu)

    variance = numpy.zeros((len(u), len(nu)))

    gainvariance = numpy.sqrt(gain_variance(nu, path))

    gain_covariance = numpy.outer(gainvariance, gainvariance)*ripple_cov

    pyplot.imshow(gain_covariance)
    #pyplot.show()
    for i in range(len(u)):


        residual_covariance = sky_covariance(u[i], 0, nu) + beam_covariance(u[i], v=0, nu=nu)
        model_covariance = sky_covariance(u[i], 0, nu, S_low=1, S_high = 5)


        nu_cov = gain_covariance*model_covariance + (1 - 2*gain_covariance)*residual_covariance
        #diagonal = 2*gainvariance*

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

    if plot:
        plot_PS(u, eta[:int(len(eta)/2)], nu, variance[:, :int(len(eta)/2)], cosmological=True, title="Total", save = save,
                save_name = plot_name)
    return eta[:int(len(eta)/2)], variance[:, :int(len(eta)/2)]








if __name__ == "__main__":
    path = "./Data/MWA_All_Coordinates_Cath.txt"
    output_folder = "../../Plots/Analytic_Covariance/"


    nu = numpy.linspace(145, 155, 101)*1e6
    u = numpy.logspace(0.1, 2.5, 100)
    eta, res_PS = calculate_residual_2DPS(u, nu, path=path)
    eta1, original_PS = calculate_total_2DPS(u, nu, plot = False)


    plot_PS(u, eta, nu, original_PS, cosmological=True, title="Uncalibrated", save = True, save_name = output_folder + "residuals_uncalibrated.pdf")
    plot_PS(u, eta, nu, res_PS, cosmological=True, title="Calibrated", save = True, save_name = output_folder + "residuals_calibrated.pdf")
    plot_PS(u, eta, nu, numpy.abs(res_PS-original_PS), cosmological=True, title="Calibrated-Uncalibrated", save = True, save_name = output_folder + "residuals_difference.pdf")
    plot_PS(u, eta, nu, numpy.abs(res_PS - original_PS)/original_PS, cosmological=True, ratio =True, title="Calibrated -Uncalibrated/Uncalibrated", save=True, save_name=output_folder + "residuals_ratio.pdf")

    pyplot.show()
