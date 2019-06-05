import numpy

from matplotlib import pyplot
from radiotelescope import RadioTelescope
from radiotelescope import beam_width

from analytic_covariance import sky_covariance
from analytic_covariance import beam_covariance
from analytic_covariance import moment_returner
from analytic_covariance import dft_matrix
from analytic_covariance import blackman_harris_taper

def gain_variance(path):
    # Step 1 Load the MWA tile positions
    # Step 2 Calculate the baselines
    # Step 3 Pick a tile (core, redundant, outrigger)
    # Step 4 Calculate the Covariances for those baselines at all frequencies
    # Step 4 Calculate the mean signals for those baselines
    # Step 5 Calculate the gain variance by summing by the rations per frequency
    # DFT that gain covariance matrix (off-diagonals == 0)
    # What is the frequency structure
    tile_id = [31, 81, 1036]  # 81 #1036

    nu = numpy.linspace(145, 155, 100)*1e6
    mwa_telescope = RadioTelescope(load=True, path=path, frequency_channels=nu)
    average_sky_brightness = moment_returner(n_order=1, S_low= 1, S_mid=1, S_high= 5)
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

            signal =  2*numpy.pi*sigma**2*average_sky_brightness#*numpy.exp(-2*numpy.pi**2*sigma**2*(u**2 + v**2)*(nu[j]/nu[0])**2)
            ratios[k, i, :] = signal**2/numpy.diag(data_covariance)
            #print("signal", average_sky_brightness)
            #print("noise", numpy.diag(data_covariance))

        variance[k, :] = numpy.sqrt(1/(2*numpy.real(numpy.sum(ratios[k,...], axis=0))))
        pyplot.plot(nu/1e6, variance[k, ...], label =f"Antenna {tile_id}")


    #nu_cov = numpy.diag(variance)

    #window_function = blackman_harris_taper(nu)
    #taper1, taper2 = numpy.meshgrid(window_function, window_function)
    #dftmatrix, eta = dft_matrix(nu)

    #tapered_cov = nu_cov * taper1 * taper2
    #eta_cov = numpy.dot(numpy.dot(dftmatrix.conj().T, tapered_cov), dftmatrix)

    #pyplot.plot(nu, variance)
    #pyplot.show()
    #pyplot.plot(eta, numpy.diag(numpy.real(eta_cov)))
    #pyplot.show()
    alt_variance = numpy.sqrt(numpy.diag(data_covariance)/(2*signal**2*127))
    pyplot.plot(nu/1e6, alt_variance, label = "analytic")
    pyplot.legend()
    pyplot.show()
    print(variance[0,...] - alt_variance)
    return

if __name__ == "__main__":
    path = "./Data/MWA_All_Coordinates_Cath.txt"
    gain_variance(path)
