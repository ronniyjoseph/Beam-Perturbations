import numpy
from radiotelescope import RadioTelescope
from radiotelescope import beam_width

from analytic_covariance import sky_covariance
from analytic_covariance import beam_covariance
from analytic_covariance import moment_returner

def gain_variance(path):
    # Step 1 Load the MWA tile positions
    # Step 2 Calculate the baselines
    # Step 3 Pick a tile (core, redundant, outrigger)
    # Step 4 Calculate the Covariances for those baselines at all frequencies
    # Step 4 Calculate the mean signals for those baselines
    # Step 5 Calculate the gain variance by summing by the rations per frequency
    # DFT that gain covariance matrix (off-diagonals == 0)
    # What is the frequency structure
    tile_id = 1036  # 81 #1036

    nu = numpy.linspace(145, 155, 2)*1e6
    mwa_telescope = RadioTelescope(load=True, path=path)
    average_sky_brightness = moment_returner(n_order=1, S_low= 1, S_mid=1)
    print(average_sky_brightness)

    # select baseline table indices in which tile_id is present
    baseline_indices = numpy.where((mwa_telescope.baseline_table.antenna_id1 == tile_id) |
                                   (mwa_telescope.baseline_table.antenna_id2 == tile_id))[0]

    baseline_table_selection = mwa_telescope.baseline_table.sub_table((baseline_indices))

    ratios = numpy.zeros((len(baseline_indices)))
    j = 0

    for i in range(len(baseline_indices)):




        u = baseline_table_selection.u(frequency=nu[j])[i]
        v = baseline_table_selection.v(frequency=nu[j])[i]

        data_covariance = sky_covariance(u, v, nu) + beam_covariance(u, v, nu)
        sigma = beam_width(frequency=nu[j])

        signal = 2*numpy.pi*sigma**2*average_sky_brightness*numpy.exp(-2*numpy.pi**2*sigma**2*(u**2 + v**2))
        print("Signal", signal)
        ratios[i] = signal**2/data_covariance[j, j]

    variance = 1/(2*numpy.real(numpy.sum(ratios, axis=0)))

    print(variance)


    return

if __name__ == "__main__":
    path = "./Data/MWA_All_Coordinates_Cath.txt"
    gain_variance(path)
