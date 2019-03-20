import numpy
import powerbox

from matplotlib import pyplot
import matplotlib.colors as colors

from scipy.constants import c

from generaltools import colorbar
from skymodel import SkyRealisation
from radiotelescope import RadioTelescope


def main(verbose=True):

    path = "./hex_pos.txt"
    frequency_range = numpy.linspace(135, 165, 100) * 1e6
    faulty_dipole = 1
    faulty_tile = 81
    sky_param = ["random"]
    sky_seed = 0
    calibration = True
    beam_type = "gaussian"
    load = False

    telescope = RadioTelescope(load = True, path=path, verbose = verbose)
    baseline_table = telescope.baseline_table
    source_population = SkyRealisation(sky_type="random", verbose = verbose)

    #Determine maximum resolution
    max_frequency = frequency_range[-1]
    max_u = numpy.max(baseline_table.u(max_frequency))
    max_v = numpy.max(baseline_table.v(max_frequency))
    max_b = max(max_u, max_v)
    # sky_resolutions
    min_l = 1. / max_b

    for frequency_index in len(frequency_range):

        sky_image, l_coordinates = source_population.create_sky_image(
            frequency_channels = frequency_range[frequency_index], resolution=min_l, oversampling=1)
        ll, mm = numpy.meshgrid(l_coordinates, l_coordinates)

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


    return

def visibility_extractor_OO(baseline_table_object, source_population_object, frequency, antenna1_response,
                            antenna2_response, padding_factor = 3):

    apparent_sky = sky_cube * antenna1_response * numpy.conj(antenna2_response)

    padded_sky = numpy.pad(apparent_sky, padding_factor * apparent_sky.shape[0], mode="constant")
    shifted_image = numpy.fft.ifftshift(padded_sky, axes=(0, 1))
    visibility_grid, uv_coordinates = powerbox.dft.fft(shifted_image, L=2*(2 * padding_factor + 1), axes=(0, 1))


    measured_visibilities = uv_list_to_baseline_measurements_OO(baseline_table, frequency, visibility_grid, uv_coordinates)

    return measured_visibilities, uv_coordinates



if __name__ == "__main__":
    start = time.clock()
    main()
    end = time.clock()
    print(f"Time is {end-start}")