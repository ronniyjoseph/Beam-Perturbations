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
    source_population = SkyRealisation(sky_type="random", verbose = verbose)



    return


if __name__ == "__main__":
    start = time.clock()
    main()
    end = time.clock()
    print(f"Time is {end-start}")