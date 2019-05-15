import numpy
import radiotelescope
import powerbox
from scipy import interpolate

import sys
sys.path.append('../../../redundant_calibration/code/SCAR')

from single_dipole_PS_impact import main as old_code
from Single_Dipole_PS_Impact_OO import main as new_code



from radiotelescope import RadioTelescope
from skymodel import SkyRealisation

import matplotlib
from matplotlib import pyplot

from time import process_time

def main():
    coordinated_old, ideal_cube1, ideal_weights1, broken_cube1, broken_weights1 = old_code()
    coordinates_new, ideal_cube2, ideal_weights2, broken_cube2, broken_weights2 = new_code()

    print((ideal_cube1 - ideal_cube2).shape)
    print((ideal_weights1 - ideal_weights2).shape)
    print((broken_cube1 - broken_cube1).shape)
    print((broken_weights1 - broken_weights2).shape)
    pyplot.plot(coordinated_old - coordinates_new)
    pyplot.show()
    print("I am just not going to do what you want")

    for i in range(2):
        pyplot.pcolor(coordinated_old, coordinates_new, numpy.abs(ideal_cube1 - ideal_cube2)[..., i])
        pyplot.title("Visibilities")
        pyplot.colorbar()
        pyplot.show()

        pyplot.pcolor(coordinated_old, coordinates_new, (ideal_weights1 - ideal_weights2)[..., i])
        pyplot.title("Weights")
        pyplot.colorbar()
        pyplot.show()

        pyplot.pcolor(coordinated_old, coordinates_new, numpy.abs(broken_cube1 - broken_cube2)[..., i])
        pyplot.title("Visibilities")
        pyplot.colorbar()
        pyplot.show()

        pyplot.pcolor(coordinated_old, coordinates_new, (broken_weights1 - broken_weights2)[..., i])
        pyplot.title("Weights")
        pyplot.colorbar()
        pyplot.show()

    return
if __name__ == "__main__":
    main()