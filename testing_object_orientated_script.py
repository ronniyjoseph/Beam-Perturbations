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

from matplotlib import pyplot

from time import process_time

def main():
    ideal_cube1, ideal_weights1, broken_cube1, broken_weights1 = old_code()
    ideal_cube2, ideal_weights2, broken_cube2, broken_weights2 = new_code()

    print(ideal_cube1 - ideal_cube2)
    print(ideal_weights1 - ideal_weights2)
    print(broken_cube1 - broken_cube1)
    print(broken_weights1 - broken_weights2)

    pyplot.imshow(numpy.abs(ideal_cube1 - ideal_cube2))
    pyplot.show()
    pyplot.imshow(ideal_weights1 - ideal_weights2)
    pyplot.show()
    pyplot.imshow(broken_cube1 - broken_cube1)
    pyplot.show()
    pyplot.imshow(broken_weights1 - broken_weights2)
    pyplot.show()
if __name__ == "__main__":
    main()