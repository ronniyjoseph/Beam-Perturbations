import numpy

class RadioTelescope:
    def __init__(self, load = True, path, shape = 'linear'):

        self.antenna_positions = AntennaPositions(load)
        self.baseline_table = 0
        self.gains = 0

    return

class AntennaPositions:
    def __init__(self, load = True, path = None, shape = 'linear'):
        if load:
            if path == None:
                raise ValueError("Specificy the antenna position path if loading position data")
            else:
                antenna_data = numpy.loadtxt(path)
                self.antenna_ids = antenna_data[:,0]
                self.x_coordinates = antenna_data[:,1]
                self.y_coordinates = antenna_data[:,2]
                self.z_coordinates = numpy.zeros_like(antenna_data[:,0])
        else:
            x= 0

    return


def xyz_position_creator(shape, verbose=True):
    # type: (object) -> object
    """
	Generates an array lay-out defined by input parameters, returns
	x,y,z coordinates of each antenna in the array

	shape	: list of array parameters
	shape[0]	: string value 'square', 'hex', 'doublehex', 'linear'

		'square': produces a square array
			shape[1]: 1/2 side of the square in meters
			shape[2]: minimum baseline length
			shape[3]: x position of square
			shape[4]: y position of square

		'hex': produces a hex array

		'doublehex': produces a double hex array

		'linear': produces a linear array
			shape[1]: x-outeredges of the array
			shape[2]: number of elements in the EW-linear array

	"""
    if shape[0] == "square" or shape[0] == 'doublesquare':
        if verbose:
            print("")
            print("Creating x- y- z-positions of a square array")
        x_coordinates = numpy.arange(-shape[1], shape[1], shape[2])
        y_coordinates = numpy.arange(-shape[1], shape[1], shape[2])

        block1 = numpy.zeros((len(x_coordinates) * len(y_coordinates), 4))
        k = 0
        for i in range(len(x_coordinates)):
            for j in range(len(y_coordinates)):
                block1[k, 0] = 1001 + k
                block1[k, 1] = x_coordinates[i]
                block1[k, 2] = y_coordinates[j]
                block1[k, 3] = 0
                k += 1
        if shape[0] == 'square':
            block1[:, 1] += shape[3]
            block1[:, 2] += shape[4]
            xyz_coordinates = block1.copy()
        elif shape[0] == 'doublesquare':
            block2 = block1.copy()

            block2[:, 0] += 1000 + len(block1[:, 0])
            block2[:, 1] += shape[3]
            block2[:, 2] += shape[4]
            xyz_coordinates = numpy.vstack((block1, block2))

    elif shape[0] == 'hex' or shape[0] == 'doublehex':
        if verbose:
            print("")
            print("Creating x- y- z-positions of a " + shape[0] + " array")

        dx = shape[1]
        dy = dx * numpy.sqrt(3.) / 2.

        line1 = numpy.array([numpy.arange(4) * dx, numpy.zeros(4), numpy.zeros(4)]).transpose()

        # define the second line
        line2 = line1[0:3, :].copy()
        line2[:, 0] += dx / 2.
        line2[:, 1] += dy
        # define the third line
        line3 = line1[0:3].copy()
        line3[:, 1] += 2 * dy
        # define the fourth line
        line4 = line2[0:2, :].copy()
        line4[:, 1] += 2 * dy

        block1 = numpy.vstack((line1[1:], line2, line3, line4))

        block2 = numpy.vstack((line1[1:], line2, line3[1:], line4))
        block2[:, 0] *= -1

        block3 = numpy.vstack((line2, line3, line4))
        block3[:, 1] *= -1

        block4 = numpy.vstack((line2, line3[1:], line4))
        block4[:, 0] *= -1
        block4[:, 1] *= -1
        hex_block = numpy.vstack((block1, block2, block3, block4))

        if shape[0] == 'hex':
            hex_block[:, 0] += shape[2]
            hex_block[:, 1] += shape[3]
            antenna_numbers = numpy.arange(len(hex_block[:, 0])) + 1001
            xyz_coordinates = numpy.vstack((antenna_numbers, hex_block.T)).T
        elif shape[0] == 'doublehex':
            antenna_numbers = numpy.arange(len(hex_block[:, 0])) + 1001
            first_hex = numpy.vstack((antenna_numbers, hex_block.T)).T

            second_hex = first_hex.copy()

            first_hex[:, 1] += shape[2]
            first_hex[:, 2] += shape[3]

            second_hex[:, 0] += 1000 + len(first_hex[:, 0])
            second_hex[:, 1] += shape[4]
            second_hex[:, 2] += shape[5]
            xyz_coordinates = numpy.vstack((first_hex, second_hex))

    elif shape[0] == 'linear':
        if verbose:
            print("")
            print("Creating x- y- z-positions of a " + str(shape[2]) + " element linear array")
        xyz_coordinates = numpy.zeros((shape[2], 4))
        xyz_coordinates[:, 0] = numpy.arange(shape[2]) + 1001
        xyz_coordinates[:, 1] = numpy.linspace(-shape[1], shape[1], shape[2])
    elif shape[0] == 'file':
        xyz_coordinates = antenna_table_loader(shape[1])

    return xyz_coordinates





