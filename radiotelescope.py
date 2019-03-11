import numpy

class RadioTelescope:
    def __init__(self, load = True, path = None, shape = 'linear'):

        self.antenna_positions = AntennaPositions(load, path, shape)
        self.gains = 0
        self.baseline_table = BaselineTable(self.antenna_positions, self.gains)

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
                raise ValueError("Custom shapes are note supported yet")

        return

class BaselineTable:
    def __init__(self, ):
        self.first_antenna = 0
        self.second_antenna= 0
        self.u_coordinates = 0
        self.v_coordinates = 0
        self.w_coordinates = 0
        #update all attributes
        baseline_converter()

        return

    def baseline_converter(self, gain_table, frequency_channels, verbose=True):
        if verbose:
            print("")
            print("Converting xyz to uvw-coordinates")

        assert min(frequency_channels) > 1e6, "Frequency range is smaller 1 MHz, probably wrong units"

        # calculate the wavelengths of the adjecent channels
        wavelength_range = scipy.constants.c / frequency_channels
        # Count the number of antenna
        number_of_antenna = len(xy_positions[:, 0])
        # Calculate the number of possible baselines
        number_of_baselines = int(0.5 * number_of_antenna * (number_of_antenna - 1.))
        # count the number of channels
        n_channels = len(frequency_channels)
        # Create an empty array for the baselines
        # baselines x Antenna1, Antenna2, u, v, w, gain product, phase sum x channels
        uv_positions = numpy.zeros((number_of_baselines, 7, n_channels))

        if verbose:
            print("")
            print("Number of antenna =", number_of_antenna)
            print("Total number of baselines =", number_of_baselines)

        # arbitrary counter to keep track of the baseline table
        k = 0
        for i in range(number_of_antenna):
            for j in range(i + 1, number_of_antenna):
                # save the antenna numbers in the uv table
                uv_positions[k, 0, :] = xy_positions[i, 0]
                uv_positions[k, 1, :] = xy_positions[j, 0]

                # rescale and write uvw to multifrequency baseline table
                uv_positions[k, 2, :] = (xy_positions[i, 1] - xy_positions[j, 1]) / \
                                        wavelength_range
                uv_positions[k, 3, :] = (xy_positions[i, 2] - xy_positions[j, 2]) / \
                                        wavelength_range
                uv_positions[k, 4, :] = (xy_positions[i, 3] - xy_positions[j, 3]) / \
                                        wavelength_range

                # Find the gains
                amp_gain1 = gain_table[gain_table[:, 0, 0] == xy_positions[i, 0], 1, :][0]
                amp_gain2 = gain_table[gain_table[:, 0, 0] == xy_positions[j, 0], 1, :][0]

                phase_gain1 = gain_table[gain_table[:, 0, 0] == xy_positions[i, 0], 2, :][0]
                phase_gain2 = gain_table[gain_table[:, 0, 0] == xy_positions[j, 0], 2, :][0]

                # calculate the complex baseline gain
                uv_positions[k, 5, :] = amp_gain1 * amp_gain2
                uv_positions[k, 6, :] = -(phase_gain1 - phase_gain2)

                k += 1

        return uv_positions
