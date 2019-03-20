import numpy
import copy

from scipy.constants import c


class RadioTelescope:
    def __init__(self, load=True, path=None, shape='linear', frequency_channels=None, verbose=False):
        if verbose:
            print("Creating the radio telescope")
        self.antenna_positions = AntennaPositions(load, path, shape)
        self.baseline_table = BaselineTable(self.antenna_positions, frequency_channels, verbose)
        return


class AntennaPositions:
    def __init__(self, load=True, path=None, shape='linear'):
        if load:
            if path == None:
                raise ValueError("Specificy the antenna position path if loading position data")
            else:
                antenna_data = numpy.loadtxt(path)

                antenna_data = antenna_data[numpy.argsort(antenna_data[:, 0])]

                self.antenna_ids = antenna_data[:, 0]
                self.x_coordinates = antenna_data[:, 1]
                self.y_coordinates = antenna_data[:, 2]
                self.z_coordinates = antenna_data[:, 3]
        else:
            raise ValueError("Custom shapes are note supported yet")
        return

    def number_antennas(self):
        return len(self.antenna_ids)


class BaselineTable:
    def __init__(self, position_table, frequency_channels=None, verbose=False):
        self.antenna_id1 = None
        self.antenna_id2 = None
        self.u_coordinates = None
        self.v_coordinates = None
        self.w_coordinates = None
        self.reference_frequency = None
        self.number_of_baselines = None
        self.selection = None
        # update all attributes
        self.baseline_converter(position_table, frequency_channels, verbose)
        return

    def baseline_converter(self, position_table, frequency_channels=None, verbose=True):
        if verbose:
            print("")
            print("Converting xyz to uvw-coordinates")

        if frequency_channels is None:
            self.reference_frequency = 150e6
        elif type(frequency_channels) == numpy.ndarray:
            assert min(frequency_channels) > 1e6, "Frequency range is smaller 1 MHz, probably wrong units"
            self.reference_frequency = frequency_channels[0]
        elif numpy.isscalar(frequency_channels):
            assert frequency_channels > 1e6, "Frequency range is smaller 1 MHz, probably wrong units"
            self.reference_frequency = frequency_channels
        else:
            raise ValueError(
                f"frequency_channels should be 'numpy.ndarray', or scalar not type({self.reference_frequency})")

        # calculate the wavelengths of the adjecent channels
        reference_wavelength = c / self.reference_frequency
        # Count the number of antenna
        number_of_antenna = position_table.number_antennas()
        # Calculate the number of possible baselines
        self.number_of_baselines = int(0.5 * number_of_antenna * (number_of_antenna - 1.))

        # Create arrays for the baselines
        # baselines x Antenna1, Antenna2, u, v, w, gain product, phase sum x channels
        antenna_1 = numpy.zeros(self.number_of_baselines)
        antenna_2 = antenna_1.copy()

        u_coordinates = antenna_1.copy()
        v_coordinates = antenna_1.copy()
        w_coordinates = antenna_1.copy()

        if verbose:
            print("")
            print("Number of antenna =", number_of_antenna)
            print("Total number of baselines =", self.number_of_baselines)

        # arbitrary counter to keep track of the baseline table
        k = 0

        for i in range(number_of_antenna):
            for j in range(i + 1, number_of_antenna):
                # save the antenna numbers in the uv table
                antenna_1[k] = position_table.antenna_ids[i]
                antenna_2[k] = position_table.antenna_ids[j]

                # rescale and write uvw to multifrequency baseline table
                u_coordinates[k] = (position_table.x_coordinates[i] - position_table.x_coordinates[
                    j]) / reference_wavelength
                v_coordinates[k] = (position_table.y_coordinates[i] - position_table.y_coordinates[
                    j]) / reference_wavelength
                w_coordinates[k] = (position_table.z_coordinates[i] - position_table.z_coordinates[
                    j]) / reference_wavelength

                k += 1

        self.antenna_id1 = antenna_1
        self.antenna_id2 = antenna_2

        self.u_coordinates = u_coordinates
        self.v_coordinates = v_coordinates
        self.w_coordinates = w_coordinates
        return

    def u(self, frequency=None):
        rescaled_u = rescale_baseline(self.u_coordinates, self.reference_frequency, frequency)
        selected_rescaled_u = select_baselines(rescaled_u, self.selection)

        return selected_rescaled_u

    def v(self, frequency=None):
        rescaled_v = rescale_baseline(self.v_coordinates, self.reference_frequency, frequency)
        selected_rescaled_v = select_baselines(rescaled_v, self.selection)

        return selected_rescaled_v

    def w(self, frequency=None):
        rescaled_w = rescale_baseline(self.w_coordinates, self.reference_frequency, frequency)
        selected_rescaled_w = select_baselines(rescaled_w, self.selection)

        return selected_rescaled_w

    def sub_table(self, baseline_selection_indices):
        subtable = copy.copy(self)
        subtable.selection = baseline_selection_indices

        return subtable

def beam_width(frequency, diameter=4, epsilon=1):
    sigma = epsilon * c / (frequency * diameter)
    width = numpy.sin(0.5 * sigma)
    return width


def ideal_gaussian_beam(source_l, source_m, nu, diameter=4, epsilon=1):
    sigma = beam_width(nu, diameter, epsilon)

    beam_attenuation = numpy.exp(-(source_l ** 2. + source_m ** 2.) / (2 * sigma ** 2))

    return beam_attenuation

def broken_gaussian_beam(faulty_dipole, ideal_beam, source_l, source_m, nu, diameter=4, epsilon=1, dx=1.1):
    wavelength = c / nu
    x_offsets = numpy.array([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5, -1.5,
                             -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5], dtype=numpy.float32) * dx

    y_offsets = numpy.array([1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5,
                             -0.5, -0.5, -1.5, -1.5, -1.5, -1.5], dtype=numpy.float32) * dx

    dipole_beam = ideal_gaussian_beam(source_l, source_m, nu, diameter / 4.)
    broken_beam = ideal_beam - 1 / 16 * dipole_beam * numpy.exp(
        -2. * numpy.pi * 1j * (x_offsets[faulty_dipole] * numpy.abs(source_l) +
                               y_offsets[faulty_dipole] * numpy.abs(source_m)) / wavelength)

    return broken_beam

def ideal_mwa_beam_loader(theta, phi, frequency, load=True, verbose = False):
    if not load:
        if verbose:
            print("Creating the idealised MWA beam\n")
        ideal_beam = mwa_tile_beam(theta, phi, frequency=frequency)
        if not os.path.exists("beam_maps"):
            print("")
            print("Creating beam map folder locally!")
            os.makedirs("beam_maps")
        numpy.save(f"beam_maps/ideal_beam_map.npy", ideal_beam)
    if load:
        if verbose:
            print("Loading the idealised MWA beam\n")
        ideal_beam = numpy.load(f"beam_maps/ideal_beam_map.npy")

    return ideal_beam


def broken_mwa_beam_loader(theta, phi, frequency, faulty_dipole, load=True):
    dipole_weights = numpy.zeros(16) + 1
    dipole_weights[faulty_dipole] = 0
    if load:
        print(f"Loading perturbed tile beam for dipole {faulty_dipole}")
        perturbed_beam = numpy.load(f"beam_maps/perturbed_dipole_{faulty_dipole}_map.npy")
    elif not load:
        # print(f"Generating perturbed tile beam for dipole {faulty_dipole}")
        perturbed_beam = mwa_tile_beam(theta, phi, weights=dipole_weights, frequency=frequency)
        if not os.path.exists("beam_maps"):
            print("")
            print("Creating beam map folder locally!")
            os.makedirs("beam_maps")
        numpy.save(f"beam_maps/perturbed_dipole_{faulty_dipole}_map.npy", perturbed_beam)

    return perturbed_beam


def rescale_baseline(baseline_coordinates, reference_frequency, frequency):
    if frequency is None:
        rescaled_coordinates = baseline_coordinates
    elif numpy.isscalar(frequency):
        rescaling_factor = frequency / reference_frequency
        rescaled_coordinates = baseline_coordinates * rescaling_factor
    elif type(frequency) == numpy.ndarray:
        rescaling_factor = frequency / reference_frequency
        coordinate_mesh, rescale_mesh = numpy.meshgrid(rescaling_factor, baseline_coordinates)
        rescaled_coordinates = coordinate_mesh * rescale_mesh
    else:
        raise ValueError(f"frequency should be scalar or numpy.ndarray not {type(frequency)}")

    return rescaled_coordinates


def select_baselines(baseline_coordinates, baseline_selection_indices):
    if baseline_selection_indices is None:
        selected_baseline_coordinates = baseline_coordinates
    else:
        selected_baseline_coordinates = baseline_coordinates[baseline_selection_indices, ...]
    return selected_baseline_coordinates

