import numpy
from powerspectrum import get_power_spectrum
from radiotelescope import RadioTelescope


def main():
    verbose = True
    path = "./hex_pos.txt"
    data_path = "/data/rjoseph/Hybrid_Calibration/Tile_Pertubation/Simulation_Output/" + "gaussian_tile1036_dipole1_corrected_True/"
    frequency_range = numpy.linspace(135, 165, 100) * 1e6
    telescope = RadioTelescope(load=True, path=path, verbose=verbose)

    ideal_data = numpy.load(data_path + "ideal_simulated_data.npy")
    broken_data = numpy.load(data_path + "broken_simulated_data.npy")

    get_power_spectrum(frequency_range, telescope, ideal_data, broken_data, faulty_tile=1036,
                       plot_file_name=data_path + "1036_1_repr_data.pdf", verbose=verbose)

    return


if __name__ == "__main__":
    main()
