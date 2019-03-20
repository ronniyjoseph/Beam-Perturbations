import numpy
import radiotelescope

import sys
sys.path.append('../../../redundant_calibration/code/SCAR')
from RadioTelescope import antenna_gain_creator
from RadioTelescope import baseline_converter
from RadioTelescope import antenna_table_loader

from single_dipole_PS_impact import visibility_extractor
from single_dipole_PS_impact import uv_list_to_baseline_measurements
from single_dipole_PS_impact import ideal_gaussian_beam
from single_dipole_PS_impact import broken_gaussian_beam




from radiotelescope import RadioTelescope
from skymodel import SkyRealisation

from matplotlib import pyplot

from time import process_time

def main():
    path = "./hex_pos.txt"
    frequency_range = numpy.linspace(135, 165, 100) * 1e6
    faulty_dipole = 1
    faulty_tile = 81
    verbose = False
    frequency_index = 1

    start_old = process_time()
    #ideal_old, broken_old = generate_visibilities_old(path, frequency_range, faulty_dipole, faulty_tile, frequency_index)
    end_old = process_time()

    start_OO = process_time()
    ideal_OO, broken_OO = generate_visibilities_OO(path, frequency_range, faulty_dipole, faulty_tile, frequency_index)
    end_OO = process_time()

    print(end_OO - start_OO)
    print(ideal_old - ideal_OO)

    return

def generate_visibilities_old(path, frequency_range, faulty_dipole, faulty_tile, frequency_index, verbose= False,
                              beam_type = 'gaussian'):
    xyz_positions = antenna_table_loader(path)
    gain_table = antenna_gain_creator(xyz_positions, frequency_range)
    baseline_table = baseline_converter(xyz_positions, gain_table, frequency_range, verbose=verbose)

    source_population = SkyRealisation(sky_type="random")
    sky_cube, l_coordinates = source_population.create_sky_image(frequency_channels=frequency_range,
                                                             baseline_table=baseline_table)
    ll, mm, ff = numpy.meshgrid(l_coordinates, l_coordinates, frequency_range)

    if beam_type == "MWA":
        tt, pp, = lm_to_theta_phi(ll, mm)
        ideal_beam = ideal_mwa_beam_loader(tt, pp, ff, load)
        broken_beam = broken_mwa_beam_loader(tt, pp, ff, faulty_dipole, load)

    elif beam_type == "gaussian":
        ideal_beam = ideal_gaussian_beam(ll, mm, ff)
        broken_beam = broken_gaussian_beam(faulty_dipole, ideal_beam, ll, mm, ff)
    else:
        raise ValueError("The only valid option for the beam are 'MWA' or 'gaussian'")

    ideal_measured_visibilities = numpy.zeros((baseline_table.shape[0], len(frequency_range)), dtype=complex)
    broken_measured_visibilities= ideal_measured_visibilities.copy()

    ##### Select perfect baselines #####
    perfect_baseline_indices = numpy.where((baseline_table[:, 0, 0] != faulty_tile) &
                                           (baseline_table[:, 1, 0] != faulty_tile))[0]
    broken_baseline_indices = numpy.where((baseline_table[:, 0, 0] == faulty_tile) |
                                          (baseline_table[:, 1, 0] == faulty_tile))[0]

    # Sample all baselines, and get the relevant uv_grid coordinates
    ideal_measured_visibilities[..., frequency_index], full_uv_grid = visibility_extractor(
        baseline_table[..., frequency_index], sky_cube[..., frequency_index], ideal_beam[..., frequency_index],
        ideal_beam[..., frequency_index])

    # Copy good baselines to broken table
    broken_measured_visibilities[perfect_baseline_indices, frequency_index] = ideal_measured_visibilities[
        perfect_baseline_indices, frequency_index]

    broken_measured_visibilities[broken_baseline_indices, frequency_index], partial_uv_grid = visibility_extractor(
        baseline_table[broken_baseline_indices, :, frequency_index], sky_cube[..., frequency_index],
        ideal_beam[..., frequency_index], broken_beam[..., frequency_index])

    return ideal_measured_visibilities[..., frequency_index], broken_measured_visibilities[..., frequency_index]

def generate_visibilities_OO(path, frequency_range, faulty_dipole, faulty_tile, frequency_index, beam_type = 'gaussian'):
    telescope = radiotelescope.RadioTelescope(load=True, path=path)
    baseline_table = telescope.baseline_table

    source_population = SkyRealisation(sky_type="random")
    sky_cube, l_coordinates = source_population.create_sky_image(frequency_channels=frequency_range[frequency_index],
                                                             radiotelescope=telescope)
    ll, mm = numpy.meshgrid(l_coordinates, l_coordinates)

    if beam_type == "MWA":
        tt, pp, = lm_to_theta_phi(ll, mm)
        ideal_beam = ideal_mwa_beam_loader(tt, pp, frequency_range[frequency_index], load)
        broken_beam = broken_mwa_beam_loader(tt, pp, frequency_range[frequency_index], faulty_dipole, load)

    elif beam_type == "gaussian":
        ideal_beam = ideal_gaussian_beam(ll, mm, frequency_range[frequency_index])
        broken_beam = broken_gaussian_beam(faulty_dipole, ideal_beam, ll, mm, frequency_range[frequency_index])
    else:
        raise ValueError("The only valid option for the beam are 'MWA' or 'gaussian'")


    ideal_measured_visibilities = numpy.zeros((baseline_table.number_of_baselines), dtype=complex)
    broken_measured_visibilities= ideal_measured_visibilities.copy()

    ##### Select perfect baselines #####
    perfect_baseline_indices = numpy.where((baseline_table.antenna_id1 != faulty_tile) &
                                           (baseline_table.antenna_id2 != faulty_tile))[0]
    broken_baseline_indices = numpy.where((baseline_table.antenna_id1 == faulty_tile) |
                                          (baseline_table.antenna_id2 == faulty_tile))[0]

    # Sample all baselines, and get the relevant uv_grid coordinates
    ideal_measured_visibilities[:], full_uv_grid = visibility_extractor_OO(
        baseline_table, sky_cube, frequency_range[frequency_index], ideal_beam[..., frequency_index],
        ideal_beam[..., frequency_index])

    print(sys.getsizeof(ideal_measured_visibilities)/1e9)

    # Copy good baselines to broken table
    broken_measured_visibilities[perfect_baseline_indices] = ideal_measured_visibilities[
        perfect_baseline_indices]

    broken_measured_visibilities[broken_baseline_indices], partial_uv_grid = visibility_extractor_OO(
        baseline_table.sub_table(broken_baseline_indices), sky_cube, frequency_range[frequency_index],
        ideal_beam, broken_beam)

    return ideal_measured_visibilities, broken_measured_visibilities


def visibility_extractor_OO(baseline_table, sky_cube, frequency, antenna1_response, antenna2_response, padding_factor = 3):
    apparent_sky = sky_cube * antenna1_response * numpy.conj(antenna2_response)

    padded_sky = numpy.pad(apparent_sky, padding_factor * apparent_sky.shape[0], mode="constant")
    shifted_image = numpy.fft.ifftshift(padded_sky, axes=(0, 1))
    visibility_grid, uv_coordinates = powerbox.dft.fft(shifted_image, L=2* (2 * padding_factor + 1), axes=(0, 1))
    measured_visibilities = uv_list_to_baseline_measurements_OO(baseline_table, frequency, visibility_grid, uv_coordinates)

    return measured_visibilities, uv_coordinates

def uv_list_to_baseline_measurements_OO(baseline_table, frequency, visibility_grid, uv_grid):

    u_bin_centers = uv_grid[0]
    v_bin_centers = uv_grid[1]

    # now we have the bin edges we can start binning our baseline table
    # Create an empty array to store our baseline measurements in
    visibility_data = visibility_grid

    real_component = interpolate.RegularGridInterpolator([u_bin_centers, v_bin_centers], numpy.real(visibility_data),
                                                         bounds_error=False, fill_value= 0)
    imag_component = interpolate.RegularGridInterpolator([u_bin_centers, v_bin_centers], numpy.imag(visibility_data),
                                                         bounds_error=False, fill_value= 0)

    visibilities = real_component([baseline_table.u(frequency), baseline_table.v(frequency)]) + \
                   1j*imag_component([baseline_table.u(frequency), baseline_table.v(frequency)])

    return visibilities

if __name__ == "__main__":
    main()