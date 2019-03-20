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


    ideal_powerspectrum = PowerSpectrumData()
    broken_powerspectrum = PowerSpectrumData()

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

        ##Determine the indices of the broken baselines and calulcate the visibility measurements
        ##################################################################

        broken_baseline_indices = numpy.where((baseline_table.antenna_id1 == faulty_tile) |
                                              (baseline_table.antenna_id2 == faulty_tile))[0]

        ideal_measured_visibilities = visibility_extractor(baseline_table, sky_cube, frequency_range[frequency_index],
                                                           ideal_beam, ideal_beam)

        broken_measured_visibilities = ideal_measured_visibilities.copy()
        broken_measured_visibilities[broken_baseline_indices] = visibility_extractor(
            baseline_table.sub_table(broken_baseline_indices), sky_cube, frequency_range[frequency_index],
            ideal_beam, broken_beam)

        ideal_powerspectrum.append_frequency_slice(new_data=ideal_measured_visibilities,
                                                   new_u=baseline_table.u(frequency_range[frequency_index]),
                                                   new_v=baseline_table.v(frequency_range[frequency_index]),
                                                   new_frequency=frequency_range[frequency_index])

        broken_powerspectrum.append_frequency_slice(new_data=broken_measured_visibilities,
                                                   new_u=baseline_table.u(frequency_range[frequency_index]),
                                                   new_v=baseline_table.v(frequency_range[frequency_index]),
                                                   new_frequency=frequency_range[frequency_index])

    return

def visibility_extractor(baseline_table_object, sky_image, frequency, antenna1_response,
                            antenna2_response, padding_factor = 3):

    apparent_sky = sky_image * antenna1_response * numpy.conj(antenna2_response)

    padded_sky = numpy.pad(apparent_sky, padding_factor * apparent_sky.shape[0], mode="constant")
    shifted_image = numpy.fft.ifftshift(padded_sky, axes=(0, 1))
    visibility_grid, uv_coordinates = powerbox.dft.fft(shifted_image, L=2*(2 * padding_factor + 1), axes=(0, 1))


    measured_visibilities = uv_list_to_baseline_measurements(baseline_table_object, frequency, visibility_grid,
                                                             uv_coordinates)

    return measured_visibilities

def uv_list_to_baseline_measurements(baseline_table_object, frequency, visibility_grid, uv_grid):

    u_bin_centers = uv_grid[0]
    v_bin_centers = uv_grid[1]

    baseline_coordinates = numpy.array([baseline_table_object.u(frequency), baseline_table_object.v(frequency)])
    # now we have the bin edges we can start binning our baseline table
    # Create an empty array to store our baseline measurements in
    visibility_data = visibility_grid

    real_component = interpolate.RegularGridInterpolator([u_bin_centers, v_bin_centers], numpy.real(visibility_data),
                                                         bounds_error=False, fill_value= 0)
    imag_component = interpolate.RegularGridInterpolator([u_bin_centers, v_bin_centers], numpy.imag(visibility_data),
                                                         bounds_error=False, fill_value= 0)

    visibilities = real_component(baseline_coordinates.T) + 1j*imag_component(baseline_coordinates.T)

    return visibilities

def regrid_visibilities(measured_visibilities, baseline_u, baseline_v, u_grid):
    u_shifts = numpy.diff(u_grid) / 2.

    u_bin_edges = numpy.concatenate((numpy.array([u_grid[0] - u_shifts[0]]), u_grid[1:] - u_shifts,
                                     numpy.array([u_grid[-1] + u_shifts[-1]])))

    weights_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u,
                                                     baseline_v,
                                                     bins=(u_bin_edges, u_bin_edges))

    real_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u,
                                                     baseline_v,
                                                     bins=(u_bin_edges, u_bin_edges),
                                                     weights=
                                                     numpy.real(measured_visibilities))

    imag_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u,
                                                     baseline_v,
                                                     bins=(u_bin_edges, u_bin_edges),
                                                     weights=
                                                     numpy.imag(measured_visibilities))

    regridded_visibilities = real_regrid + 1j*imag_regrid
    return regridded_visibilities, weights_regrid


if __name__ == "__main__":
    main()

