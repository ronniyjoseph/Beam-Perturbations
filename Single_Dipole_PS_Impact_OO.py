import numpy
import powerbox
import sys
import time

from matplotlib import pyplot
import matplotlib.colors as colors

from scipy.constants import c
from scipy import interpolate

from generaltools import colorbar
from generaltools import symlog_bounds

from skymodel import SkyRealisation
from radiotelescope import RadioTelescope
from radiotelescope import ideal_gaussian_beam
from radiotelescope import broken_gaussian_beam

from powerspectrum import PowerSpectrumData
from powerspectrum import regrid_visibilities
def main(verbose=True):

    path = "./HexCoords_Luke.txt"
    frequency_range = numpy.linspace(135, 165, 2) * 1e6
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
    max_u = numpy.max(numpy.abs(baseline_table.u(max_frequency)))
    max_v = numpy.max(numpy.abs(baseline_table.v(max_frequency)))
    max_b = max(max_u, max_v)
    # sky_resolutions
    min_l = 1. / (2*max_b)

    re_gridding_resolution = 0.5  # lambda
    n_regridded_cells = int(numpy.ceil(2*max_b/re_gridding_resolution))
    regridded_uv = numpy.linspace(-max_b, max_b, n_regridded_cells)


    ideal_measured_visibilities = numpy.zeros((baseline_table.number_of_baselines, len(frequency_range)), dtype = complex)
    broken_measured_visibilities= ideal_measured_visibilities.copy()

    ideal_regridded_cube = numpy.zeros((n_regridded_cells, n_regridded_cells, len(frequency_range)), dtype = complex)
    broken_regridded_cube= ideal_regridded_cube.copy()

    ideal_regridded_weights = numpy.zeros((n_regridded_cells, n_regridded_cells, len(frequency_range)))
    broken_regridded_weights = ideal_regridded_weights.copy()

    for frequency_index in range(len(frequency_range)):
        sky_image, l_coordinates = source_population.create_sky_image(
            frequency_channels = frequency_range[frequency_index], resolution=min_l, oversampling=1)
        ll, mm = numpy.meshgrid(l_coordinates, l_coordinates)

        # Create Beam
        #############################################################################
        if beam_type == "MWA":
            tt, pp, = lm_to_theta_phi(ll, mm)
            ideal_beam = ideal_mwa_beam_loader(tt, pp, frequency_range[frequency_index], load)
            broken_beam = broken_mwa_beam_loader(tt, pp, frequency_range[frequency_index], faulty_dipole, load)

        elif beam_type == "gaussian":
            ideal_beam = ideal_gaussian_beam(ll, mm, frequency_range[frequency_index])
            broken_beam = broken_gaussian_beam(faulty_dipole, ideal_beam, ll, mm, frequency_range[frequency_index])
        else:
            raise ValueError("The only valid option for the beam are 'MWA' or 'gaussian'")

        ##Determine the indices of the broken baselines and calulcate the visibility measurements
        ##################################################################

        broken_baseline_indices = numpy.where((baseline_table.antenna_id1 == faulty_tile) |
                                              (baseline_table.antenna_id2 == faulty_tile))[0]

        ideal_measured_visibilities[:, frequency_index] = visibility_extractor(baseline_table, sky_image, frequency_range[frequency_index],
                                                           ideal_beam, ideal_beam)

        broken_measured_visibilities[:, frequency_index] = ideal_measured_visibilities[:, frequency_index].copy()
        broken_measured_visibilities[broken_baseline_indices, frequency_index] = visibility_extractor(
            baseline_table.sub_table(broken_baseline_indices), sky_image, frequency_range[frequency_index],
            ideal_beam, broken_beam)

    if verbose:
        print("Gridding data for Power Spectrum Estimation")
    #Create empty_uvf_cubes:
    print(numpy.max(regridded_uv),numpy.min(regridded_uv))

    for frequency_index in range(len(frequency_range)):

        ideal_regridded_cube[..., frequency_index], ideal_regridded_weights[..., frequency_index] = regrid_visibilities(
            ideal_measured_visibilities[:, frequency_index], baseline_table.u(frequency_range[frequency_index]),
            baseline_table.v(frequency_range[frequency_index]), regridded_uv)

        broken_regridded_cube[..., frequency_index], broken_regridded_weights[..., frequency_index] = regrid_visibilities(
            broken_measured_visibilities[:, frequency_index], baseline_table.u(frequency_range[frequency_index]),
            baseline_table.v(frequency_range[frequency_index]), regridded_uv)

    return regridded_uv, ideal_regridded_cube, ideal_regridded_weights, broken_regridded_cube, broken_regridded_weights
    """
    # visibilities have now been re-gridded
    if verbose:
        print("Taking Fourier Transform over frequency and averaging")
    ideal_shifted = numpy.fft.ifftshift(ideal_regridded_cube, axes=2)
    broken_shifted = numpy.fft.ifftshift(broken_regridded_cube, axes=2)

    ideal_uvn, eta_coords = powerbox.dft.fft(ideal_shifted,
                                             L=numpy.max(frequency_range) - numpy.min(frequency_range), axes=(2,))
    broken_uvn, eta_coords = powerbox.dft.fft(broken_shifted,
                                              L=numpy.max(frequency_range) - numpy.min(frequency_range), axes=(2,))

    ideal_PS, uv_bins = powerbox.tools.angular_average_nd(numpy.abs(ideal_uvn) ** 2,
                                                          coords=[regridded_uv, regridded_uv,
                                                                  eta_coords], bins=50,
                                                          n=2, weights=numpy.sum(ideal_regridded_weights, axis=2))
    broken_PS, uv_bins = powerbox.tools.angular_average_nd(numpy.abs(broken_uvn) ** 2,
                                                           coords=[regridded_uv, regridded_uv,
                                                                   eta_coords], bins=50,
                                                           n=2, weights=numpy.sum(broken_regridded_weights, axis=2))

    diff_PS = ideal_PS - broken_PS
    selection = int(len(eta_coords[0]) / 2) + 1

    if verbose:
        print("Making 2D PS Plots")
    fontsize = 15
    figure = pyplot.figure(figsize=(40, 8))
    ideal_axes = figure.add_subplot(131)
    broken_axes = figure.add_subplot(132)
    difference_axes = figure.add_subplot(133)

    ideal_plot = ideal_axes.pcolor(uv_bins, eta_coords[0, selection:], numpy.real(ideal_PS[:, selection:].T),
                                   cmap='Spectral_r',
                                   norm=colors.LogNorm(vmin=numpy.nanmin(numpy.real(ideal_PS[:, selection:].T)),
                                                       vmax=numpy.nanmax(numpy.real(ideal_PS[:, selection:].T))))

    broken_plot = broken_axes.pcolor(uv_bins, eta_coords[0, selection:], numpy.real(broken_PS[:, selection:].T),
                                     cmap='Spectral_r',
                                     norm=colors.LogNorm(vmin=numpy.nanmin(numpy.real(broken_PS[:, selection:].T)),
                                                         vmax=numpy.nanmax(numpy.real(broken_PS[:, selection:].T))))

    symlog_min, symlog_max, symlog_threshold, symlog_scale = symlog_bounds(numpy.real(diff_PS[:, selection:]))

    diff_plot = difference_axes.pcolor(uv_bins, eta_coords[0, selection:], numpy.real(diff_PS[:, selection:].T),
                                       norm=colors.SymLogNorm(linthresh=symlog_threshold, linscale=symlog_scale,
                                                              vmin=symlog_min, vmax=symlog_max), cmap='coolwarm')

    ideal_axes.set_xscale("log")
    ideal_axes.set_yscale("log")

    broken_axes.set_xscale("log")
    broken_axes.set_yscale("log")

    difference_axes.set_xscale("log")
    difference_axes.set_yscale("log")

    x_labeling = r"$ k_{\perp} \, [\mathrm{h}\,\mathrm{Mpc}^{-1}]$"
    y_labeling = r"$k_{\parallel} $"

    x_labeling = r"$ |u |$"
    y_labeling = r"$ \eta $"

    ideal_axes.set_xlabel(x_labeling, fontsize=fontsize)
    ideal_axes.set_ylabel(y_labeling, fontsize=fontsize)

    broken_axes.set_xlabel(x_labeling, fontsize=fontsize)

    difference_axes.set_xlabel(x_labeling, fontsize=fontsize)

    figure.suptitle(f"Tile {faulty_tile}")
    # ideal_axes.set_xlim(10**-2.5, 10**-0.5)
    # broken_axes.set_xlim(10**-2.5, 10**-0.5)
    # difference_axes.set_xlim(10**-2.5, 10**-0.5)

    #ideal_cax = colorbar(ideal_plot)
    #broken_cax = colorbar(broken_plot)
    #diff_cax = colorbar(diff_plot)
    #diff_cax.set_label(r"$[Jy^2]$", fontsize=fontsize)

    pyplot.show()

    return
"""
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

    real_component = interpolate.RegularGridInterpolator([u_bin_centers, v_bin_centers], numpy.real(visibility_data))
    imag_component = interpolate.RegularGridInterpolator([u_bin_centers, v_bin_centers], numpy.imag(visibility_data))

    visibilities = real_component(baseline_coordinates.T) + 1j*imag_component(baseline_coordinates.T)

    return visibilities

def regrid_visibilities(measured_visibilities, baseline_u, baseline_v, u_grid):
    u_shifts = numpy.diff(u_grid) / 2.

    u_bin_edges = numpy.concatenate((numpy.array([u_grid[0] - u_shifts[0]]), u_grid[1:] - u_shifts,
                                     numpy.array([u_grid[-1] + u_shifts[-1]])))
    weights_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u, baseline_v, bins=(u_bin_edges, u_bin_edges))

    real_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u, baseline_v, bins=(u_bin_edges, u_bin_edges),
                                                     weights=numpy.real(measured_visibilities))

    imag_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u, baseline_v, bins=(u_bin_edges, u_bin_edges),
                                                     weights=numpy.imag(measured_visibilities))

    regridded_visibilities = real_regrid + 1j*imag_regrid
    return regridded_visibilities, weights_regrid


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print("Total time is", end - start)
