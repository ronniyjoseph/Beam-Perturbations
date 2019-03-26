import numpy
import powerbox

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors as colors

from generaltools import colorbar
from generaltools import symlog_bounds
"""
This file is going to contain all relevant power spectrum functions, i.e data gridding, (frequency tapering), frequency
fft, angular averaging, plotting

"""


class PowerSpectrumData:
    def __init__(self, visibility_data = None, u_coordinate = None, v_coordinate = None, frequency_coordinate = None):
        self.data_raw = visibility_data
        self.u_raw = u_coordinate
        self.v_raw = v_coordinate
        self.f_raw = frequency_coordinate

        self.data_regrid = None
        self.u_regrid = None
        self.v_regrid = None
        self.f_regrid = None
        self.eta = None
        return

    def append_frequency_slice(self, new_data, new_u, new_v, new_frequency):

        if self.data is None:
            self.data = new_data
            self.u = new_u
            self.v = new_v
            self.f = numpy.array([new_frequency])
        else:
            current_data = self.data
            current_u = self.u
            current_v = self.v
            current_f = self.f

            self.data = numpy.vstack((current_data, new_data))
            self.u = numpy.vstack((current_u, new_u))
            self.v = numpy.vstack((current_v, new_v))
            self.f = numpy.vstack((current_f, numpy.array([new_frequency])))
        return

    def regrid_data(self, keep_raw = True):
        return


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


def get_power_spectrum(frequency_range, radio_telescope, ideal_measured_visibilities, broken_measured_visibilities,
                       faulty_tile, plot_file_name,  verbose = False):
    baseline_table = radio_telescope.baseline_table

    # Determine maximum resolution
    max_frequency = frequency_range[-1]
    max_u = numpy.max(numpy.abs(baseline_table.u(max_frequency)))
    max_v = numpy.max(numpy.abs(baseline_table.v(max_frequency)))
    max_b = max(max_u, max_v)

    re_gridding_resolution = 0.5  # lambda
    n_regridded_cells = int(numpy.ceil(2 * max_b / re_gridding_resolution))
    regridded_uv = numpy.linspace(-max_b, max_b, n_regridded_cells)

    if verbose:
        print("Gridding data for Power Spectrum Estimation")

    #Create empty_uvf_cubes:
    ideal_regridded_cube = numpy.zeros((n_regridded_cells,n_regridded_cells, len(frequency_range)), dtype = complex)
    broken_regridded_cube= ideal_regridded_cube.copy()

    ideal_regridded_weights = numpy.zeros((n_regridded_cells,n_regridded_cells, len(frequency_range)))
    broken_regridded_weights= ideal_regridded_weights.copy()

    for frequency_index in range(len(frequency_range)):

        ideal_regridded_cube[..., frequency_index], ideal_regridded_weights[..., frequency_index] = regrid_visibilities(
            ideal_measured_visibilities[:, frequency_index], baseline_table.u(frequency_range[frequency_index]),
            baseline_table.v(frequency_range[frequency_index]), regridded_uv)

        broken_regridded_cube[..., frequency_index], broken_regridded_weights[..., frequency_index] = regrid_visibilities(
            broken_measured_visibilities[:, frequency_index], baseline_table.u(frequency_range[frequency_index]),
            baseline_table.v(frequency_range[frequency_index]), regridded_uv)

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
    power_spectrum_plot(uv_bins, eta_coords[0, selection:], ideal_PS[:, selection:], broken_PS[:, selection:],
                        diff_PS[:, selection:],plot_file_name, faulty_tile)
    return

def power_spectrum_plot(uv_bins, eta_coords, ideal_PS, broken_PS, diff_PS, plot_file_name, faulty_tile = -1, ):
    fontsize = 15
    figure = pyplot.figure(figsize=(30, 10))
    ideal_axes = figure.add_subplot(131)
    broken_axes = figure.add_subplot(132)
    difference_axes = figure.add_subplot(133)

    ideal_plot = ideal_axes.pcolor(uv_bins, eta_coords, numpy.real(ideal_PS.T),
                                   cmap='Spectral_r',
                                   norm=colors.LogNorm(vmin=numpy.nanmin(numpy.real(ideal_PS.T)),
                                                       vmax=numpy.nanmax(numpy.real(ideal_PS.T))))

    broken_plot = broken_axes.pcolor(uv_bins, eta_coords, numpy.real(broken_PS.T),
                                     cmap='Spectral_r',
                                     norm=colors.LogNorm(vmin=numpy.nanmin(numpy.real(broken_PS.T)),
                                                         vmax=numpy.nanmax(numpy.real(broken_PS.T))))

    symlog_min, symlog_max, symlog_threshold, symlog_scale = symlog_bounds(numpy.real(diff_PS))

    diff_plot = difference_axes.pcolor(uv_bins, eta_coords, numpy.real(diff_PS.T),
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

    ideal_axes.set_xlim(5, 200)
    broken_axes.set_xlim(5, 200)
    difference_axes.set_xlim(5, 200)

    ideal_cax = colorbar(ideal_plot)
    broken_cax = colorbar(broken_plot)
    diff_cax = colorbar(diff_plot)
    diff_cax.set_label(r"$[Jy^2]$", fontsize=fontsize)

    figure.savefig("../../Plots/Tile_Beam_Perturbation_Plots/" + plot_file_name)
    return