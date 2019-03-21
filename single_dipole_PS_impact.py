import numpy
import powerbox
from matplotlib import pyplot
import matplotlib.colors as colors
from scipy.constants import c
from scipy import interpolate
from scipy.integrate import quad
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import sys
sys.path.append('../../../redundant_calibration/code/SCAR')
from RadioTelescope import antenna_gain_creator
from RadioTelescope import baseline_converter
from RadioTelescope import antenna_table_loader
from skymodel import SkyRealisation
from quick_simulation_visibility_covariance import lm_to_theta_phi
from quick_simulation_visibility_covariance import mwa_tile_beam
from scipy.constants import c

from generaltools import colorbar
import time
"""
We calculate the power spectrum for the MWA, when 1 dipole is offline in the array, in the presence of a stochastic 
foreground of point sources.
"""

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

    if verbose:
        print("Creating Radio Telescope")

    #Create Radio Telescope
    #####################################################################
    xyz_positions = antenna_table_loader(path)
    gain_table = antenna_gain_creator(xyz_positions, frequency_range)
    baseline_table = baseline_converter(xyz_positions, gain_table, frequency_range, verbose=verbose)
    #####################################################################

    if verbose:
        print("Creating the source population")
    #Create Sky
    ######################################################################
    source_population = SkyRealisation(sky_type="random")
    sky_cube, l_coordinates = source_population.create_sky_image(frequency_channels= frequency_range,
                                                                 baseline_table = baseline_table)
    ll, mm, ff = numpy.meshgrid(l_coordinates, l_coordinates, frequency_range)
    ############################################################################
    print(f"The size of the sky is {sky_cube.shape}")


    if verbose:
        print("Calculating beam patterns")
    #Create Beam
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
    ##################################################################


    if verbose:
        print("Generating observed visibilities")
    #Create visibilities
    ##############################################################################
    ideal_measured_visibilities = numpy.zeros((baseline_table.shape[0], len(frequency_range)), dtype = complex)
    broken_measured_visibilities= ideal_measured_visibilities.copy()

    ##### Select perfect baselines #####
    perfect_baseline_indices = numpy.where((baseline_table[:, 0, 0] != faulty_tile) &
                                           (baseline_table[:, 1, 0] != faulty_tile))[0]
    broken_baseline_indices = numpy.where((baseline_table[:, 0, 0] == faulty_tile) |
                                          (baseline_table[:, 1, 0] == faulty_tile))[0]

    print(f"there are broken {len(broken_baseline_indices)} baselines")

    if verbose:
        print(" Generating Visbilities for each frequencies")
    for frequency_index in range(len(frequency_range)):
        #Sample all baselines, and get the relevant uv_grid coordinates
        ideal_measured_visibilities[...,frequency_index], full_uv_grid = visibility_extractor(
            baseline_table[...,frequency_index], sky_cube[...,frequency_index], ideal_beam[...,frequency_index],
            ideal_beam[...,frequency_index])

        #Copy good baselines to broken table
        broken_measured_visibilities[perfect_baseline_indices, frequency_index] = ideal_measured_visibilities[
            perfect_baseline_indices, frequency_index]

        broken_measured_visibilities[broken_baseline_indices, frequency_index], partial_uv_grid = visibility_extractor(
            baseline_table[broken_baseline_indices, :, frequency_index], sky_cube[...,frequency_index],
            ideal_beam[...,frequency_index], broken_beam[...,frequency_index])
    ############################################################################################

    ###Get Power Spectrum
    ############################################################################################
    if verbose:
        print("Gridding data for Power Spectrum Estimation")
    #Create empty_uvf_cubes:
    print(numpy.max(full_uv_grid[0]),numpy.min(full_uv_grid[0]) )
    print(numpy.max(baseline_table[:,2:4,:]), numpy.min(baseline_table[:,2:4,:]))

    re_gridding_resolution = 0.5 #lambda
    n_regridded_cells = int(numpy.ceil((numpy.max(full_uv_grid[0]) - numpy.min(full_uv_grid[0]))/re_gridding_resolution))
    regridded_u_coordinates = numpy.linspace(numpy.min(full_uv_grid[0]), numpy.max(full_uv_grid[0]), n_regridded_cells)

    print(numpy.min(regridded_u_coordinates), numpy.max(regridded_u_coordinates) )

    ideal_regridded_vis = numpy.zeros((n_regridded_cells, n_regridded_cells, len(frequency_range)), dtype=complex)
    broken_regridded_vis= ideal_regridded_vis.copy()

    ideal_weights = numpy.zeros((n_regridded_cells, n_regridded_cells, len(frequency_range)))
    broken_weights = ideal_weights

    #Regridding the visibilities
    for frequency_index in range(len(frequency_range)):

        ideal_regridded_vis[...,frequency_index], ideal_weights[...,frequency_index] = regrid_visibilities(
            ideal_measured_visibilities[:, frequency_index],
                                                                        baseline_table[:, 2, frequency_index],
                                                                        baseline_table[:, 2, frequency_index],
                                                                        regridded_u_coordinates)

        broken_regridded_vis[..., frequency_index], broken_weights[..., frequency_index] = regrid_visibilities(
            broken_measured_visibilities[:, frequency_index],
                                                                        baseline_table[:, 2, frequency_index],
                                                                        baseline_table[:, 2, frequency_index],
                                                                        regridded_u_coordinates)

        if calibration:
            broken_regridded_vis[..., frequency_index] *= calibration_correction(faulty_dipole,
                                                                                 frequency_range[frequency_index])

    return regridded_u_coordinates, ideal_regridded_vis, ideal_weights, broken_regridded_vis, broken_weights

    """
    #visibilities have now been re-gridded
    if verbose:
        print("Taking Fourier Transform over frequency and averaging")
    ideal_shifted = numpy.fft.ifftshift(ideal_regridded_vis, axes=2)
    broken_shifted = numpy.fft.ifftshift(broken_regridded_vis, axes=2)

    ideal_uvn, eta_coords = powerbox.dft.fft(ideal_shifted, L=numpy.max(frequency_range) - numpy.min(frequency_range), axes = (2,) )
    broken_uvn, eta_coords = powerbox.dft.fft(broken_shifted, L=numpy.max(frequency_range) - numpy.min(frequency_range), axes = (2,) )

    ideal_PS, uv_bins = powerbox.tools.angular_average_nd(numpy.abs(ideal_uvn)**2,
                                                          coords = [regridded_u_coordinates,regridded_u_coordinates, eta_coords], bins = 50,
                                                          n = 2, weights=numpy.sum(ideal_weights, axis=2))
    broken_PS, uv_bins = powerbox.tools.angular_average_nd(numpy.abs(broken_uvn)**2,
                                                           coords = [regridded_u_coordinates,regridded_u_coordinates, eta_coords], bins = 50,
                                                           n = 2, weights=numpy.sum(broken_weights, axis=2))

    diff_PS = ideal_PS - broken_PS
    selection = int(len(eta_coords[0])/2) + 1


    k_perpendicular = u_to_k_perpendicular(uv_bins, frequency_range)
    k_parallel = eta_to_k_parallel(eta_coords[0, selection:], frequency_range)
    if verbose:
        print("Plotting")
    fontsize = 15
    figure = pyplot.figure(figsize=(40,8))
    ideal_axes = figure.add_subplot(131)
    broken_axes = figure.add_subplot(132)
    difference_axes = figure.add_subplot(133)

    ideal_plot = ideal_axes.pcolor(uv_bins, eta_coords[0, selection:], numpy.real(ideal_PS[:, selection:].T),
                                   cmap = 'Spectral_r',
                                   norm = colors.LogNorm(vmin = numpy.nanmin(numpy.real(ideal_PS[:, selection:].T)),
                                                         vmax = numpy.nanmax(numpy.real(ideal_PS[:, selection:].T))))



    broken_plot = broken_axes.pcolor(uv_bins, eta_coords[0, selection:], numpy.real(broken_PS[:, selection:].T),
                                     cmap = 'Spectral_r',
                                     norm=colors.LogNorm(vmin=numpy.nanmin(numpy.real(broken_PS[:, selection:].T)),
                                                         vmax=numpy.nanmax(numpy.real(broken_PS[:, selection:].T))))

    symlog_min, symlog_max, symlog_threshold, symlog_scale = symlog_bounds(numpy.real(diff_PS[:, selection:]))

    diff_plot = difference_axes.pcolor(uv_bins, eta_coords[0, selection:], numpy.real(diff_PS[:, selection:].T),
                                       norm=colors.SymLogNorm(linthresh=symlog_threshold, linscale=symlog_scale,
                                        vmin= symlog_min, vmax= symlog_max), cmap = 'coolwarm')


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

    ideal_axes.set_xlabel(x_labeling, fontsize = fontsize)
    ideal_axes.set_ylabel(y_labeling, fontsize = fontsize)

    broken_axes.set_xlabel(x_labeling, fontsize = fontsize)

    difference_axes.set_xlabel(x_labeling, fontsize = fontsize)

    figure.suptitle(f"Tile {faulty_tile}")
    #ideal_axes.set_xlim(10**-2.5, 10**-0.5)
    #broken_axes.set_xlim(10**-2.5, 10**-0.5)
    #difference_axes.set_xlim(10**-2.5, 10**-0.5)


    ideal_cax = colorbar(ideal_plot)
    broken_cax = colorbar(broken_plot)
    diff_cax = colorbar(diff_plot)
    diff_cax.set_label(r"$[Jy^2]$", fontsize = fontsize)

    pyplot.show()

    return
    """

def symlog_bounds(data):
    data_min = numpy.nanmin(data)
    data_max = numpy.nanmax(data)

    if data_min == 0:
        indices = numpy.where(data > 0)[0]
        if len(indices) == 0:
            lower_bound = -0.1
        else:
            lower_bound = numpy.nanmin(data[indices])
    else:
        lower_bound = numpy.abs(data_min)/data_min*numpy.abs(data_min)

    if data_max == 0:
        indices = numpy.where(data < 0)[0]
        if len(indices) == 0:
            upper_bound = 0.1
        else:
            upper_bound = numpy.nanmax(data[indices])
    else:
        upper_bound = numpy.abs(data_max)/data_max*numpy.abs(data_max)

    ### Figure out what the lintresh is (has to be linear)
    threshold = 1e-3*min(numpy.abs(lower_bound), numpy.abs(upper_bound))
    #### Figure out the linscale parameter (has to be in log)
    scale = numpy.log10(upper_bound - lower_bound)/7
    return lower_bound, upper_bound, threshold, scale



def u_to_k_perpendicular(uv_bins, frequency_range):

    distance_comoving = comoving_distance(frequency_range[int(len(frequency_range)/2)])
    k_perpendicular = 2*numpy.pi*uv_bins/distance_comoving

    return k_perpendicular

def eta_to_k_parallel(eta, frequency_range, H_0 = 70.4, rest_frequency_21cm = 1.420e9):
    z = redshift(frequency_range[int(len(frequency_range)/2)])
    k_parallel = 2*numpy.pi*H_0*1000*rest_frequency_21cm*E(z)*eta/(c*(1 + z)**2.)

    return k_parallel


def E(z, omega_M = 0.27, omega_k = 0, omega_Lambda = 0.73):
    return numpy.sqrt(omega_M*(1+z)**2 + omega_k*(1+z)**2 + omega_Lambda)

def comoving_distance(frequency,  H_0 = 70.4):
    z = redshift(frequency)
    z_range = numpy.linspace(0, z, 100)

    y = E(z_range)
    distance = c/(1000*H_0)*numpy.trapz(1/y, z_range)
    print(distance)
    return distance


def redshift(observed_frequency, rest_frequency_21cm = 1.420e9):

    z = (rest_frequency_21cm - observed_frequency)/observed_frequency
    return z


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


def visibility_extractor(baseline_table, sky_cube, antenna1_response, antenna2_response, padding_factor = 3):
    apparent_sky = sky_cube * antenna1_response * numpy.conj(antenna2_response)

    padded_sky = numpy.pad(apparent_sky, padding_factor * apparent_sky.shape[0], mode="constant")
    shifted_image = numpy.fft.ifftshift(padded_sky, axes=(0, 1))
    visibility_grid, uv_coordinates = powerbox.dft.fft(shifted_image, L=2* (2 * padding_factor + 1), axes=(0, 1))
    measured_visibilities = uv_list_to_baseline_measurements(baseline_table, visibility_grid, uv_coordinates)

    return measured_visibilities, uv_coordinates

def uv_list_to_baseline_measurements(baseline_table, visibility_grid, uv_grid):

    u_bin_centers = uv_grid[0]
    v_bin_centers = uv_grid[1]

    # now we have the bin edges we can start binning our baseline table
    # Create an empty array to store our baseline measurements in
    visibility_data = visibility_grid

    real_component = interpolate.RegularGridInterpolator([u_bin_centers, v_bin_centers], numpy.real(visibility_data))#,
                                                         #bounds_error=False, fill_value= 0)
    imag_component = interpolate.RegularGridInterpolator([u_bin_centers, v_bin_centers], numpy.imag(visibility_data))#,
                                                         #bounds_error=False, fill_value= 0)

    visibilities = real_component(baseline_table[:, 2:4]) + 1j*imag_component(baseline_table[:, 2:4])

    return visibilities


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

def calibration_correction(faulty_dipole, nu, dx = 1.1):
    wavelength = c / nu
    x_offsets = numpy.array([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5, -1.5,
                             -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5], dtype=numpy.float32) * dx

    y_offsets = numpy.array([1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5,
                             -0.5, -0.5, -1.5, -1.5, -1.5, -1.5], dtype=numpy.float32) * dx

    correction = -16/15*wavelength**2./(4*numpy.pi**2*x_offsets[faulty_dipole]*y_offsets[faulty_dipole])*\
                 (numpy.exp(2*numpy.pi*1j*(x_offsets[faulty_dipole] + y_offsets[faulty_dipole]) / wavelength) -
                  numpy.exp(-2*numpy.pi*1j*(-x_offsets[faulty_dipole] + y_offsets[faulty_dipole])/wavelength) -
                  numpy.exp(-2*numpy.pi*1j*(x_offsets[faulty_dipole] - y_offsets[faulty_dipole])/wavelength) +
                  numpy.exp(2*numpy.pi*1j*(x_offsets[faulty_dipole] + y_offsets[faulty_dipole])/wavelength))

    return correction

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




if __name__ == "__main__":
    start = time.clock()
    main()
    end = time.clock()
    print(f"Time is {end-start}")