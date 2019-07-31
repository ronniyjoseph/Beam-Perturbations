import sys
import numpy
import powerbox
from matplotlib import pyplot
from radiotelescope import RadioTelescope

from skymodel import SkyRealisation
from radiotelescope import ideal_gaussian_beam
from generaltools import from_lm_to_theta_phi
from generaltools import colorbar
import matplotlib.colors as colors

from scipy.signal import convolve2d

sys.path.append("../")

def main():
    path = "./Data/MWA_Compact_Coordinates.txt"
    plot_folder = "../../Plots/Analytic_Covariance/"
    plot_u_dist = False
    plot_array_matrix = False
    plot_inverse_matrix = False
    plot_weights = False
    grid_weights = True
    telescope = RadioTelescope(load=True, path=path)
    baseline_lengths = numpy.sqrt(telescope.baseline_table.u_coordinates**2 + telescope.baseline_table.v_coordinates**2)

    if plot_u_dist:
        figure_u, axes_u = pyplot.subplots(1,1)
        axes_u.hist(baseline_lengths, density = True, bins = 100, label = "MWA Phase II Compact")
        axes_u.set_xlabel(r"$u\,[\lambda]$")
        axes_u.set_ylabel("Baseline PDF")
        axes_u.legend()
        figure_u.savefig(plot_folder + "MWA_Phase_II_Baseline_PDF.pdf")

    array_matrix = matrix_constructor_alternate(telescope)
    #
    # pyplot.rcParams['xtick.bottom'] = pyplot.rcParams['xtick.labelbottom'] = False
    # pyplot.rcParams['xtick.top'] = pyplot.rcParams['xtick.labeltop'] = True
    if plot_array_matrix:
        figure_amatrix = pyplot.figure(figsize=(250, 10))
        axes_amatrix = figure_amatrix.add_subplot(111)
        plot_amatrix = axes_amatrix.imshow(array_matrix.T, origin = 'lower')
        colorbar(plot_amatrix)
        axes_amatrix.set_xlabel("Baseline Number", fontsize = 20)
        axes_amatrix.set_ylabel("Antenna Number", fontsize = 20)
        figure_amatrix.savefig(plot_folder + "Array_Matrix_Double.pdf")


    inverse_array_matrix = numpy.linalg.pinv(array_matrix)
    if plot_inverse_matrix:
        figure_inverse = pyplot.figure(figsize = (110, 20))
        axes_inverse = figure_inverse.add_subplot(111)
        plot_inverse = axes_inverse.imshow(numpy.abs(inverse_array_matrix))
        colorbar(plot_inverse)

    baseline_weights = numpy.sqrt((numpy.abs(inverse_array_matrix[::2, ::2])**2 + numpy.abs(inverse_array_matrix[1::2, 1::2])**2))
    # baseline_weights = numpy.sqrt(numpy.abs(inverse_array_matrix[:int(len(telescope.antenna_positions.antenna_ids) - 1), :int(len(baseline_lengths))])**2 + \
    #                    numpy.abs(inverse_array_matrix[int(len(telescope.antenna_positions.antenna_ids) -1 ):, :int(len(baseline_lengths)):])**2)
    if plot_weights:
        figure_weights, axes_weights = pyplot.subplots(1,1)
        weights_plot = axes_weights.imshow(baseline_weights)
        axes_weights.set_title("Antenna Baseline Weights")
        colorbar(weights_plot)

    if grid_weights:
        u_u_weights = numpy.zeros((len(baseline_lengths), len(baseline_lengths)))
        baselines = telescope.baseline_table
        antennas = telescope.antenna_positions.antenna_ids
        for i in range(len(baseline_lengths)):
            index1 = numpy.where(antennas == baselines.antenna_id1[i])[0]
            index2 = numpy.where(antennas == baselines.antenna_id2[i])[0]

            if index1 == 0:
                baseline_weights1 = 0
            else:
                baseline_weights1 = baseline_weights[index1 - 1, :]

            if index2 == 0:
                baseline_weights2 = 0
            else:
                baseline_weights2 = baseline_weights[index2 - 1, :]
            u_u_weights[i, :] = (baseline_weights1 + baseline_weights2)/2

        u_bins = numpy.linspace(0, numpy.max(baseline_lengths), 101)


        sorted_indices = numpy.argsort(baseline_lengths)
        sorted_weights = u_u_weights[sorted_indices, :][:, sorted_indices]
        fig_cal, axes_cal = pyplot.subplots(1,2)
        cal_plot = axes_cal[0].imshow(u_u_weights, origin = 'lower', interpolation = 'none')
        axes_cal[0].set_xlabel("Uncalibrated Baseline Index")
        axes_cal[0].set_ylabel("Calibrated Baseline Index")
        axes_cal[0].set_title("Baseline-Baseline' Weights Quadrature")
        colorbar(cal_plot)


        bin_indices = numpy.digitize(baseline_lengths[sorted_indices], u_bins)

        sorted_plot = axes_cal[1].imshow(sorted_weights, interpolation='none', origin='lower')
        #axes_cal[1].set_xlabel("Uncalibrated Baseline Index")
        #axes_cal[1].set_ylabel("Calibrated Baseline Index")
        axes_cal[1].set_title(" Sorted Baseline-Baseline' Weights")
        colorbar(sorted_plot)

        uu1, uu2 = numpy.meshgrid(baseline_lengths, baseline_lengths)
        flattened_weights = u_u_weights.flatten()
        flattened_uu1 = uu1.flatten()
        flattened_uu2 = uu2.flatten()

        computed_weights = numpy.histogram2d(flattened_uu1, flattened_uu2,  bins = u_bins ,
                                                                 weights = flattened_weights)
        bin_counter = numpy.zeros_like(computed_weights)
        bin_counter += 1e-10
        bin_counter[computed_weights != 0] = 2

        computed_counts = numpy.histogram2d(flattened_uu1, flattened_uu2,  bins = u_bins ,
                                                                 weights = bin_counter.flatten() )

        figure_uu, axes_uu = pyplot.subplots(1,1)
        norm = colors.LogNorm(vmax= 1e-3 )
        weights_plot = axes_uu.pcolor(u_bins, u_bins, computed_counts[0])
        # weights_plot = axes_uu.imshow(binned_weights, origin = 'lower', norm = norm)

        cbar_uu = colorbar(weights_plot)
        axes_uu.set_title(r"Computed Weights")
        axes_uu.set_xlabel(r"$u\,[\lambda]$")
        axes_uu.set_ylabel(r"$u^{\prime}\,[\lambda]$")
        cbar_uu.set_label("Poorly Defined Weights")
        figure_uu.savefig(plot_folder + "Baseline_Weights_uu.pdf")

        baseline_pdf = numpy.histogram(baseline_lengths, bins = u_bins, density = True)
        ww1, ww2 = numpy.meshgrid(baseline_pdf[0], baseline_pdf[0])
        approx_weights = ww1*ww2
        figure_approx, axes_approx = pyplot.subplots(1,1)
        dirty_plot = axes_approx.pcolor(u_bins, u_bins , approx_weights)


        # dirty_plot = axes_dirty.imshow(ww1*ww2, origin = 'lower', norm=norm)

        cbar_dirty = colorbar(dirty_plot)
        axes_approx.set_title(r"Approximated Weights")
        axes_approx.set_xlabel(r"$u\,[\lambda]$")
        axes_approx.set_ylabel(r"$u^{\prime}\,[\lambda]$")
        cbar_dirty.set_label("Poorly Defined Weights")

        fig, ax = pyplot.subplots(1,1)
        blaah = ax.pcolor(u_bins, u_bins, computed_weights[0] -approx_weights)
        ax.set_xlabel(r"$u\,[\lambda]$")
        ax.set_ylabel(r"$u^{\prime}\,[\lambda]$")
        colorbar(blaah)

        figure_approx.savefig(plot_folder + "Baseline_Weights_uu_from_PDF.pdf")
    pyplot.show()
    return


def matrix_constructor_alternate(telescope):
    antennas = telescope.antenna_positions.antenna_ids
    baselines = telescope.baseline_table
    array_matrix = numpy.zeros((2 * baselines.number_of_baselines, 2 * len(antennas)))

    for i in range(baselines.number_of_baselines):
        index1 = numpy.where(antennas == baselines.antenna_id1[i])[0]
        index2 = numpy.where(antennas == baselines.antenna_id2[i])[0]

        # Fill in the real rows
        array_matrix[2 * i, 2 * index1] = 1
        array_matrix[2 * i, 2 * index2] = 1

        # Fill in the imaginary rows

        array_matrix[2 * i + 1, 2 * index1 + 1] = 1
        array_matrix[2 * i + 1, 2 * index2 + 1] = -1

    constrained_matrix = array_matrix[:, 2:]
    return constrained_matrix

def matrix_constructor_sep(telescope):
    antennas = telescope.antenna_positions.antenna_ids
    baselines = telescope.baseline_table
    array_matrix = numpy.zeros((2 * baselines.number_of_baselines, 2 * len(antennas)))

    for i in range(baselines.number_of_baselines):
        index1 = numpy.where(antennas == baselines.antenna_id1[i])[0]
        index2 = numpy.where(antennas == baselines.antenna_id2[i])[0]

        # Fill in the real rows
        array_matrix[2 * i, 2 * index1] = 1
        array_matrix[2 * i, 2 * index2] = 1

        # Fill in the imaginary rows

        array_matrix[2 * i + 1, 2 * index1 + 1] = 1
        array_matrix[2 * i + 1, 2 * index2 + 1] = -1

    return array_matrix


def matrix_constructor_sep_shift(telescope):
    antennas = telescope.antenna_positions.antenna_ids
    baselines = telescope.baseline_table
    array_matrix = numpy.zeros((2 * baselines.number_of_baselines, 2 * len(antennas)))

    for i in range(baselines.number_of_baselines):
        index1 = numpy.where(antennas == baselines.antenna_id1[i])[0]
        index2 = numpy.where(antennas == baselines.antenna_id2[i])[0]

        # Fill in the real rows
        array_matrix[i, index1] = 1
        array_matrix[i, index2] = 1

        # Fill in the imaginary rows

        array_matrix[baselines.number_of_baselines + i, len(antennas) + index1] = 1
        array_matrix[baselines.number_of_baselines + i, len(antennas) + index2] = -1

    return array_matrix


def matrix_constructor_double(telescope):
    antennas = telescope.antenna_positions.antenna_ids
    baselines = telescope.baseline_table
    array_matrix = numpy.zeros((baselines.number_of_baselines, 2 * len(antennas)))

    for i in range(baselines.number_of_baselines):
        index1 = numpy.where(antennas == baselines.antenna_id1[i])[0]
        index2 = numpy.where(antennas == baselines.antenna_id2[i])[0]

        # Fill in the real rows
        array_matrix[i, index1] = 1
        array_matrix[i, len(antennas) + index2] = 1

    return array_matrix


if __name__ == "__main__":
    main()