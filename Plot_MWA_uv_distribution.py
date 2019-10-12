import numpy
import argparse
import matplotlib
from matplotlib import colors

from generaltools import from_eta_to_k_par
from analytic_covariance import gain_error_covariance
from analytic_covariance import blackman_harris_taper
from analytic_covariance import compute_ps_variance
from analytic_covariance import dft_matrix
from analytic_covariance import compute_weights

from radiotelescope import RadioTelescope
from generaltools import from_u_to_k_perp


def main(ssh=False, labelfontsize=13, tickfontsize=11, plot_name="Baseline_Distribution_MWA.pdf"):
    plot_path = "../../Plots/Analytic_Covariance/"
    u_range = numpy.logspace(0, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165, 251) * 1e6


    k_perp = from_u_to_k_perp(u_range, frequency_range[int(len(frequency_range) / 2)])
    x_label = r"$k_{\perp}$ [Mpc$^{-1}$]"

    mwa_position_path = "./Data/MWA_Compact_Coordinates.txt"
    mwa_telescope = RadioTelescope(load=True, path=mwa_position_path)

    log_steps = numpy.diff(numpy.log10(u_range))
    u_bin_edges = numpy.zeros(len(u_range) + 1)
    u_bin_edges[1:] = 10**(numpy.log10(u_range) + 0.5*log_steps[0])
    u_bin_edges[0] = 10**(numpy.log10(u_range[0] - 0.5*log_steps[0]))
    baseline_lengths = numpy.sqrt(mwa_telescope.baseline_table.u_coordinates ** 2 +
                                  mwa_telescope.baseline_table.u_coordinates ** 2)

    counts, edges = numpy.histogram(baseline_lengths, bins=u_bin_edges)
    figure, axes = pyplot.subplots(1, 1, figsize=(5, 5))
    axes.plot(k_perp, counts/len(baseline_lengths))
    axes.set_xscale('log')
    axes.set_xlabel(x_label, fontsize=labelfontsize)
    axes.set_ylabel('Fraction of Baselines', fontsize=labelfontsize)

    axes.tick_params(axis='both', which='major', labelsize=tickfontsize)

    figure.tight_layout()

    figure.savefig(plot_path + plot_name)

    if not ssh:
        print("I am plotting")

        pyplot.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot and compare the sky and beam modelling errors')
    parser.add_argument('-ssh', action='store_true', default=False, help='flag to use when remote plotting')
    args = parser.parse_args()

    if args.ssh:
        matplotlib.use('Agg')
    from matplotlib import pyplot
    main(ssh = args.ssh)