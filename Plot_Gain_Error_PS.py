import numpy
import argparse
import matplotlib
from matplotlib import colors

from generaltools import from_eta_to_k_par
from generaltools import from_u_to_k_perp
from analytic_covariance import gain_error_covariance
from analytic_covariance import blackman_harris_taper
from analytic_covariance import compute_ps_variance
from analytic_covariance import dft_matrix
from analytic_covariance import compute_weights

from radiotelescope import RadioTelescope
from plottools import plot_power_spectrum


def main(ssh= False, labelfontsize = 12, tickfontsize= 11, plot_name = "Gain_PS_Window.pdf"):
    plot_path = "../../Plots/Analytic_Covariance/"
    u_range = numpy.logspace(0, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165, 251) * 1e6

    mwa_position_path = "./Data/MWA_Compact_Coordinates.txt"
    mwa_telescope = RadioTelescope(load=True, path=mwa_position_path)
    weights = compute_weights(u_range, mwa_telescope.baseline_table.u_coordinates,
                              mwa_telescope.baseline_table.v_coordinates)

    gain_error_sky = gain_error_covariance(u_range, frequency_range, residuals='sky')
    gain_error_both = gain_error_covariance(u_range, frequency_range, residuals='both')
    gain_error_sky_MWA = gain_error_covariance(u_range, frequency_range, residuals='sky', weights = weights, broken_baseline_weight=0.3)
    gain_error_both_MWA = gain_error_covariance(u_range, frequency_range, residuals='both', weights = weights, broken_baseline_weight=0.3)

    window_function = blackman_harris_taper(frequency_range)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)
    dftmatrix, eta = dft_matrix(frequency_range)
    k_par = from_eta_to_k_par(eta[:int(len(eta) / 2)], nu_observed=150e6)
    gain_error_ps = numpy.tile(compute_ps_variance(taper1, taper2, gain_error_both, dftmatrix), (len(u_range), 1))

    mwa_sky_window = numpy.zeros((len(u_range), len(frequency_range)))
    mwa_both_window = mwa_sky_window.copy()

    k_perp = from_u_to_k_perp(u_range, frequency_range[int(len(frequency_range) / 2)])
    x_label = r"$k_{\perp}$ [Mpc$^{-1}$]"

    mwa_position_path = "./Data/MWA_Compact_Coordinates.txt"
    mwa_telescope = RadioTelescope(load=True, path=mwa_position_path)

    log_steps = numpy.diff(numpy.log10(u_range))
    u_bin_edges = numpy.zeros(len(u_range) + 1)
    u_bin_edges[1:] = 10 ** (numpy.log10(u_range) + 0.5 * log_steps[0])
    u_bin_edges[0] = 10 ** (numpy.log10(u_range[0] - 0.5 * log_steps[0]))
    baseline_lengths = numpy.sqrt(mwa_telescope.baseline_table.u_coordinates ** 2 +
                                  mwa_telescope.baseline_table.u_coordinates ** 2)

    counts, edges = numpy.histogram(baseline_lengths, bins=u_bin_edges)

    print(k_perp.shape)
    print(counts.shape)
    for i in range(len(u_range)):
        mwa_sky_window[i, :] = compute_ps_variance(taper1, taper2, gain_error_sky_MWA[i, ...], dftmatrix)
        mwa_both_window[i, :] = compute_ps_variance(taper1, taper2, gain_error_both_MWA[i, ...], dftmatrix)

    figure, axes = pyplot.subplots(1, 3, figsize=(15, 5))



    axes[0].plot(k_perp, counts / len(baseline_lengths))
    axes[0].set_xscale('log')
    axes[0].set_xlabel(x_label, fontsize=labelfontsize)
    axes[0].set_ylabel('Fraction of Baselines', fontsize=labelfontsize)


    axes[1].plot(k_par, compute_ps_variance(taper1, taper2, gain_error_sky, dftmatrix)[:int(len(eta) / 2)], label = r"$\mathbf{C}_{g}(sky)$")
    axes[1].plot(k_par, compute_ps_variance(taper1, taper2, gain_error_both, dftmatrix)[:int(len(eta) / 2)], label = r"$\mathbf{C}_{g}(sky + beam)$")

    axes[1].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]", fontsize=labelfontsize)
    axes[1].set_ylabel("Dimensionless Power", fontsize=labelfontsize)

    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e-18, 1e-5)
    axes[1].legend(fontsize = labelfontsize*0.8)

    norm = colors.LogNorm()
    plot_power_spectrum(u_range, eta[:int(len(eta) / 2)], frequency_range, mwa_sky_window[:, :int(len(eta) / 2)], axes=axes[2],
                         axes_label_font=labelfontsize, tickfontsize=tickfontsize, colorbar_show=True,
                         xlabel_show=True, ylabel_show=True, z_label="Dimensionless Power", ratio=True, norm = norm)
    # axes[2].plot(k_perp, mwa_sky_window[:, :int(len(eta) / 2)].T)
    # axes[2].set_xscale('log')
    # axes[2].set_yscale('log')
    # axes[2].set_xlim(1e-18, 1e-5)

    axes[0].tick_params(axis='both', which='major', labelsize=tickfontsize)
    axes[1].tick_params(axis='both', which='major', labelsize=tickfontsize)
    axes[2].tick_params(axis='both', which='major', labelsize=tickfontsize)


    print("I am plotting")
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

