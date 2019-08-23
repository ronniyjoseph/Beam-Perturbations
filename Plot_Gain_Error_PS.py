import numpy
import matplotlib
from matplotlib import pyplot
from matplotlib import colors

from generaltools import from_eta_to_k_par

from analytic_covariance import gain_error_covariance
from analytic_covariance import blackman_harris_taper
from analytic_covariance import compute_ps_variance
from analytic_covariance import dft_matrix

from Plot_Calibrated_Error_Comparison import plot_power_spectrum


def main(labelfontsize = 10, ticksize= 10, plot_name = "Gain_PS_Window.pdf"):
    plot_path = "../../Plots/Analytic_Covariance/"
    u_range = numpy.logspace(0, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165, 101) * 1e6

    gain_error_sky = gain_error_covariance(u_range, frequency_range, residuals='sky')
    gain_error_both = gain_error_covariance(u_range, frequency_range, residuals='both')

    window_function = blackman_harris_taper(frequency_range)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)
    dftmatrix, eta = dft_matrix(frequency_range)
    k_perp = from_eta_to_k_par(eta[:int(len(eta) / 2)], nu_observed=150e6)
    gain_error_ps = numpy.tile(compute_ps_variance(taper1, taper2, gain_error_both, dftmatrix), (len(u_range), 1))

    figure, axes = pyplot.subplots(1, 2, figsize = (10, 5))

    axes[0].plot(k_perp, compute_ps_variance(taper1, taper2, gain_error_sky, dftmatrix)[:int(len(eta) / 2)], label = r"$\mathbf{C}_{g}(sky)$")
    axes[0].plot(k_perp, compute_ps_variance(taper1, taper2, gain_error_both, dftmatrix)[:int(len(eta) / 2)], label = r"$\mathbf{C}_{g}(sky + beam)$")

    axes[0].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].legend()

    # ps_norm = colors.LogNorm(vmin=1e3, vmax = 1e15)
    plot_power_spectrum(u_range, eta[:int(len(eta) / 2)], frequency_range, gain_error_ps[:, :int(len(eta) / 2)], axes=axes[1],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, ylabel_show=True, z_label="Dimensionless", ratio=True)

    figure.tight_layout()
    figure.savefig(plot_path + plot_name)
    pyplot.show()
    return


if __name__ == "__main__":
    main()
