import numpy
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib import colors

from analytic_covariance import residual_ps_error
from generaltools import from_eta_to_k_par
from generaltools import from_u_to_k_perp
from generaltools import from_jansky_to_milikelvin
from generaltools import colorbar

from Plot_Calibrated_Error_Comparison import plot_power_spectrum


def main(labelfontsize=10, ticksize=10):
    plot_path = "../../Plots/Analytic_Covariance/"
    u_range = numpy.logspace(0, numpy.log10(500), 100)

    # 100 frequency channels is fine for now, maybe later do a higher number to push up the k_par range
    frequency_range = numpy.linspace(135, 165, 101) * 1e6

    eta, sky_only_raw, sky_only_cal = residual_ps_error(u_range, frequency_range, residuals='sky')
    eta, sky_and_beam_raw, sky_and_beam_cal = residual_ps_error(u_range, frequency_range, residuals='both',
                                                                broken_baselines_weight=0.3)
    difference_cal = sky_and_beam_cal - sky_only_cal

    figure, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    ps_norm = colors.LogNorm(vmin=1e3, vmax=1e15)
    plot_power_spectrum(u_range, eta, frequency_range, sky_and_beam_cal,
                        title=r"$\mathbf{C}_{r}$(sky + beam)", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, norm=ps_norm,
                        xlabel_show=True, colorbar_show=True)

    diff_norm = colors.SymLogNorm(linthresh=1e2, linscale=1.5, vmin=-1e14, vmax=1e12)
    diff_norm = colors.LogNorm(vmin=1e5, vmax=1e12)

    plot_power_spectrum(u_range, eta, frequency_range, sky_and_beam_cal - sky_only_cal,
                        axes=axes[1], axes_label_font=labelfontsize, tickfontsize=ticksize,
                        norm=diff_norm, colorbar_show=True, xlabel_show=True,
                        title=r"$\mathbf{C}_{r}$(sky + beam) - $\mathbf{C}_{r}$(sky) ", diff=True)

    ratio_norm = colors.SymLogNorm(linthresh=1e3, linscale=1, vmin=-1e9, vmax=1e14)
    plot_power_spectrum(u_range, eta, frequency_range, (sky_and_beam_cal - sky_only_cal) / sky_only_raw,
                        axes=axes[2], axes_label_font=labelfontsize, tickfontsize=ticksize,
                        norm=ratio_norm, colorbar_show=True, xlabel_show=True,
                        title=r"$(\mathbf{C}_{r}$(sky + beam) - $\mathbf{C}_{r}$(sky))/EoR ", diff=True)

    # ratio_norm = colors.LogNorm(1e-2, 1e2)
    # # Plot ratios with uncalibrated
    # plot_power_spectrum(u_range, eta, frequency_range, (sky_and_beam_cal - sky_only_cal)/sky_only_cal,
    #          ratio= True, axes=axes[2], axes_label_font= labelfontsize, tickfontsize = ticksize,
    #         xlabel_show= True, colorbar_show=True, norm =ratio_norm, title="Fraction of Fiducial EoR Power")

    figure.tight_layout()
    figure.savefig(plot_path + "Comparing_Sky_and_Beam_Errors_Post_Calibration_MWA.pdf")
    pyplot.show()

    return


if __name__ == "__main__":
    main()