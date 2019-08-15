import os
import numpy
from matplotlib import pyplot
from matplotlib import colors

from analytic_covariance import residual_ps_error
from generaltools import from_eta_to_k_par
from generaltools import from_u_to_k_perp
from generaltools import from_jansky_to_milikelvin
from generaltools import colorbar

from Error_Comparison_Plots import plot_power_spectrum

def main(labelfontsize = 10, ticksize= 10):
    u_range = numpy.logspace(0, numpy.log10(200), 100)

    # 100 frequency channels is fine for now, maybe later do a higher number to push up the k_par range
    frequency_range = numpy.linspace(135, 165, 101) * 1e6

    eta, sky_only_raw, sky_only_cal = residual_ps_error(u_range, frequency_range, residuals='sky')

    figure, axes = pyplot.subplots(1, 3, figsize = (15, 5))
    ps_norm = plot_power_spectrum(u_range, eta, frequency_range, sky_only_cal, title="Calibrated", axes=axes[0],
                                  axes_label_font= labelfontsize, tickfontsize = ticksize, return_norm = True, colorbar_show=True, xlabel_show= True)

    # Plot Difference with uncalibrated
    diff_norm = colors.LogNorm(vmin=1e0, vmax=1e5)
    difference_label = r"Difference [mK$^2$ Mpc$^3$ ]"
    plot_power_spectrum(u_range, eta, frequency_range, sky_only_cal - sky_only_raw,
                        axes=axes[1], axes_label_font= labelfontsize, tickfontsize = ticksize,
                        norm=diff_norm, colorbar_show=True,xlabel_show= True, title="Difference")

    ratio_norm = colors.LogNorm(1e-3, 5e-1)
    # Plot ratios with uncalibrated
    plot_power_spectrum(u_range, eta, frequency_range, (sky_only_cal - sky_only_raw)/sky_only_raw,
             ratio= True, axes=axes[2], axes_label_font= labelfontsize, tickfontsize = ticksize,
            xlabel_show= True, colorbar_show=True, norm =ratio_norm, title="Fractional Difference")

    figure.tight_layout()
    pyplot.show()

    return

if __name__ == "__main__":
    main()