import os
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib import colors

from scipy.interpolate import interp2d
from analytic_covariance import residual_ps_error
from analytic_covariance import dft_matrix
from generaltools import from_eta_to_k_par
from generaltools import from_u_to_k_perp
from generaltools import from_jansky_to_milikelvin
from generaltools import colorbar

from Plot_Calibrated_Error_Comparison import plot_power_spectrum
from Plot_Fiducial_PS import fiducial_eor


def main(labelfontsize = 10, ticksize= 10):
    plot_path = "../../Plots/Analytic_Covariance/"

    u_range = numpy.logspace(0, numpy.log10(500), 100)

    # 100 frequency channels is fine for now, maybe later do a higher number to push up the k_par range
    ###### set to 251 to fill up k from -2
    frequency_range = numpy.linspace(135, 165, 251) * 1e6

    eta, sky_only_raw, sky_only_cal = residual_ps_error(u_range, frequency_range, residuals='sky')
    difference_data = sky_only_cal - sky_only_raw
    fiducial_ps = fiducial_eor(u_range, eta)

    print(difference_data.max())
    print(fiducial_ps.max())
    figure, axes = pyplot.subplots(1, 3, figsize = (15, 5))
    ps_norm = colors.LogNorm(vmin=1e2, vmax=1e15)

    plot_power_spectrum(u_range, eta, frequency_range, sky_only_cal, title=r"$\mathbf{C}_{r}$(sky)", axes=axes[0],
                                  axes_label_font= labelfontsize, tickfontsize = ticksize, norm = ps_norm,
                                  colorbar_show=True, xlabel_show= True)

    # Plot Difference with uncalibrated
    diff_norm = colors.LogNorm(vmin=1e5, vmax=1e9)
    difference_label = r"Difference [mK$^2$ Mpc$^3$ ]"
    plot_power_spectrum(u_range, eta, frequency_range, difference_data,
                        axes=axes[1], axes_label_font= labelfontsize, tickfontsize = ticksize,
                        norm=diff_norm, colorbar_show=True,xlabel_show= True,
                        title=r"$\mathbf{C}_{r}$(sky) - $\mathbf{C}_{\mathrm{sky}}$ ")

    ratio_norm = colors.LogNorm(vmin=1e2, vmax=1e5)
    # Plot ratios with uncalibrated
    plot_power_spectrum(u_range, eta, frequency_range, difference_data/fiducial_ps,
                        axes=axes[2], axes_label_font= labelfontsize, tickfontsize = ticksize,
                        xlabel_show= True, colorbar_show=True, norm =ratio_norm,
                        title=r"$(\mathbf{C}_{r}$(sky) - $\mathbf{C}_{\mathrm{sky}}$)/EoR ")

    figure.tight_layout()


    figure.savefig(plot_path + "Comparing_Calibated_and_Uncalibrated.pdf")

    #pyplot.show()

    return

if __name__ == "__main__":
    main()