import os
import numpy
from matplotlib import pyplot
from matplotlib import colors

from analytic_covariance import residual_ps_error
from analytic_covariance import dft_matrix
from generaltools import from_eta_to_k_par
from generaltools import from_u_to_k_perp
from generaltools import from_jansky_to_milikelvin
from generaltools import colorbar

from Plot_Calibrated_Error_Comparison import plot_power_spectrum

def main(labelfontsize = 10, ticksize= 10):
    ps_data_path = "/home/ronniyjoseph/Sync/PhD/Projects/beam_perturbations/code/tile_beam_perturbations/" + \
                   "Data/power_2d_kperp50_kpar50.dat"

    u_range = numpy.logspace(0, numpy.log10(500), 100)
    # 100 frequency channels is fine for now, maybe later do a higher number to push up the k_par range
    ###### set to 251 to fill up k from -2

    frequency_range = numpy.linspace(135, 165, 251) * 1e6

    eta, sky_only_raw, sky_only_cal = residual_ps_error(u_range, frequency_range, residuals='sky')

    z_fiducial = 8
    bandwidth_fiducial = 30e6
    central_frequency_fiducial = 1.42e9/(z_fiducial + 1)

    power_spectrum_fiducial_eor = numpy.fromfile(ps_data_path, dtype=numpy.float32).reshape(50, 50)
    u_range_fiducial = numpy.linspace(0, 397.15 * numpy.sqrt(2), 51)[1:]
    frequency_range_fiducial = numpy.linspace(central_frequency_fiducial - bandwidth_fiducial/2,
                                              central_frequency_fiducial + bandwidth_fiducial/2, len(frequency_range))
    dftmatrix, eta_fiducial = dft_matrix(frequency_range_fiducial)
    hist, eta_fiducial = numpy.histogram(eta_fiducial[:int(len(eta_fiducial)/2)], bins = 50)


    figure, axes = pyplot.subplots(1, 3, figsize = (15, 5))
    ps_norm = plot_power_spectrum(u_range, eta, frequency_range, sky_only_cal, title="Calibrated", axes=axes[0],
                                  axes_label_font= labelfontsize, tickfontsize = ticksize, return_norm = True,
                                  colorbar_show=True, xlabel_show= True)

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

    fiducial_figure, fiducial_axes = pyplot.subplots(1, 1, figsize=(5, 5))
    norm = colors.LogNorm(vmin=0.5e4, vmax=1.3e6)
    plot_power_spectrum(u_range_fiducial, eta_fiducial[:-1], frequency_range,
                        power_spectrum_fiducial_eor,
                        ratio=True, axes=fiducial_axes, axes_label_font=labelfontsize, tickfontsize=ticksize,
                        xlabel_show=True, colorbar_show=True, norm=norm, title="Fiducial EoR Power Spectrum z = 8")
    fiducial_figure.tight_layout()

    pyplot.show()

    return

if __name__ == "__main__":
    main()