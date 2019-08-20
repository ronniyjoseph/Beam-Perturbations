import numpy
from matplotlib import pyplot
from matplotlib import colors

from analytic_covariance import calculate_sky_power_spectrum
from analytic_covariance import calculate_beam_power_spectrum_averaged
from analytic_covariance import calculate_beam_power_spectrum

from Plot_Calibrated_Error_Comparison import plot_power_spectrum

def main(labelfontsize = 10, ticksize= 10):
    u_range = numpy.logspace(0, numpy.log10(200), 100)
    frequency_range = numpy.linspace(135, 165, 101) * 1e6

    eta, sky_power_spectrum = calculate_sky_power_spectrum(u_range, frequency_range)
    eta, beam_power_spectrum_averaged = calculate_beam_power_spectrum_averaged(u_range, frequency_range)
    eta, beam_power_spectrum_1direction = calculate_beam_power_spectrum(u_range, frequency_range)
    eta, beam_power_spectrum_total1 = calculate_beam_power_spectrum(u_range, frequency_range)
    eta, beam_power_spectrum_total2 = calculate_beam_power_spectrum(u_range, frequency_range)

    figure, axes = pyplot.subplots(1, 3, figsize = (15, 5))

    plot_power_spectrum(u_range, eta, frequency_range, sky_power_spectrum, title="Sky Noise", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, return_norm=True, colorbar_show=True,
                        xlabel_show=True)

    plot_power_spectrum(u_range, eta, frequency_range, beam_power_spectrum_averaged, title="Beam Noise", axes=axes[1],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, return_norm=True, colorbar_show=True,
                        xlabel_show=True)
    plot_power_spectrum(u_range, eta, frequency_range, beam_power_spectrum_1direction, title="Beam 1d", axes=axes[2],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, return_norm=True, colorbar_show=True,
                        xlabel_show=True)







