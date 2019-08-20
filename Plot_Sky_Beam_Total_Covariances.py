import numpy
from matplotlib import pyplot
from matplotlib import colors

from analytic_covariance import calculate_sky_power_spectrum
from analytic_covariance import calculate_beam_power_spectrum_averaged
from analytic_covariance import calculate_beam_power_spectrum
from analytic_covariance import calculate_total_power_spectrum

from Plot_Calibrated_Error_Comparison import plot_power_spectrum
from analytic_covariance import plot_PS


def main(labelfontsize = 10, ticksize= 10):
    k_perp_range = numpy.array([1e-4, 1.1e-1])

    u_range = numpy.logspace(-1, numpy.log10(200), 100)
    frequency_range = numpy.linspace(135, 165, 101) * 1e6

    eta, sky_power_spectrum = calculate_sky_power_spectrum(u_range, frequency_range)
    #eta, beam_power_spectrum_averaged = calculate_beam_power_spectrum_averaged(u_range, frequency_range)
    eta, beam_power_spectrum_1direction = calculate_beam_power_spectrum(u_range, frequency_range)
    eta, total_power_spectrum_1 = calculate_total_power_spectrum(u_range, frequency_range)
    #eta, total_power_spectrum_total_2 = calculate_total_power_spectrum(u_range, frequency_range)

    figure, axes = pyplot.subplots(1, 3, figsize = (15, 5))

    ps_norm = colors.LogNorm(vmin=1e3, vmax = 1e15)
    plot_power_spectrum(u_range, eta, frequency_range, sky_power_spectrum, title="Sky Noise", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm = ps_norm, x_range=k_perp_range)

    beamnorm = colors.SymLogNorm(linthresh=1e7, linscale = 1, vmin = -1e14, vmax = 1e14)
    plot_power_spectrum(u_range, eta, frequency_range, total_power_spectrum_1 - sky_power_spectrum, title="Beam Noise", axes=axes[1],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=beamnorm, diff=True, colormap='coolwarm', x_range=k_perp_range)

    # plot_PS(u_range, eta, frequency_range, total_power_spectrum_1, cosmological=True)
    plot_power_spectrum(u_range, eta, frequency_range, total_power_spectrum_1 , title="Total Noise", axes=axes[2],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm = ps_norm, x_range=k_perp_range)

    figure.tight_layout()

    pyplot.show()
    return


if __name__ == "__main__":
    main()




