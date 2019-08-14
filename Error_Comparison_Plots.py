import os
import numpy
from matplotlib import pyplot
from matplotlib import colors

from analytic_covariance import residual_ps_error
from generaltools import from_eta_to_k_par
from generaltools import from_u_to_k_perp
from generaltools import from_jansky_to_milikelvin
from generaltools import colorbar

def main(labelfontsize = 10, ticksize= 10):
    u_range = numpy.logspace(0, numpy.log10(200), 100)

    # 100 frequency channels is fine for now, maybe later do a higher number to push up the k_par range
    frequency_range = numpy.linspace(135, 165, 101) * 1e6

    eta, sky_only_raw, sky_only_cal = residual_ps_error(u_range, frequency_range, residuals='sky')
    #eta, beam_only_raw, beam_only_cal = residual_ps_error(u_range, frequency_range, residuals='sky')
    eta, sky_and_beam_raw, sky_and_beam_cal = residual_ps_error(u_range, frequency_range, residuals='both')

    figure, axes = pyplot.subplots(1, 4, figsize = (20, 5))
    ps_norm = plot_power_spectrum(u_range, eta, frequency_range, sky_and_beam_cal, title="Sky + Beam", axes=axes[1],
                                  axes_label_font= labelfontsize, tickfontsize = ticksize, return_norm = True, colorbar_show=True, xlabel_show= True)

    plot_power_spectrum(u_range, eta, frequency_range, sky_only_cal, title="Sky Only", axes=axes[0],
                        axes_label_font= labelfontsize, tickfontsize = ticksize, ylabel_show= True, norm=ps_norm,colorbar_show=True, xlabel_show= True)

    # Plot Difference with uncalibrated
    diff_norm = colors.LogNorm(vmin=1e0, vmax=1e14)
    difference_label = r"Difference [mK$^2$ Mpc$^3$ ]"
    plot_power_spectrum(u_range, eta, frequency_range, sky_and_beam_cal - sky_only_cal,
                        axes=axes[2], axes_label_font= labelfontsize, tickfontsize = ticksize,
                        norm=ps_norm, colorbar_show=True,xlabel_show= True, title="Difference")

    ratio_norm = colors.LogNorm(1e-2, 1e2)
    # Plot ratios with uncalibrated
    plot_power_spectrum(u_range, eta, frequency_range, (sky_and_beam_cal - sky_only_cal)/sky_only_cal,
             ratio= True, axes=axes[3], axes_label_font= labelfontsize, tickfontsize = ticksize,
            xlabel_show= True, colorbar_show=True, norm =ratio_norm, title="Ratio")

    figure.tight_layout()
    pyplot.show()

    return





def plot_power_spectrum(u_bins, eta_bins, nu, data, norm = None, title=None, axes=None,
                        colormap = "viridis", axes_label_font=20, tickfontsize=15, xlabel_show=False, ylabel_show=False,
                        zlabel_show=False, z_label = None, return_norm = False, colorbar_show = False, ratio = False):

    central_frequency = nu[int(len(nu) / 2)]
    x_values = from_u_to_k_perp(u_bins, central_frequency)
    y_values = from_eta_to_k_par(eta_bins, central_frequency)

    if ratio:
        z_values = data
    else:
        z_values = from_jansky_to_milikelvin(data, nu)


    x_label = r"$k_{\perp}$ [Mpc$^{-1}$]"
    y_label = r"$k_{\parallel}$ [Mpc$^{-1}$]"
    if z_label is None:
        z_label = r"Variance [mK$^2$ Mpc$^3$ ]"

    axes.set_xlim(1e-3, 1e-1)
    axes.set_ylim(9e-3, 5e-1)

    z_values[data < 0] = numpy.abs(z_values[data < 0])
    if norm is None:
        norm = colors.LogNorm(vmin=numpy.real(z_values).min(), vmax=numpy.real(z_values).max())

    if title is not None:
        axes.set_title(title)

    psplot = axes.pcolor(x_values, y_values, z_values.T, cmap=colormap, rasterized=True, norm=norm)
    if colorbar_show:
        cax = colorbar(psplot)
        cax.ax.tick_params(axis='both', which='major', labelsize=tickfontsize)
        cax.set_label(z_label, fontsize=axes_label_font)

    axes.set_xscale('log')
    axes.set_yscale('log')

    if xlabel_show:
        axes.set_xlabel(x_label, fontsize=axes_label_font)
    if ylabel_show:
        axes.set_ylabel(y_label, fontsize=axes_label_font)


    axes.tick_params(axis='both', which='major', labelsize=tickfontsize)

    return norm if return_norm else None


if __name__ == "__main__":
    main()
