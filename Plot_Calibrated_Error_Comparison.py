import numpy
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib import colors

from analytic_covariance import residual_ps_error
from generaltools import from_eta_to_k_par
from generaltools import from_u_to_k_perp
from generaltools import from_jansky_to_milikelvin
from generaltools import colorbar


def main(labelfontsize = 10, ticksize= 10):
    plot_path = "../../Plots/Analytic_Covariance/"
    u_range = numpy.logspace(0, numpy.log10(500), 100)

    # 100 frequency channels is fine for now, maybe later do a higher number to push up the k_par range
    frequency_range = numpy.linspace(135, 165, 101) * 1e6

    eta, sky_only_raw, sky_only_cal = residual_ps_error(u_range, frequency_range, residuals='sky')
    eta, sky_and_beam_raw, sky_and_beam_cal = residual_ps_error(u_range, frequency_range, residuals='both')
    difference_cal = sky_and_beam_cal - sky_only_cal


    figure, axes = pyplot.subplots(1, 3, figsize = (15, 5))

    ps_norm = colors.LogNorm(vmin = 1e3, vmax = 1e15)
    plot_power_spectrum(u_range, eta, frequency_range, sky_and_beam_cal,
                                  title=r"$\mathbf{C}_{r}$(sky + beam)", axes=axes[0],
                                  axes_label_font= labelfontsize, tickfontsize = ticksize, norm = ps_norm,
                        xlabel_show= True, colorbar_show=True)

    diff_norm = colors.SymLogNorm(linthresh= 1e2, linscale = 1.5, vmin = -1e14, vmax = 1e12)
    diff_norm = colors.LogNorm(vmin = 1e5, vmax = 1e12)


    plot_power_spectrum(u_range, eta, frequency_range, sky_and_beam_cal - sky_only_cal,
                        axes=axes[1], axes_label_font= labelfontsize, tickfontsize = ticksize,
                        norm=diff_norm, colorbar_show=True, xlabel_show= True,
                        title=r"$\mathbf{C}_{r}$(sky + beam) - $\mathbf{C}_{r}$(sky) ", diff=True)


    ratio_norm = colors.SymLogNorm(linthresh= 1e3, linscale = 1, vmin = -1e9, vmax = 1e14)
    plot_power_spectrum(u_range, eta, frequency_range, (sky_and_beam_cal - sky_only_cal)/sky_only_raw,
                        axes=axes[2], axes_label_font= labelfontsize, tickfontsize = ticksize,
                        norm=ratio_norm, colorbar_show=True, xlabel_show= True,
                        title=r"$(\mathbf{C}_{r}$(sky + beam) - $\mathbf{C}_{r}$(sky))/EoR ", diff=True)



    # ratio_norm = colors.LogNorm(1e-2, 1e2)
    # # Plot ratios with uncalibrated
    # plot_power_spectrum(u_range, eta, frequency_range, (sky_and_beam_cal - sky_only_cal)/sky_only_cal,
    #          ratio= True, axes=axes[2], axes_label_font= labelfontsize, tickfontsize = ticksize,
    #         xlabel_show= True, colorbar_show=True, norm =ratio_norm, title="Fraction of Fiducial EoR Power")

    figure.tight_layout()
    figure.savefig(plot_path + "Comparing_Sky_and_Beam_Errors_Post_Calibration.pdf")
    #pyplot.show()

    return


def plot_power_spectrum(u_bins, eta_bins, nu, data, norm = None, title=None, axes=None,
                        colormap = "viridis", axes_label_font=20, tickfontsize=15, xlabel_show=False, ylabel_show=False,
                        zlabel_show=False, z_label = None, return_norm = False, colorbar_show = False, ratio = False,
                        diff = False, x_range = None, y_range = None):

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

    if x_range is None:
        axes.set_xlim(9e-3, 3e-1)
    if y_range is None:
        axes.set_ylim(9e-3, 1.2e0)

    if diff:
        print(numpy.log10(-z_values.min()))
        pass
    else:
        z_values[data < 0] = numpy.nan
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
