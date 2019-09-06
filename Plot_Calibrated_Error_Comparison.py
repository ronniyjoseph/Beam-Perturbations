import numpy
import matplotlib
from matplotlib import colors
from plottools import plot_power_spectrum
from analytic_covariance import residual_ps_error
from Plot_Fiducial_PS import fiducial_eor
import argparse


def main(ssh = False, labelfontsize = 10, ticksize= 10):
    plot_path = "../../Plots/Analytic_Covariance/"
    u_range = numpy.logspace(0, numpy.log10(500), 100)

    # 100 frequency channels is fine for now, maybe later do a higher number to push up the k_par range
    frequency_range = numpy.linspace(135, 165, 251) * 1e6

    eta, sky_only_raw, sky_only_cal = residual_ps_error(u_range, frequency_range, residuals='sky')
    eta, sky_and_beam_raw, sky_and_beam_cal = residual_ps_error(u_range, frequency_range, residuals='both')
    difference_cal = sky_and_beam_cal - sky_only_cal

    fiducial_ps = fiducial_eor(u_range, eta)

    figure, axes = pyplot.subplots(1, 3, figsize = (15, 5))

    ps_norm = colors.LogNorm(vmin = 1e2, vmax = 1e15)
    plot_power_spectrum(u_range, eta, frequency_range, sky_and_beam_cal,
                                  title=r"$\mathbf{C}_{r}$(sky + beam)", axes=axes[0],
                                  axes_label_font= labelfontsize, tickfontsize = ticksize, norm = ps_norm,
                        xlabel_show= True, colorbar_show=True)

    diff_norm = colors.SymLogNorm(linthresh= 1e2, linscale = 1.5, vmin = -1e15, vmax = 1e15)
    plot_power_spectrum(u_range, eta, frequency_range, difference_cal,
                        axes=axes[1], axes_label_font= labelfontsize, tickfontsize = ticksize,
                        norm=diff_norm, colorbar_show=True, xlabel_show= True,
                        title=r"$\mathbf{C}_{r}$(sky + beam) - $\mathbf{C}_{r}$(sky) ", diff=True,
                        colormap='coolwarm')

    ratio_norm = colors.SymLogNorm(linthresh= 1e1, linscale = 1, vmin = -1e1, vmax = 1e5)
    plot_power_spectrum(u_range, eta, frequency_range, difference_cal/fiducial_ps,
                        axes=axes[2], axes_label_font= labelfontsize, tickfontsize = ticksize,
                        norm=ratio_norm, colorbar_show=True, xlabel_show= True,
                        title=r"$(\mathbf{C}_{r}$(sky + beam) - $\mathbf{C}_{r}$(sky))/EoR ", diff=True)


    figure.tight_layout()
    figure.savefig(plot_path + "Comparing_Sky_and_Beam_Errors_Post_Calibration.pdf")
    if not ssh:
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
