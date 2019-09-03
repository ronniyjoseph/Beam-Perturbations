import numpy
import argparse

import matplotlib
from matplotlib import colors

from analytic_covariance import residual_ps_error
from analytic_covariance import compute_weights

from radiotelescope import RadioTelescope
from plottools import plot_power_spectrum
from Plot_Fiducial_PS import fiducial_eor


def main(labelfontsize=10, ticksize=10):
    plot_path = "../../Plots/Analytic_Covariance/"
    mwa_position_path = "./Data/MWA_Compact_Coordinates.txt"
    mwa_telescope = RadioTelescope(load=True, path=mwa_position_path)

    u_range = numpy.logspace(1, numpy.log10(500), 100)

    # 100 frequency channels is fine for now, maybe later do a higher number to push up the k_par range
    frequency_range = numpy.linspace(135, 165, 251) * 1e6
    weights = compute_weights(u_range, mwa_telescope.baseline_table.u_coordinates,
                              mwa_telescope.baseline_table.v_coordinates)
    eta, sky_only_raw, sky_only_cal = residual_ps_error(u_range, frequency_range, residuals='sky',
                                                                broken_baselines_weight=0.3, weights = weights) #residual_ps_error(u_range, frequency_range, residuals='sky')
    eta, sky_and_beam_raw, sky_and_beam_cal = residual_ps_error(u_range, frequency_range, residuals='both',
                                                                broken_baselines_weight=0.3, weights= weights)
    fiducial_ps = fiducial_eor(u_range, eta)

    difference_cal = sky_and_beam_cal - sky_only_cal

    figure, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    ps_norm = colors.LogNorm(vmin=1e3, vmax=1e15)
    plot_power_spectrum(u_range, eta, frequency_range, sky_and_beam_cal,
                        title=r"MWA $\mathbf{C}_{r}$(sky + beam)", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, norm=ps_norm,
                        xlabel_show=True, colorbar_show=True)

    diff_norm = colors.SymLogNorm(linthresh=1e2, linscale=1.5, vmin=-1e12, vmax=1e12)

    plot_power_spectrum(u_range, eta, frequency_range, difference_cal,
                        axes=axes[1], axes_label_font=labelfontsize, tickfontsize=ticksize,
                        norm=diff_norm, colorbar_show=True, xlabel_show=True,
                        title=r" MWA $\mathbf{C}_{r}$(sky + beam) - $\mathbf{C}_{r}$(sky) ", diff=True, colormap='coolwarm')

    ratio_norm = colors.SymLogNorm(linthresh= 1e1, linscale = 1, vmin = -1e2, vmax = 1e2)

    plot_power_spectrum(u_range, eta, frequency_range, difference_cal / fiducial_ps,
                        axes=axes[2], axes_label_font=labelfontsize, tickfontsize=ticksize,
                        norm=ratio_norm, colorbar_show=True, xlabel_show=True,
                        title=r"$(\mathbf{C}_{r}$(sky + beam) - $\mathbf{C}_{r}$(sky))/EoR ", diff=True)

    figure.tight_layout()
    figure.savefig(plot_path + "Comparing_Sky_and_Beam_Errors_Post_Calibration_MWA.pdf")
    pyplot.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot and compare the sky and beam modelling errors')
    parser.add_argument('-ssh', type=bool, action='store_true', default=False, help='flag to use when remote plotting')
    if parser.ssh:
        matplotlib.use('Agg')
    from matplotlib import pyplot
    main()