import numpy
from matplotlib import pyplot
from matplotlib import colors

from analytic_covariance import gain_error_covariance
from analytic_covariance import blackman_harris_taper
from analytic_covariance import compute_ps_variance
from analytic_covariance import dft_matrix
from Plot_Calibrated_Error_Comparison import plot_power_spectrum


def main(labelfontsize = 10, ticksize= 10):

    u_range = numpy.logspace(0, numpy.log10(200), 100)
    frequency_range = numpy.linspace(135, 165, 101) * 1e6

    gain_error = gain_error_covariance(u_range, frequency_range)
    window_function = blackman_harris_taper(frequency_range)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)
    dftmatrix, eta = dft_matrix(frequency_range)
    gain_error_ps = numpy.tile(compute_ps_variance(taper1, taper2, gain_error, dftmatrix), (len(u_range), 1))

    figure, axes = pyplot.subplots(1, 1, figsize = (5, 5))

    # ps_norm = colors.LogNorm(vmin=1e3, vmax = 1e15)
    plot_power_spectrum(u_range, eta[:int(len(eta) / 2)], frequency_range, gain_error_ps[:, :int(len(eta) / 2)], axes=axes,
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True,ylabel_show=True, z_label="Dimensionless")

    figure.tight_layout()

    pyplot.show()
    return


if __name__ == "__main__":
    main()
