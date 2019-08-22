import numpy
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib import colors
from analytic_covariance import dft_matrix

from scipy.interpolate import interp2d

from Plot_Calibrated_Error_Comparison import plot_power_spectrum


def main(labelfontsize = 10, ticksize= 10):
    frequency_range = numpy.linspace(135, 165, 101) * 1e6
    u_range = numpy.logspace(0, numpy.log10(500), 100)
    dftmatrix, eta = dft_matrix(frequency_range)

    u_fiducial, eta_fiducial, fiducial_eor = read_data()

    fiducial_figure, fiducial_axes = pyplot.subplots(1, 2, figsize=(10, 5))
    norm = colors.LogNorm(vmin=0.5e4, vmax=1.3e6)
    interpolation = interp2d(u_fiducial, eta_fiducial, fiducial_eor)

    plot_power_spectrum(u_range, eta_fiducial, frequency_range, fiducial_eor,
                        ratio=True, axes=fiducial_axes[0], axes_label_font=labelfontsize, tickfontsize=ticksize,
                        xlabel_show=True, colorbar_show=True, norm=norm, title="Fiducial EoR Power Spectrum z = 8")

    plot_power_spectrum(u_range, eta, frequency_range, interpolation(u_range, eta),
                        ratio=True, axes=fiducial_axes[1], axes_label_font=labelfontsize, tickfontsize=ticksize,
                        xlabel_show=True, colorbar_show=True, norm=norm, title="Fiducial EoR Power Spectrum z = 8")


    fiducial_figure.tight_layout()
    pyplot.show()
    return



def read_data():
    ps_data_path = "/home/ronniyjoseph/Sync/PhD/Projects/beam_perturbations/code/tile_beam_perturbations/" + \
                   "Data/power_z8_kperp50_kpa50.dat"

    z_fiducial = 8
    bandwidth_fiducial = 30e6
    central_frequency_fiducial = 1.42e9 / (z_fiducial + 1)

    power_spectrum_fiducial_eor = numpy.fromfile(ps_data_path, dtype=numpy.float32).reshape(50, 50)
    u_range_fiducial = numpy.linspace(0, 397.15 * numpy.sqrt(2), 51)[1:]
    frequency_range_fiducial = numpy.linspace(central_frequency_fiducial - bandwidth_fiducial / 2,
                                              central_frequency_fiducial + bandwidth_fiducial / 2, 100)

    dftmatrix, eta_fiducial = dft_matrix(frequency_range_fiducial)
    hist, eta_fiducial = numpy.histogram(eta_fiducial[:int(len(eta_fiducial) / 2)], bins=50)

    return u_range_fiducial, eta_fiducial[:-1], power_spectrum_fiducial_eor


if __name__ == "__main__":
    main()
