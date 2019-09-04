import numpy
import argparse
import matplotlib
from matplotlib import colors
from scipy.interpolate import interp1d

from analytic_covariance import dft_matrix
from plottools import plot_power_spectrum


def main(ssh = False, labelfontsize = 10, ticksize= 10):
    output_path = "../../Plots/Analytic_Covariance/"
    path = "/home/ronniyjoseph/Sync/PhD/Projects/beam_perturbations/code/tile_beam_perturbations/Data/"
    file = "redshift8.csv"

    frequency_range = numpy.linspace(135, 165, 101) * 1e6
    u_range = numpy.logspace(0, numpy.log10(500), 100)
    dftmatrix, eta = dft_matrix(frequency_range)
    eta = eta[:len(eta)//2]
    u_fiducial, eta_fiducial, fiducial_eor = read_data(path+file)
    norm = colors.LogNorm(vmin=1e2, vmax=1e5)

    fiducial_figure, fiducial_axes = pyplot.subplots(1, 1, figsize=(5, 5))
    interpolation = interpolate(u_range, eta, u_fiducial, eta_fiducial, fiducial_eor)

    plot_power_spectrum(u_range, eta, frequency_range, interpolation,
                        ratio=True, axes=fiducial_axes, axes_label_font=labelfontsize, tickfontsize=ticksize,
                        xlabel_show=True, colorbar_show=True, norm=norm, title="Fiducial EoR Power Spectrum z = 8")

    fiducial_figure.tight_layout()
    fiducial_figure.savefig(output_path + "Fiducial_EoR_PS_z8.pdf" )

    if not ssh:
        pyplot.show()
    return


def read_data(ps_data_path):

    z_fiducial = 8
    bandwidth_fiducial = 30e6
    central_frequency_fiducial = 1.42e9 / (z_fiducial + 1)

    power_spectrum_fiducial_eor = numpy.loadtxt(ps_data_path, delimiter=',')
    u_range_fiducial = numpy.linspace(0, 500, 100)
    frequency_range_fiducial = numpy.linspace(central_frequency_fiducial - bandwidth_fiducial / 2,
                                              central_frequency_fiducial + bandwidth_fiducial / 2, 251)

    dftmatrix, eta_fiducial = dft_matrix(frequency_range_fiducial)
    # hist, eta_fiducial = numpy.histogram(eta_fiducial[:int(len(eta_fiducial) / 2)], bins=251//2)

    return u_range_fiducial, eta_fiducial[:int(len(eta_fiducial) // 2)], power_spectrum_fiducial_eor.T


def interpolate(u, eta, u_original, eta_original, ps_data):

    eta_interpolated = numpy.zeros((len(u_original), len(eta)))
    for i in range(len(u_original)):
        eta_1d_interp = interp1d(eta_original, ps_data[i, :], kind='cubic')
        eta_interpolated[i, :] = eta_1d_interp(eta)

    fully_interpolated = numpy.zeros((len(u), len(eta)))
    for i in range(len(eta)):
        u_1d_interp = interp1d(u_original, eta_interpolated[:, i], kind='cubic')
        fully_interpolated[:, i] = u_1d_interp(u)

    return fully_interpolated


def fiducial_eor(u, eta, path = "./Data/"
                 , file = "redshift8.csv"):
    u_fiducial, eta_fiducial, ps_fiducial = read_data(path + file)
    ps_interpolated = interpolate(u, eta, u_fiducial, eta_fiducial, ps_fiducial)
    return ps_interpolated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot and compare the sky and beam modelling errors')
    parser.add_argument('-ssh',  action='store_true', default=False, help='flag to use when remote plotting')
    args = parser.parse_args()

    if args.ssh:
        matplotlib.use('Agg')
    from matplotlib import pyplot
    main(ssh = args.ssh)
