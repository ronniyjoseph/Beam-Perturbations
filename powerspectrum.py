import numpy

"""
This file is going to contain all relevant power spectrum functions, i.e data gridding, (frequency tapering), frequency
fft, angular averaging, plotting

"""


class PowerSpectrumData:
    def __init__(self, visibility_data = None, u_coordinate = None, v_coordinate = None, frequency_coordinate = None):
        self.data_raw = visibility_data
        self.u_raw = u_coordinate
        self.v_raw = v_coordinate
        self.f_raw = frequency_coordinate

        self.data_regrid = None
        self.u_regrid = None
        self.v_regrid = None
        self.f_regrid = None
        self.eta = None
        return

    def append_frequency_slice(self, new_data, new_u, new_v, new_frequency):

        if self.data is None:
            self.data = new_data
            self.u = new_u
            self.v = new_v
            self.f = numpy.array([new_frequency])
        else:
            current_data = self.data
            current_u = self.u
            current_v = self.v
            current_f = self.f

            self.data = numpy.vstack((current_data, new_data))
            self.u = numpy.vstack((current_u, new_u))
            self.v = numpy.vstack((current_v, new_v))
            self.f = numpy.vstack((current_f, numpy.array([new_frequency])))
        return

    def regrid_data(self, keep_raw = True):
        return


def regrid_visibilities(measured_visibilities, baseline_u, baseline_v, u_grid):
    u_shifts = numpy.diff(u_grid) / 2.

    u_bin_edges = numpy.concatenate((numpy.array([u_grid[0] - u_shifts[0]]), u_grid[1:] - u_shifts,
                                     numpy.array([u_grid[-1] + u_shifts[-1]])))

    weights_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u,
                                                     baseline_v,
                                                     bins=(u_bin_edges, u_bin_edges))

    real_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u,
                                                     baseline_v,
                                                     bins=(u_bin_edges, u_bin_edges),
                                                     weights=
                                                     numpy.real(measured_visibilities))

    imag_regrid, u_bins, v__bins = numpy.histogram2d(baseline_u,
                                                     baseline_v,
                                                     bins=(u_bin_edges, u_bin_edges),
                                                     weights=
                                                     numpy.imag(measured_visibilities))

    regridded_visibilities = real_regrid + 1j*imag_regrid
    return regridded_visibilities, weights_regrid
