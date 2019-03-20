import numpy

"""
This file is going to contain all relevant power spectrum functions, i.e data gridding, (frequency tapering), frequency
fft, angular averaging, plotting

"""


class PowerSpectrumData:
    def __init__(self, visibility_data = None, u_coordinate = None, v_coordinate = None, frequency_coordinate = None):
        self.data = visibility_data
        self.u = u_coordinate
        self.v = v_coordinate
        self.f = frequency_coordinate
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

    def regrid_data(self):

