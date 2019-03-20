import numpy


class SkyRealisation:
    def __init__(self, sky_type, fluxes=0, l_coordinates=0, m_coordinates=0, spectral_indices=0,
                 seed=0, k1=4100, gamma1=1.59, k2=4100, gamma2=2.5, flux_low=400e-3, flux_mid=1, flux_high=5.,
                 verbose = False):

        if verbose:
            print("Creating the sky realisation")

        if sky_type == "random":
            numpy.random.seed(seed)
            self.fluxes = stochastic_sky(seed, k1, gamma1, k2, gamma2, flux_low, flux_mid, flux_high)
            all_r = numpy.sqrt(numpy.random.uniform(0, 1, len(self.fluxes)))
            all_phi = numpy.random.uniform(0, 2. * numpy.pi, len(self.fluxes))

            self.l_coordinates = all_r * numpy.cos(all_phi)
            self.m_coordinates = all_r * numpy.sin(all_phi)
            self.spectral_indices = numpy.zeros_like(self.fluxes) + spectral_indices
        elif sky_type == "point":
            self.fluxes = fluxes
            self.l_coordinates = l_coordinates
            self.m_coordinates = m_coordinates
            self.spectral_indices = spectral_indices
        else:
            raise ValueError(f"sky_type must be 'random' or 'point' NOT {sky_type}")
        return

    def create_sky_image(self, frequency_channels, baseline_table = None, radiotelescope = None,
                        resolution = None, oversampling= 1):

        #####################################
        # Assume the sky is flat
        #####################################

        source_flux = self.fluxes
        source_l = self.l_coordinates
        source_m = self.m_coordinates


        if baseline_table is not None:
            n_frequencies = len(frequency_channels)
            #Find longest baseline to determine sky_image sampling, pick highest frequency for longest baseline
            max_u = numpy.max(baseline_table[:,2,-1])
            max_v = numpy.max(baseline_table[:,3,-1])
            max_b = max(max_u, max_v)
            #sky_resolutions
            min_l = 1./max_b
            delta_l = min_l/oversampling
        elif radiotelescope is not None:
            n_frequencies = 1

            max_u = numpy.max(radiotelescope.baseline_table.u(frequency_channels))
            max_v = numpy.max(radiotelescope.baseline_table.v(frequency_channels))
            max_b = max(max_u, max_v)
            #sky_resolutions
            min_l = 1./max_b
            delta_l = min_l/oversampling
        elif resolution == None and resolution == None:
            raise ValueError("Input either a RadioTelescope object or specify a resolution")

        l_pixel_dimension = int(2./delta_l)

        if l_pixel_dimension % 2 == 0:
            l_pixel_dimension += 1

        #empty sky_image
        sky_image = numpy.zeros((l_pixel_dimension, l_pixel_dimension, n_frequencies))

        l_coordinates = numpy.linspace(-1, 1, l_pixel_dimension)


        l_shifts = numpy.diff(l_coordinates)/2.

        l_bin_edges = numpy.concatenate((numpy.array([l_coordinates[0] - l_shifts[0]]),
                                         l_coordinates[1:] - l_shifts,
                                         numpy.array([l_coordinates[-1] + l_shifts[-1]])))


        for frequency_index in range(n_frequencies):
            sky_image[:, :, frequency_index], l_bins, m_bins = numpy.histogram2d(source_l, source_m,
                                                               bins=(l_bin_edges, l_bin_edges),
                                                               weights=source_flux)

        #normalise sky image for pixel size Jy/beam
        normalised_sky_image = sky_image/(2/l_pixel_dimension)**2.

        return normalised_sky_image, l_coordinates

    def create_visibility_measurements(self):
        print(2)
        return




def stochastic_sky(seed = 0, k1=4100, gamma1=1.59, k2=4100, \
                      gamma2=2.5, S_low=400e-3, S_mid=1, S_high=5.):
    numpy.random.seed(seed)

    # Franzen et al. 2016
    # k1 = 6998, gamma1 = 1.54, k2=6998, gamma2=1.54
    # S_low = 0.1e-3, S_mid = 6.0e-3, S_high= 400e-3 Jy

    # Cath's parameters
    # k1=4100, gamma1 =1.59, k2=4100, gamma2 =2.5
    # S_low = 0.400e-3, S_mid = 1, S_high= 5 Jy

    if S_low > S_mid:
        norm = k2 * (S_high ** (1. - gamma2) - S_low ** (1. - gamma2)) / (1. - gamma2)
        n_sources = numpy.random.poisson(norm * 2. * numpy.pi)
        # generate uniform distribution
        uniform_distr = numpy.random.uniform(size=n_sources)
        # initialize empty array for source fluxes
        source_fluxes = numpy.zeros(n_sources)
        source_fluxes = \
            (uniform_distr * norm * (1. - gamma2) / k2 +
             S_low ** (1. - gamma2)) ** (1. / (1. - gamma2))
    else:
        # normalisation
        norm = k1 * (S_mid ** (1. - gamma1) - S_low ** (1. - gamma1)) / (1. - gamma1) + \
               k2 * (S_high ** (1. - gamma2) - S_mid ** (1. - gamma2)) / (1. - gamma2)
        # transition between the one power law to the other
        mid_fraction = k1 / (1. - gamma1) * (S_mid ** (1. - gamma1) - S_low ** (1. - gamma1)) / norm
        n_sources = numpy.random.poisson(norm * 2. * numpy.pi)

        #########################
        # n_sources = 1e5
        #########################

        # generate uniform distribution
        uniform_distr = numpy.random.uniform(size=n_sources)
        # initialize empty array for source fluxes
        source_fluxes = numpy.zeros(n_sources)

        source_fluxes[uniform_distr < mid_fraction] = \
            (uniform_distr[uniform_distr < mid_fraction] * norm * (1. - gamma1) / k1 +
             S_low ** (1. - gamma1)) ** (1. / (1. - gamma1))

        source_fluxes[uniform_distr >= mid_fraction] = \
            ((uniform_distr[uniform_distr >= mid_fraction] - mid_fraction) * norm * (1. - gamma2) / k2 +
             S_mid ** (1. - gamma2)) ** (1. / (1. - gamma2))
    return source_fluxes


