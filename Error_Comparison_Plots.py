import os
import numpy
from matplotlib import pyplot

from analytic_covariance import dft_matrix
from analytic_covariance import blackman_harris_taper
from analytic_covariance import sky_covariance
from analytic_covariance import beam_covariance
from analytic_covariance import plot_PS
from analytic_covariance import compute_ps_variance
from generaltools import from_eta_to_k_par

def main():
    u_range = numpy.logspace(0, numpy.log10(200), 100)
    frequency_range= numpy.linspace(135,165, 100)*1e6

    sky_only_raw, sky_only_cal = residual_ps_error(u_range, frequency_range, residuals='sky')
    beam_only_raw, beam_only_cal = residual_ps_error(u_range, frequency_range, residuals='sky')
    sky_and_beam_raw, sky_and_beam_cal = residual_ps_error(u_range, frequency_range, residuals='sky')


    figure, axes = pyplot.subplots()

    plot_PS(u, eta[:int(len(eta) / 2)], frequency_range, cal_variance[:, :int(len(eta) / 2)], cosmological=True,
    #             title="Calibrated Residuals", save=True, save_name=path + "/residual_calibrated_PS.pdf")
    #     plot_PS(u, eta[:int(len(eta)/2)], frequency_range, raw_variance[:, :int(len(eta) / 2)], cosmological=True,
    #             title="Uncalibrated Residuals", save=True, save_name=path + "/residual_uncalibrated_PS.pdf")
    #     plot_PS(u, eta[:int(len(eta)/2)], frequency_range, cal_variance[:, :int(len(eta) / 2)] - raw_variance[:, :int(len(eta) / 2)], cosmological=True,
    #             title="Difference", save=True, save_name=path + "/residual_difference_PS.pdf")

    return


def residual_ps_error(u_range, frequency_range, residuals ='both', path="./", plot = True):
    cal_variance = numpy.zeros((len(u_range), len(frequency_range)))
    raw_variance = numpy.zeros((len(u_range), len(frequency_range)))

    model_variance = numpy.diag(sky_covariance(0,0, frequency_range, S_low = 1, S_high = 10))
    model_normalisation = numpy.sqrt(numpy.outer(model_variance, model_variance))
    gain_error_covariance = numpy.zeros((len(u_range), len(frequency_range), len(frequency_range)))

    #Compute all residual to model ratios at different u scales
    for u_index in range(len(u_range)):
        if residuals == "sky":
            residual_covariance = sky_covariance(u_range[u_index], 0, frequency_range)
        elif residuals == "beam":
            residual_covariance = beam_covariance(u_range[u_index], v=0, nu=frequency_range)
        elif residuals == 'both':
            residual_covariance = sky_covariance(u_range[u_index], 0, frequency_range) + \
                                  beam_covariance(u_range[u_index], v=0, nu=frequency_range)
        gain_error_covariance[u_index, :, :] = residual_covariance/model_normalisation

    gain_averaged_covariance = numpy.sum(gain_error_covariance, axis=0) / (baseline_table.number_of_baselines**4)

    window_function = blackman_harris_taper(frequency_range)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)
    dftmatrix, eta = dft_matrix(frequency_range)

    #Compute the gain corrected residuals at all u scales
    for i in range(len(u_range)):
        if residuals == "sky":
            residual_covariance = sky_covariance(u_range[i], 0, frequency_range)
        elif residuals == "beam":
            residual_covariance = beam_covariance(u_range[i], v=0, nu=frequency_range)
        elif residuals == 'both':
            residual_covariance = sky_covariance(u_range[i], 0, frequency_range) + \
                                  beam_covariance(u_range[i], v=0, nu=frequency_range)

        model_covariance = sky_covariance(u_range[i], 0, frequency_range, S_low=1, S_high= 5)

        nu_cov = 2*gain_averaged_covariance*model_covariance + (1 + 2*gain_averaged_covariance)*residual_covariance

        cal_variance[i, :] = compute_ps_variance(taper1, taper2, nu_cov, dftmatrix)
        raw_variance[i, :] = compute_ps_variance((taper1, taper2, residual_covariance, dftmatrix)

    return raw_variance, cal_variance

    # if plot:
    #     plot_PS(u, eta[:int(len(eta)/2)], frequency_range, cal_variance[:, :int(len(eta) / 2)], cosmological=True,
    #             title="Calibrated Residuals", save=True, save_name=path + "/residual_calibrated_PS.pdf")
    #     plot_PS(u, eta[:int(len(eta)/2)], frequency_range, raw_variance[:, :int(len(eta) / 2)], cosmological=True,
    #             title="Uncalibrated Residuals", save=True, save_name=path + "/residual_uncalibrated_PS.pdf")
    #     plot_PS(u, eta[:int(len(eta)/2)], frequency_range, cal_variance[:, :int(len(eta) / 2)] - raw_variance[:, :int(len(eta) / 2)], cosmological=True,
    #             title="Difference", save=True, save_name=path + "/residual_difference_PS.pdf")
    #     plot_PS(u, eta[:int(len(eta)/2)], frequency_range,
    #             (cal_variance[:, :int(len(eta) / 2)] - raw_variance[:, :int(len(eta) / 2)])/raw_variance[:, :int(len(eta) / 2)], cosmological=True,
    #             title="Ratio", save=True, save_name=path + "/residual_ratio_PS.pdf", ratio=True)
    #
    #     pyplot.show()


if __name__ == "__main__":
    main())