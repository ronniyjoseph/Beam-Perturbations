import numpy
from mpl_toolkits.axes_grid1 import make_axes_locatable


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    return fig.colorbar(mappable, cax=cax)

def symlog_bounds(data):
    data_min = numpy.nanmin(data)
    data_max = numpy.nanmax(data)

    if data_min == 0:
        indices = numpy.where(data > 0)[0]
        if len(indices) == 0:
            lower_bound = -0.1
        else:
            lower_bound = numpy.nanmin(data[indices])
    else:
        lower_bound = numpy.abs(data_min)/data_min*numpy.abs(data_min)

    if data_max == 0:
        indices = numpy.where(data < 0)[0]
        if len(indices) == 0:
            upper_bound = 0.1
        else:
            upper_bound = numpy.nanmax(data[indices])
    else:
        upper_bound = numpy.abs(data_max)/data_max*numpy.abs(data_max)

    ### Figure out what the lintresh is (has to be linear)
    threshold = 1e-3*min(numpy.abs(lower_bound), numpy.abs(upper_bound))
    #### Figure out the linscale parameter (has to be in log)
    scale = numpy.log10(upper_bound - lower_bound)/7

    return lower_bound, upper_bound, threshold, scale
