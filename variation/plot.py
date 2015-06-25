
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def _get_mplot_axes(axes, fhand):
    if axes is not None:
        return axes, None
    if fhand is None:
        fig = plt.figure()
    else:
        fig = Figure()

    canvas = FigureCanvas(fig)
    axes = fig.add_subplot(111)
    return axes, canvas


def _print_figure(canvas, fhand):
    if fhand is None:
        plt.show()
        return
    canvas.print_figure(fhand)


def plot_histogram(mat, bins=20, range_=None, fhand=None, axes=None):

    axes, canvas = _get_mplot_axes(axes, fhand)

    axes.hist(mat, bins=bins, range=range_)

    _print_figure(canvas, fhand)
