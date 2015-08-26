
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


def _print_figure(canvas, fhand, no_interactive_win):
    if fhand is None:
        if not no_interactive_win:
            plt.show()
        return
    canvas.print_figure(fhand)


def plot_histogram(mat, bins=10, range_=None, fhand=None, axes=None,
                   no_interactive_win=False):

    axes, canvas = _get_mplot_axes(axes, fhand)

    result = axes.hist(mat, bins=bins, range=range_)
    _print_figure(canvas, fhand, no_interactive_win=no_interactive_win)
    return result
