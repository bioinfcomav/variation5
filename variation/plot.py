
from itertools import cycle
import bisect

import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.core.defchararray import decode
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame
from scipy.stats.morestats import probplot


plt.style.use('ggplot')


def _get_mplot_axes(axes, fhand, figsize=None, plot_type=111):
    if axes is not None:
        return axes, None
    if fhand is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = Figure(figsize=figsize)

    canvas = FigureCanvas(fig)
    axes = fig.add_subplot(plot_type)
    return axes, canvas


def _print_figure(canvas, fhand, no_interactive_win):
    if fhand is None:
        if not no_interactive_win:
            plt.show()
        return
    canvas.print_figure(fhand)


def _calc_boxplot_stats(distribs, whis=1.5, show_fliers=False):
    cum_distribs = numpy.cumsum(distribs, axis=1)
    cum_distribs_norm = cum_distribs / cum_distribs[:, -1][:, None]
    series_stats = []
    for distrib, cum_distrib, cum_distrib_norm in zip(distribs, cum_distribs,
                                                      cum_distribs_norm):
        quartiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        quartiles = [numpy.searchsorted(cum_distrib_norm, q) + 1
                     for q in quartiles]
        stats = {'q1': quartiles[1], 'med': quartiles[2], 'q3': quartiles[3]}
        stats['iqr'] = stats['q3'] - stats['q1']
        mean = numpy.sum(numpy.array(range(0, distrib.shape[0])) * distrib)
        stats['mean'] = mean / cum_distrib[-1]
        stats['whishi'] = quartiles[4]
        stats['whislo'] = quartiles[0]
        stats['cihi'] = stats['med']
        stats['cihi'] += 1.57 * stats['iqr'] / numpy.sqrt(cum_distrib[-1])
        stats['cilo'] = stats['med']
        stats['cilo'] -= 1.57 * stats['iqr'] / numpy.sqrt(cum_distrib[-1])
        stats['fliers'] = numpy.array([])
        if show_fliers:
            fliers_indices = list(range(0, int(stats['whislo'])))
            fliers_indices += list(range(int(stats['whishi']),
                                         distrib.shape[0]))
            stats['fliers'] = numpy.repeat(numpy.arange(len(fliers_indices)),
                                           fliers_indices)
        series_stats.append(stats)
    return series_stats


def calc_and_plot_boxplot(mat, by_row=True, fhand=None, axes=None,
                          no_interactive_win=False, figsize=None,
                          show_fliers=False, mpl_params={}):
    print_figure = False
    if axes is None:
        print_figure = True
    axes, canvas = _get_mplot_axes(axes, fhand, figsize=figsize)

    if by_row:
        mat = mat.transpose()
    result = axes.boxplot(mat)

    for function_name, params in mpl_params.items():
        function = getattr(axes, function_name)
        function(*params['args'], **params['kwargs'])
    if print_figure:
        _print_figure(canvas, fhand, no_interactive_win=no_interactive_win)
    return result


def plot_boxplot_from_distribs(distribs, by_row=True, fhand=None, axes=None,
                               no_interactive_win=False, figsize=None,
                               show_fliers=False, mpl_params={}, color=None):
    '''It assumes that there are as many bins in the distributions as integers
    in their range'''

    mat = distribs
    print_figure = False
    if axes is None:
        print_figure = True
    axes, canvas = _get_mplot_axes(axes, fhand, figsize=figsize)

    if not by_row:
        mat = mat.transpose()
    bxp_stats = _calc_boxplot_stats(mat)
    result = axes.bxp(bxp_stats, patch_artist=True)
    if color is not None:
        _set_box_color(result, color)

    for function_name, params in mpl_params.items():
        function = getattr(axes, function_name)
        function(*params['args'], **params['kwargs'])
    if print_figure:
        _print_figure(canvas, fhand, no_interactive_win=no_interactive_win)
    return result


def _set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['caps'], color='black')
    plt.setp(bp['medians'], color='black')


def plot_boxplot_from_distribs_series(distribs_series, by_row=True, fhand=None,
                                      axes=None, no_interactive_win=False,
                                      figsize=None, show_fliers=False,
                                      mpl_params={}, colors=None, labels=None,
                                      xticklabels=None):
    '''It assumes that there are as many bins in the distributions as integers
    in their range'''
    print_figure = False
    if axes is None:
        print_figure = True
    axes, canvas = _get_mplot_axes(axes, fhand, figsize=figsize)

    n_series = len(distribs_series)
    if colors is None:
        colors = cm.rainbow(numpy.linspace(0, 1, n_series))
    if labels is None:
        labels = [str(x) for x in range(n_series)]

    for dist_n, (distribs, color) in enumerate(zip(distribs_series, colors)):
        mat = distribs
        if not by_row:
            mat = mat.transpose()
        positions = numpy.arange(1, (n_series + 1) * mat.shape[0] + 1,
                                 n_series + 1)
        positions += dist_n
        bxp_stats = _calc_boxplot_stats(mat)
        result = axes.bxp(bxp_stats, positions=positions, patch_artist=True)
        _set_box_color(result, color)

    axes.set_xlim(0, positions[-1] + 1)
    xticks = numpy.arange((n_series + 1) / 2.0,
                          (n_series + 1) * (mat.shape[0] + 1) + 1,
                          n_series + 1)
    if xticklabels is None:
        xticklabels = numpy.arange(1, mat.shape[0] + 1)
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticklabels, rotation=90)
    # draw temporary red and blue lines and use them to create a legend
    for color, label in zip(colors, labels):
        axes.plot([], c=color, label=label)
    axes.legend()

    for function_name, params in mpl_params.items():
        function = getattr(axes, function_name)
        function(*params['args'], **params['kwargs'])
    if print_figure:
        _print_figure(canvas, fhand, no_interactive_win=no_interactive_win)
    return result


def plot_distrib(distrib, bins, fhand=None, axes=None,
                 no_interactive_win=False, figsize=None,
                 mpl_params={}, n_ticks=10, **kwargs):
    '''Plots the histogram of an already calculated distribution'''

    print_figure = False
    if axes is None:
        print_figure = True
    axes, canvas = _get_mplot_axes(axes, fhand, figsize=figsize)
    width = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
    result = axes.bar(bins[:-1], distrib, width=width, **kwargs)

    ticks = numpy.arange(0, bins.shape[0], int(bins.shape[0] / n_ticks))
    axes.set_xticks(bins[ticks])
    xticklabels = [str(x)[:len(str(x).split('.')[0]) + 4] for x in bins[ticks]]
    axes.set_xticklabels(xticklabels)

    for function_name, params in mpl_params.items():
        function = getattr(axes, function_name)
        function(*params.get('args', []), **params.get('kwargs', {}))
    if print_figure:
        _print_figure(canvas, fhand, no_interactive_win=no_interactive_win)
    return result


def plot_barplot(matrix, columns, fpath=None, stacked=True, mpl_params={},
                 axes=None, color=None, figsize=(70, 10), **kwargs):
    df = DataFrame(matrix, columns=columns)
    axes = df.plot(kind='bar', stacked=stacked, figsize=figsize, axes=axes,
                   color=color, **kwargs)
    for function_name, params in mpl_params.items():
        function = getattr(axes, function_name)
        function(*params['args'], **params['kwargs'])
    if fpath is not None:
        figure = axes.get_figure()
        figure.savefig(fpath)
    return axes


class _AxesMod():
    def __init__(self, axes):
        self.axes = axes

    def hist2d(self, matrix, xbins, ybins):
        pc = self.axes.pcolorfast(xbins, ybins, matrix)
        self.axes.set_xlim(xbins[0], xbins[-1])
        self.axes.set_ylim(ybins[0], ybins[-1])
        return matrix, xbins, ybins, pc


def plot_hist2d(matrix, xbins, ybins, fhand=None, axes=None, fig=None,
                no_interactive_win=False, figsize=None,
                mpl_params={}, colorbar_label='', **kwargs):
    print_figure = False
    if axes is None:
        print_figure = True
        fig = Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
        axes = fig.add_subplot(111)
    axesmod = _AxesMod(axes)
    result = axesmod.hist2d(matrix, xbins, ybins)
    fig.colorbar(result[3], ax=axes, label=colorbar_label)
    for function_name, params in mpl_params.items():
        function = getattr(axes, function_name)
        function(*params['args'], **params['kwargs'])
    if print_figure:
        _print_figure(canvas, fhand, no_interactive_win=no_interactive_win)
    return result


def qqplot(x, distrib, distrib_params, axes=None, mpl_params={},
           no_interactive_win=False, figsize=None, fhand=None):
    print_figure = False
    if axes is None:
        print_figure = True
        axes, canvas = _get_mplot_axes(axes, fhand, figsize=figsize)
    result = probplot(x, dist=distrib, sparams=distrib_params, plot=axes)
    for function_name, params in mpl_params.items():
        function = getattr(axes, function_name)
        function(*params['args'], **params['kwargs'])
    if print_figure:
        _print_figure(canvas, fhand, no_interactive_win=no_interactive_win)
    return result


def _look_for_first_different(items, index):
    '''It returns the index of the first index to the left of the givin index
    that is different'''
    item = items[index]
    right = bisect.bisect_right(items, item, lo=index + 1)
    return right


def manhattan_plot(chrom, pos, values, axes=None, mpl_params={},
                   no_interactive_win=False, figsize=None, fhand=None,
                   colors=['darkorchid', 'darkturquoise'],
                   show_chroms=True):

    print_figure = False
    if axes is None:
        print_figure = True
        axes, canvas = _get_mplot_axes(axes, fhand, figsize=figsize)

    colors = cycle(colors)

    start = 0
    x_offset = 0
    chroms = []
    while True:
        if start >= len(chrom):
            break
        end = _look_for_first_different(chrom, start)
        this_chrom = chrom[start]
        x = pos[start: end] + x_offset
        y = values[start: end]
        col = next(colors)
        axes.scatter(x, y, c=col, alpha=0.8,
                     edgecolors='none')
        x_offset = x[-1]
        chroms.append((this_chrom, x[-1]))
        start = end

    if show_chroms:
        xlabels = []
        xticks = []
        x_offset = 0
        for chrom, x1 in chroms:
            chrom_mean_x_pos = (x_offset + x1) / 2
            axes.axvline(x1, color='0.75')
            xlabels.append(chrom)
            xticks.append(chrom_mean_x_pos)
            x_offset = x1

        axes.set_xticks(xticks)
        axes.set_xticklabels(xlabels)

    if print_figure:
        _print_figure(canvas, fhand, no_interactive_win=no_interactive_win)

    return


def plot_lines(x, y, fhand=None, axes=None,
               no_interactive_win=False, figsize=None,
               mpl_params={}, **kwargs):
    print_figure = False
    if axes is None:
        print_figure = True
    axes, canvas = _get_mplot_axes(axes, fhand, figsize=figsize)
    result = axes.plot(x, y, **kwargs)
    for function_name, params in mpl_params.items():
        function = getattr(axes, function_name)
        function(*params['args'], **params['kwargs'])
    if print_figure:
        _print_figure(canvas, fhand, no_interactive_win=no_interactive_win)
    return result


def _write_line(fhand, items, sep):
    line = sep.join(items)
    fhand.write(line)
    fhand.write('\n')


def write_curlywhirly(fhand, labels, coords, dim_labels=None,
                      classifications=None):
    sep = '\t'

    header_items = []

    if classifications is None:
        classifications = {'all': ['all'] * len(labels)}

    categories = list(classifications.keys())
    cat_labels = ['categories:%s' % cat for cat in categories]
    header_items.extend(cat_labels)

    header_items.append('label')
    if dim_labels is None:
        dim_labels = ['dim_%i' % idx for idx in range(coords.shape[1])]
    header_items.extend(dim_labels)

    _write_line(fhand, header_items, sep)

    for idx, (label, coord) in enumerate(zip(labels, coords)):
        line_items = []
        if classifications is not None:
            classes = [classifications[cat][idx] for cat in categories]
            line_items.extend(classes)

        line_items.append(label)
        line_items.extend(map(str, coord))
        _write_line(fhand, line_items, sep)
