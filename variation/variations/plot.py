
from variation.plot import _get_mplot_axes, _print_figure
from allel.stats.ld import rogers_huff_r, plot_pairwise_ld


def pairwise_ld(variations, axes=None, fhand=None, figsize=None,
                no_interactive_win=False):
    print_figure = False
    if axes is None:
        print_figure = True
        axes, canvas, _ = _get_mplot_axes(axes, fhand, figsize=figsize)

    gts012 = variations.gts_as_mat012

    r_ld = rogers_huff_r(gts012) ** 2
    plot_pairwise_ld(r_ld, ax=axes)

    if print_figure:
        _print_figure(canvas, fhand, no_interactive_win=no_interactive_win)
