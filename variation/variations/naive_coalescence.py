import numpy


def get_mean_alleles_per_window(variations, win_size, win_step=None,
                                count_unknown_alleles=False):

    for window in variations.iterate_wins(win_size=win_size, win_step=win_step):
        yield _calc_mean_alleles_per_window(window, count_unknown_alleles=count_unknown_alleles)


def _calc_mean_alleles_per_window(window, count_unknown_alleles=False):

    window_view = window["/calls/GT"].view().astype('float')
    num_vars, num_samples, _ = window_view.shape
    window_view = window_view.reshape(num_vars, num_samples*2)
    if not count_unknown_alleles:
        window_view[window_view == -1] = numpy.nan
    window_view = numpy.sort(window_view, axis=1)
    # it also filters nan
    allele_count_per_locus = (numpy.diff(window_view) > 0).sum(axis=1)+1
    mean_alleles_per_window = numpy.mean(allele_count_per_locus,
                                         dtype=numpy.float64)
    return (mean_alleles_per_window)
