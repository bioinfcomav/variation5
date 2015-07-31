from itertools import zip_longest


def first(iterable):
    for item in iterable:
        return item
    raise ValueError('No items in iterable, no first item')


def group_items(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

