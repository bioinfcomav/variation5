from itertools import zip_longest, islice


def first(iterable):
    for item in iterable:
        return item
    raise ValueError('No items in iterable, no first item')


def group_items(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


class PeekableIterator(object):
    def __init__(self, iterable):
        self._stream = iterable
        self._buffer = []
        self._peek_buffer_idx = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._buffer:
            if self._peek_buffer_idx:
                self._peek_buffer_idx -= 1
            return self._buffer.pop(0)
        return next(self._stream)

    def peek(self):
        if self._peek_buffer_idx is None:
            try:
                item = next(self._stream)
            except StopIteration:
                raise
            self._buffer.append(item)
            return item
        else:
            item = self._buffer[self._peek_buffer_idx]
            self._peek_buffer_idx += 1
            if self._peek_buffer_idx >= len(self._buffer):
                self._peek_buffer_idx = None
            return item

    def reset_peek(self):
        if not self._buffer:
            self._peek_buffer_idx = None
        else:
            self._peek_buffer_idx = 0


def group_in_packets(iterable, packet_size):
    'ABCDE -> (A, B), (C, D), (E,)'
    iterable = iter(iterable)
    while True:
        chunk = tuple(islice(iterable, packet_size))
        if not chunk:
            break
        yield chunk
