
from io import BytesIO
import gzip
import pickle
import sys


class CCache():
    def __init__(self, compresslevel=4):
        self._fhand = BytesIO()
        self._cache_to_write = gzip.GzipFile(fileobj=self._fhand, mode='wb',
        compresslevel=compresslevel)
        self.items_read = 0

    def put(self, item):
        pickle.dump(item, self._cache_to_write, protocol=-1)
        self.items_read += 1

    def put_iterable(self, iterable, max_items=0, max_size=0):
        for i, item in enumerate(iterable):
            self.put(item)
            if max_items and i + 1 >= max_items:
                break
            if max_size and sys.getsizeof(self._fhand) > max_size:
                break

    def _stop_put(self):
        self._cache_to_write.flush()
        self._fhand.seek(0)
        self._cache_to_read = gzip.GzipFile(fileobj=self._fhand, mode='rb')

    @property
    def items(self):
        self._stop_put()
        while True:
            try:
                item = pickle.load(self._cache_to_read)
            except EOFError:
                return
            yield item


def test():
    queue = CCache()
    queue.put('hola')
    queue.put('adios')
    queue.put_iterable([1, 2])
    queue.put_iterable([3, 4], max_items=1)
    assert list(queue.items) == ['hola', 'adios', 1, 2, 3]


if __name__ == '__main__':
    test()
