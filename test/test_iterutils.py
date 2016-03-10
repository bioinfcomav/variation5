import unittest

from variation.iterutils import PeekableIterator, group_in_packets


class PeekTest(unittest.TestCase):
    def test_peekable_iter(self):
        iterator = iter(['a', 'b', 'c', 'd', 'e', 'f'])
        iterator = PeekableIterator(iterator)
        assert 'a' == iterator.peek()
        assert 'b' == iterator.peek()
        assert 'a' == next(iterator)
        assert 'c' == iterator.peek()
        iterator.reset_peek()
        assert 'b' == iterator.peek()
        assert 'b' == next(iterator)

        iterator = iter(['a', 'b', 'c', 'd', 'e', 'f'])
        iterator = PeekableIterator(iterator)
        iterator.peek()
        iterator.peek()
        iterator.peek()
        iterator.reset_peek()
        assert 'a' == iterator.peek()
        iterator = iter(['a', 'b', 'c', 'd', 'e', 'f'])
        iterator = PeekableIterator(iterator)
        iterator.peek()
        iterator.peek()
        iterator.peek()
        iterator.peek()
        iterator.peek()
        iterator.peek()

        try:
            iterator.peek()
            self.fail('stopIteration expected')
        except StopIteration:
            pass
        assert 'a' == next(iterator)

        iterator = iter(['a', 'b', 'c', 'd', 'e', 'f'])
        iterator = PeekableIterator(iterator)
        assert 'a' == iterator.peek()
        assert 'a' == next(iterator)
        assert 'b' == iterator.peek()

    def test_group_in_packets(self):
        'It groups an iterator in packets of items'
        packets = list(group_in_packets(range(4), 2))
        assert packets == [(0, 1), (2, 3)]

        packets = [packet for packet in group_in_packets(range(5), 2)]
        assert packets == [(0, 1), (2, 3), (4,)]

        packets = list(group_in_packets([], 2))
        assert packets == []

if __name__ == '__main__':
    unittest.main()
