import unittest
from variation.iterutils import PeekableIterator


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

if __name__ == '__main__':
    unittest.main()
