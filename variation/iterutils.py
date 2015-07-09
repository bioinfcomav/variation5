
def first(iterable):
    for item in iterable:
        return item
    raise ValueError('No items in iterable, no first item')
