
from variation import SNPS_PER_CHUNK


class Pipeline():
    def __init__(self, variations, chunk_size=SNPS_PER_CHUNK, kept_fields=None,
                 ignored_fields=None):
        self._steps = []
        self._variations = variations
        self.chunk_size = chunk_size
        self.kept_fields = kept_fields
        self.ignored_fields = ignored_fields

    def append(self, funct, args=None, kwargs=None, id_=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if id_ is None:
            id_ = str(len(self._steps))

        funct = self._wrap_funct(funct)

        step = {'funct': funct, 'args': args, 'kwargs': kwargs, 'id_': id_}
        self._steps.append(step)

    def _wrap_funct(self, funct):
        ig_fields = self.ignored_fields

        def process_slice(slic3):
            chunk = self._variations.get_chunk(kept_fields=self.kept_fields,
                                               ignored_fields=ig_fields)
            return funct(chunk)
        return funct

    def run(self, snps_out=None):

        slices = self.variations._create_iterate_chunk_slices(self.chunk_size)
        reducer = list
        for step in self._steps:
            funct = step['funct']
            chunk_results = map(funct, slices)
            reduce(reducer, chunk_results)
