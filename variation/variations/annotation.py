import numpy
from variation import GT_FIELD, MISSING_INT
from variation.matrix.stats import counts_and_allels_by_row
from variation.variations.filters import SampleFilter, FLT_VARS
from variation.variations.vars_matrices import VariationsArrays

ANNOTATED_VARS = 'annotated_vars'


def _filter_samples_for_stats(variations, samples=None):
    if samples is None:
        vars_for_stat = variations
    else:
        filter_samples = SampleFilter(samples)
        vars_for_stat = filter_samples(variations)[FLT_VARS]
    return vars_for_stat


def is_variable(variations, samples):
    sample_variation = _filter_samples_for_stats(variations, samples=samples)
    gts = sample_variation[GT_FIELD]
    counts, _ = counts_and_allels_by_row(gts, missing_value=MISSING_INT)
    all_missing_gts = numpy.sum(counts, axis=1) == 0
    is_variable_ = (numpy.sum(counts > 0, axis=1) > 1).astype(int)
    is_variable_[all_missing_gts] = MISSING_INT
    return is_variable_


class IsVariableAnnotator():
    def __init__(self, annot_id, samples=None):
        self.samples = samples
        self.annot_id = annot_id
        self._description = None

    def _get_description(self, variations, samples):
        if self._description is not None:
            return self._description

        if not samples or (len(samples) == len(variations.samples)):
            samples = 'All'
        else:
            samples = ', '.join(samples)

        desc = 'Is variable information between this samples: {}'.format(samples)
        self._description = desc
        return desc

    def __call__(self, variations):
        _variations = VariationsArrays()
        _variations.put_chunks(variations.iterate_chunks())
        variations = _variations

        description = self._get_description(variations, self.samples)

        variable_data = is_variable(variations, samples=self.samples)
        annotation_field = '/variations/info/{}'.format(self.annot_id)

        # add metadata to variation
        metadata = variations.metadata
        metadata[annotation_field] = {'Type': 'String', 'Number': '1',
                                      'Description': description}
        variations._set_metadata(metadata)

        # add matrix to matrix
        variations[annotation_field] = variable_data
        result = {}
        result[ANNOTATED_VARS] = variations
        return result
