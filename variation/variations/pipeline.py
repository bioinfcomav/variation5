
import numpy

from variation import SNPS_PER_CHUNK
from variation.variations.filters import (COUNTS, EDGES, FLT_VARS, FLT_STATS,
                                          N_KEPT, TOT, N_FILTERED_OUT,
                                          SELECTED_VARS)
from collections import OrderedDict
from variation.variations.annotation import ANNOTATED_VARS


class Pipeline():
    def __init__(self):
        self._pipeline = []

    def append(self, callable_instance, id_=None):

        if id_ is None:
            id_ = str(len(self._pipeline))
        filter_name = callable_instance.__class__.__name__

        step = {'callable': callable_instance, 'id': id_, 'name': filter_name,
                'order': len(self._pipeline)}
        self._pipeline.append(step)

    def _pipeline_funct(self, chunk):
        results = []
        for step in self._pipeline:
            # This for should be more internal than the for for the HDF5
            # chunks because in that way the chunk won't abandond the
            # processor cache until of the processing is done for that chunk

            if chunk.num_variations == 0:
                continue

            callable_instance = step['callable']
            result = callable_instance(chunk)
            if FLT_VARS in result:
                chunk = result[FLT_VARS]
                del result[FLT_VARS]
            elif ANNOTATED_VARS in result:
                chunk = result[ANNOTATED_VARS]
                del result[ANNOTATED_VARS]

            results.append(result)
        return results, chunk

    def _reduce_results(self, results, vars_out):
        result = OrderedDict()
        for slice_result, chunk in results:
            if vars_out is not None:
                vars_out.put_chunks([chunk])
            for step_result, step in zip(slice_result, self._pipeline):
                step_id = step['id']
                callable_instance = step['callable']

                if step_id not in result:
                    result[step_id] = {'name': step['name'],
                                       'order': step['order']}

                if not hasattr(callable_instance, 'do_histogram'):
                    do_hist = False
                else:
                    do_hist = callable_instance.do_histogram
                if not step_result:
                    continue
                if do_hist:
                    if COUNTS not in result[step_id]:
                        result[step_id][COUNTS] = step_result[COUNTS]
                        result[step_id][EDGES] = step_result[EDGES]
                    else:
                        if not numpy.allclose(step_result[EDGES],
                                              result[step_id][EDGES]):
                            msg = 'The bin edges for a pipeline '
                            msg += 'result in step %s '
                            msg += 'funct %s do not match'
                            msg %= step['id'], step['name']
                            raise RuntimeError(msg)
                        result[step_id][COUNTS] += step_result[COUNTS]

                if SELECTED_VARS in step_result:
                    result[step_id][SELECTED_VARS] = step_result[SELECTED_VARS]
                if FLT_STATS in step_result:
                    if FLT_STATS not in result[step_id]:
                        result[step_id][FLT_STATS] = step_result[FLT_STATS]
                    else:
                        n_kept = step_result[FLT_STATS][N_KEPT]
                        tot = step_result[FLT_STATS][TOT]
                        flt_out = step_result[FLT_STATS][N_FILTERED_OUT]
                        result[step_id][FLT_STATS][N_KEPT] += n_kept
                        result[step_id][FLT_STATS][TOT] += tot
                        result[step_id][FLT_STATS][N_FILTERED_OUT] += flt_out
        return result

    def _check_and_fix_histogram_ranges(self, vars_in, chunk_size, kept_fields,
                                        ignored_fields):
        callables_to_check = []
        for step in self._pipeline:
            callable_instance = step['callable']
            if not hasattr(callable_instance, 'do_histogram'):
                continue
            if (callable_instance.do_histogram and
               callable_instance.range is None):
                callables_to_check.append(callable_instance)
        if not callables_to_check:
            return

        original_do_filterings = []
        for callable_instance in callables_to_check:
            original_do_filterings.append(callable_instance.do_filtering)
            callable_instance.do_filtering = False

        mins = []
        maxs = []
        for callable_instance in callables_to_check:
            mins.append(None)
            maxs.append(None)

        for chunk in vars_in.iterate_chunks(kept_fields=kept_fields,
                                            ignored_fields=ignored_fields,
                                            chunk_size=chunk_size):
            for idx, callable_instance in enumerate(callables_to_check):
                result = callable_instance(chunk)
                min_, max_ = result[EDGES][0], result[EDGES][-1]
                if mins[idx] is None or mins[idx] > min_:
                    mins[idx] = min_
                if maxs[idx] is None or maxs[idx] < max_:
                    maxs[idx] = max_

        for idx, callable_instance in enumerate(callables_to_check):
            callable_instance.do_filtering = original_do_filterings[idx]
            callable_instance.range = mins[idx], maxs[idx]

    def run(self, vars_in, vars_out=None, chunk_size=SNPS_PER_CHUNK,
            kept_fields=None, ignored_fields=None):

        self._check_and_fix_histogram_ranges(vars_in, chunk_size,
                                             kept_fields=kept_fields,
                                             ignored_fields=ignored_fields)

        chunks = vars_in.iterate_chunks(kept_fields=kept_fields,
                                        ignored_fields=ignored_fields,
                                        chunk_size=chunk_size)

        results_and_chunks = map(self._pipeline_funct, chunks)

        return self._reduce_results(results_and_chunks, vars_out)
