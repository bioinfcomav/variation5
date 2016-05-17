
import functools

import numpy

from variation import SNPS_PER_CHUNK
from variation.variations.filters import COUNTS, EDGES, FLT_VARS
from variation.variations.vars_matrices import VariationsArrays


class Pipeline():
    def __init__(self):
        self._pipeline = []

    def append(self, callable_instance, id_=None):
        if id_ is None:
            id_ = str(len(self._pipeline))
        filter_name = callable_instance.__class__.__name__            
        
        step = {'callable': callable_instance, 'id': id_, 'name': filter_name}
        self._pipeline.append(step)

    def _pipeline_funct(self, slic3, vars_in, kept_fields=None,
                        ignored_fields=None):
        chunk = vars_in.get_chunk(slic3, kept_fields=kept_fields,
                                  ignored_fields=ignored_fields)
        results = []
        for step in self._pipeline:
            # This for should be more internal than the for for the HDF5
            # chunks because in that way the chunk won't abandond the
            # processor cache until of the processing is done for that chunk
            callable_instance = step['callable']
            result = callable_instance(chunk)
            if FLT_VARS in result:
                chunk = result[FLT_VARS]
                del result[FLT_VARS]
            results.append(result)

        return results, chunk

    def _reduce_results(self, results, vars_out):
        result = {}
        for slice_result, chunk in results:
            vars_out.put_chunks([chunk])
            for step_result, step in zip(slice_result, self._pipeline):
                step_id = step['id']

                if step_id not in result:
                    callable_instance = step['callable']
                    result[step_id] = {'name': step['name']}

                if callable_instance.do_histogram:
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

        return result

    def _check_and_fix_histogram_ranges(self, vars_in, chunk_size, kept_fields,
                                        ignored_fields):
        callables_to_check = []
        for step in self._pipeline:
            callable_instance = step['callable']
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
        
        for slic3 in vars_in._create_iterate_chunk_slices(chunk_size):
            chunk = vars_in.get_chunk(slic3, kept_fields=kept_fields,
                                      ignored_fields=ignored_fields)
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
                

    def run(self, vars_in, vars_out, chunk_size=SNPS_PER_CHUNK,
            kept_fields=None, ignored_fields=None):

        self._check_and_fix_histogram_ranges(vars_in, chunk_size,
                                             kept_fields=kept_fields,
                                             ignored_fields=ignored_fields)
        
        pipeline_funct = functools.partial(self._pipeline_funct,
                                           vars_in=vars_in,
                                           kept_fields=kept_fields,
                                           ignored_fields=ignored_fields)
        # To use the Single Writer Multiple Reader feature of HDF5 we have
        # to spawn several readers, one per mapping process, but only one
        # writer. So the results should go through a socket to the main
        # process to be written
        slices = vars_in._create_iterate_chunk_slices(chunk_size)
        results_and_chunks = map(pipeline_funct, slices)
        
        return self._reduce_results(results_and_chunks, vars_out)
