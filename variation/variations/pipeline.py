
import functools

import numpy

from variation import SNPS_PER_CHUNK


class Pipeline():
    def __init__(self):
        self._pipeline = []

    def append(self, funct, args=None, kwargs=None, id_=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if id_ is None:
            id_ = str(len(self._pipeline))
        funct_name = funct.__name__
        funct = functools.partial(funct, *args, **kwargs)
        
        step = {'funct': funct, 'id': id_, 'funct_name': funct_name}
        self._pipeline.append(step)

    def _pipeline_funct(self, slic3, vars_in, vars_out, kept_fields=None,
                        ignored_fields=None):
        chunk = vars_in.get_chunk(slic3, kept_fields=kept_fields,
                                  ignored_fields=ignored_fields)
        results = []
        for step in self._pipeline:
            funct = step['funct']
            chunk, bins, edges = funct(chunk)
            result = {'bins': bins,
                      'edges': edges}
            results.append(result)

        vars_out.put_chunks([chunk])

        return results

    def _reduce_results(self, results):
        result = {}
        for slice_result in results:
            for step_result, step in zip(slice_result, self._pipeline):
                step_id = step['id']
                if step_id not in result:
                    result[step_id] = {'bins': None, 'edges': None,
                                       'funct': step['funct_name']}
                if result[step_id]['bins'] is None:
                    result[step_id]['bins'] = step_result['bins']
                    result[step_id]['edges'] = step_result['edges']
                else:
                    if not numpy.allclose(step_result['edges'],
                                          result[step_id]['edges']):
                        msg = 'The bin edges for a pipeline result in step %s '
                        msg += 'funct %s do not match'
                        msg %= step['id'], step['funct_name']
                        raise RuntimeError(msg)
                    result[step_id]['bins'] += step_result['bins']
        return result

    def run(self, vars_in, vars_out=None, chunk_size=SNPS_PER_CHUNK,
            kept_fields=None, ignored_fields=None):

        slices = vars_in._create_iterate_chunk_slices(chunk_size)
        
        pipeline_funct = functools.partial(self._pipeline_funct,
                                           vars_in=vars_in,
                                           vars_out=vars_out,
                                           kept_fields=kept_fields,
                                           ignored_fields=ignored_fields)
        logs_and_hists = map(pipeline_funct, slices)
        
        return self._reduce_results(logs_and_hists)
