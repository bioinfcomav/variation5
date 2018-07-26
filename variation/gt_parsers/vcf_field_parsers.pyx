import numpy as numpy
from libc.stdlib cimport atoi
from libc.stdlib cimport atof
from libc.string cimport strcmp
from cpython cimport bool
cimport cython

cimport numpy as numpy


'''Code in variation.init'''
DTYPE_INT8 = numpy.int8
DTYPE_INT16 = numpy.int16
DTYPE_INT32 = numpy.int32
DTYPE_FLOAT16 = numpy.float16
DTYPE_FLOAT32 = numpy.float32
DTYPE_BOOL = numpy.bool_


cdef:
    int MISSING_INT = -1
    float MISSING_FLOAT = float('nan')
    char * MISSING_STR = ''
    bytes MISSING_BYTE = b''
    bool MISSING_BOOL = False


class _MissingValues():
    def __init__(self):
        self._missing_values = {int: MISSING_INT,
                                float: MISSING_FLOAT,
                                str: MISSING_STR,
                                DTYPE_INT8: MISSING_INT,
                                DTYPE_INT16: MISSING_INT,
                                DTYPE_INT32: MISSING_INT,
                                DTYPE_FLOAT16: MISSING_FLOAT,
                                DTYPE_FLOAT32: MISSING_FLOAT,
                                DTYPE_BOOL: MISSING_BOOL,
                                bool: MISSING_BOOL}

    def __getitem__(self, dtype):
        str_dtype = str(dtype)
        if dtype in self._missing_values:
            return self._missing_values[dtype]
        elif isinstance(dtype, str):
            if 'str' in dtype:
                return MISSING_STR
            elif 'int' in dtype:
                return MISSING_INT
            elif 'float' in dtype:
                return MISSING_FLOAT
            elif dtype[0] == 'S':
                return MISSING_BYTE
            elif dtype[:2] == '|S':
                return MISSING_BYTE
        elif 'int' in str_dtype:
            return MISSING_INT
        elif 'float' in str_dtype:
            return MISSING_FLOAT
        elif 'bool' in str_dtype:
            return MISSING_BOOL
        elif str_dtype[:2] == '|S':
            return MISSING_BYTE
        else:
            raise ValueError('No missing type defined for type: ' + str(dtype))


MISSING_VALUES = _MissingValues()

DEF_DSET_PARAMS = {
    'compression': 'gzip',  # Not much slower than lzf, but compresses much
                            # more
    'shuffle': True,
    'fletcher32': True  # checksum, slower but safer
}

STATS_DEPTHS = ','.join([str(x) for x in range(0, 75, 5)])
cdef:
    int PRE_READ_MAX_SIZE = 10000
    int MAX_DEPTH = 100
    int MIN_N_GENOTYPES = 10
    int SNP_DENSITY_WINDOW_SIZE = 100
    int DIVERSITY_WINDOW_SIZE = 100000
    int MAX_N_ALLELES = 4
    int MAX_ALLELE_COUNTS = 50
# Speed is related to chunksize, so if you change snps-per-chunk check the
# performance
    int SNPS_PER_CHUNK = 200

    char * VCF_FORMAT = 'VCFv4.2'

DEF_METADATA = {'CALLS': {b'GT': {'Description': 'Genotype',
                                  'dtype': 'int'}},
                'INFO': {}, 'FILTER': {}, 'OTHER': {},
                'VARIATIONS': {'alt': {'dtype': 'str'},
                               'chrom': {'dtype': 'str'},
                               'id': {'dtype': 'str'},
                               'pos': {'dtype': 'int32'},
                               'qual': {'dtype': 'float16'},
                               'ref': {'dtype': 'str'}}}

'''end init code'''

MAPPED_INTS = {}

cdef:
    bytes NOT_VALUE = b'.'
    bytes COMMA = b','
    bytes TWO_DOTS = b':'
    bytes DOT_COMMA = b';'
    bytes EQUAL = b'='

    int i
    bytes bytes_i
for i in range(200):
    MAPPED_INTS[str(i).encode('utf-8')] = i
MAPPED_INTS[MISSING_BYTE] = MISSING_INT
MAPPED_INTS[NOT_VALUE] = MISSING_INT

cdef int _to_int(bytes string):
    try:
        return MAPPED_INTS[string]
    except KeyError:
        return atoi(string)


cdef float _to_float(string):
    try:
        return parse_charptr_to_py_float(string)
    except TypeError:
        return MISSING_FLOAT


cdef float parse_charptr_to_py_float(char * s):
    assert s is not NULL, "byte string value is NULL"
    if strcmp(s, MISSING_BYTE) == 0:
        return MISSING_FLOAT
    elif strcmp(s, NOT_VALUE) == 0:
        return MISSING_FLOAT
    else:
        return atof(s)


TYPE_CASTS = {int: _to_int,
              float: _to_float,
              numpy.int8: _to_int, 'int8': _to_int,
              numpy.int16: _to_int, 'int16': _to_int,
              numpy.int32: _to_int, 'int32': _to_int,
              numpy.float16: _to_float, 'float16': _to_float,
              numpy.float32: _to_float, 'float32': _to_float}


cdef _get_type_cast(dtype):
    if dtype == 'str':
        type_ = None
    elif dtype == 'bool':
        type_ = None
    else:
        type_ = TYPE_CASTS[dtype]
    return type_


PARSED_GT_FMT_CACHE = {}


cdef _parse_gt_fmt(fmt, metadata):
    global PARSED_GT_FMT_CACHE
    cdef:
        char * orig_fmt = fmt
        char * format
        char * msg
        bool number
    try:
        return PARSED_GT_FMT_CACHE[fmt]
    except KeyError:
        pass

    meta = metadata['CALLS']
    format_ = []
    for fmt in fmt.split(TWO_DOTS):
        try:
            fmt_meta = meta[fmt]
        except KeyError:
            msg = 'FORMAT metadata was not defined in header: '
            msg += fmt.decode('utf-8')
            raise RuntimeError(msg)
        dtype_ = fmt_meta['dtype']

        type_cast = _get_type_cast(dtype_)
        number = fmt_meta['Number'] != 1
        format_.append((fmt, type_cast,
                        number,  # Is list
                        fmt_meta,
                        MISSING_VALUES[fmt_meta['dtype']]))
    PARSED_GT_FMT_CACHE[orig_fmt] = format_
    return format_

PARSED_INFO_TYPE_CACHE = {}

cpdef _parse_info(info, ignored_fields, metadata):
    global PARSED_INFO_TYPE_CACHE
    if NOT_VALUE == info:
        return None
    infos = info.split(DOT_COMMA)
    parsed_infos = {}
    for info in infos:
        if EQUAL in info:
            key, val = info.split(EQUAL, 1)
        else:
            key, val = info, True
        if key in ignored_fields:
            continue
        try:
            meta = metadata['INFO'][key]
        except KeyError:
            msg = 'INFO metadata was not defined in header: '
            msg += key.decode('utf-8')
            raise RuntimeError(msg)
        try:
            type_ = PARSED_INFO_TYPE_CACHE[key]
        except KeyError:
            try:
                type_ = _get_type_cast(meta['dtype'])
                PARSED_INFO_TYPE_CACHE[key] = type_
            except KeyError:
                print(info)
                print(metadata['INFO'])
                print(meta)
                raise

        if isinstance(val, bool):
            pass
        elif COMMA in val:
            if type_ is None:
                val = val.split(COMMA)
            else:
                val = [type_(val) for val in val.split(COMMA)]
            val_to_check_len = val
        else:
            if type_ is not None:
                val = type_(val)
            val_to_check_len = [val]
        if not isinstance(meta['Number'], int):
            if not isinstance(val, list):
                val = val_to_check_len

        parsed_infos[key] = val
    return parsed_infos


PARSED_GT_CACHE = {}

cdef _parse_gt(bytes gt, list empty_gt):
    cdef bytes phased = b'|'
    cdef bytes not_phased = b'/'
    global PARSED_GT_CACHE
#     DEF ploidy = len(empty_gt)
    cdef int c_gt[2]
    cdef list gt_splited
    try:
        return PARSED_GT_CACHE[gt]
    except KeyError:
        pass
    if gt == b'.' or gt == b'./.':
        c_gt = empty_gt
    else:
        if phased in gt:
            is_phased = True
            gt_splited = gt.split(phased)
        else:
            is_phased = False
            gt_splited = gt.split(not_phased)
        for i from 0 <= i < len(gt_splited) by 1:
            c_gt[i] = _to_int(gt_splited[i])
    #             gt = [_to_int(allele) for allele in gt]
    PARSED_GT_CACHE[gt] = c_gt
    return c_gt

cdef list _gt_data_to_list(gt_data, mapper_function, missing_val,
                           int max_len_tip=1):
    cdef int max_len = 0
    cdef list missing_item
    gt_parsed_data = []
    for gt_sample_data in gt_data:
        if max_len == 0:
            # we're in the first item
            if gt_sample_data == NOT_VALUE or gt_sample_data is None:
                # it might be just one. that works in our data most of the time
                max_len = max_len_tip
                missing_item = [missing_val] * max_len
            else:
                gt_sample_data_parsed = gt_sample_data.split(COMMA)
                max_len = len(gt_sample_data_parsed)
                missing_item = [missing_val] * max_len
        if gt_sample_data == NOT_VALUE or gt_sample_data is None:
            gt_sample_data = missing_item
        else:
            gt_sample_data = gt_sample_data.split(COMMA)
            gt_sample_data = [mapper_function(item) for item in gt_sample_data]
            if len(gt_sample_data) > max_len:
                max_len = len(gt_sample_data)
                missing_item = [missing_val] * max_len
                return _gt_data_to_list(gt_data, mapper_function, missing_val,
                                        max_len_tip=max_len)
        gt_parsed_data.append(gt_sample_data)
    return gt_parsed_data


cpdef _parse_calls(fmt, calls, list ignored_fields, list kept_fields,
                   metadata, empty_gt):
    fmt = _parse_gt_fmt(fmt, metadata)
    empty_call = [NOT_VALUE] * len(fmt)
    calls = [empty_call if gt == NOT_VALUE else gt.split(TWO_DOTS)
             for gt in calls]

    for call_data in calls:
        if len(call_data) < len(fmt):
            call_data.append(b'.')

    calls = zip(*calls)
    parsed_gts = []

    cdef int max_len
    for fmt_data, gt_data in zip(fmt, calls):
        if fmt_data[0] in ignored_fields:
            continue
        if kept_fields and fmt_data[0] not in kept_fields:
            continue

        if fmt_data[0] == b'GT':
            gt_data = [_parse_gt(sample_gt, empty_gt) for sample_gt in gt_data]
        else:
            if fmt_data[2]:     # the info for a sample in this field is
                                # or should be a list
                gt_data = _gt_data_to_list(gt_data, fmt_data[1],
                                           fmt_data[4])
            else:
                gt_data = [fmt_data[1](sample_gt) for sample_gt in gt_data]

        meta = fmt_data[3]

        parsed_gts.append((fmt_data[0], gt_data))
    return parsed_gts
