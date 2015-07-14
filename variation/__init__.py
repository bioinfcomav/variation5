
import numpy

MISSING_INT = -1
MISSING_FLOAT = float('nan')
MISSING_STR = ''
MISSING_BYTE = b''
MISSING_BOOL = False


class _MissingValues():
    def __init__(self):
        self._missing_values = {int: MISSING_INT,
                                float: MISSING_FLOAT,
                                str: MISSING_STR,
                                numpy.int8: MISSING_INT,
                                numpy.int16: MISSING_INT,
                                numpy.int32: MISSING_INT,
                                numpy.float16: MISSING_FLOAT,
                                numpy.float32: MISSING_FLOAT,
                                numpy.bool_: MISSING_BOOL,
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


# Speed is related to chunksize, so if you change snps-per-chunk check the
# performance
SNPS_PER_CHUNK = 200

DEF_DSET_PARAMS = {
    'compression': 'gzip',  # Not much slower than lzf, but compresses much
                            # more
    'shuffle': True,
    'fletcher32': True  # checksum, slower but safer
}