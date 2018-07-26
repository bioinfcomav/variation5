
import numpy

MISSING_INT = -1
MISSING_FLOAT = float('nan')
MISSING_STR = ''
MISSING_BYTE = b''
MISSING_BOOL = False
TRUE_INT = 1
FALSE_INT = 0

GT_FIELD = '/calls/GT'
GQ_FIELD = '/calls/GQ'
ALT_FIELD = '/variations/alt'
REF_FIELD = '/variations/ref'
QUAL_FIELD = '/variations/qual'
DP_FIELD = '/calls/DP'
AO_FIELD = '/calls/AO'
RO_FIELD = '/calls/RO'
AD_FIELD = '/calls/AD'
CHROM_FIELD = '/variations/chrom'
POS_FIELD = '/variations/pos'
INFO_FIELD = '/variations/info'


class _MissingValues():
    def __init__(self):
        self._missing_values = {int: MISSING_INT,
                                'Integer': MISSING_INT,
                                float: MISSING_FLOAT,
                                'Float': MISSING_FLOAT,
                                str: MISSING_STR,
                                'String': MISSING_STR,
                                numpy.int8: MISSING_INT,
                                numpy.int16: MISSING_INT,
                                numpy.int32: MISSING_INT,
                                numpy.float16: MISSING_FLOAT,
                                numpy.float32: MISSING_FLOAT,
                                numpy.bool_: MISSING_BOOL,
                                numpy.bytes_: MISSING_BYTE,
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
SNPS_PER_CHUNK = 600

MIN_NUM_GENOTYPES_FOR_POP_STAT = 10
MIN_CALL_DP_FOR_HET = 20

DEF_DSET_PARAMS = {
                   # Not much slower than lzf but compresses much more
                   'compression': 'gzip',
                   'shuffle': True,
                   # checksum, slower but safer
                   'fletcher32': True}

PRE_READ_MAX_SIZE = 10000
STATS_DEPTHS = ','.join([str(x) for x in range(0, 75, 5)])
MAX_DEPTH = 100
MIN_N_GENOTYPES = 10
SNP_DENSITY_WINDOW_SIZE = 100
MANHATTAN_WINDOW_SIZE = 100000
MAX_N_ALLELES = 4
MAX_ALLELE_COUNTS = 50
DEF_MIN_DEPTH = 20

VCF_FORMAT = 'VCFv4.2'
DEF_METADATA = {'CALLS': {b'GT': {'Description': 'Genotype',
                                  'dtype': 'int'}},
                'INFO': {}, 'FILTER': {}, 'OTHER': {},
                'VARIATIONS': {'alt': {'dtype': 'str'},
                               'chrom': {'dtype': 'str'},
                               'id': {'dtype': 'str'},
                               'pos': {'dtype': 'int32'},
                               'qual': {'dtype': 'float32'},
                               'ref': {'dtype': 'str'}}}
