
MISSING_INT = -1
MISSING_GT = MISSING_INT
MISSING_FLOAT = float('nan')
MISSING_STR = ''
FILLING_INT = -2
FILLING_FLOAT = float('-inf')
FILLING_STR = MISSING_STR

# Speed is related to chunksize, so if you change snps-per-chunk check the
# performance
SNPS_PER_CHUNK = 200

DEF_DSET_PARAMS = {
    'compression': 'gzip',  # Not much slower than lzf, but compresses much
                            # more
    'shuffle': True,
    'fletcher32': True  # checksum, slower but safer
}