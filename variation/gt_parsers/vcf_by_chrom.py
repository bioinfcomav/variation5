import os
import subprocess
import gzip
import pickle
from multiprocessing import Pool
from tempfile import NamedTemporaryFile, gettempdir
from functools import partial

from variation.variations.vars_matrices import VariationsH5
from variation.gt_parsers.vcf import VCFParser


def get_chroms_in_vcf(vcf_fpath):
    cmd = ['tabix', '-l', vcf_fpath]
    output = subprocess.check_output(cmd)
    return output.splitlines()


def get_vcf_lines_for_chrom(chrom, vcf_fpath, header=True):
    if header:
        fhand = gzip.open(vcf_fpath, 'rb')
        for line in fhand:
            if line.startswith(b'#'):
                yield line
            else:
                break

    cmd = ['tabix', vcf_fpath, chrom]
    tabix = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in tabix.stdout:
        yield line


def _parse_vcf(chrom, vcf_fpath, tmp_dir, max_field_lens, max_field_str_lens,
               kept_fields, ignored_fields):
    tmp_h5_fhand = NamedTemporaryFile(prefix=chrom.decode() + '.',
                                      suffix='.tmp.h5', dir=tmp_dir)
    max_field_lens = pickle.loads(max_field_lens)
    max_field_str_lens = pickle.loads(max_field_str_lens)

    tmp_h5_fpath = tmp_h5_fhand.name
    tmp_h5_fhand.close()
    tmp_h5 = VariationsH5(tmp_h5_fpath, 'w', ignore_overflows=True,
                          ignore_undefined_fields=True)

    vcf_parser = VCFParser(get_vcf_lines_for_chrom(chrom, vcf_fpath),
                           kept_fields=kept_fields,
                           ignored_fields=ignored_fields,
                           max_field_lens=max_field_lens,
                           max_field_str_lens=max_field_str_lens,
                           max_n_vars=None)

    tmp_h5.put_vars(vcf_parser, max_field_lens=max_field_lens,
                    max_field_str_lens=max_field_str_lens)
    return tmp_h5_fpath


def _get_max_field(vcf_fpath, preread_nvars, kept_fields, ignored_fields):
    vcf_parser = VCFParser(gzip.open(vcf_fpath, 'rb'),
                           max_n_vars=preread_nvars, kept_fields=kept_fields,
                           ignored_fields=ignored_fields)
    for var in vcf_parser.variations:
        var
    max_field_lens = pickle.dumps(vcf_parser.max_field_lens)
    max_field_str_lens = pickle.dumps(vcf_parser.max_field_str_lens)
    return max_field_lens, max_field_str_lens


def _merge_h5(h5_chroms_fpaths, out_h5_fpath):
    outh5 = VariationsH5(out_h5_fpath, 'w')
    for h5_chrom_fpath in h5_chroms_fpaths:
        inh5 = VariationsH5(h5_chrom_fpath, 'r')
        outh5.put_chunks(inh5.iterate_chunks())


def _remove_temp_chrom_h5s(h5_chroms_fpaths):
    for h5_chrom_fpath in h5_chroms_fpaths:
        if os.path.exists(h5_chrom_fpath):
            os.remove(h5_chrom_fpath)


def _remove_temp_chrom_in_dir(tmp_dir):
    if tmp_dir is None:
        tmp_dir = gettempdir()
    for fname in os.listdir(tmp_dir):
        fpath = os.path.join(tmp_dir, fname)
        if fpath.endswith('.tmp.h5'):
            os.remove(fpath)


def vcf_to_h5(vcf_fpath, out_h5_fpath, n_threads, preread_nvars, tmp_dir,
               kept_fields=None, ignored_fields=None):
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    chroms = get_chroms_in_vcf(vcf_fpath)
    max_field_lens, max_field_str_lens = _get_max_field(vcf_fpath,
                                                        preread_nvars,
                                                        kept_fields=kept_fields,
                                                        ignored_fields=ignored_fields)

    partial_parse_vcf = partial(_parse_vcf, vcf_fpath=vcf_fpath,
                                tmp_dir=tmp_dir,
                                max_field_lens=max_field_lens,
                                max_field_str_lens=max_field_str_lens,
                                kept_fields=kept_fields,
                                ignored_fields=ignored_fields)
    with Pool(n_threads) as pool:
        try:
            h5_chroms_fpaths = pool.map(partial_parse_vcf, chroms)
        except Exception:
            _remove_temp_chrom_in_dir(tmp_dir)
            raise

    try:
        _merge_h5(h5_chroms_fpaths, out_h5_fpath)
    except Exception:
        raise
    finally:
        _remove_temp_chrom_h5s(h5_chroms_fpaths)
