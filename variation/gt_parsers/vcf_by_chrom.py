import os
import subprocess
import gzip
from multiprocessing import Pool
from tempfile import NamedTemporaryFile
from functools import partial

from variation.variations.vars_matrices import VariationsH5
from variation.gt_parsers.vcf import VCFParser
from variation.utils.file_utils import remove_temp_file_in_dir


def get_chroms_in_vcf(vcf_fpath):
    cmd = ['tabix', '-l', vcf_fpath]
    output = subprocess.check_output(cmd)
    return output.splitlines()


def get_vcf_lines_for_chrom(chrom, vcf_fpath, header=True):
    cmd = ['tabix']
    if header:
        cmd.append('-h')
    cmd.extend([vcf_fpath, chrom])

    # tabix = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # for line in tabix.stdout:
    #     yield line
    tabix_process = subprocess.run(cmd, stdout=subprocess.PIPE)
    for line in tabix_process.stdout.split(b'\n'):
        if line:
            yield line

    # with NamedTemporaryFile() as out_fhand:
    #     tabix_process = subprocess.run(cmd, stdout=out_fhand)
    #     out_fhand.seek(0)
    #     for line in out_fhand:
    #         yield line


def _parse_vcf(chrom, vcf_fpath, tmp_dir, kept_fields, ignored_fields):
    tmp_h5_fhand = NamedTemporaryFile(prefix=chrom.decode() + '.',
                                      suffix='.tmp.h5', dir=tmp_dir)

    tmp_h5_fpath = tmp_h5_fhand.name
    tmp_h5_fhand.close()
    tmp_h5 = VariationsH5(tmp_h5_fpath, 'w', ignore_undefined_fields=True,
                          kept_fields=kept_fields,
                          ignored_fields=ignored_fields)

    vcf_parser = VCFParser(get_vcf_lines_for_chrom(chrom, vcf_fpath),
                           kept_fields=kept_fields,
                           ignored_fields=ignored_fields)

    tmp_h5.put_vars(vcf_parser)
    tmp_h5.close()
    return tmp_h5_fpath


def _merge_h5(h5_chroms_fpaths, out_h5_fpath):
    outh5 = VariationsH5(out_h5_fpath, 'w')
    for h5_chrom_fpath in h5_chroms_fpaths:
        inh5 = VariationsH5(h5_chrom_fpath, 'r')
        outh5.put_chunks(inh5.iterate_chunks())
        inh5.close()
    outh5.close()


def _remove_temp_chrom_h5s(h5_chroms_fpaths):
    for h5_chrom_fpath in h5_chroms_fpaths:
        if os.path.exists(h5_chrom_fpath):
            os.remove(h5_chrom_fpath)


def vcf_to_h5(vcf_fpath, out_h5_fpath, n_threads, tmp_dir, kept_fields=None,
              ignored_fields=None):
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    chroms = get_chroms_in_vcf(vcf_fpath)

    partial_parse_vcf = partial(_parse_vcf, vcf_fpath=vcf_fpath,
                                tmp_dir=tmp_dir,
                                kept_fields=kept_fields,
                                ignored_fields=ignored_fields)
    with Pool(n_threads) as pool:
        try:
            h5_chroms_fpaths = pool.map(partial_parse_vcf, chroms)
        except Exception:
            remove_temp_file_in_dir(tmp_dir, '.tmp.h5')
            raise

    try:
        _merge_h5(h5_chroms_fpaths, out_h5_fpath)
    except Exception:
        raise
    finally:
       _remove_temp_chrom_h5s(h5_chroms_fpaths)
