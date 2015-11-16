#!/usr/bin/env python

import sys
import argparse

from variation.vcf import read_gzip_file
from variation.variations.vars_matrices import VariationsH5
from variation.genotypes_matrix import (GenotypesMatrixParser, STANDARD_GT,
                                        IUPAC_CODING)


def _setup_argparse(**kwargs):
    'It prepares the command line argument parsing.'
    parser = argparse.ArgumentParser(**kwargs)

    help_msg = 'Input CSV file containing genotypes (default STDIN)'
    parser.add_argument('input', nargs=1, help=help_msg, default=None)
    parser.add_argument('-o', '--output', required=True,
                        help='Output HDF5 file path')
    help_msg = 'File containing SNP information if not given in the input. '
    help_msg += 'id field must match for all of the SNPs'
    parser.add_argument('-m', '--metadata_fpath', default=None, help=help_msg)
    help_msg = 'Genotypes are encoded using IUPAC nomenclature. By default '
    help_msg += 'They are encoded using alleles directly (ex: CT)'
    parser.add_argument('-i', '--iupac_coding', default=False,
                        action='store_true', help=help_msg)
    parser.add_argument('-s', '--separator', default=',',
                        help='Sep caracter between different fields (def: ,)')
    parser.add_argument('-a', '--max_alt_allele', default=1, type=int,
                        help='Max number of alternative alleles (def: 1)')
    help_msg = 'Comma separated list of fieldnames containing SNPs information'
    help_msg += '. By default it includes id, pos and chrom'
    parser.add_argument('-f', '--snp_fieldnames', default='id,pos,chrom',
                        help=help_msg)
    return parser


def _parse_args(parser):
    parsed_args = parser.parse_args()
    args = {}
    if parsed_args.input is None:
        args['in_fhand'] = sys.stdin
    else:
        in_fpath = parsed_args.input[0]
        if in_fpath.split('.')[-1] == 'gz':
            args['in_fhand'] = read_gzip_file(in_fpath)
        else:
            args['in_fhand'] = open(in_fpath, 'r')
    args['out_fpath'] = parsed_args.output
    if parsed_args.metadata_fpath is None:
        args['metadata_fhand'] = None
    else:
        args['metadata_fhand'] = open(args['metadata_fpath'])
    args['sep'] = parsed_args.separator
    args['max_alt_allele'] = parsed_args.max_alt_allele
    if parsed_args.iupac_coding:
        args['gt_coding'] = IUPAC_CODING
    else:
        args['gt_coding'] = STANDARD_GT
    args['snp_fieldnames'] = parsed_args.snp_fieldnames.split(',')
    return args


def main():
    description = 'Transforms CSV file into HDF5 format'
    parser = _setup_argparse(description=description)
    args = _parse_args(parser)
    
    in_fhand = args['in_fhand']
    snp_fieldnames = args['snp_fieldnames']
    metadata_fhand = args['metadata_fhand']
    max_alt_allele = args['max_alt_allele']
    gt_matrix_parser = GenotypesMatrixParser(fhand=in_fhand,
                                             gt_coding=args['gt_coding'], 
                                             max_alt_allele=max_alt_allele,
                                             sep=args['sep'],
                                             metadata_fhand=metadata_fhand,
                                             snp_fieldnames=snp_fieldnames)
    h5 = VariationsH5(args['out_fpath'], mode='w')
    h5.put_vars_from_csv(gt_matrix_parser)


if __name__ == '__main__':
    main()
