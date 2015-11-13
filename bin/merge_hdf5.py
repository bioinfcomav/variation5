#!/usr/bin/env python
import argparse
from variation.variations.vars_matrices import VariationsH5
from variation.genotypes_matrix import merge_variations
from numpy import mean


def _setup_argparse(**kwargs):
    'It prepares the command line argument parsing.'
    parser = argparse.ArgumentParser(**kwargs)

    parser.add_argument('input',
                        help='Input HDF5 files', nargs=2)
    parser.add_argument('-o', '--output', required=True,
                        help='Output HDF5 file path')
    help_msg = 'Ignore SNPs with overlaps in the same file'
    parser.add_argument('-i', '--ignore_overlaps', action='store_true',
                        default=False, help=help_msg)
    help_msg = 'Ignore SNPs with more than one overlaps between files. '
    help_msg += 'It is not possible merge more than two SNPs without phase info'
    parser.add_argument('-di', '--ignore_more_overlaps', action='store_true',
                        default=False, help=help_msg)
    help_msg = 'Merge two values in one with an operation. For example: merge '
    help_msg += 'two snps with different quality, what do you want? The '
    help_msg += 'minimum of two, the maximum, the mean...'
    help_msg += 'To maximum = max, minimum = min, mean = mean'
    help_msg += 'Example of use: qual=min. By default both values are stored'
    parser.add_argument('-ff', '--fields_function', action='append',
                        help=help_msg, default=[],)
    parser.add_argument('-if', '--ignore_fields', default=[],
                        action='append',
                        help='Fields to avoid writing to HDF5 file (None)')
    return parser


def _parse_args(parser):
    parsed_args = parser.parse_args()
    args = {}
    args['in_fpaths'] = parsed_args.input
    args['out_fpath'] = parsed_args.output
    args['ignore_overlaps'] = parsed_args.ignore_overlaps
    args['ignore_more_overlaps'] = parsed_args.ignore_more_overlaps
    args['fields_func'] = parsed_args.fields_function
    args['ignore_fields'] = parsed_args.ignore_fields
    return args


def main():
    description = 'Merge HDF5 files into a new HDF5 file'
    parser = _setup_argparse(description=description)
    args = _parse_args(parser)
    fields_function = {}
    allowed_functions = {'min': min, 'max': max, 'mean': mean}
    for field_f in args['fields_func']:
        field, function = field_f.split('=')
        if function not in allowed_functions:
            raise ('Function not supported')
        fields_function[field] = allowed_functions[function]
    merged_fpath = args['out_fpath']
    h5_1 = VariationsH5(args['in_fpaths'][0], 'r')
    h5_2 = VariationsH5(args['in_fpaths'][1], 'r')
    try:
        merge_variations(h5_1, h5_2, merged_fpath,
                         ignore_overlaps=args['ignore_overlaps'],
                         ignore_2_or_more_overlaps=args['ignore_more_overlaps'],
                         fields_funct=fields_function,
                         ignore_fields=args['ignore_fields'])
    except FileExistsError:
        raise ('The output file already exists. Remove it to create a new one')


if __name__ == '__main__':
    main()
