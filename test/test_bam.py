# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join

from test.test_utils import TEST_DATA_DIR
from variation.variations.vars_matrices import VariationsArrays
from variation.gt_parsers.bam import BAMParser
from variation.gt_parsers.bam import _parse_bams as parse_bams
from variation.gt_parsers.bam import _infer_alleles as infer_alleles
from variation.gt_parsers.bam import _generate_regions as generate_regions
from variation import GT_FIELD, CHROM_FIELD, POS_FIELD, AD_FIELD, REF_FIELD


class BamParserTest(unittest.TestCase):

    def test_parse_bam(self):
        bam_fpath = join(TEST_DATA_DIR, 'example.rg.bam')
        parser = BAMParser([bam_fpath], kmer_size=4, ploidy=2,
                           min_num_samples=2,
                           max_field_lens={'alt': 1, 'CALLS': {b'AD': 3}},
                           max_field_str_lens={'chrom': 20})

        snps = VariationsArrays(ignore_undefined_fields=True)
        snps.put_vars(parser)
        assert snps.ploidy
        assert list(snps.chroms) == ['ref']
        assert snps.num_variations == 4
        assert len(snps[REF_FIELD]) == 4
        assert len(snps[REF_FIELD][0]) == 4

        assert list(snps[CHROM_FIELD]) == ['ref', 'ref', 'ref', 'ref']
        assert list(snps[POS_FIELD]) == [15, 16, 17, 36]
        assert AD_FIELD in snps
        assert GT_FIELD in snps


class TestBamParsing(unittest.TestCase):
    def test_bam_parsing(self):
        bam_fpath = join(TEST_DATA_DIR, 'example.rg.bam')
        loci = list(parse_bams([bam_fpath], kmer_size=4)['loci'])
        locus = loci[0]
        assert locus['location'] == (0, 6)
        assert locus['coverage'] == 1
        assert locus['kmer_counts']['sample1']['TTAG'] == 1

    def test_bam_parsing_no_rg(self):
        bam_fpath = join(TEST_DATA_DIR, 'example.no_rg.bam')
        loci = list(parse_bams([bam_fpath], kmer_size=4)['loci'])
        locus = loci[0]
        assert locus['location'] == (0, 6)
        assert locus['coverage'] == 1
        assert locus['kmer_counts'][None]['TTAG'] == 1

    def test_min_sample_coverage(self):
        bam_fpath = join(TEST_DATA_DIR, 'example.rg.bam')

        loci = list(parse_bams([bam_fpath], kmer_size=4,
                    min_sample_coverage=2)['loci'])
        assert len(loci) == 7

        loci = list(parse_bams([bam_fpath], kmer_size=4,
                    min_sample_coverage=3)['loci'])
        assert len(loci) == 3

        loci = list(parse_bams([bam_fpath], kmer_size=4,
                    min_sample_coverage=4)['loci'])
        assert not loci

    def test_min_num_samples(self):
        bam_fpath = join(TEST_DATA_DIR, 'example.rg.bam')

        loci = list(parse_bams([bam_fpath], kmer_size=4)['loci'])
        assert len(loci) == 24

        loci = list(parse_bams([bam_fpath], kmer_size=4,
                               min_num_samples=2)['loci'])
        assert len(loci) == 4

    def test_bams_parsing(self):
        bam_fpath1 = join(TEST_DATA_DIR, 'example.rg.bam')
        bam_fpath2 = join(TEST_DATA_DIR, 'example.rg.bam')
        bams = [bam_fpath1, bam_fpath2]
        loci = list(parse_bams(bams, kmer_size=4)['loci'])
        locus = loci[0]
        assert locus['location'] == (0, 6)
        assert locus['coverage'] == 2
        assert locus['kmer_counts']['sample1']['TTAG'] == 2

    def test_generate_regions(self):
        regions = generate_regions(references=['ref'],
                                   reference_lens={'ref': 3},
                                   step=1)
        assert list(regions) == [('ref', 0, 1), ('ref', 1, 2), ('ref', 2, 3)]

        regions = generate_regions(references=['ref1', 'ref2'],
                                   reference_lens={'ref1': 2, 'ref2': 2},
                                   step=1)
        assert list(regions) == [('ref1', 0, 1), ('ref1', 1, 2),
                                 ('ref2', 0, 1), ('ref2', 1, 2)]

        regions = generate_regions(references=['ref1', 'ref2'],
                                   reference_lens={'ref1': 4, 'ref2': 4},
                                   step=2)
        assert list(regions) == [('ref1', 0, 1), ('ref1', 2, 3),
                                 ('ref2', 0, 1), ('ref2', 2, 3)]

    def test_step(self):
        bam_fpath = join(TEST_DATA_DIR, 'example.rg.bam')

        loci = list(parse_bams([bam_fpath], kmer_size=4, step=1)['loci'])
        assert len(loci) == 24

        loci = list(parse_bams([bam_fpath], kmer_size=4, step=2)['loci'])
        assert len(loci) == 12



class TestInferAlleles(unittest.TestCase):
    def _prepare_counts(self, kmer_counts):
        return [{'kmer_counts': {'sample1': kmer_counts}}]

    def _check_alleles(self, loci, sample_alleles):
        # print(loci[0]['alleles']['sample1'], sample_alleles)
        assert loci[0]['alleles']['sample1'] == sample_alleles

    def test_infer_alleles(self):

        alleles = [({'ACTG': 9, 'TTTT': 2}, {'ACTG': 9, 'TTTT': 2})]
        alleles.append(({'ACTG': 10}, {'ACTG': 10}))
        alleles.append(({'ACTG': 6, 'TTTT': 2, 'extra_for_ploidy': 1},
                        {'ACTG': 6, 'TTTT': 2}))
        alleles.append(({'ACTG': 6, 'TTTT': 5, 'TTTA': 1},
                        {'ACTG': 6, 'TTTT': 6}))
        alleles.append(({'ACTG': 60, 'TTTT': 25},
                        {'ACTG': 60}))
        alleles.append(({'ACTG': 11, 'TTTT': 1},
                        {'ACTG': 11}))
        alleles.append(({'ACTG': 60, 'TTTT': 45},
                        {'ACTG': 60, 'TTTT': 45}))

        for in_alleles, out_alleles in alleles:
            loci = self._prepare_counts(in_alleles)
            loci = list(infer_alleles(loci, ploidy=2))
            self._check_alleles(loci, out_alleles)

if __name__ == "__main__":
    # import sys; sys.argv = ['', 'BamParserTest']
    unittest.main()
