# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest

from ete3 import Tree

from variation.clustering import do_tree



class ClusteringTest(unittest.TestCase):
    def test_upgma(self):
        dists = [1.45, 1.51, 1.57, 2.98, 2.94, 3.04, 7.51, 7.55, 7.39, 7.10]
        labels = ["H", "C", "G", "O", "R"]
        tree = do_tree(dists, labels, method="upgma")
        newick_tree = "((((H,C):0.73,G):0.77,O):1.49,R):3.69;"
        expected_tree = Tree(newick_tree)
        result = expected_tree.compare(tree)
        assert result["source_edges_in_ref"] - 1 < 0.00001
        assert result["ref_edges_in_source"] - 1 < 0.00001

    def test_nj(self):
        dists = [5, 9, 10, 9, 10, 8, 8, 9, 7, 3]
        labels = ["A", "B", "C", "D", "E"]
        tree = do_tree(dists, labels, method="nj")
        newick_tree = "(C:2,((A:2,B:3):3,(D:2,E:1):2):2);"
        expected_tree = Tree(newick_tree)
        result = expected_tree.compare(tree, unrooted=True)
        assert result["source_edges_in_ref"] - 1 < 0.00001
        assert result["ref_edges_in_source"] - 1 < 0.00001


if __name__ == "__main__":
    unittest.main()
