# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

from tempfile import NamedTemporaryFile
import unittest

from ete3 import Tree


from variation.clustering import (do_tree, get_subtrees, annotate_tree,
                                  write_tree_to_nexus_file, FigtreeConfig)




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

    def test_cluster_selection_by_cutoff(self):
        dists = [1.45, 1.51, 1.57, 2.98, 2.94, 3.04, 7.51, 7.55, 7.39, 7.10]
        labels = ["H", "C", "G", "O", "R"]
        ultrametric_length = 10
        tree = do_tree(dists, labels, method="upgma")
        tree.convert_to_ultrametric(tree_length=ultrametric_length)

        leaves = {frozenset(node.get_leaf_names())
                  for node in get_subtrees(tree, dist_treshold=4)}
        assert leaves == {frozenset({'O'}), frozenset({'C', 'H', 'G'}),
                          frozenset({'R'})}

        leaves = {frozenset(node.get_leaf_names())
                  for node in get_subtrees(tree, dist_treshold=0)}
        assert leaves == {frozenset({'C', 'H', 'G', 'O', 'R'})}

        leaves = {frozenset(node.get_leaf_names())
                  for node in get_subtrees(tree, dist_treshold=15)}
        assert not leaves

        leaves = {frozenset(node.get_leaf_names())
                  for node in get_subtrees(tree, dist_treshold=7)}
        assert leaves == {frozenset({'R'}), frozenset({'G'}),
                          frozenset({'C', 'H'}), frozenset({'O'})}

        leaves = {frozenset(node.get_leaf_names())
                  for node in get_subtrees(tree, dist_treshold=8)}
        assert leaves == {frozenset({'R'}), frozenset({'C'}),
                          frozenset({'H'}), frozenset({'O'}),
                          frozenset({'G'})}

        dists = [1.45, 1.51, 1.57, 2.98, 2.94, 3.04, 7.51, 7.55, 7.39, 7.10]
        labels = ["H", "C", "G", "O", "R"]
        tree = do_tree(dists, labels, method="nj")

        leaves = {frozenset(node.get_leaf_names())
                  for node in get_subtrees(tree, dist_treshold=4)}
        assert leaves == {frozenset({'R'})}

    def test_draw_figtree(self):
        dists = [1.45, 1.51, 1.57, 2.98, 2.94, 3.04, 7.51, 7.55, 7.39, 7.10]
        labels = ["H", "C", "G", "O", "R"]
        tree = do_tree(dists, labels, method="nj")
        leaf_annotations = {"R": {"group1": "A", "group2": "C", "group3": "A"},
                            "O": {"group1": "A", "group2": "A", "group3": "A"},
                            "H": {"group1": "B", "group2": "B", "group3": "B"},
                            "C": {"group1": "B", "group2": "A", "group3": "B"},
                            "G": {"group1": "C", "group2": "D", "group3": "B"}
                            }
        annotate_tree(tree, leaf_annotations)
        figtree_config = FigtreeConfig(branch_color_attribute="group1",
                                       leaf_label_color_attribute="group2",
                                       )
        with NamedTemporaryFile(mode="w") as test_fhand:
            write_tree_to_nexus_file(tree, test_fhand,
                                     figtree_config=figtree_config)
            test_fhand.flush()
            test_string = open(test_fhand.name, "r").read()
            tree_string = test_string.split(";")[5]
            leaves = tree_string.split("],")
            assert "H:0.71[&" in leaves[0]
            assert "group1=B" in leaves[0]
            assert "G:0.755[&" in leaves[2]
            assert "group1=C" in leaves[2]
            assert "set appearance.branchColorAttribute=\"group1\";" in test_string
            assert "set tipLabels.colorAttribute=\"group2\";" in test_string

        figtree_config = FigtreeConfig(branch_color_attribute="group1")
        chosen_features = ["group1", "group2"]
        with NamedTemporaryFile(mode="w") as test_fhand:
            write_tree_to_nexus_file(tree, test_fhand,
                                     figtree_config=figtree_config,
                                     chosen_features=chosen_features)
            test_fhand.flush()
            test_string = open(test_fhand.name, "r").read()
            tree_string = test_string.split(";")[5]
            leaves = tree_string.split("],")
            assert "group1=A" in leaves[2]
            assert "set appearance.branchColorAttribute=\"group1\";" in test_string
            assert "set tipLabels.colorAttribute=\"group2\";" not in test_string
            assert "group3" not in leaves[0]
            assert "group3" not in leaves[2]


if __name__ == "__main__":
    unittest.main()
