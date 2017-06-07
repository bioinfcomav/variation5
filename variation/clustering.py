import io
import re

from Bio.Phylo.TreeConstruction import _DistanceMatrix as BioDistMatrix
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.NewickIO import write as write_newick

from ete3 import TreeNode


def _convert_biotree_to_etetree(bio_tree):
    fhand = io.StringIO()
    write_newick([bio_tree], fhand)
    newick = fhand.getvalue()
    newick = re.sub("Inner[0-9]+:", ":", newick)
    ete_tree = TreeNode(newick)
    return (ete_tree)


def _create_biopython_dist(labels, dists):
    idx = 0
    row = 0
    biopython_triangular = [[0]]
    while True:
        next_idx = idx + row + 1
        row_dists = list(dists[idx: next_idx])
        if not row_dists:
            break
        row_dists.append(0)
        biopython_triangular.append(row_dists)
        row += 1
        idx = next_idx
    return BioDistMatrix(labels, biopython_triangular)


def _do_tree(dists, labels, do_tree_funct):
    dists = _create_biopython_dist(labels, dists)
    bio_tree = do_tree_funct(dists)
    ete_tree = _convert_biotree_to_etetree(bio_tree)
    return ete_tree


def do_tree(dists, labels, method):
    constructor = DistanceTreeConstructor()
    if method == "upgma":
        do_upgma = constructor.upgma
        return _do_tree(dists, labels, do_upgma)
    elif method == "nj":
        do_nj = constructor.nj
        return _do_tree(dists, labels, do_nj)

    raise ValueError("Unknown Method: {}".format(method))


def get_subtrees(tree, dist_treshold):
    if dist_treshold == 0:
        yield tree
    else:
        root = tree.get_tree_root()
        for node in tree.traverse():
            distance = tree.get_distance(node, target2=root, topology_only=False)
            if distance < dist_treshold:
                continue
            parent = node.up
            parent_dist_to_root = tree.get_distance(parent, target2=root,
                                                    topology_only=False)
            if parent_dist_to_root < dist_treshold:
                yield node
