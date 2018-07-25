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


def annotate_tree(tree, leaf_annotations):
    for leaf_name in leaf_annotations:
        node = tree.search_nodes(name=leaf_name)
        if len(node) > 1:
            print("Ambiguous Leaf Name: {}".format(leaf_name))
        else:
            node = node[0]
            node.add_features(**leaf_annotations[leaf_name])


def _create_taxa_block(tree):
    taxa_names = tree.get_leaf_names()
    number_of_taxa = len(taxa_names)
    block = "begin taxa;\n"
    block += "\tdimensions ntax={};\n".format(str(number_of_taxa))
    block += "\ttaxlabels\n"
    for name in taxa_names:
        block += "\t\"{}\"\n".format(name)
    block += ";\nend;\n"
    return block


def _fix_tree_format_for_figtree(tree, choose_features=None):
    if choose_features:
        features = choose_features
    else:
        features = []
    tree_string = tree.write(features=features, format=2)
    tree_string = tree_string.replace("&&NHX:", "&")
    tree_string = re.sub(":(?=\D)", ",", tree_string)
    return tree_string


def _create_tree_block(tree, choose_features):
    block = "begin trees;\n"
    block += "\ttree tree1 = [&R] {}\nend;\n"
    tree_string = _fix_tree_format_for_figtree(tree, choose_features)
    block = block.format(tree_string)
    return block


class FigtreeConfig(dict):

    def __init__(self, branch_color_attribute=None,
                 leaf_label_color_attribute=None,
                 branch_line_width=None,
                 branch_labels_font_size=None,
                 leaf_labels_font_size=None
                 ):
        self["branch_color_attribute"] = branch_color_attribute
        self["leaf_label_color_attribute"] = leaf_label_color_attribute
        self["branch_line_width"] = branch_line_width
        self["branch_labels_font_size"] = branch_labels_font_size
        self["leaf_labels_font_size"] = leaf_labels_font_size

    def build_figtree_nexus_block(self):
        block = "begin figtree;\n"
        if self["branch_labels_font_size"]:
            block += "\tset branchLabels.fontSize={};\n".format(str(self["branch_labels_font_size"]))
        if self["branch_line_width"]:
            block += "\tset appearance.branchLineWidth={};\n".format(str(self["branch_line_width"]))
        if self["branch_color_attribute"]:
            block += "\tset appearance.branchColorAttribute=\"{}\";\n".format(self["branch_color_attribute"])
        if self["leaf_label_color_attribute"]:
            block += "\tset tipLabels.colorAttribute=\"{}\";\n".format(self["leaf_label_color_attribute"])
        if self["leaf_labels_font_size"]:
            block += "\tset tipLabels.fontSize={};\n".format(str(self["leaf_labels_font_size"]))
        block += "end;\n"
        return block


def write_tree_to_nexus_file(tree, fhand, figtree_config=None,
                             chosen_features=None):
    start_block = "#NEXUS\n"
    fhand.write(start_block)
    taxa_block = _create_taxa_block(tree)
    fhand.write(taxa_block)
    tree_block = _create_tree_block(tree, chosen_features)
    fhand.write(tree_block)
    if figtree_config is not None:
        options_block = figtree_config.build_figtree_nexus_block()
        fhand.write(options_block)
