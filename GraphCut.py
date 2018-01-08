import numpy as np
import maxflow
from utils import warn


def graphcut(img1, img2, mask):
    """
    Inputs:
        Mask:
            The 80px area out of the boundary. 2 dims.
            Pixels on the edge of Img1 are marked with 1 and same for Img2. Internal pixels are marked with 3.
        Img1 & Img2:
            Here Img1 means source img, aka. the input img. Img2 is the candidate.
            Both Img1 and Img2 should be in OpenCV format (y, x, c). Both are actually gradient maps.
    Outputs:
        SegMap:
            0 if not in masked area. 1 if be with img1 and 2 if img2.
    """
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    assert len(mask.shape) == 2
    if not (img1.shape[:2] == img2.shape[:2] == mask.shape):
        warn("Image/Mask shape does not match! Img1 = {}, Img2 = {}, Mask = {}".format(
            img1.shape, img2.shape, mask.shape), domain=__file__)

    shape = mask.shape
    estimated_nodes = (mask > 0).sum()
    estimated_edges = estimated_nodes * 4

    g = maxflow.Graph[int](estimated_nodes, estimated_edges)
    nodes = g.add_nodes(estimated_nodes)
    nodemap = {}
    nodemap_inv = {}

    for y in range(shape[0]):
        for x in range(shape[1]):
            if mask[y, x] > 0:
                nodemap[len(nodemap)] = (y, x)
                nodemap_inv[(y, x)] = len(nodemap_inv)

    for y in range(shape[0]):
        for x in range(shape[1]):
            if y + 1 < shape[0] and mask[y + 1, x] > 0:
                value = np.abs(img1[y, x] - img2[y, x]).sum() + \
                    np.abs(img1[y + 1, x] - img2[y + 1, x]).sum()
                g.add_edge(nodes[nodemap_inv[(y, x)]],
                           nodes[nodemap_inv[(y + 1, x)]], value, value)
            if x + 1 < shape[1] and mask[y, x + 1] > 0:
                value = np.abs(img1[y, x] - img2[y, x]).sum() + \
                    np.abs(img1[y, x + 1] - img2[y, x + 1]).sum()
                g.add_node(nodes[nodemap_inv[(y, x)]],
                           nodes[nodemap_inv[(y, x + 1)]], value, value)
            if mask[y, x] == 1:  # Connected to src
                g.add_tedge(nodes[nodemap_inv[(y, x)]], np.inf, 0)
            if mask[y, x] == 2:  # Connected to dst
                g.add_tedge(nodes[nodemap_inv[(y, x)]], 0, np.inf)

    flow = g.maxflow()
    # PyMaxflow: 1 if node is segmented with src node else 0
    segmentation_result = 2 - np.array(list(map(g.get_segment, nodes)))
    segmap = np.zeros_like(mask)

    for i, result in enumerate(segmentation_result):
        segmap[nodemap_inv[i]] = result

    return segmap, flow


if __name__ == "__main__":
    pass
