import sys

sys.path.extend(['../'])
from graph import tools


num_node = 5
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 3), (1, 4), (1, 5), (4, 5)]
# inward_ori_index = [(1,0), (2,1), (3,2), (4,0), (5,4), (6,5), (0,7), (8,7),
#                 (10,8), (11,8), (12,11), (13,12), (14,8), (15, 14), (16,15)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
