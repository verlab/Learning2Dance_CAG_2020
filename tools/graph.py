import numpy as np
import pdb


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """
    def __init__(self,
                 num_nodes,
                 edges,
                 center,
                 k=2,
                 strategy='spatial',
                 max_hop=1,
                 dilation=1):

        self.num_node = num_nodes
        self.edges = edges
        self.center = center
        self.k = k
        
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge()
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        # self.print_matrix(self.hop_dis)
        self.get_adjacency(strategy)
        self.adj_matrix = np.where(self.hop_dis!=np.inf,self.hop_dis,0)
        self.hop_matrix = get_hop_distance(self.num_node,self.edge,max_hop=k)
        self.lower_adj_matrix = np.where(self.hop_matrix!=np.inf,1,0)

    def __str__(self):
        return self.A

    def edge_list():
        return np.argwhere(self.adj_matrix==1)

    def getA(self,cols):
        cols = sorted(cols)
        identity = np.eye(self.num_node)[:,cols]
        return np.stack((identity,self.adj_matrix[:,cols]))
    
    def getLowAjd(self,cols):
        cols = sorted(cols)
        aux = self.lower_adj_matrix[np.ix_(cols,cols)]-np.eye(len(cols))
        aux1 = np.argwhere(aux==1).tolist()
        return aux, sorted(set(tuple(sorted(x)) for x in aux1))

    def print_matrix(self,matrix):
        np.set_printoptions(linewidth=26*26,precision=2)
        print('idx ',list(range(25)))
        for i in range(matrix.shape[0]):
            print(i,matrix[i],'\n')

    def get_edge(self):
        # edge is a list of [child, parent] paris
        self_link = [(i, i) for i in range(self.num_node)]
        self.edge = self_link + self.edges
        

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

