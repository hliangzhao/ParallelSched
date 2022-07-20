"""
Cluster construction.
"""
import parameter as param
import numpy as np


class Cluster(object):
    def __init__(self, logger):
        self.logger = logger

        self.CLUSTER_RES_CAPS = np.array(
            [param.CLUSTER_NUM_NODES * param.NUM_RES_SLOTS for _ in range(param.NUM_RES_TYPES)]
        )
        self.NODE_RES_CAPS = np.array(
            [param.NUM_RES_SLOTS for _ in range(param.NUM_RES_TYPES)]
        )
        self.cluster_state = np.zeros((param.NUM_RES_TYPES, param.CLUSTER_NUM_NODES * param.NUM_RES_SLOTS))
        self.node_used_res = np.zeros((param.CLUSTER_NUM_NODES, param.NUM_RES_TYPES))

    def clear(self):
        self.cluster_state.fill(0)
        self.node_used_res.fill(0)

    def alloc(self, res_req, node):
        """
        Allocate res for one task from node.
        :param res_req:
        :param node:
        :return:
        """
        if np.any(np.greater(self.node_used_res[node] + res_req, self.NODE_RES_CAPS)):
            return False, self.node_used_res[node]
        else:
            self.node_used_res[node] += res_req
            # update cluster state
            for i in range(param.NUM_RES_TYPES):
                req = res_req[i]
                if req > 0:
                    start_idx = node * param.NUM_RES_SLOTS
                    for j in range(param.NUM_RES_SLOTS):
                        if self.cluster_state[i, j + start_idx] == 0:
                            self.cluster_state[i, j + start_idx] = 1
                            req -= 1
                            if req == 0:
                                break
            return True, self.node_used_res[node]

    def get_cluster_state(self):
        return self.cluster_state.copy()

    def get_cluster_utilizations(self):
        utilizations = list([
            float(np.sum(self.node_used_res[:, i])) / self.CLUSTER_RES_CAPS[i] for i in range(param.NUM_RES_TYPES)
        ])
        return utilizations
