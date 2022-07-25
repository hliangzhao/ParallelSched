"""
The Tetris algorithm.
"""
from queue import PriorityQueue
import parameter as param
import time
import numpy as np
from scheduler.base import Scheduler


class Tetris(Scheduler):
    def _schedule(self):
        tic = time.time()
        if len(self.uncompleted_jobs) > 0:
            node_used_res_queue = PriorityQueue()
            for i in range(param.CLUSTER_NUM_NODES):
                # this queue is sorted based on node id instead of available resources
                node_used_res_queue.put((i, np.zeros(param.NUM_RES_TYPES)))

            while not node_used_res_queue.empty():
                node, used_res = node_used_res_queue.get()
                mean_res_score = dict()
                mean_align_score = dict()
                for job in self.uncompleted_jobs:
                    if param.PS_WORKER:
                        res = job.res_ps + job.res_worker
                    else:
                        res = job.res_worker
                    mean_res_score[job] = np.sum(res) * (1 - job.progress / job.num_epochs)
                    mean_align_score[job] = np.sum((param.NUM_RES_SLOTS - used_res) * res)
                weight = (float(sum(mean_align_score.values())) / len(mean_align_score)) / (
                        float(sum(mean_res_score.values())) / len(mean_res_score))
                if weight == 0:
                    continue
                score_queue = PriorityQueue()
                for job in self.uncompleted_jobs:
                    score = mean_align_score[job] + weight * mean_res_score[job]
                    score_queue.put((-score, job))

                while not score_queue.empty():
                    _, job = score_queue.get()
                    if job.num_workers >= param.MAX_NUM_WORKERS:
                        continue
                    else:
                        if param.PS_WORKER:
                            res_req = job.res_worker + job.res_ps
                        else:
                            res_req = job.res_worker
                        succ, node_used_res = self.cluster.alloc(res_req, node)
                        if succ:
                            if param.PS_WORKER and param.BUNDLE_ACTION:
                                self._state(job.j_id, "bundle")
                                job.num_workers += 1
                                job.cur_worker_placement.append(node)
                                job.num_ps += 1
                                job.cur_ps_placement.append(node)
                                job.dom_share = np.max(
                                    1. * (job.num_workers * job.resr_worker + job.num_ps * job.resr_ps) /
                                    self.cluster.CLUSTER_RESR_CAPS
                                )
                            else:
                                self._state(job.j_id, "worker")
                                job.num_workers += 1
                                job.cur_worker_placement.append(node)
                                job.dom_share = np.max(
                                    1. * (job.num_workers * job.resr_worker + job.num_ps * job.resr_ps) /
                                    self.cluster.CLUSTER_RESR_CAPS
                                )
                                if param.PS_WORKER:
                                    self._state(job.j_id, "ps")
                                    job.num_ps += 1
                                    job.cur_ps_placement.append(node)
                                    job.dom_share = np.max(
                                        1. * (job.num_workers * job.resr_worker + job.num_ps * job.resr_ps) /
                                        self.cluster.CLUSTER_RESR_CAPS
                                    )
                            self.running_jobs.add(job)
                            node_used_res_queue.put((node, node_used_res))
                            break
                        else:
                            break

        toc = time.time()
        self.logger.debug(self.name + ":: scheduling time: " + "%.3f" % (toc - tic) + " seconds.")
        for job in self.uncompleted_jobs:
            self.logger.debug(
                self.name + ":: scheduling results" +
                " num_worker: " + str(job.num_workers) +
                " num_ps: " + str(job.num_ps))
