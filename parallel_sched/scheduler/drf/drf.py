"""
Dominant Resource Fairness (DRF).
"""
from queue import PriorityQueue
import time
import numpy as np
import parameter as param
from scheduler.base import Scheduler


class DRF(Scheduler):
    """
    TODO: This is just a FIFO. Sth. wrong in the implementation?
    """
    def _schedule(self):
        tic = time.time()
        drf_queue = PriorityQueue()
        for job in self.uncompleted_jobs:
            drf_queue.put((0, job.arrive_time, job))

        while not drf_queue.empty():
            _, job_arrival, job = drf_queue.get()
            _, node = self.node_used_res_queue.get()
            if param.PS_WORKER:
                res_req = job.res_worker + job.res_ps
            else:
                res_req = job.res_worker
            succ, node_used_res = self.cluster.alloc(res_req, node)
            self.node_used_res_queue.put((np.sum(node_used_res), node))
            if succ:
                if param.PS_WORKER and param.BUNDLE_ACTION and False:
                    self._state(job.j_id, "bundle")
                    job.num_workers += 1
                    job.cur_worker_placement.append(node)
                    job.num_ps += 1
                    job.cur_ps_placement.append(node)
                    job.dom_share = np.max(1. * (job.num_workers * job.res_worker + job.num_ps * job.res_ps) /
                                           self.cluster.CLUSTER_RES_CAPS)

                else:
                    self._state(job.j_id, "worker")
                    job.num_workers += 1
                    job.cur_worker_placement.append(node)
                    job.dom_share = np.max(1. * (job.num_workers * job.res_worker + job.num_ps * job.res_ps) /
                                           self.cluster.CLUSTER_RES_CAPS)

                    if param.PS_WORKER:
                        job.num_ps += 1
                        job.cur_ps_placement.append(node)
                        job.dom_share = np.max(1. * (job.num_workers * job.res_worker + job.num_ps * job.res_ps) /
                                               self.cluster.CLUSTER_RES_CAPS)

                self.running_jobs.add(job)
                if job.num_workers < param.MAX_NUM_WORKERS and job.num_ps < param.MAX_NUM_WORKERS:
                    drf_queue.put((job.dom_share, job_arrival, job))

            else:
                break

        toc = time.time()
        self.logger.debug(self.name + ":: scheduling time: " + "%.3f" % (toc - tic) + " seconds.")
        for job in self.uncompleted_jobs:
            self.logger.debug(
                self.name + ":: scheduling results" +
                " num_worker: " + str(job.num_workers) +
                " num_ps: " + str(job.num_ps))
