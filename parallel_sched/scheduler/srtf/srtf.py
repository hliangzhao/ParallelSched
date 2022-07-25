"""
Shorted Remaining Time First (SRTF).
"""
from scheduler.base import Scheduler
import time
from queue import PriorityQueue
import parameter as param
import numpy as np


class SRTF(Scheduler):
    def _schedule(self):
        """
        SRTF schedules jobs in the sequence of their remaining time.
        :return:
        """
        tic = time.time()

        srtf_queue = PriorityQueue()
        for job in self.uncompleted_jobs:
            srtf_queue.put((1 - job.progress / job.num_epochs, job.arrive_time, job))

        flag = False
        while not srtf_queue.empty():
            _, _, job = srtf_queue.get()
            """
            Allocate maximal number of workers.
            Bundle one ps and one worker together by default.
            """
            for _ in range(param.MAX_NUM_WORKERS):
                _, node = self.node_used_res_queue.get()
                if param.PS_WORKER:
                    res_req = job.res_worker + job.res_ps
                else:
                    res_req = job.res_worker

                succ, node_used_res = self.cluster.alloc(res_req, node)
                self.node_used_res_queue.put((np.sum(node_used_res), node))
                if succ:
                    # TODO: why always False?
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
                                               self.cluster.CLUSTER_RESR_CAPS)
                        if param.PS_WORKER:
                            self._state(job.j_id, "ps")
                            job.num_ps += 1
                            job.cur_ps_placement.append(node)
                            job.dom_share = np.max(1. * (job.num_workers * job.res_worker + job.num_ps * job.res_ps) /
                                                   self.cluster.CLUSTER_RESR_CAPS)

                    self.running_jobs.add(job)
                else:
                    flag = True
                    break
            if flag:
                break

        toc = time.time()
        self.logger.debug(self.name + ":: scheduling time: " + "%.3f" % (toc - tic) + " seconds.")
        for job in self.uncompleted_jobs:
            self.logger.debug(self.name + ":: scheduling results num_worker: " + str(job.num_workers))
