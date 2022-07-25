"""
The Optimus algorithm.
"""
from queue import PriorityQueue
import time
import numpy as np
import parameter as param
from scheduler.base import Scheduler

EST_ERROR = 0.0  # change to 0.05 with estimation error


class Optimus(Scheduler):
    def est_util(self, job):
        if job.num_workers == 0:
            return -np.iinfo(np.int32).max, "worker"
        if param.PS_WORKER and job.num_ps == 0:
            return -np.iinfo(np.int32).max, "ps"

        speed = job.step(False) * (1 + EST_ERROR * np.random.choice([-1, 1], 1))
        node_used_res, node = self.node_used_res_queue.get()
        self.node_used_res_queue.put((np.sum(node_used_res), node))

        job.num_workers += 1
        job.cur_worker_placement.append(node)
        speed2 = job.step(False) * (1 + EST_ERROR * np.random.choice([-1, 1], 1))
        worker_utility = (job.num_epochs - job.progress) / speed - (job.num_epochs - job.progress) / speed2
        job.num_workers -= 1
        job.cur_worker_placement = job.cur_worker_placement[:-1]

        if param.PS_WORKER:
            job.num_ps += 1
            job.cur_ps_placement.append(node)
            speed3 = job.step(False)
            ps_utility = (job.num_epochs - job.progress) / speed - (job.num_epochs - job.progress) / speed3
            job.num_ps -= 1
            job.cur_ps_placement = job.cur_ps_placement[:-1]
            if ps_utility >= worker_utility:
                return -ps_utility, "ps"
            else:
                return -worker_utility, "worker"
        else:
            return -worker_utility, "worker"

    def _schedule(self):
        tic = time.time()
        opt_queue = PriorityQueue()
        for job in self.uncompleted_jobs:
            util, role = self.est_util(job)
            opt_queue.put((util, job, role))

        while not opt_queue.empty():
            util, job, role = opt_queue.get()
            if util >= 0:
                break
            elif role == "worker" and job.num_workers >= param.MAX_NUM_WORKERS:
                continue
            else:
                if param.PS_WORKER and role == "ps":
                    res_req = job.res_ps
                else:
                    res_req = job.res_worker
                _, node = self.node_used_res_queue.get()
                succ, node_used_res = self.cluster.alloc(res_req, node)
                self.node_used_res_queue.put((np.sum(node_used_res), node))
                if succ:
                    if param.PS_WORKER and role == "ps":
                        self._state(job.j_id, "ps")
                        job.num_ps += 1
                        job.cur_ps_placement.append(node)
                    else:
                        self._state(job.j_id, "worker")
                        job.num_workers += 1
                        job.cur_worker_placement.append(node)
                    self.running_jobs.add(job)
                    util, role = self.est_util(job)
                    opt_queue.put(util, job, role)
                else:
                    break

        toc = time.time()
        self.logger.debug(self.name + ":: scheduling time: " + "%.3f" % (toc - tic) + " seconds.")
        for job in self.uncompleted_jobs:
            self.logger.debug(self.name + ":: scheduling results type: " + str(job.j_type) +
                              " num_worker: " + str(job.num_workers) +
                              " num_ps: " + str(job.num_ps))
