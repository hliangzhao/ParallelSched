"""
Dominant Resource Fairness (DRF).
"""
from queue import PriorityQueue
import time
import parameter as param
from scheduler.base import Scheduler


class DRF(Scheduler):
    def _schedule(self):
        """
        Only cpu and gpu resources will be considered.
        :return:
        """
        tic = time.time()
        drf_queue = PriorityQueue()
        for job in self.uncompleted_jobs:
            job.num_workers = 0
            job.num_ps = 0
            drf_queue.put((0, job.arrive_time, job))
        self.running_jobs = set()

        # keep track of available resources on each node
        node_used_cpu_list = [0] * range(param.CLUSTER_NUM_NODES)
        node_used_mem_list = [0] * range(param.CLUSTER_NUM_NODES)
        node_used_gpu_list = [0] * range(param.CLUSTER_NUM_NODES)
        node_used_bw_list = [0] * range(param.CLUSTER_NUM_NODES)

        node_used_res_queue = PriorityQueue()
        for i in range(param.CLUSTER_NUM_NODES):
            node_used_res_queue.put((0, i))
        placements = dict()

        while not drf_queue.empty():
            dom_share, job_arrival, job = drf_queue.get()
            cpu_req = job.res_worker[0] + job.res_ps[0]
            mem_req = 0
            bw_req = 0
            gpu_req = job.res_worker[1] + job.res_ps[1]

            _, node = node_used_res_queue.get()
            res_sufficient = True
            if node_used_cpu_list[node] + cpu_req > 8 or \
                    node_used_mem_list[node] + mem_req > 48 or \
                    node_used_bw_list[node] + bw_req > 10 or \
                    node_used_gpu_list[node] + gpu_req > 8:
                res_sufficient = False

            if res_sufficient:
                job.num_workers += 1
                job.num_ps += 1
                node_used_cpu_list[node] += cpu_req
                node_used_mem_list[node] += mem_req
                node_used_bw_list[node] += bw_req
                node_used_gpu_list[node] += gpu_req
                node_used_res_queue.put((node_used_cpu_list[node] + node_used_gpu_list[node], node))
                if job.j_id in placements:
                    placements[job.j_id].append(node)
                else:
                    placements[job.j_id] = [node]
                job.cur_ps_placement.append(node)
                job.cur_worker_placement.append(node)

                # update dominant resource share
                cpu_share = 1. * (job.num_workers * job.res_worker[0] + job.num_ps * job.res_ps[0]) / 48
                # mem_share = 1. * (job.num_worker * job.worker_mem + job.num_ps * job.ps_mem) / 288
                # bw_share = 1. * (job.num_worker * job.worker_bw + job.num_ps * job.ps_bw) / 60
                gpu_share = 1. * (job.num_workers * job.resr_worker[1]) / 48
                dom_share = max(cpu_share, gpu_share)
                if job.num_workers < 16 and job.num_ps < 16:
                    drf_queue.put((dom_share, job_arrival, job))

                if job not in self.running_jobs:
                    self.running_jobs.add(job)
            else:
                self.cluster_used_cpu = sum(node_used_cpu_list)
                self.cluster_used_mem = sum(node_used_mem_list)
                self.cluster_used_bw = sum(node_used_bw_list)
                self.cluster_used_gpu = sum(node_used_gpu_list)
                break

        toc = time.time()
        self.logger.debug(self.name + ":: scheduling time: " + "%.3f" % (toc - tic) + " seconds.")

        toc = time.time()
        self.logger.debug(self.name + ":: scheduling time: " + "%.3f" % (toc - tic) + " seconds.")
        for job in self.uncompleted_jobs:
            self.logger.debug(self.name + ":: scheduling results" + "job id: " + str(job.id) +
                              " num_worker: " + str(job.num_workers) +
                              " num_ps: " + str(job.num_ps))
        # a = input()
