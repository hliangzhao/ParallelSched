"""
The base class for Scheduler.
"""
from queue import PriorityQueue
import numpy as np
import parameter as param
from cluster import Cluster
import log


class Scheduler(object):
    def __init__(self, name, trace, logger):
        self.name = name
        self.trace = trace
        if logger is None:
            assert name
            self.logger = log.get_logger(name=name, file_handler=False)
        else:
            self.logger = logger

        self.cluster = Cluster(self.logger)
        self.cur_ts = 0
        self.end = False

        self.running_jobs = set()
        self.uncompleted_jobs = set()
        self.completed_jobs = set()

        # all state-action pairs in one ts
        self.data = None
        self.rewards = []

        self.window_jobs = None

    def step(self):
        """
        On step forward in a time slot.
        :return:
        """
        assert not self.end
        self._prepare()
        self._schedule()
        self._progress()

        if len(self.completed_jobs) == param.TOT_NUM_JOBS:
            self.end = True
        self.cur_ts += 1

        return self.data

    def get_results(self):
        """
        Return completed jobs number, avg. complete time, makespan, and avg. reward.
        :return:
        """
        jcts = [(job.end_time - job.arrive_time + 1.) for job in self.completed_jobs]
        makespan = max([job.end_time + 1. for job in self.completed_jobs])
        assert jcts
        return len(self.completed_jobs), 1. * sum(jcts) / len(jcts), makespan, sum(self.rewards) / len(self.rewards)

    def get_job_cts(self):
        """
        Get the complete time of all the completed jobs.
        :return:
        """
        jcts = dict()
        for job in self.completed_jobs:
            jcts[job.j_id] = job.end_time = job.arrive_time + 1.
        return jcts

    def _prepare(self):
        self.cluster.clear()
        self.data = []
        self.running_jobs.clear()
        if self.cur_ts in self.trace:
            for job in self.trace[self.cur_ts]:
                # must reset since it is trained for multiple epochs
                job.reset()
                self.uncompleted_jobs.add(job)
                self.logger.debug(job.info())

        for job in self.uncompleted_jobs:
            job.num_workers = 0
            job.cur_worker_placement = []
            if param.PS_WORKER:
                job.num_ps = 0
                job.cur_ps_placement = []

        # sort based on used resources from smallest to largest for load balancing
        # TODO: Further, optimize the job placement simultaneously
        self.node_used_res_queue = PriorityQueue()
        for i in range(param.CLUSTER_NUM_NODES):
            self.node_used_res_queue.put((0, i))

    def _schedule(self):
        self.logger.info("Implement this!")

    def _progress(self):
        """
        Get reward of the scheduling decision in this time slot.
        :return:
        """
        reward = 0
        for job in self.running_jobs.copy():
            epoch = job.step()
            reward += epoch / job.num_epochs
            if job.progress >= job.real_num_epochs:
                if param.FINE_GRAIN_JCT:
                    job.end_time = self.cur_ts - 1 + job.get_run_time_in_ts()
                else:
                    job.end_time = self.cur_ts
                # self.running_jobs.remove(job)
                self.uncompleted_jobs.remove(job)
                self.completed_jobs.add(job)

        if param.NUM_UNCOMPLETED_JOB_REWARD:
            reward = len(self.uncompleted_jobs)

        self.rewards.append(reward)

    def observe(self):
        # for test, first use dominant resource share of each job as input state
        q = PriorityQueue()
        for job in self.uncompleted_jobs:
            if param.PS_WORKER:
                if job.num_workers >= param.MAX_NUM_WORKERS and job.num_ps >= param.MAX_NUM_WORKERS:
                    continue
            else:
                if job.num_workers >= param.MAX_NUM_WORKERS:
                    continue

            if param.JOB_SORT_PRIORITY == param.PRIORITIES[0]:
                q.put((job.dom_share, job.arrive_time, job))
            elif param.JOB_SORT_PRIORITY == param.PRIORITIES[1]:
                q.put((job.arrive_time, job.arrive_time, job))
            elif param.JOB_SORT_PRIORITY == param.PRIORITIES[2]:
                q.put((1 - job.progress / job.num_epochs, job.arrive_time, job))

        if param.ZERO_PADDING:
            state = np.zeros(shape=param.STATE_DIM)
        else:
            state = -1 * np.ones(shape=param.STATE_DIM)

        self.window_jobs = [None for _ in range(param.SCHED_WINDOW_SIZE)]

        shuffle = np.array([i for i in range(param.SCHED_WINDOW_SIZE)])
        if param.JOB_ORDER_SHUFFLE:
            shuffle = np.random.choice(param.SCHED_WINDOW_SIZE, param.SCHED_WINDOW_SIZE, replace=False)

        for order in shuffle:
            if not q.empty():
                _, _, job = q.get()
                j = 0
                for inputs, enable in param.INPUTS_GATE:
                    if enable:
                        if inputs == "TYPE":
                            if not param.INPUT_RESCALE:
                                if not param.TYPE_BINARY:
                                    state[j][order] = job.type
                                else:
                                    bin_str = "{0:b}".format(job.type).zfill(4)
                                    for bin_ch in bin_str:
                                        state[j][order] = int(bin_ch)
                                        j += 1
                                    j -= 1
                            else:
                                state[j][order] = float(job.type) / 8

                        elif inputs == "STAY":
                            if not param.INPUT_RESCALE:
                                state[j][order] = self.cur_ts - job.arrv_time
                            else:
                                state[j][order] = float(self.cur_ts - job.arrv_time) / 100

                        elif inputs == "PROGRESS":
                            state[j][order] = 1 - job.progress / job.num_epochs

                        elif inputs == "DOM_RES":
                            state[j][order] = job.dom_share

                        elif inputs == "WORKERS":
                            if not param.INPUT_RESCALE:
                                state[j][order] = job.num_workers
                            else:
                                state[j][order] = float(job.num_workers) / param.MAX_NUM_WORKERS

                        elif inputs == "PS":
                            if not param.INPUT_RESCALE:
                                state[j][order] = job.num_ps
                            else:
                                state[j][order] = float(job.num_ps) / param.MAX_NUM_WORKERS

                        else:
                            raise RuntimeError
                        j += 1

                self.window_jobs[order] = job

            self.logger.debug("ts: {}, backlog: {}, completed jobs: {}, uncompleted jobs: {}".format(
                str(self.cur_ts),
                str(max(len(self.uncompleted_jobs) - param.SCHED_WINDOW_SIZE, 0)),
                str(len(self.completed_jobs)),
                str(len(self.uncompleted_jobs)),
            ))
            return state

    def _state(self, label_job_id, role="worker"):
        """
        Whether this action leads to worker increment or ps increment.
        """
        state_inputs = self.observe()
        labels = np.zeros(param.ACTION_DIM)

        for i in range(param.SCHED_WINDOW_SIZE):
            job = self.window_jobs[i]
            if job and job.j_id == label_job_id:
                if param.PS_WORKER:
                    if param.BUNDLE_ACTION:
                        if role == "worker":
                            labels[i * 3] = 1
                        elif role == "ps":
                            labels[i * 3 + 1] = 1
                        elif role == "bundle":
                            labels[i * 3 + 2] = 1
                    else:
                        if role == "worker":
                            labels[i * 2] = 1
                        elif role == "ps":
                            labels[i * 2 + 1] = 1
                else:
                    labels[i] = 1

        self.data.append((state_inputs, labels))
