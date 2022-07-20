"""
Job construction.
"""
import collections
import parameter as param
import numpy as np


class Job(object):
    def __init__(self, j_id, j_type, logger=None):
        self.j_id = j_id
        self.j_type = j_type
        self.logger = logger

        self.num_epochs = None
        self.real_num_epochs = None
        self.progress = 0.

        self.arrive_time = None
        self.start_time = None
        self.end_time = None

        self.num_workers = 0
        self.num_ps = 0
        self.res_worker = None
        self.res_ps = None

        self.model = None
        self.epoch_size = None
        self.local_comp_time = None
        self.model_size = None
        self.inter_bw = None
        self.intra_bw = None

        self.prev_worker_placement = None
        self.cur_worker_placement = None
        self.prev_ps_placement = None
        self.cur_ps_placement = None

        self.dom_share = 0
        self.speed_func = None
        self.training = True
        self.run_time_in_ts = 0  # only valid immediately after step() call

    def step(self, flag=True):
        """
        Adjust future epochs according to current allocated PSs and workers.
        TODO: not fully understand
        :param flag:
        :return:
        """
        assert self.progress < self.real_num_epochs
        assert self.num_workers == len(self.cur_worker_placement)
        try:
            if flag:
                assert self.num_workers <= param.MAX_NUM_WORKERS and self.num_ps <= param.MAX_NUM_WORKERS
            else:
                assert self.num_workers <= param.MAX_NUM_WORKERS + 1 and self.num_ps <= param.MAX_NUM_WORKERS + 1
        except AssertionError:
            print("num workers:", self.num_workers, "num_ps:", self.num_ps)

        if self.num_workers == 0:
            return 0.
        if param.PS_WORKER and self.num_ps == 0:
            return 0.

        if param.REAL_SPEED_TRACE or not self.training:  # always use real trace for validation
            if param.PS_WORKER:
                epoch = self.speed_func(self.num_ps, self.num_workers) * param.TS_DURATION / self.epoch_size
                num_epoch_error = param.TRAIN_SPEED_ERROR * (2 * np.random.rand() - 1)
                epoch = (1 + num_epoch_error) * epoch
            else:
                epoch = self.speed_func(self.num_workers, self.num_workers) * param.TS_DURATION / self.epoch_size
        else:
            if param.PS_WORKER:
                iter_times = []  # each worker's step time
                ps_on_node = dict()  # number of ps on each cluster node
                worker_on_node = dict()  # number of workers on each cluster node
                for node in self.cur_worker_placement:
                    if node in worker_on_node:
                        worker_on_node[node] += 1
                    else:
                        worker_on_node[node] = 1
                    if node not in ps_on_node:
                        ps_on_node[node] = 0
                for node in self.cur_ps_placement:
                    if node in ps_on_node:
                        ps_on_node[node] += 1
                    else:
                        ps_on_node[node] = 1
                    if node not in worker_on_node:
                        worker_on_node[node] = 0

                for node in self.cur_worker_placement:
                    effective_intra_bw = self.intra_bw / max(ps_on_node[node], worker_on_node[node])
                    if len(self.cur_ps_placement) == ps_on_node[node]:  # all ps in this worker node
                        worker_side_inter_bw = ps_side_inter_bw = self.inter_bw
                    else:
                        worker_side_inter_bw = self.inter_bw / (len(self.cur_ps_placement) - ps_on_node[node])
                        num_worker_list = []
                        for n in ps_on_node:
                            if ps_on_node[n] > 0:
                                num_worker_list.append(worker_on_node[n])
                        ps_side_inter_bw = self.inter_bw / (len(self.cur_worker_placement) - min(num_worker_list))

                    effective_inter_bw = min(worker_side_inter_bw, ps_side_inter_bw)
                    inter_trans_time = 2.0 * (self.model_size / len(self.cur_ps_placement)) / effective_inter_bw
                    intra_trans_time = 2.0 * (self.model_size / len(self.cur_ps_placement)) / effective_intra_bw
                    # training time of one step at a worker
                    iter_time = self.local_comp_time + max(inter_trans_time, intra_trans_time)
                    iter_times.append(iter_time)
                # each time slot is 20 minutes
                epoch = self.num_workers * param.TS_DURATION / max(iter_times) / self.epoch_size
            else:
                colocations = collections.Counter(self.cur_worker_placement)
                max_inter_trans_time = 2.0 * (1 - min(colocations.values()) / len(
                    self.cur_worker_placement)) * self.model_size / self.inter_bw
                intra_trans_time = 2.0 * min(colocations.values()) / len(
                    self.cur_worker_placement) * self.model_size / self.intra_bw
                iter_time = self.local_comp_time + max(max_inter_trans_time, intra_trans_time)
                if self.num_workers <= 8:
                    epoch = self.num_workers * param.TS_DURATION / iter_time / self.epoch_size
                else:
                    epoch = max(
                        (12 - self.num_workers / 2.0) * param.TS_DURATION / iter_time / self.epoch_size,
                        param.TS_DURATION / iter_time / self.epoch_size,
                    )

        if flag:
            if self.progress + epoch > self.real_num_epochs:
                self.run_time_in_ts = (self.real_num_epochs - self.progress) / epoch
                # TODO: is this a bug?
                epoch = self.real_num_epochs - self.progress
                self.progress = float(self.real_num_epochs)
            else:
                self.progress += epoch
                self.run_time_in_ts = 1
        return

    def get_run_time_in_ts(self):
        return self.run_time_in_ts

    def reset(self):
        """
        Reset all. Used for validation where the trace should be kept the same.
        :return:
        """
        self.progress = 0.
        self.end_time = None

        self.num_workers = 0
        self.num_ps = 0

        self.prev_worker_placement = None
        self.cur_worker_placement = None
        self.prev_ps_placement = None
        self.cur_ps_placement = None

        self.dom_share = 0

    def info(self):
        return "Job id: " + str(self.j_id) + " type: " + str(self.j_type) + " arrv time: " + str(self.arrive_time) \
               + " progress: " + str(self.progress) + " total epochs: " + str(self.real_num_epochs)
