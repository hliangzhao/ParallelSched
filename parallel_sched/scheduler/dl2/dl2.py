"""
The DL2 algorithm.
"""
from queue import PriorityQueue
import numpy as np
import parameter as param
from scheduler.base import Scheduler


class DL2(Scheduler):
    def __init__(self, name, trace, logger, training_mode=True):
        super().__init__(name, trace, logger)

        self.eps = 0.
        self.training_mode = training_mode
        self.sched_seq = []
        self.job_prog_in_ts = dict()
        self.window_jobs = None

        self.job_stats = dict()
        for stats_name in ["arrival", "ts_completed", "tot_completed", "duration",
                           "uncompleted", "running", "total", "backlog", "cpu_util", "gpu_util"]:
            self.job_stats[stats_name] = []

        if param.PS_WORKER and param.BUNDLE_ACTION:
            self.action_freq = [0, 0, 0]

        # for the first ts
        self._prepare()

    def _prepare(self):
        num_arrived_jobs = 0
        if self.cur_ts in self.trace:
            for job in self.trace[self.cur_ts]:
                job.reset()
                self.uncompleted_jobs.add(job)
                if not self.training_mode:
                    job.training = False
                num_arrived_jobs += 1
                self.logger.debug(job.info())
        self.job_stats["arrival"].append(num_arrived_jobs)
        self.job_stats["total"].append(len(self.completed_jobs) + len(self.uncompleted_jobs))
        self.job_stats["backlog"].append(max(len(self.uncompleted_jobs) - param.SCHED_WINDOW_SIZE, 0))

        # reset
        self._sched_states()
        self.running_jobs.clear()
        self.node_used_res_queue = PriorityQueue()
        for i in range(param.CLUSTER_NUM_NODES):
            self.node_used_res_queue.put((0, i))
        self.cluster.clear()

        for job in self.uncompleted_jobs:
            # assign each job a bundle of ps and worker first to avoid job starvation
            if param.ASSIGN_BUNDLE and param.PS_WORKER:
                _, node = self.node_used_res_queue.get()
                res_req = job.res_worker + job.res_ps
                succ, node_used_res = self.cluster.alloc(res_req, node)
                if succ:
                    job.num_workers = 1
                    job.cur_worker_placement = [node]
                    job.num_ps = 1
                    job.cur_ps_placement = [node]
                    job.dom_share = np.max(1. * (job.num_workers * job.res_worker + job.num_ps * job.res_ps) /
                                           self.cluster.CLUSTER_RES_CAPS)
                    self.running_jobs.add(job)
                else:
                    job.num_workers = 0
                    job.cur_worker_placement = []
                    job.num_ps = 0
                    job.cur_ps_placement = []
                    job.dom_share = 0

                self.node_used_res_queue.put((np.sum(node_used_res), node))

            else:
                job.num_workers = 0
                job.cur_worker_placement = []
                if param.PS_WORKER:
                    job.num_ps = 0
                    job.cur_ps_placement = []
                job.dom_share = 0

        if param.VARYING_SKIP_NUM_WORKERS:
            self.skip_num_workers = np.random.randint(1, param.MAX_NUM_WORKERS)
        else:
            self.skip_num_workers = 8  # np.random.randint(0, param.MAX_NUM_WORKERS)

        if param.VARYING_PS_WORKER_RATIO:
            self.ps_worker_ratio = np.random.randint(3, 8)
        else:
            self.ps_worker_ratio = 5

    def _move(self):
        """
        Move to next time slot.
        :return:
        """
        self._progress()
        if len(self.completed_jobs) == param.TOT_NUM_JOBS:
            self.end = True
        else:
            self.cur_ts += 1
            if self.cur_ts > param.MAX_TS_LEN:
                self.logger.error("Exceed the maximal number of time slot for one trace")
                self.logger.error("Results: " + str(self.get_results()))
                self.logger.error("Stats: " + str(self.get_job_stats()))
                for job in self.uncompleted_jobs:
                    self.logger.error("Uncompleted job " + str(job.j_id) +
                                      " total epoch: " + str(job.num_epochs) +
                                      " progress: " + str(job.progress) +
                                      " workers: " + str(job.num_workers))
                raise RuntimeError
            self._prepare()

    def step(self, output):
        """
        TODO: wrap this?
        :param output:
        :return:
        """
        mask = np.ones(param.ACTION_DIM)
        for i in range(len(self.window_jobs)):
            if self.window_jobs[i] is None:
                if param.PS_WORKER:
                    if param.BUNDLE_ACTION:
                        mask[3 * i] = 0.
                        mask[3 * i + 1] = 0.
                        mask[3 * i + 2] = 0.
                    else:
                        mask[2 * i] = 0.
                        mask[2 * i + 1] = 0.
                else:
                    mask[i] = 0.
            else:
                if param.PS_WORKER:
                    worker_full = False
                    ps_full = False
                    if self.window_jobs[i].num_workers >= param.MAX_NUM_WORKERS:
                        worker_full = True
                    if self.window_jobs[i].num_ps >= param.MAX_NUM_WORKERS:
                        ps_full = True
                    if worker_full:
                        if param.BUNDLE_ACTION:
                            mask[3 * i] = 0.
                        else:
                            mask[2 * i] = 0.
                    if ps_full:
                        if param.BUNDLE_ACTION:
                            mask[3 * i + 1] = 0.
                        else:
                            mask[2 * i + 1] = 0.
                    if (worker_full or ps_full) and param.BUNDLE_ACTION:
                        mask[3 * i + 2] = 0.

        masked_output = np.reshape(output[0] * mask, (1, len(mask)))
        sum_prob = np.sum(masked_output)
        action_vec = np.zeros(len(mask))
        move_on = True
        valid_state = False
        if ((not param.PS_WORKER) and sum(mask[:len(self.window_jobs)]) == 0) \
                or (param.PS_WORKER and (not param.BUNDLE_ACTION) and sum(mask[:2 * len(self.window_jobs)]) == 0) \
                or (param.PS_WORKER and param.BUNDLE_ACTION and sum(mask[:3 * len(self.window_jobs)]) == 0):
            self.logger.debug("All jobs are None, move on and do not save it as a sample")
            self._move()
        elif sum_prob <= 0:
            self.logger.info("All actions are masked or some action with probability 1 is masked")
            if param.EXPERIMENT_NAME is None:
                # Output: [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. 0.  1.  0.]], WHY?
                self.logger.info("Output: " + str(output))
                self.logger.info("Mask: " + str(mask))
                self.logger.info("Window jobs: " + str(self.window_jobs))
                num_worker_ps_str = ""
                for job in self.window_jobs:
                    if job:
                        num_worker_ps_str += str(job.id) + ": " + str(job.num_ps) + " " + str(job.num_workers) + ","
                self.logger.info("Job: " + num_worker_ps_str)
            self._move()
        else:
            # to decide the action
            action = -1
            masked_output = masked_output / sum_prob
            if self.training_mode:
                if np.random.rand() > param.MASK_PROB:
                    masked_output = np.reshape(output[0], (1, len(mask)))
                    action_cumsum = np.cumsum(masked_output)
                    action = (action_cumsum > np.random.randint(1, param.RAND_RANGE) / float(param.RAND_RANGE)).argmax()

                    if param.EPSILON_GREEDY:
                        if np.random.rand() < self.eps:
                            valid_actions = []
                            for i in range(len(masked_output[0])):
                                if masked_output[0][i] > param.MIN_ACTION_PROB_FOR_SKIP:
                                    valid_actions.append(i)
                            action = valid_actions[np.random.randint(0, len(valid_actions))]

                    if param.INJECT_SAMPLES:
                        if (not param.REAL_SPEED_TRACE) and (not param.PS_WORKER):
                            all_max_res = True
                            for job in self.window_jobs:
                                if job:
                                    if job.num_workers > self.skip_num_workers:
                                        continue
                                    else:
                                        all_max_res = False
                                        break
                            if all_max_res and masked_output[0][len(action_vec) - 1] > param.MIN_ACTION_PROB_FOR_SKIP \
                                    and np.random.rand() <= param.SAMPLE_INJECTION_PROB:
                                action = len(action_vec) - 1
                                self.logger.debug("Got 1.")

                        elif param.REAL_SPEED_TRACE and param.PS_WORKER:
                            if param.JOB_RES_BALANCE and param.BUNDLE_ACTION:
                                max_num_ps_worker = 0
                                min_num_ps_worker = 10 ** 10
                                min_job_idx = -1
                                for i in range(len(self.window_jobs)):
                                    job = self.window_jobs[i]
                                    if job:
                                        num_ps_worker = job.num_ps + job.num_workers
                                        if num_ps_worker > max_num_ps_worker:
                                            max_num_ps_worker = num_ps_worker
                                        if num_ps_worker < min_num_ps_worker:
                                            min_num_ps_worker = num_ps_worker
                                            min_job_idx = i
                                if min_num_ps_worker and min_job_idx != -1 and \
                                        max_num_ps_worker / min_num_ps_worker > np.random.randint(3, 6):
                                    if masked_output[0][3 * min_job_idx + 2] > param.MIN_ACTION_PROB_FOR_SKIP and \
                                            masked_output[0][3 * min_job_idx] > param.MIN_ACTION_PROB_FOR_SKIP:
                                        if np.random.rand() < 0.5:
                                            action = 3 * min_job_idx + 2
                                        else:
                                            action = 3 * min_job_idx

                            shuffle = [_ for _ in range(len(self.window_jobs))]
                            for i in shuffle:
                                job = self.window_jobs[i]
                                if job:
                                    if param.BUNDLE_ACTION:
                                        # if one of three actions: ps/worker/bundle has low probability, enforce to select it
                                        if min(self.action_freq) > 0 and \
                                                min(self.action_freq) * 1.0 / sum(self.action_freq) < 0.001:
                                            act_idx = np.argmin(self.action_freq)
                                            if mask[3 * i + act_idx] > 0 and \
                                                    masked_output[0][3 * i + act_idx] > param.MIN_ACTION_PROB_FOR_SKIP:
                                                action = 3 * i + act_idx
                                                self.logger.debug("Got 0: " + str(act_idx))
                                                break

                                        if job.num_workers == 0 or job.num_ps == 0:

                                            if job.num_ps == 0 and job.num_workers == 0 and \
                                                    mask[3 * i + 2] > 0 and \
                                                    masked_output[0][3 * i + 2] > param.MIN_ACTION_PROB_FOR_SKIP and \
                                                    np.random.rand() < 0.5:
                                                action = 3 * i + 2
                                                self.logger.debug("Got 1")

                                            if job.num_workers == 0 and mask[3 * i] > 0 and \
                                                    masked_output[0][3 * i] > param.MIN_ACTION_PROB_FOR_SKIP:
                                                action = 3 * i

                                            if job.num_ps == 0 and mask[3 * i + 1] > 0 and \
                                                    masked_output[0][3 * i] > param.MIN_ACTION_PROB_FOR_SKIP:
                                                action = 3 * i + 1

                                            break

                                        elif job.num_ps > job.num_workers * self.ps_worker_ratio and \
                                                np.random.rand() < 0.5:

                                            if mask[3 * i + 2] > 0 and \
                                                    masked_output[0][3 * i + 2] > param.MIN_ACTION_PROB_FOR_SKIP and \
                                                    mask[3 * i] > 0 and \
                                                    masked_output[0][3 * i] > param.MIN_ACTION_PROB_FOR_SKIP:

                                                if np.random.rand() < 0.5:
                                                    # increase this job's bundle
                                                    action = 3 * i + 2
                                                    self.logger.debug("Got 2.")

                                                else:
                                                    action = 3 * i
                                                    self.logger.debug("Got 2.")
                                                break
                                        elif job.num_workers >= job.num_ps * 0.5 and np.random.rand() < 0.5:
                                            if mask[3 * i + 2] > 0 and \
                                                    masked_output[0][3 * i + 2] > param.MIN_ACTION_PROB_FOR_SKIP and \
                                                    mask[3 * i + 1] > 0 and \
                                                    masked_output[0][3 * i + 1] > param.MIN_ACTION_PROB_FOR_SKIP:

                                                if np.random.rand() < 0.01:
                                                    # increase this job's bundle
                                                    action = 3 * i + 2
                                                    self.logger.debug("Got 3.")

                                                else:
                                                    # increase ps
                                                    action = 3 * i + 1
                                                    self.logger.debug("Got 4.")

                                                break

                                    else:
                                        if job.num_workers == 0 and mask[2 * i] > 0 and \
                                                masked_output[0][2 * i] > param.MIN_ACTION_PROB_FOR_SKIP and \
                                                np.random.rand() < 0.01:
                                            action = 2 * i
                                            self.logger.debug("Got 1.")
                                            break

                                        elif job.num_ps == 0 and mask[2 * i + 1] > 0 and \
                                                masked_output[0][2 * i + 1] > param.MIN_ACTION_PROB_FOR_SKIP and \
                                                np.random.rand() < 0.01:
                                            action = 2 * i + 1
                                            self.logger.debug("Got 2.")
                                            break

                                        elif job.num_ps >= job.num_workers * self.ps_worker_ratio and \
                                                mask[2 * i] > 0 and \
                                                masked_output[0][2 * i] > param.MIN_ACTION_PROB_FOR_SKIP and \
                                                np.random.rand() < 0.5:
                                            # increase this job's worker
                                            action = 2 * i
                                            self.logger.debug("Got 3.")
                                            break

                                        elif job.num_workers >= job.num_ps * self.ps_worker_ratio and \
                                                mask[2 * i + 1] > 0 and \
                                                masked_output[0][2 * i + 1] > param.MIN_ACTION_PROB_FOR_SKIP and \
                                                np.random.rand() < 0.5:
                                            # increase this job's ps
                                            action = 2 * i + 1
                                            self.logger.debug("Got 4.")
                                            break

            else:
                if param.SELECT_ACTION_MAX_PROB:
                    # output is [[...]] # always select the action with max probability
                    action = np.argmax(masked_output)
                else:
                    action_cumsum = np.cumsum(masked_output)
                    action = (action_cumsum > np.random.randint(1, param.RAND_RANGE) / float(param.RAND_RANGE)).argmax()

            action_vec[action] = 1
            # check whether skip this timeslot
            if param.SKIP_TS and action == len(action_vec) - 1:
                self._move()
                # filter out the first action that causes 0 reward??? NO
                # if sum([job.num_workers+job.num_ps for job in self.uncompleted_jobs]) > 0:
                valid_state = True
                self.sched_seq.append(None)
                self.logger.debug("Skip action is selected!")
                self.logger.debug("Output: " + str(output))
                self.logger.debug("Masked output: " + str(masked_output))
            else:
                # count action freq
                if param.PS_WORKER and param.BUNDLE_ACTION:
                    self.action_freq[action % 3] += 1

                # allocate resource
                if param.PS_WORKER:
                    if param.BUNDLE_ACTION:
                        job = self.window_jobs[action / 3]
                    else:
                        job = self.window_jobs[action / 2]
                else:
                    job = self.window_jobs[action]
                if job is None:
                    self._move()
                    self.logger.debug("The selected action is None!")
                else:
                    _, node = self.node_used_res_queue.get()
                    # get resource requirement of the selected action
                    if param.PS_WORKER:
                        if param.BUNDLE_ACTION:
                            if action % 3 == 0:
                                res_req = job.res_worker
                            elif action % 3 == 1:
                                res_req = job.res_ps
                            else:
                                res_req = job.res_worker + job.res_ps
                        else:
                            if action % 2 == 0:  # worker
                                res_req = job.res_worker
                            else:
                                res_req = job.res_ps
                    else:
                        res_req = job.res_worker

                    succ, node_used_resrs = self.cluster.alloc(res_req, node)
                    if succ:
                        move_on = False
                        # change job tasks and placement
                        if param.PS_WORKER:
                            if param.BUNDLE_ACTION:
                                if action % 3 == 0:  # worker
                                    job.num_workers += 1
                                    job.cur_worker_placement.append(node)
                                elif action % 3 == 1:  # ps
                                    job.num_ps += 1
                                    job.cur_ps_placement.append(node)
                                else:  # bundle
                                    job.num_ps += 1
                                    job.cur_ps_placement.append(node)
                                    job.num_workers += 1
                                    job.cur_worker_placement.append(node)
                            else:
                                if action % 2 == 0:  # worker
                                    job.num_workers += 1
                                    job.cur_worker_placement.append(node)
                                else:  # ps
                                    job.num_ps += 1
                                    job.cur_ps_placement.append(node)
                        else:
                            job.num_workers += 1
                            job.cur_worker_placement.append(node)

                        job.dom_share = np.max(1.0 * (job.num_workers * job.res_worker + job.num_ps * job.res_ps) /
                                               self.cluster.CLUSTER_RESR_CAPS)
                        self.node_used_res_queue.put(
                            (np.sum(node_used_resrs), node))
                        self.running_jobs.add(job)
                        valid_state = True
                        self.sched_seq.append(job)
                    else:
                        self._move()
                        self.logger.debug("No enough resources!")

        if move_on:
            r = self.rewards[-1] * move_on
        else:
            r = 0

        # invalid state, action and output when move on except for skip ts
        return masked_output, action_vec, r, move_on, valid_state

    def get_job_stats(self):
        self.job_stats["duration"] = [(job.end_time - job.arrive_time + 1) for job in self.completed_jobs]
        for name, value in self.job_stats.items():
            self.logger.debug(name + ": length " + str(len(value)) + " " + str(value))
        return self.job_stats

    def _sched_states(self):
        self.states = []
        for job in self.running_jobs:
            self.states.append((job.j_id, job.j_type, job.num_workers, job.num_ps))

    def get_sched_states(self):
        return self.states

    def get_job_reward(self):
        r = []
        for job in self.sched_seq:
            if job is None:
                if len(self.job_prog_in_ts) > 0:
                    r.append(self.rewards[-1] / len(self.job_prog_in_ts))
                else:
                    r.append(0)
            else:
                r.append(self.job_prog_in_ts[job])

        self.sched_seq = []
        self.job_prog_in_ts.clear()

        self.logger.info("Action frequency: " + str(self.action_freq))
        return r

    def _progress(self):
        r = 0
        num_ts_completed = 0
        for job in self.running_jobs:
            norm_progress = job.step() / job.num_epochs
            self.job_prog_in_ts[job] = norm_progress
            r += norm_progress
            if job.progress >= job.real_num_epochs:
                if param.FINE_GRAIN_JCT:
                    job.end_time = self.cur_ts - 1 + job.get_run_time_in_ts()
                else:
                    job.end_time = self.cur_ts
                self.uncompleted_jobs.remove(job)
                self.completed_jobs.add(job)
                num_ts_completed += 1
        self.rewards.append(r)

        self.job_stats["running"].append(len(self.running_jobs))
        self.job_stats["tot_completed"].append(len(self.completed_jobs))
        self.job_stats["uncompleted"].append(len(self.uncompleted_jobs))
        self.job_stats["ts_completed"].append(num_ts_completed)
        cpu_util, gpu_util = self.cluster.get_cluster_util()
        self.job_stats["cpu_util"].append(cpu_util)
        self.job_stats["gpu_util"].append(gpu_util)
