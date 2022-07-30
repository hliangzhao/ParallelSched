"""
Validation.
"""
import numpy as np
import time
import parameter as param
from scheduler.drf import drf
from scheduler.fifo import fifo
from scheduler.tetris import tetris
from scheduler.srtf import srtf
from scheduler.optimus import optimus
from scheduler.dl2 import dl2


def val_loss(net, val_traces, logger, global_step):
    """
    Validate the loss of the heuristic.
    """
    avg_loss = 0
    step = 0
    data = []
    sched = None
    for episode in range(len(val_traces)):
        job_trace = val_traces[episode]
        if param.HEURISTIC == "DRF":
            sched = drf.DRF("DRF", job_trace, logger)
        elif param.HEURISTIC == "FIFO":
            sched = fifo.FIFO("FIFO", job_trace, logger)
        elif param.HEURISTIC == "SRTF":
            sched = srtf.SRTF("SRTF", job_trace, logger)
        elif param.HEURISTIC == "Tetris":
            sched = tetris.Tetris("Tetris", job_trace, logger)
        elif param.HEURISTIC == "Optimus":
            sched = optimus.Optimus("Optimus", job_trace, logger)

        ts = 0
        while not sched.end:
            data += sched.step()
            ts += 1
            if len(data) >= param.MINI_BATCH_SIZE:
                # prepare a validation batch
                indexes = np.random.choice(len(data), size=param.MINI_BATCH_SIZE, replace=False)
                inputs = []
                labels = []
                for idx in indexes:
                    ipt, lbl = data[idx]
                    inputs.append(ipt)
                    labels.append(lbl)
                # supervised learning to calculate gradients
                outputs, loss = net.get_sl_loss(np.stack(inputs), np.vstack(labels))
                avg_loss += loss

                step += 1
                data = []
    return avg_loss / step


def val_jmr(net, val_traces, logger, global_step, tb_logger):
    """
    Validate the job mean response of DL2.
    """
    avg_jct = []
    avg_makespan = []
    avg_reward = []
    step = 0.0

    tic = time.time()

    stats = dict()
    stats["step"] = global_step
    stats["jcts"] = []

    states = dict()
    states["step"] = global_step
    states["states"] = []

    for episode in range(len(val_traces)):
        job_trace = val_traces[episode]
        sched = dl2.DL2("DL2", job_trace, logger, False)
        ts = 0
        while not sched.end:
            inputs = sched.observe()
            outputs = net.predict(np.reshape(inputs, (1, param.STATE_DIM[0], param.STATE_DIM[1])))
            masked_output, action, reward, move_on, valid_state = sched.step(outputs)
            if episode == 0 and move_on:
                stt = sched.get_sched_states()
                states["states"].append(stt)
                """
                Log the "job id: job type: num_workers".
                """
                s = "ts: " + str(ts) + " "
                for id, tp, num_workers, num_ps in states:
                    if param.PS_WORKER:
                        s += "(id: " + str(id) + " type: " + str(type) + " num_workers: " + str(
                            num_workers) + " num_ps: " + str(num_ps) + ")\n"
                    else:
                        s += "(id: " + str(id) + " type: " + str(type) + " num_workers: " + str(num_workers) + ")\n"
                tb_logger.add_text(tag="rl:res_allocation:" + str(episode) + str(global_step), value=s,
                                   step=global_step)
                ts += 1

            if episode == 0:
                if step % 50 == 0:
                    i = 0
                    value = "input:"
                    for (key, enabled) in param.INPUTS_GATE:
                        if enabled:
                            # [("TYPE", True), ("STAY", False), ("PROGRESS", False), ("DOM_RESR", True), ("WORKERS", False)]
                            if key == "TYPE":
                                value += " type: " + str(inputs[i]) + "\n\n"
                            elif key == "STAY":
                                value += " stay_ts: " + str(inputs[i]) + "\n\n"
                            elif key == "PROGRESS":
                                value += " rt: " + str(inputs[i]) + "\n\n"
                            elif key == "DOM_RESR":
                                value += " resr: " + str(inputs[i]) + "\n\n"
                            elif key == "WORKERS":
                                value += " workers: " + str(inputs[i]) + "\n\n"
                            elif key == "PS":
                                value += " ps: " + str(inputs[i]) + "\n\n"
                            i += 1
                    value += " output: " + str(outputs) + "\n\n" + " masked_output: " + str(masked_output) + "\n\n" + \
                             " action: " + str(action)

                    tb_logger.add_text(
                        tag="rl:input+output+action:" + str(global_step) + "_" +
                            str(episode) + "_" + str(ts) + "_" + str(step),
                        value=value, step=global_step)
            step += 1

        num_jobs, jct, makespan, reward = sched.get_results()
        stats["jcts"].append(sched.get_job_cts().values())
        avg_jct.append(jct)
        avg_makespan.append(makespan)
        avg_reward.append(reward)

    elapsed_t = time.time() - tic
    logger.info("time for making one decision: " + str(elapsed_t / step) + " seconds")
    with open("DL2_JCTs.txt", 'a') as f:
        f.write(str(stats) + '\n')
    with open("DL2_states.txt", 'a') as f:
        f.write(str(states) + "\n")

    return (1. * sum(avg_jct) / len(avg_jct),
            1. * sum(avg_makespan) / len(avg_makespan),
            sum(avg_reward) / len(avg_reward))
