# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from multiprocessing import Queue
import multiprocessing
import logging
from joblib import Parallel, delayed

from omegaconf import open_dict

from hydra._internal.pathlib import Path
from hydra.plugins.common.utils import (
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
    HydraConfig,
)
from hydra.plugins import Launcher

log = logging.getLogger(__name__)


def execute_job(idx, overrides, config_loader, config, task_function, q):
    gpu = q.get()
    log.info("\t#{} : {}".format(idx, " ".join(filter_overrides(overrides))))
    sweep_config = config_loader.load_sweep_config(
        config, list(overrides)
    )
    with open_dict(sweep_config):
        sweep_config.hydra.job.id = idx
        sweep_config.hydra.job.num = idx
    HydraConfig().set_config(sweep_config)
    os.environ["CUDA_VISIBLE_DEVICE"] = str(gpu)
    ret = run_job(
        config=sweep_config,
        task_function=task_function,
        job_dir_key="hydra.sweep.dir",
        job_subdir_key="hydra.sweep.subdir",
    )
    configure_log(config.hydra.hydra_logging, config.hydra.verbose)
    q.put(gpu)
    return ret


class ParallelLauncher(Launcher):
    def __init__(self):
        self.config = None
        self.config_loader = None
        self.task_function = None

    def setup(self, config, config_loader, task_function):
        self.config = config
        self.config_loader = config_loader
        self.task_function = task_function
        print(task_function)
        self.config.gpus = [int(gpu) for gpu in str(self.config.gpus)]


    def launch(self, job_overrides):
        setup_globals()
        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = self.config.hydra.sweep.dir
        Path(str(sweep_dir)).mkdir(parents=True, exist_ok=True)
        log.info("Launching {} jobs locally".format(len(job_overrides)))

        m = multiprocessing.Manager()
        total_n_jobs = self.config.n_jobs * len(self.config.gpus)
        print('self.config.gpus', self.config.gpus)
        print('self.config.n_jobs', self.config.n_jobs)
        print('total_n_jobs', total_n_jobs)
        q = m.Queue(maxsize=total_n_jobs)
        for i in self.config.gpus:
            for _ in range(int(self.config.n_jobs)):
                q.put(i)

        runs = Parallel(n_jobs=total_n_jobs)(delayed(execute_job)(idx, overrides, self.config_loader, self.config, self.task_function, q) for idx, overrides in enumerate(job_overrides))

        return runs
#  to join this conversation on GitHub. Already have an account? Sign in to comment
# Footer
# © 2024 GitHub, Inc.
# Footer navigation
# Terms
# Privacy
# Security
# Status
# Docs
# Contact
# Manage cookies
# Do not share my personal information
