from spinup.user_config import DEFAULT_DATA_DIR
from spinup.utils.mpi_tools import proc_id
import joblib
import os
import warnings
import torch
from pathlib import Path
from itertools import accumulate
import logging
import sys
from gymnasium.spaces import Box, Discrete

def setup_logger_kwargs(exp_name: str, seed: int=None):

    seed = f'_s{seed}' if seed is not None else ''
    subfolder = "".join([exp_name, seed])

    output_dir = Path(DEFAULT_DATA_DIR) / exp_name / subfolder


    return dict(output_dir=output_dir)

class Logger:
    """
    A general purpose logger (shared among all reimplemented algorithms).
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # set up logger
        self.logger = logging.getLogger("training_logger")
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler(stream=sys.stdout)
        file_handler = logging.FileHandler(self.output_dir / 'run.log')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def save_state(self, state_dict: dict, iteration: int=None):
        if proc_id() == 0:
            iter_str = '' if iteration is None else str(iteration)
            filename = f"vars{iter_str}.pkl"
            joblib.dump(state_dict, os.path.join(self.output_dir, filename))

            # This automatically saves the pytorch model
            # (Even though this is suboptimal, I keep this to be compatible with the rest of the spinup functionalities)
            if hasattr(self, 'pytorch_saver_elements'):
                self._pytorch_simple_save(iteration)

    def _pytorch_simple_save(self, itr=None):
        """
        Saves the PyTorch model (or models). [Copied over from logx.py]
        """
        if proc_id()==0:
            assert hasattr(self, 'pytorch_saver_elements'), \
                "First have to setup saving with self.setup_pytorch_saver"
            fpath = 'pyt_save'
            fpath = os.path.join(self.output_dir, fpath)
            fname = 'model' + ('%d'%itr if itr is not None else '') + '.pt'
            fname = os.path.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # We are using a non-recommended way of saving PyTorch models,
                # by pickling whole objects (which are dependent on the exact
                # directory structure at the time of saving) as opposed to
                # just saving network weights. This works sufficiently well
                # for the purposes of Spinning Up, but you may want to do
                # something different for your personal PyTorch project.
                # We use a catch_warnings() context to avoid the warnings about
                # not being able to save the source code.
                torch.save(self.pytorch_saver_elements, fname)


    def setup_pytorch_saver_elements(self, elements):
        self.pytorch_saver_elements = elements

def discounted_cumsum(values: list, discount_factor: float) -> list:
    # x is the previously accumulated sum; y is the element at the current index
    return list(accumulate(values[::-1], lambda x, y: discount_factor * x + y))[::-1]

def get_act_dim(action_space) -> int:
    if isinstance(action_space, Box):
        return action_space.shape[0]
    elif isinstance(action_space, Discrete):
        return 1
    else:
        raise NotImplementedError