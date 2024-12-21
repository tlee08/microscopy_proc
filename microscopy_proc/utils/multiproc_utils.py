"""
Utility functions.
"""

from __future__ import annotations

from multiprocessing import current_process

from dask.distributed import get_worker

from microscopy_proc.utils.logging_utils import init_logger

logger = init_logger(__name__)


@staticmethod
def get_cpid() -> int:
    """Get child process ID for multiprocessing."""
    return current_process()._identity[0] if current_process()._identity else 0


def get_dask_pid() -> int:
    """Get the Dask process ID."""
    logger.debug(get_worker())
    logger.debug(get_worker().id)
    worker_id = get_worker().id
    return int(worker_id)
