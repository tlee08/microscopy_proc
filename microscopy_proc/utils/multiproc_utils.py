"""
Utility functions.
"""

from __future__ import annotations

from multiprocessing import current_process

from dask.distributed import get_worker


@staticmethod
def get_cpid() -> int:
    """Get child process ID for multiprocessing."""
    return current_process()._identity[0] if current_process()._identity else 0


def get_dask_pid() -> int:
    """Get the Dask process ID."""
    print(get_worker())
    print(get_worker().id)
    return get_worker().id