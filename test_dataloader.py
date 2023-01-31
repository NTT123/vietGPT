"""Test dataloader module."""

import contextlib
import shutil
import tempfile

import numpy as np

from dataloader import create_data_iterator


@contextlib.contextmanager
def create_temp_dataset():
    """Create a temporary dataset for testing."""

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    # Create a temporary dataset
    tokens_1 = np.random.randint(0, 100, size=(100,), dtype=np.int16)
    tokens_2 = np.random.randint(0, 100, size=(100,), dtype=np.int16)
    # save tokens_1 and tokens_2 to the temporary directory
    np.save(temp_dir + "/tokens_1.npy", tokens_1)
    np.save(temp_dir + "/tokens_2.npy", tokens_2)
    yield temp_dir
    # remove the temporary directory
    shutil.rmtree(temp_dir)


def test_dataloader():
    """Test the dataloader function."""
    with create_temp_dataset() as data_dir:
        batch_size = 1024
        seq_len = 32
        data_iterator = create_data_iterator(data_dir, batch_size, seq_len)
        batch = next(data_iterator)
        batch = next(data_iterator)
        assert batch.shape == (batch_size, seq_len)
        assert batch.dtype == np.int16
