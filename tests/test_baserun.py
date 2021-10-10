import pytest
import subprocess
import os


def test_basic_run():
    s = subprocess.run(['python', 'batch_job.py'], text=True, capture_output=True)
    assert s.returncode == 0
    assert os.path.isdir('./temp_data')
    assert os.path.isdir('./errors')
    assert len(os.listdir('./errors')) == 0
    assert os.path.isfile('./run_config.log')
    assert sum([('main_log' in x) for x in os.listdir('.')]) == 1
