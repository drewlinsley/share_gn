import os
from setuptools import setup, find_packages
from db import credentials
from utils import logger
from config import Config


"""e.g. python setup.py install"""

setup(
    name="cluttered_nist",
    version="0.1",
    packages=find_packages(),
)

config = Config()
log = logger.get(os.path.join(config.log_dir, 'setup'))
log.info('Installed required packages and created paths.')

params = credentials.postgresql_connection()
sys_password = credentials.machine_credentials()['password']
os.popen(
    'sudo -u postgres createuser -sdlP %s' % params['user'], 'w').write(
    sys_password)
os.popen(
    'sudo -u postgres createdb %s -O %s' % (
        params['database'],
        params['user']), 'w').write(sys_password)
log.info('Created DB.')
