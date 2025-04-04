"""
Global configuration for the CPU-based hybrid recommendation system.
Contains default parameters, paths, and settings.
"""
import os
import multiprocessing
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_OUTPUT_PATH = './cpu_models'

# Model parameters
DEFAULT_FACTORS = 100
DEFAULT_ITERATIONS = 15
DEFAULT_REGULARIZATION = 0.01
DEFAULT_USE_NATIVE = True

# Column names
DEFAULT_ITEM_ID_COL = 'itemid'
DEFAULT_USER_ID_COL = 'visitorid'
DEFAULT_TIMESTAMP_COL = 'timestamp'
DEFAULT_EVENT_TYPE_COL = 'event'

# Event weights for different user interactions
DEFAULT_EVENT_WEIGHTS = {
    'view': 1.0,
    'addtocart': 4.0,
    'transaction': 10.0
}

# Parallelization settings
DEFAULT_N_JOBS = max(1, multiprocessing.cpu_count() - 1)

# Artifact reuse flag
DEFAULT_REUSE_ARTIFACTS = True

# ANN Library preference order
ANN_LIBRARIES = ['nmslib', 'annoy', 'sklearn']