"""
DeepDrugDomain Logging Configuration

This module provides a centralized logging configuration for the entire package.

Users of the package can customize the logging configuration by providing their
own configuration in a JSON file and then setting it up as shown below:


Usage:
>>> from deepdrugdomain.utils import logging_config
>>> logger = logging_config.get_logger(__name__)
>>> logger.info("This is an informational message.")

>>> # For customizing the logging configuration:
>>> import json
>>>
>>> # Load the user's logging config
>>> with open('path_to_user_logging_config.json', 'r') as file:
>>>     user_config = json.load(file)
>>>
>>> # Setup logging using the user's config
>>> logging_config.setup_logging(user_config)

"""

import logging
import logging.config
from typing import Optional, Dict

DEFAULT_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
}


def setup_logging(config: Optional[Dict] = None):
    """Set up logging from an external configuration or use the default."""
    if config is None:
        config = DEFAULT_CONFIG
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Set up default logging when this module is imported.
setup_logging()
