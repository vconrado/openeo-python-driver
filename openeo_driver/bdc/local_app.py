"""
Script to start a local server. This script can serve as the entry-point for doing spark-submit.
"""

import logging
import os
import sys
from logging.config import dictConfig

import openeo_driver
from openeo_driver.bdc.bdc_backend import BDCBackendImplementation
from openeo_driver.server import show_log_level, run_gunicorn
from openeo_driver.views import build_app

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(process)s %(levelname)s in %(name)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    },
    'loggers': {
        "gunicorn": {"level": "INFO"},
        'werkzeug': {'level': 'DEBUG'},
        'flask': {'level': 'DEBUG'},
        'openeo': {'level': 'DEBUG'},
        'openeo_driver': {'level': 'DEBUG'},
        'kazoo': {'level': 'WARN'},
    }
})

_log = logging.getLogger('openeo-bdc-local')

if __name__ == '__main__':
    _log.info(repr({"pid": os.getpid(), "interpreter": sys.executable,
              "version": sys.version, "argv": sys.argv}))

    app = build_app(backend_implementation=BDCBackendImplementation())
    app.config.from_mapping(
        OPENEO_TITLE="BDC Backend",
        OPENEO_DESCRIPTION="openEO API using BDC backend",
        OPENEO_BACKEND_VERSION=openeo_driver.__version__,
    )

    show_log_level(logging.getLogger('openeo'))
    show_log_level(logging.getLogger('openeo_driver'))
    show_log_level(app.logger)

    run_gunicorn(
        app=app,
        threads=4,
        host="0.0.0.0",
        port=9007
    )
