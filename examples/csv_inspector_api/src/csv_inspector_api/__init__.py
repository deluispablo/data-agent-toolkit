"""Example FastAPI host that embeds ``csv-inspector``.

This is executable documentation of how to embed the agent in an HTTP
service, not a distribution. Serve it with the application factory::

    uvicorn --app-dir src csv_inspector_api.app:create_app --factory
"""

from __future__ import annotations

from .app import __version__, create_app

__all__ = ["__version__", "create_app"]
