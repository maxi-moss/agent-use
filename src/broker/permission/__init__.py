"""Permission triage. ``PermissionModule`` is this package's whole surface.

Everything else here is internal: the classifier's model, its prompt and its
log are the package's own, and a caller that reached past this module could
re-pin any of them. ``render_permission_log`` is the one exception: the log is
written here but read by whoever answers the developer.
"""

from broker.permission.module import PermissionModule
from broker.permission.permission_log import render_permission_log

__all__ = ["PermissionModule", "render_permission_log"]
