"""Tools Jittor offers its users, as opposed to tools that build Jittor.

``jittor/utils`` used to hold both, plus a source translator and a Flask app.
The split is by audience:

* here — things a user imports or runs against their own model: an NVTX range
  marker for profiler timelines, ``jtune`` for re-running one generated
  kernel by hand, model tracing, and the lazy-safe ``jt.benchmark`` API;
* repository ``tools/`` — things a maintainer runs against the checkout;
* ``jittor/build`` — installed compiler utilities and their declared resources.

Nothing is imported here: ``jittor.tools.nvtx`` loads the NVTX shared library
at import, and it must not be paid for by ``import jittor``.
"""
