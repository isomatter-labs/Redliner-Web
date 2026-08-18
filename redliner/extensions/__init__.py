"""Drop-in extensions.

Every module in this package is imported once at startup, so a plugin only has
to define a class and decorate it with the relevant registry. Nothing here is
imported by the core, and a module that raises is logged and skipped.

This is the simplest route when you have forked Redliner. If you would rather
not fork, ship a separate package that declares entry points instead -- see
EXTENDING.md.
"""
