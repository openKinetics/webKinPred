# api/methods/__init__.py
#
# This package defines the method registry system for webKinPred.
# Each prediction method (DLKcat, UniKP, etc.) has a corresponding module
# in this directory that declares a `descriptor` object of type MethodDescriptor.
# The registry auto-discovers and loads all descriptors at startup.
#
# To add a new prediction method, see docs/project/contributing.rst.
