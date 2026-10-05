"""Thin wrappers around the four ONNX models.

Each wrapper is lazily constructed and owns exactly one concern: load the
graph, pre-process an image the way the graph expects, post-process raw tensors
into plain Python. Extractors stay free of tensor plumbing, and swapping a
model (e.g. FER+ for a stronger expression model when real webcam data
arrives) touches one file.
"""
