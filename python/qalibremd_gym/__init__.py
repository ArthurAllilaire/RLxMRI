"""Gymnasium wrapper for the QalibreMDPhantom Julia environments."""

# Lazy: do not import juliacall envs at package import time so that the
# MRzero venv (no juliacall) can still use env_paradigm_a.
__all__ = ["QalibreMDE1Env"]


def __getattr__(name):
    if name == "QalibreMDE1Env":
        from .env import QalibreMDE1Env  # noqa: F401
        return QalibreMDE1Env
    raise AttributeError(name)
