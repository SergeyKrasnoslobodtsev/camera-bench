"""Name-based factory for pluggable backends.

A :class:`Registry` maps short names such as ``"ultralytics"`` or ``"ffmpeg-cuda"``
to backend classes. Classes are referenced either directly or by an import path
(``"package.module:Class"``), so heavy dependencies are imported only when a backend
is actually created.

Third-party packages can add backends without touching this project by declaring
an entry point in the registry's group::

    [project.entry-points."camera_bench.inference"]
    onnxruntime = "my_package.backends:OrtBackend"

Examples
--------
>>> class Base:
...     pass
>>> registry = Registry("demo", base=Base)
>>> @registry.register("plain")
... class Plain(Base):
...     pass
>>> registry.create("plain").__class__.__name__
'Plain'
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, Generic, TypeVar, overload

T = TypeVar("T")

__all__ = ["Registry", "UnknownBackendError"]


class UnknownBackendError(KeyError):
    """Raised when a backend name is not registered."""

    def __init__(self, kind: str, name: str, known: list[str]) -> None:
        super().__init__(name)
        self.kind = kind
        self.name = name
        self.known = known

    def __str__(self) -> str:
        known = ", ".join(self.known) or "none"
        return f"unknown {self.kind} backend {self.name!r}; available: {known}"


class Registry(Generic[T]):
    """Registry of backend classes that share a common base class.

    Parameters
    ----------
    kind : str
        Human-readable registry name used in error messages, e.g. ``"inference"``.
    base : type
        Class every registered backend must derive from.
    group : str, optional
        Entry point group scanned for third-party backends. ``None`` disables discovery.

    See Also
    --------
    Registry.register : Add a backend.
    Registry.create : Instantiate a backend by name.
    """

    def __init__(self, kind: str, base: type[T], group: str | None = None) -> None:
        self.kind = kind
        self.base = base
        self.group = group
        self._targets: dict[str, type[T] | str] = {}
        self._discovered = False

    @overload
    def register(self, name: str) -> Callable[[type[T]], type[T]]: ...

    @overload
    def register(self, name: str, target: type[T] | str) -> type[T] | str: ...

    def register(self, name: str, target: type[T] | str | None = None) -> Any:
        """Register a backend under ``name``.

        Parameters
        ----------
        name : str
            Short name used on the command line.
        target : type or str, optional
            Backend class, or its import path ``"module:attr"`` for lazy loading.
            When omitted, ``register`` works as a class decorator.

        Returns
        -------
        type or str or callable
            The target itself, or a decorator when ``target`` is omitted.

        Raises
        ------
        ValueError
            If ``name`` is already taken by a different target.
        """
        if target is None:

            def decorator(cls: type[T]) -> type[T]:
                self._add(name, cls)
                return cls

            return decorator
        self._add(name, target)
        return target

    def _add(self, name: str, target: type[T] | str) -> None:
        current = self._targets.get(name)
        if current is not None and current != target:
            raise ValueError(f"{self.kind} backend {name!r} is already registered")
        self._targets[name] = target

    def names(self) -> list[str]:
        """Return registered backend names in alphabetical order."""
        self._discover()
        return sorted(self._targets)

    def get(self, name: str) -> type[T]:
        """Resolve a backend class by name.

        Parameters
        ----------
        name : str
            Registered backend name.

        Returns
        -------
        type
            The backend class. Lazy import paths are imported and cached.

        Raises
        ------
        UnknownBackendError
            If no backend is registered under ``name``.
        TypeError
            If the resolved object does not derive from the registry base class.
        """
        self._discover()
        try:
            target = self._targets[name]
        except KeyError:
            raise UnknownBackendError(self.kind, name, self.names()) from None
        if isinstance(target, str):
            target = _import_path(target)
            if not (isinstance(target, type) and issubclass(target, self.base)):
                raise TypeError(f"{name!r} does not resolve to a {self.base.__name__} subclass")
            self._targets[name] = target
        return target

    def create(self, name: str, /, **options: Any) -> T:
        """Instantiate the backend registered under ``name``.

        Parameters
        ----------
        name : str
            Registered backend name.
        **options
            Keyword arguments forwarded to the backend constructor.

        Returns
        -------
        object
            A new backend instance.
        """
        return self.get(name)(**options)

    def _discover(self) -> None:
        # Entry points are scanned once, and built-in names always win over plugins.
        if self._discovered or self.group is None:
            return
        self._discovered = True
        for point in entry_points(group=self.group):
            self._targets.setdefault(point.name, point.value)


def _import_path(path: str) -> Any:
    """Import ``"package.module:attr"`` and return the attribute."""
    module_name, _, attr = path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr) if attr else module
