"""Expose CubOS hardware as plain-object devices for ivoryOS.

ivoryOS's edge server discovers hardware by instantiating driver objects at
module scope and introspecting their public methods (name, type-hinted
parameters, docstring, return annotation) via ``inspect``. It needs nothing
more than a plain Python object -- no base class, decorator, or registration
call.

CubOS's instrument drivers already satisfy that contract directly: every
concrete driver in ``cubos.instruments.*.vendors`` is a typed
:class:`~cubos.instruments.base_instrument.BaseInstrument` subclass with
docstringed public methods and dataclass return values, so :func:`load_instrument`
below is just a config-driven constructor -- the returned object needs no
further wrapping.

``GantrySession`` is different: it's a single persistent object meant to be
shared across CubOS's own API/UI, so its public surface also includes
calibration and limit-recovery flows that assume a specific operator
sequence and are unsafe to expose as standalone, one-shot ivoryOS actions.
:class:`CubOSGantry` narrows that surface to plain motion.

For a mounted, multi-instrument rig (gantry + pipette + capper + ...),
neither of those fits: the useful unit of action isn't the gantry alone or
one instrument alone, it's CubOS's own protocol-command layer (see
``cubos.protocol_engine.commands``) -- ``aspirate``, ``transfer``,
``pick_up_tip``, ``cap``/``decap``, ``measure``, ``move`` -- which already
resolves deck/labware positions and sequences gantry motion with the
instrument action. That layer also carries opinions worth keeping rather
than re-deciding: e.g. ``aspirate`` refuses to run standalone once durable
fluid-state tracking is active, and plain ``dispense`` isn't registered as
a command at all (only ``transfer``/``serial_transfer`` track a fluid's
source and destination correctly). :class:`CubOSProtocolOps` exposes every
registered command as an ivoryOS action by binding it to one
``ProtocolContext``, so those rules travel with it automatically.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import Any, Callable

from cubos.gantry.gantry import Gantry
from cubos.gantry.session import GantryPositionSnapshot, GantrySession
from cubos.instruments.base_instrument import BaseInstrument
from cubos.instruments.registry import get_instrument_class
from cubos.protocol_engine.registry import CommandRegistry
from cubos.protocol_engine.setup import ProtocolInput, setup_protocol


class CubOSGantry:
    """ivoryOS-facing motion actions for one CubOS gantry.

    Wraps a :class:`~cubos.gantry.session.GantrySession`, connecting
    immediately so the object is ready to drive as soon as ivoryOS
    instantiates it. Only plain motion is exposed here -- calibration and
    limit-recovery need an operator present and stay behind CubOS's own
    Operator UI or a direct ``GantrySession`` reference (:attr:`session`).
    """

    def __init__(self, config_path: str | Path, *, offline: bool = False) -> None:
        factory: Callable[..., Gantry] = (
            (lambda config: Gantry(config=config, offline=True)) if offline else Gantry
        )
        self.session = GantrySession(gantry_factory=factory)
        self.session.connect(config_path)

    def position(self) -> GantryPositionSnapshot:
        """Return the gantry's current cached position."""
        return self.session.position()

    def home(self) -> GantryPositionSnapshot:
        """Home the gantry and return its resulting position."""
        return self.session.home()

    def move_to(self, x: float, y: float, z: float) -> GantryPositionSnapshot:
        """Move to an absolute work-frame position and wait until idle."""
        return self.session.move_to_blocking(x=x, y=y, z=z)

    def jog(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0, feed_rate: float = 2000
    ) -> GantryPositionSnapshot:
        """Jog by a relative offset on one or more axes and wait until idle."""
        return self.session.jog_blocking(x=x, y=y, z=z, feed_rate=feed_rate)

    def disconnect(self) -> GantryPositionSnapshot:
        """Disconnect from the gantry."""
        return self.session.disconnect()


def load_instrument(instrument_type: str, vendor: str, **config: Any) -> BaseInstrument:
    """Instantiate and connect a registered CubOS instrument driver.

    ``instrument_type``/``vendor`` select the driver class from CubOS's
    instrument registry (see ``cubos.instruments.registry.yaml``);
    ``config`` is passed straight through to the driver's constructor. The
    returned driver's own public methods (e.g. ``run_CV``, ``capture_cap``)
    are already typed and docstringed, so ivoryOS can introspect it as-is.
    """
    driver_cls = get_instrument_class(instrument_type, vendor)
    driver = driver_cls(**config)
    driver.connect()
    return driver


# Registered protocol commands that are operator/workflow control flow, not
# hardware actions -- ``pause`` blocks on a timer and ``breakpoint`` blocks
# on operator input, neither of which makes sense as a one-shot ivoryOS
# action, and ``breakpoint`` blocking the edge server's request thread on
# ``input()`` would hang every other action until an operator is at that
# machine's own console.
_NON_HARDWARE_COMMANDS = frozenset({"pause", "breakpoint"})


class CubOSProtocolOps:
    """Every CubOS hardware protocol command as one ivoryOS action each.

    Builds a single :class:`~cubos.protocol_engine.runtime.ProtocolContext`
    from a gantry/deck config pair -- the same validated setup a real
    protocol run uses (see ``cubos.protocol_engine.setup.setup_protocol``)
    -- then binds every command in CubOS's ``CommandRegistry`` to it as a
    same-named method with the command's own parameters, docstring, and
    return type intact (minus the injected ``context`` argument), so
    ivoryOS's introspection sees exactly what a protocol YAML author would.

    ``bootstrap_protocol`` only needs to be *a* valid protocol for this
    gantry/deck pair -- ``setup_protocol`` uses it to validate configured
    motion bounds before returning the context. It does not limit which
    commands or positions can be called afterward.
    """

    def __init__(
        self,
        gantry_path: str | Path,
        deck_path: str | Path,
        bootstrap_protocol: ProtocolInput,
        *,
        mock_mode: bool = True,
    ) -> None:
        _protocol, context = setup_protocol(
            gantry_path, deck_path, bootstrap_protocol, mock_mode=mock_mode,
        )
        self.context = context
        for name in CommandRegistry.instance().command_names:
            if name in _NON_HARDWARE_COMMANDS:
                continue
            handler = CommandRegistry.instance().get(name).handler
            setattr(self, name, self._bind(handler))

    def _bind(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        context = self.context
        signature = inspect.signature(handler)
        bound_signature = signature.replace(
            parameters=[p for n, p in signature.parameters.items() if n != "context"]
        )

        @functools.wraps(handler)
        def method(*args: Any, **kwargs: Any) -> Any:
            return handler(context, *args, **kwargs)

        method.__signature__ = bound_signature
        method.__annotations__ = {
            key: value
            for key, value in getattr(handler, "__annotations__", {}).items()
            if key != "context"
        }
        return method
