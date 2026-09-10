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
neither fits: the useful unit of action is CubOS's own protocol-command
layer (``cubos.protocol_engine.commands``) -- ``aspirate``, ``transfer``,
``pick_up_tip``, ``cap``/``decap``, ``measure``, ``move`` -- which already
resolves deck/labware positions and sequences gantry motion with the
instrument action, and already encodes rules worth keeping rather than
re-deciding (e.g. plain ``dispense`` isn't registered as a command at all --
only ``transfer``/``serial_transfer`` track a fluid's source and
destination correctly). :class:`CubOSProtocolOps` binds every command
CubOS has registered to one shared hardware session, so those rules and any
future commands travel with it automatically -- nothing here hardcodes a
command list.

``connect()`` also turns every ``str``-typed deck-target parameter
(``position``, ``source``, ``vial``, ...) into a dropdown of this deck's
real vials/wells, mirroring ``isPositionArg``/``targetsForLabware`` in
CubOS's own Operator UI (apps/operator-web/src/components/editor/
ProtocolEditor.tsx) -- just built for ivoryOS's Python-typed introspection
instead of that UI's JavaScript combobox.
"""

from __future__ import annotations

import enum
import inspect
from pathlib import Path
from typing import Any

import yaml

from cubos.deck.labware.vial import Vial
from cubos.gantry.gantry import Gantry
from cubos.gantry.session import GantryPositionSnapshot, GantrySession
from cubos.instruments.base_instrument import BaseInstrument
from cubos.instruments.registry import get_instrument_class
from cubos.protocol_engine.errors import GantryHealthCheckError
from cubos.protocol_engine.protocol import Protocol
from cubos.protocol_engine.registry import CommandRegistry, RegisteredCommand
from cubos.protocol_engine.setup import setup_protocol


class CubOSGantry:
    """ivoryOS-facing motion actions for one CubOS gantry.

    Wraps a :class:`~cubos.gantry.session.GantrySession`, connecting
    immediately so the object is ready to drive as soon as ivoryOS
    instantiates it. Only plain motion is exposed here -- calibration and
    limit-recovery need an operator present and stay behind CubOS's own
    Operator UI or a direct ``GantrySession`` reference (:attr:`session`).
    """

    def __init__(self, config_path: str | Path, *, offline: bool = False) -> None:
        factory = (lambda config: Gantry(config=config, offline=True)) if offline else Gantry
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
# on operator input, and ``breakpoint`` blocking the edge server's request
# thread on ``input()`` would hang every other action until an operator is
# at that machine's own console.
_NON_HARDWARE_COMMANDS = frozenset({"pause", "breakpoint"})

# Parameter names CubOS's own Operator UI already treats as deck-target
# pickers (see ``isPositionArg`` in
# apps/operator-web/src/components/editor/ProtocolEditor.tsx), extended
# with the few command-specific names (``vial``, ``well``, ``plate``) that
# play the same role. Only applied where the parameter's real type is
# exactly ``str`` -- ``move``'s ``position`` is deliberately excluded
# because it also accepts a literal ``[x, y, z]`` coordinate, which a
# strict Enum would make impossible to pass.
_DECK_TARGET_PARAMS = frozenset({"position", "source", "destination", "vial", "well", "plate"})


class CubOSProtocolOps:
    """Every CubOS hardware protocol command as one ivoryOS action each.

    ``connect()``/``disconnect()`` own a real hardware session; every other
    method is bound from :class:`~cubos.protocol_engine.registry.CommandRegistry`
    at construction time and reads whatever session is current when called,
    so the same bound methods keep working across a disconnect/reconnect.
    Calling a command before ``connect()`` (or after ``disconnect()``)
    raises ``RuntimeError`` instead of touching stale state.

    Only two things here depend on CubOS internals rather than its
    stable, versioned protocol-command contract:

    1. Parameter types come from ``RegisteredCommand.schema`` (a Pydantic
       model CubOS itself builds from each handler's signature) rather than
       reading the handler's raw ``__annotations__`` directly. CubOS's
       command modules use postponed annotations (``from __future__ import
       annotations``), which turns every type hint into a plain string --
       the schema is where CubOS already resolves those back to real types,
       so reusing it avoids re-deriving that resolution here.
    2. ``connect()`` calls ``gantry.connect()`` /
       ``gantry.prepare_for_protocol_run()`` / ``connect_instruments()`` --
       the same three calls, in the same order,
       ``cubos.protocol_engine.setup.run_on_hardware`` makes before running
       a protocol. That function is CubOS's own primary hardware entry
       point (shared by its YAML CLI and ``Protocol.run()``), so a future
       change to this sequence breaks CubOS's own test suite too, not just
       this integration silently.
    """

    def __init__(
        self, gantry_path: str | Path, deck_path: str | Path, *, mock_mode: bool = True
    ) -> None:
        self.gantry_path = gantry_path
        self.deck_path = deck_path
        self.mock_mode = mock_mode
        self.context = None
        self._gantry: Gantry | None = None
        registry = CommandRegistry.instance()
        for name in registry.command_names:
            if name in _NON_HARDWARE_COMMANDS:
                continue
            setattr(self, name, self._bind(registry.get(name)))

    def connect(self) -> None:
        """Open a real hardware session (see class docstring for what this mirrors)."""
        if self.mock_mode:
            self._gantry = Gantry(offline=True)
        else:
            self._gantry = Gantry(config=yaml.safe_load(Path(self.gantry_path).read_text()))
        _, self.context = setup_protocol(
            self.gantry_path, self.deck_path, Protocol(steps=[]),
            gantry=self._gantry, mock_mode=self.mock_mode,
        )
        self._gantry.connect()
        self._gantry.prepare_for_protocol_run()
        self.context.gantry.connect_instruments()
        if not self._gantry.is_healthy():
            self.disconnect()
            raise GantryHealthCheckError("Gantry health check failed after connect().")
        self._apply_deck_target_choices()

    def disconnect(self) -> None:
        """Close a hardware session opened by :meth:`connect`."""
        if self.context is not None:
            self.context.gantry.disconnect_instruments()
            self.context = None
        if self._gantry is not None:
            self._gantry.disconnect()
            self._gantry = None

    def _deck_targets(self) -> list[str]:
        """Every vial/plate key and plate.well combination on this deck.

        Same pool CubOS's own Operator UI offers for position/source/
        destination fields, walking ``deck.labware`` the same way its API
        does (see ``get_deck`` in services/api/src/cubos_api/routers/deck.py)
        -- with one correction: a plain ``Vial`` also implements
        ``iter_positions()`` (a single synthetic ``"location"`` entry), but
        ``cubos.protocol_engine.commands._liquid_selection.target_position``
        -- what CubOS's own command dispatch actually produces/expects --
        uses a vial's *bare* key, never ``key.location``. Confirmed by
        running this against a real rig: without this check, every vial
        target this builds is silently wrong.
        """
        targets: list[str] = []
        for key, labware in self.context.deck.labware.items():
            if not isinstance(labware, Vial) and hasattr(labware, "iter_positions"):
                targets.extend(f"{key}.{loc_id}" for loc_id in labware.iter_positions())
            else:
                targets.append(key)
        return sorted(targets)

    def _apply_deck_target_choices(self) -> None:
        """Turn ``str``-typed deck-target parameters into a dropdown.

        ivoryOS renders an ``Enum`` parameter as a dropdown of its values.
        Unlike CubOS's own Operator UI combobox, that dropdown is strict --
        it won't accept a value outside the enum -- which is exactly why
        this only touches parameters whose real type is plain ``str``
        (see :data:`_DECK_TARGET_PARAMS`); anything typed ``Any``/``Optional``
        might need a value (like ``move``'s raw ``[x, y, z]``) that isn't
        one of this deck's named targets.
        """
        targets = self._deck_targets()
        if not targets:
            return
        deck_target = enum.Enum(
            "DeckTarget", {f"T{i}": target for i, target in enumerate(targets)}, type=str,
        )
        for name in CommandRegistry.instance().command_names:
            if name in _NON_HARDWARE_COMMANDS:
                continue
            method = getattr(self, name)
            parameters = [
                param.replace(annotation=deck_target)
                if param.name in _DECK_TARGET_PARAMS and param.annotation is str
                else param
                for param in method.__signature__.parameters.values()
            ]
            method.__signature__ = method.__signature__.replace(parameters=parameters)
            method.__annotations__ = {p.name: p.annotation for p in parameters}

    def _bind(self, registered: RegisteredCommand):
        parameters = [
            inspect.Parameter(
                field_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty if field.is_required() else field.default,
                annotation=field.annotation,
            )
            for field_name, field in registered.schema.model_fields.items()
        ]
        signature = inspect.Signature(parameters)

        def method(*args: Any, **kwargs: Any) -> Any:
            if self.context is None:
                raise RuntimeError(f"Call connect() before {registered.name}().")
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return registered.handler(self.context, **bound.arguments)

        method.__name__ = registered.name
        method.__doc__ = registered.handler.__doc__
        method.__signature__ = signature
        method.__annotations__ = {p.name: p.annotation for p in parameters}
        return method
