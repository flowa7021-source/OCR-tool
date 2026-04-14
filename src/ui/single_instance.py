"""Single-instance guard using :class:`QLocalServer` / :class:`QLocalSocket`.

When the user launches a second copy of the app, this module:

    1. Tries to connect to a named local socket (``ocr-studio-<user>``).
    2. On successful connect → we're the second instance. Send any
       command-line PDF paths over the socket and exit 0. The primary
       receives them via :attr:`SingleInstanceGuard.args_received` and
       opens them in the existing window.
    3. On connect failure → we're the primary. Bind the server and
       install a listener.

``QLocalServer`` uses named pipes on Windows and AF_UNIX sockets on
POSIX, so no TCP port is required and the handshake survives firewall
rules.
"""

from __future__ import annotations

import getpass
import logging
from collections.abc import Iterable

from PySide6.QtCore import QByteArray, QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

logger = logging.getLogger(__name__)


def _socket_name() -> str:
    """Return a per-user socket name so different accounts don't collide."""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 — Windows may fail in weird envs
        user = "default"
    return f"ocr-studio-{user}"


class SingleInstanceGuard(QObject):
    """Primary/secondary handshake over a named local socket."""

    args_received = Signal(list)  # list[str] — argv of the secondary

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._server: QLocalServer | None = None
        self._socket_name = _socket_name()

    def is_primary(self) -> bool:
        """Return True iff no other instance is running for this user.

        When primary, :meth:`start_listening` is called automatically.
        """
        probe = QLocalSocket()
        probe.connectToServer(self._socket_name)
        if probe.waitForConnected(200):
            probe.disconnectFromServer()
            return False
        return True

    def forward_and_exit(self, argv: Iterable[str]) -> bool:
        """Send ``argv`` to the primary instance. Returns True on success."""
        sock = QLocalSocket()
        sock.connectToServer(self._socket_name)
        if not sock.waitForConnected(1000):
            logger.warning(
                "forward_and_exit: could not reach primary on %s",
                self._socket_name,
            )
            return False
        # One newline-delimited path per argument; the server splits on '\n'.
        payload = "\n".join(str(a) for a in argv) + "\n"
        sock.write(QByteArray(payload.encode("utf-8")))
        sock.flush()
        sock.waitForBytesWritten(500)
        sock.disconnectFromServer()
        return True

    def start_listening(self) -> None:
        """Bind the server so the next instance can forward its argv."""
        # Clean up any stale socket left over by a previous crash
        QLocalServer.removeServer(self._socket_name)
        self._server = QLocalServer(self)
        self._server.newConnection.connect(self._on_new_connection)
        if not self._server.listen(self._socket_name):
            logger.warning(
                "SingleInstanceGuard: listen() failed on %s: %s",
                self._socket_name,
                self._server.errorString(),
            )

    def _on_new_connection(self) -> None:
        if self._server is None:
            return
        sock = self._server.nextPendingConnection()
        if sock is None:
            return
        sock.readyRead.connect(lambda s=sock: self._read_args(s))
        sock.disconnected.connect(sock.deleteLater)

    def _read_args(self, sock: QLocalSocket) -> None:
        blob = bytes(sock.readAll()).decode("utf-8", errors="replace")
        args = [line for line in blob.split("\n") if line]
        if args:
            logger.info("SingleInstanceGuard received: %r", args)
            self.args_received.emit(args)

    def stop(self) -> None:
        """Close the server. Call from closeEvent."""
        if self._server is not None:
            self._server.close()
            self._server = None
