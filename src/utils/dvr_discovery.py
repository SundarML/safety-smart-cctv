"""Pre-flight check for the DVR's RTSP address.

Home routers hand out DHCP leases with no guarantee of stability, so a
power-cycled DVR can come back on a different IP than the one baked into
config/settings.yaml. Before we hand an unreachable address to FFmpeg (which
just hangs for its full probe timeout per camera), check that the last-known
host is actually up, and if it isn't, scan the rest of its /24 for something
that talks RTSP on the same port.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 2.0
_SCAN_TIMEOUT = 0.3


def _tcp_open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _looks_like_rtsp(host: str, port: int, timeout: float) -> bool:
    """A bare OPTIONS request; any RTSP/1.0 reply (even 401) confirms the service."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.sendall(f"OPTIONS rtsp://{host}:{port}/ RTSP/1.0\r\nCSeq: 1\r\n\r\n".encode())
            reply = sock.recv(256)
        return reply.startswith(b"RTSP/1.0")
    except OSError:
        return False


def find_dvr_host(known_host: str, port: int = 554) -> str | None:
    """Scan `known_host`'s /24 for a device answering RTSP on `port`.

    Returns the new host if exactly one RTSP responder is found, else None
    (nothing found, or too many candidates to disambiguate automatically).
    """
    try:
        subnet = ipaddress.ip_network(f"{known_host}/24", strict=False)
    except ValueError:
        logger.warning("Cannot derive a subnet from %s for DVR discovery", known_host)
        return None

    logger.info("DVR at %s:%s is unreachable — scanning %s for a replacement", known_host, port, subnet)
    hosts = [str(ip) for ip in subnet.hosts() if str(ip) != known_host]

    with ThreadPoolExecutor(max_workers=64) as pool:
        open_hosts = [h for h, ok in zip(hosts, pool.map(lambda h: _tcp_open(h, port, _SCAN_TIMEOUT), hosts)) if ok]
        candidates = [
            h for h, ok in zip(open_hosts, pool.map(lambda h: _looks_like_rtsp(h, port, _CONNECT_TIMEOUT), open_hosts))
            if ok
        ]

    if len(candidates) == 1:
        logger.info("Found RTSP responder at %s — treating it as the DVR's new address", candidates[0])
        return candidates[0]
    if not candidates:
        logger.error("No RTSP responder found on %s", subnet)
        return None
    logger.error(
        "Multiple RTSP responders found on %s (%s) — cannot tell which is the DVR automatically",
        subnet, ", ".join(candidates),
    )
    return None


def resolve_dvr_hosts(cameras: list[dict], config_path: str | None = None) -> None:
    """Check each unique RTSP host used by `cameras`; rewrite in place (and in
    `config_path`, if given) with a discovered replacement if it's down."""
    cams_by_host: dict[str, list[dict]] = {}
    for cam in cameras:
        source = cam.get("source")
        if isinstance(source, str) and source.lower().startswith("rtsp://"):
            host = urlsplit(source).hostname
            if host:
                cams_by_host.setdefault(host, []).append(cam)

    for host, cams in cams_by_host.items():
        port = urlsplit(cams[0]["source"]).port or 554
        if _tcp_open(host, port, _CONNECT_TIMEOUT):
            continue

        new_host = find_dvr_host(host, port)
        if not new_host:
            logger.error(
                "%s:%s is unreachable and no replacement could be found automatically. "
                "If the DVR's IP changed, update config/settings.yaml manually.",
                host, port,
            )
            continue

        for cam in cams:
            cam["source"] = cam["source"].replace(f"@{host}:", f"@{new_host}:")
        logger.info("Updated %d camera source(s): %s -> %s", len(cams), host, new_host)

        if config_path:
            _persist_host_change(config_path, host, new_host)


def _persist_host_change(config_path: str, old_host: str, new_host: str) -> None:
    try:
        with open(config_path) as f:
            text = f.read()
        updated = text.replace(old_host, new_host)
        if updated != text:
            with open(config_path, "w") as f:
                f.write(updated)
            logger.info("Saved new DVR address %s to %s", new_host, config_path)
    except OSError as e:
        logger.warning("Could not persist new DVR address to %s: %s", config_path, e)
