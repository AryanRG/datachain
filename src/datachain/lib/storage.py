"""Shared client configuration and saved listing preparation."""

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import uuid4

from datachain.client import Client
from datachain.lib.dc.storage_pattern import validate_cloud_bucket_name
from datachain.lib.file import File
from datachain.lib.listing import (
    LISTING_BATCH_SIZE,
    get_listing,
    list_bucket,
    listing_uri_from_name,
)
from datachain.query.session import Session

if TYPE_CHECKING:
    from datachain.lib.dc.datachain import DataChain


def _backends_have_credentials(uris, client_config: dict | None) -> bool:
    """True if any backend behind ``uris`` sees credentials in client_config."""
    if not client_config:
        return False
    seen: set[type[Client]] = set()
    for uri in uris:
        try:
            seen.add(Client.get_implementation(str(uri)))
        except NotImplementedError:
            return True
    return any(c.has_explicit_credentials(client_config) for c in seen)


def _all_buckets_anonymous(uris, client_config: dict | None) -> bool:
    """Probe each unique bucket; True iff all probe as anonymous."""
    to_probe: set[tuple[type[Client], str]] = set()
    for uri in uris:
        try:
            client_cls = Client.get_implementation(str(uri))
        except NotImplementedError:
            return False
        name, _ = client_cls.split_url(str(uri))
        if not name:
            return False
        to_probe.add((client_cls, name))

    if not to_probe:
        return False

    for client_cls, name in to_probe:
        try:
            status = client_cls.bucket_status(name, **(client_config or {}))
        except NotImplementedError:
            # Backend doesn't support bucket_status (e.g. local files).
            return False
        if status.access != "anonymous":
            return False
    return True


def client_configs(
    uris: Sequence,
    session: Session | None,
    anon: bool | None,
    client_config: dict | None,
    versions: bool,
) -> tuple[dict | None, dict]:
    """The client config a session is made with, if not the active one's, and
    the client config to list ``uris`` with.

    ``client_config`` if given, else the config of ``session`` or the active
    session. ``anon`` is set
    on it, True when None and every bucket is public and has no credentials,
    and ``version_aware=False`` without ``versions``. An explicit session is
    never changed.
    """
    for single_uri in uris:
        validate_cloud_bucket_name(str(single_uri))
    config = _session_client_config(session, client_config)
    if (
        anon is None
        and not _backends_have_credentials(uris, config)
        and _all_buckets_anonymous(uris, config)
    ):
        anon = True
    if anon is not None:
        config = config | {"anon": anon}
    session_config = config if client_config is not None or anon is not None else None
    if not versions:
        config = config | {"version_aware": False}
    return session_config, config


def session_for(
    session: Session | None, client_config: dict | None, in_memory: bool
) -> Session:
    """``Session.get(session)`` with ``client_config``, if any: an explicit
    session is never changed, so a session over a copy of its catalog.
    """
    session = Session.get(session, client_config=client_config, in_memory=in_memory)
    if client_config is None or session.catalog.client_config == client_config:
        return session
    catalog = session.catalog.copy()
    catalog.client_config = client_config
    return Session("session" + uuid4().hex[:4], catalog, in_memory=in_memory)


def _session_client_config(session: Session | None, client_config: dict | None) -> dict:
    """``client_config`` if given, else the session's, without creating one."""
    if client_config is not None:
        return client_config
    session = session or Session.active()
    return session.catalog.client_config if session else {}


def prepare_listing(
    uri: str | os.PathLike[str],
    session: Session,
    client_config: dict,
    refreshed: set[str],
    *,
    file_type: type[File] = File,
    column: str = "file",
    update: bool = False,
    versions: bool = True,
    settings: dict | None = None,
    in_memory: bool = False,
    delta: bool | None = False,
    delta_on: str | Sequence[str] | None = None,
    delta_result_on: str | Sequence[str] | None = None,
    delta_compare: str | Sequence[str] | None = None,
    delta_retry: bool | str | None = None,
    delta_unsafe: bool = False,
) -> tuple["DataChain | None", str, str]:
    """Select a snapshot and attach its full-prefix refresh, without reading rows.

    A single file has no snapshot. The URI and path are kept from this selection
    so Catalog can browse the same snapshot without selecting it again.
    """
    from .dc.datasets import read_dataset

    name, list_uri, list_path, _ = get_listing(
        uri, session, update=update, versions=versions
    )
    if name is None:
        return None, list_uri, list_path

    catalog = session.catalog
    chain = read_dataset(
        name,
        namespace=catalog.metastore.system_namespace_name,
        project=catalog.metastore.listing_project_name,
        session=session,
        settings=settings,
        delta=delta,
        delta_on=delta_on,
        delta_result_on=delta_result_on,
        delta_compare=delta_compare,
        delta_retry=delta_retry,
        delta_unsafe=delta_unsafe,
    )
    # A snapshot shared by several URIs is refreshed once per operation.
    chain._query.update = update and name not in refreshed
    refreshed.add(name)
    chain.signals_schema = chain.signals_schema.mutate({column: file_type})
    # A reused snapshot can cover a parent: always refresh its whole prefix.
    chain._query.set_listing_fn(
        lambda: write_listing(
            listing_uri_from_name(name),
            name,
            session,
            client_config,
            column=column,
            file_type=file_type,
            settings=settings,
            in_memory=in_memory,
        )
    )
    return chain, list_uri, list_path


def write_listing(
    uri: str,
    ds_name: str,
    session: Session,
    client_config: dict,
    *,
    column: str = "file",
    file_type: type[File] = File,
    settings: dict | None = None,
    in_memory: bool = False,
) -> None:
    """Store the listing of the prefix ``uri``, whole, as the listing dataset
    ``ds_name``: the snapshot ``read_storage`` reads and reuses.
    """
    from .dc.records import create_records_dataset

    namespace = session.catalog.metastore.system_namespace_name
    project = session.catalog.metastore.listing_project_name
    list_func = list_bucket(uri, session.catalog.cache, client_config=client_config)
    # content_hash=None because hash_callable doesn't capture list_func's
    # closure (which holds the URI and the client config) -- auto-hashing
    # the seed would let the UDF checkpoint cache return a stale listing
    # across URIs/runs.
    (
        create_records_dataset(
            [{"seed": 0}],
            schema={"seed": int},
            content_hash=None,
            session=session,
            settings=settings,
            in_memory=in_memory,
        )
        .settings(
            prefetch=0,
            batch_size=LISTING_BATCH_SIZE,
            namespace=namespace,
            project=project,
        )
        .gen(list_func, output={column: file_type})
        # for internal listing datasets, we always bump major version
        .save(ds_name, listing=True, update_version="major")
    )
