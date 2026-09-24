"""Cloud Storage source: open a ``gs://`` object for ranged reads, never a download.

``google-cloud-storage`` is the optional ``[gcs]`` extra, so nothing here
imports ``google.*`` at module level: the API imports and serves its other
routes without it. The client, bucket and blob are typed with the small
protocols below, the slice of the SDK this module uses, which the tests'
fakes implement too.

The object is opened with ``blob.open("rb")``, a seekable ``BlobReader``:
the library samples it from its current position like a local file, so a
request costs one metadata GET (the reader's first ``seek`` learns the size)
and one ranged GET per sampled window, whatever the object's size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import BinaryIO, Protocol

from pydantic import BaseModel, Field

GCS_URI_PATTERN = r"^gs://[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]/.+$"
"""A bucket name and a non-empty object name: no bucket-only URI."""

INSTALL_HINT = "GCS support not installed (pip install 'csv-inspector-api[gcs]')"

GCS_CHUNK_BYTES = 256 * 1024
"""Granularity of a ``BlobReader`` chunk: the GCS minimum, 256 KiB."""


class GcsBlob(Protocol):
    """The part of ``google.cloud.storage.Blob`` this module uses."""

    @property
    def size(self) -> int | None:
        """Object size in bytes, once the reader has loaded the metadata."""

    @property
    def generation(self) -> int | None:
        """Object generation, pinned or loaded with the metadata."""

    def open(self, mode: str, chunk_size: int | None = None) -> BinaryIO:
        """Open the object; ``"rb"`` returns a seekable ``BlobReader``."""


class GcsBucket(Protocol):
    """The part of ``google.cloud.storage.Bucket`` this module uses."""

    def blob(self, blob_name: str, *, generation: int | None = None) -> GcsBlob:
        """Build a blob handle, without any request."""


class GcsClient(Protocol):
    """The part of ``google.cloud.storage.Client`` this module uses."""

    def bucket(self, bucket_name: str) -> GcsBucket:
        """Build a bucket handle, without any request."""


class GcsNotInstalledError(Exception):
    """``google-cloud-storage`` (the ``[gcs]`` extra) is missing; answered with 503."""


class GcsInspectRequest(BaseModel):
    """Body of ``POST /inspect/gcs``.

    Attributes:
        uri: The object, as ``gs://bucket/path/to/file.csv``.
        generation: Object generation to read; ``None`` reads the live version.
    """

    uri: str = Field(pattern=GCS_URI_PATTERN, examples=["gs://my-bucket/exports/2024/sales.csv"])
    generation: int | None = Field(default=None, ge=1, description="Pin an object version.")


@dataclass(frozen=True)
class GcsObject:
    """An opened object: the reader to sample and the blob that describes it.

    Attributes:
        blob: The blob; its ``size`` and ``generation`` are known once read.
        reader: Seekable binary reader over the object; close it when done.
    """

    blob: GcsBlob
    reader: BinaryIO


def create_client(project: str | None) -> GcsClient:
    """Build a Cloud Storage client with Application Default Credentials.

    No request is sent to Cloud Storage; finding the credentials may query
    the metadata server on Google Cloud.

    Args:
        project: Google Cloud project; ``None`` lets the SDK infer it.

    Returns:
        The client.

    Raises:
        GcsNotInstalledError: If the ``[gcs]`` extra is not installed.
        google.auth.exceptions.DefaultCredentialsError: If no credentials are found.
    """
    try:
        from google.cloud.storage import Client  # noqa: PLC0415 - the optional [gcs] extra
    except ImportError as exc:
        raise GcsNotInstalledError(INSTALL_HINT) from exc
    client: GcsClient = Client(project=project) if project else Client()
    return client


def chunk_size_for(n_bytes: int, tail_bytes: int) -> int:
    """Reader chunk size: the larger window, rounded up to 256 KiB.

    Each sampled window is then fetched with one ranged GET, instead of the
    SDK's 40 MiB default chunk.

    Args:
        n_bytes: Head window, in bytes.
        tail_bytes: Tail window, in bytes.

    Returns:
        A positive multiple of :data:`GCS_CHUNK_BYTES`.
    """
    window = max(n_bytes, tail_bytes, 1)
    return -(-window // GCS_CHUNK_BYTES) * GCS_CHUNK_BYTES


def open_gcs_object(
    uri: str, *, client: GcsClient, generation: int | None, chunk_size: int
) -> GcsObject:
    """Open a ``gs://`` object for ranged reads, without sending any request.

    The reader fetches the metadata on its first ``seek`` and then only the
    ranges read; the object is never downloaded as a whole.

    Args:
        uri: A URI matching :data:`GCS_URI_PATTERN`.
        client: The Cloud Storage client.
        generation: Object generation to read; ``None`` reads the live version.
        chunk_size: Bytes fetched per request; see :func:`chunk_size_for`.

    Returns:
        The blob and its open reader.

    Raises:
        ValueError: If ``uri`` is not a ``gs://bucket/object`` URI.
        TypeError: If the reader is not a readable, seekable stream: the
            library would then consume it in one pass instead of seeking.
    """
    if re.fullmatch(GCS_URI_PATTERN, uri) is None:
        msg = f"not a gs://bucket/object URI: {uri!r}"
        raise ValueError(msg)
    bucket_name, _, blob_name = uri.removeprefix("gs://").partition("/")
    blob = client.bucket(bucket_name).blob(blob_name, generation=generation)
    reader = blob.open("rb", chunk_size=chunk_size)
    if not (reader.readable() and reader.seekable()):
        reader.close()
        msg = f"the reader of {uri} is not a readable, seekable stream"
        raise TypeError(msg)
    return GcsObject(blob, reader)
