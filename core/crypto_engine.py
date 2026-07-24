"""
core/crypto_engine.py
========================

Encryption subsystem for StarkOS: envelope encryption for bytes, files
and whole folders.

Responsibilities
----------------
- AES-256-GCM authenticated encryption (confidentiality + integrity +
  authenticity in one primitive) for both the data itself and the key
  that protects it.
- Envelope encryption: a fresh, random 256-bit Data Encryption Key
  (DEK) is generated per encryption operation and used with AES-GCM on
  the actual payload; the DEK itself is then wrapped (encrypted) with a
  Key Encryption Key (KEK) derived from a passphrase via Argon2id (or
  supplied directly). This is the standard reason envelope encryption
  exists: the expensive, deliberately-slow KDF only has to run once per
  operation regardless of payload size, and rotating to a new KEK later
  only means re-wrapping the small DEK -- never re-encrypting the (
  possibly large) payload.
- Optional compression before encryption (correct order: compress-then-
  encrypt; see the honesty note below on when that pattern is and isn't
  safe).
- Whole-file and whole-folder encryption (folders are archived with
  `zipfile` first, then encrypted as one buffer), with zip-slip path-
  traversal protection on extraction.
- Key rotation without touching the encrypted payload.
- Integration with ConfigManager (KDF cost parameters), EventBus
  (encrypt/decrypt/rotate events), and Identity (attributing operations
  to the current session in those events) -- NOT authorization; see
  below.

Honesty about scope
--------------------
1. **This module does cryptography, not access control.** Whether a
   given caller is *allowed* to encrypt or decrypt something is
   `core.security_core.SecurityCore`'s job (RBAC/ABAC), not this
   module's -- gate calls to it with `security_core.require(principal,
   "crypto.decrypt", ...)` at the call site rather than expecting
   `CryptoEngine` to duplicate that logic.

2. **Python cannot guarantee secret key material is scrubbed from
   memory.** `bytes` objects are immutable, the interpreter may copy
   them internally, and CPython's garbage collector gives no
   scrubbing guarantee. This module does not claim to zero DEKs/KEKs
   from RAM after use -- that guarantee needs a hardware security
   module or a runtime built for it. Treat process memory (swap files,
   core dumps, debuggers) as sensitive for the lifetime of the process.

3. **Compress-then-encrypt is safe here, but *why* matters.** The well-
   known CRIME/BREACH-style attacks exploit compression only when an
   adversary can (a) inject chosen plaintext next to a secret in the
   *same* compressed stream and (b) observe the resulting ciphertext
   *length* over many attempts (a network length-oracle). Neither
   condition applies to encrypting a file or folder at rest -- there's
   no attacker-controlled data sharing a compression stream with your
   secret, and no repeated-length-oracle. This reasoning would flip if
   this compression pattern were ever reused inside a network protocol
   that mixes attacker input and secrets in one stream.

4. **Whole-buffer AES-GCM, not chunked streaming.** `encrypt_bytes`/
   `encrypt_file`/`encrypt_folder` read the entire payload into memory
   and perform one AES-GCM call (correct and simple; AES-GCM's NIST
   size limit per key+nonce is ~64 GB, far above what this reads into
   RAM anyway). This means encrypting something larger than available
   memory isn't supported -- a deliberate simplicity-over-cleverness
   choice: a custom chunked-AEAD scheme (per-chunk nonces, truncation
   protection, chunk ordering) is exactly the kind of hand-rolled
   construction most likely to introduce a subtle, dangerous bug, and
   wasn't worth that risk for a v0.4 "basic security" milestone.

Design
------
Same shape as the rest of StarkOS's security-adjacent modules: a small
`KeySource` Protocol for however the KEK is obtained, with two honest
defaults -- `PassphraseKeySource` (Argon2id, via the optional
`argon2-cffi` package) and `RawKeySource` (a pre-existing high-entropy
key, e.g. from a real secrets manager once one exists).

`CryptoEngine` satisfies the `Module` protocol (name/initialize/start/
stop) and registers with the Kernel like any other StarkOS module:

    crypto = CryptoEngine(services=services)
    crypto.set_key_source(PassphraseKeySource(passphrase=b"correct horse battery staple"))
    kernel.register_module(crypto, name="crypto_engine", priority=20)

    envelope = crypto.encrypt_file(Path("secret.pdf"), Path("secret.pdf.starkcrypt"))
    crypto.decrypt_file(Path("secret.pdf.starkcrypt"), Path("secret.decrypted.pdf"))
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import os
import secrets
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.config_manager import ConfigManager
from core.event_bus import Event, EventBus
from core.identity import Identity
from core.logger import get_logger
from core.service_container import ServiceContainer

logger = get_logger("crypto_engine")

# =============================================================================
# Exceptions
# =============================================================================

class CryptoEngineError(Exception):
    """Base exception for CryptoEngine failures."""

class KeySourceNotConfiguredError(CryptoEngineError):
    """Raised when an operation needs a KeySource but none was set."""

class KeyDerivationError(CryptoEngineError):
    """Raised when deriving/obtaining the KEK fails."""

class EncryptionError(CryptoEngineError):
    """Raised when encryption itself fails (not authorization/derivation)."""

class DecryptionError(CryptoEngineError):
    """Raised when decryption fails: wrong key, tampered/corrupt data."""

class EnvelopeFormatError(CryptoEngineError):
    """Raised when a serialized envelope is malformed or unreadable."""

# =============================================================================
# Key sources (KEK acquisition)
# =============================================================================

@runtime_checkable
class KeySource(Protocol):
    """Supplies the 32-byte Key Encryption Key used to wrap an
    envelope's DEK. `salt` is whatever was stored in that envelope
    (generated fresh per encryption) -- a passphrase-based source
    re-derives from it; a raw-key source ignores it."""

    def get_kek(self, salt: bytes) -> bytes: ...
    def check_available(self) -> bool: ...

@dataclass(slots=True, frozen=True)
class Argon2Parameters:
    """Argon2id cost parameters. Defaults are a reasonable modern
    baseline (OWASP's current guidance is time_cost=1 with a large
    memory_cost, or time_cost>=2-3 with more modest memory -- these
    defaults favor a bit more of both for headroom); tune for your
    hardware and threat model."""

    time_cost: int = 3
    memory_cost_kib: int = 262144  # 256 MiB
    parallelism: int = 4
    hash_len: int = 32  # -> 256-bit KEK, matching AES-256

    def as_dict(self) -> dict[str, int]:
        return {
            "time_cost": self.time_cost,
            "memory_cost_kib": self.memory_cost_kib,
            "parallelism": self.parallelism,
            "hash_len": self.hash_len,
        }

class PassphraseKeySource:
    """
    Derives the KEK from a passphrase via Argon2id -- the Password
    Hashing Competition winner and OWASP's current recommendation over
    PBKDF2/bcrypt/scrypt for new designs, specifically the "id" variant
    (hybrid: resistant to both GPU/ASIC cracking and side-channel
    attacks, unlike plain Argon2d or Argon2i). Requires the optional
    `argon2-cffi` package; degrades to a clear KeyDerivationError
    (never a silent wrong key) if it isn't installed.

    IMPORTANT: the exact parameters used at encryption time must be
    reproduced at decryption time (that's inherent to any KDF -- get a
    parameter wrong and you deterministically get a *different* key,
    not an error). This class does not auto-reconcile parameter
    mismatches from a stored envelope; construct it with the same
    `Argon2Parameters` you encrypted with, or decryption will correctly
    fail with DecryptionError rather than silently succeeding on wrong
    data.
    """

    def __init__(self, *, passphrase: bytes, params: Optional[Argon2Parameters] = None) -> None:
        if not passphrase:
            raise ValueError("passphrase cannot be empty.")
        self._passphrase = passphrase
        self._params = params or Argon2Parameters()

    def check_available(self) -> bool:
        try:
            import argon2  # noqa: F401
            return True
        except ImportError:
            return False

    def get_kek(self, salt: bytes) -> bytes:
        try:
            from argon2.low_level import Type, hash_secret_raw
        except ImportError as exc:
            raise KeyDerivationError("The 'argon2-cffi' package is not installed.") from exc

        try:
            return hash_secret_raw(
                secret=self._passphrase,
                salt=salt,
                time_cost=self._params.time_cost,
                memory_cost=self._params.memory_cost_kib,
                parallelism=self._params.parallelism,
                hash_len=self._params.hash_len,
                type=Type.ID,
            )
        except Exception as exc:
            raise KeyDerivationError("Argon2id key derivation failed.") from exc

class RawKeySource:
    """
    Supplies a pre-existing, high-entropy 256-bit key directly -- no
    passphrase, no derivation (e.g. a key already issued by a real
    secrets manager, once StarkOS has one). `salt` is accepted for
    interface consistency with PassphraseKeySource but ignored.
    """

    def __init__(self, *, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("key must be exactly 32 bytes (256 bits) for AES-256.")
        self._key = key

    def check_available(self) -> bool:
        return True

    def get_kek(self, salt: bytes) -> bytes:
        return self._key

# =============================================================================
# Envelope
# =============================================================================

@dataclass(slots=True, frozen=True)
class EncryptedEnvelope:
    """
    Everything needed to decrypt, except the KeySource itself. `salt`
    and `kdf_params` are recorded for auditability/debugging; the
    KeySource used at decryption time is responsible for actually
    reproducing the correct KEK (see PassphraseKeySource's docstring).
    """

    version: int
    kdf: str
    kdf_params: dict[str, Any]
    salt: bytes
    wrapped_dek_nonce: bytes
    wrapped_dek_ciphertext: bytes
    data_nonce: bytes
    data_ciphertext: bytes
    compressed: bool
    aad_context: str
    created_at: datetime = field(default_factory=datetime.utcnow)

_ENVELOPE_VERSION = 1
_AAD_PREFIX = "starkos-crypto-engine"

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class CryptoEngineConfig:
    compress: bool = True
    compression_level: int = 6
    argon2_params: Argon2Parameters = field(default_factory=Argon2Parameters)
    publish_events: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Crypto Engine
# =============================================================================

class CryptoEngine:
    """
    StarkOS envelope-encryption module. See the module docstring's
    "Honesty about scope" section before relying on this for anything
    beyond what it actually claims.

    Satisfies the `Module` protocol (name/initialize/start/stop) and
    can be registered with the Kernel like any other module.
    """

    def __init__(
        self,
        *,
        services: ServiceContainer,
        config: Optional[CryptoEngineConfig] = None,
        key_source: Optional[KeySource] = None,
    ) -> None:
        self._services = services
        self._config = config or CryptoEngineConfig()
        self._key_source: Optional[KeySource] = key_source

        self._identity: Optional[Identity] = None
        self._config_manager: Optional[ConfigManager] = None
        self._event_bus: Optional[EventBus] = None

        logger.info("CryptoEngine constructed.", extra={"compress": self._config.compress})

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "crypto_engine"

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to CryptoEngine.")

    def bind_config_manager(self, config_manager: ConfigManager) -> None:
        self._config_manager = config_manager
        self._apply_config_overrides()
        logger.debug("ConfigManager bound to CryptoEngine.")

    def bind_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        logger.debug("EventBus bound to CryptoEngine.")

    def set_key_source(self, key_source: KeySource) -> None:
        self._key_source = key_source
        logger.info("KeySource configured.", extra={"kdf": type(key_source).__name__})

    def _apply_config_overrides(self) -> None:
        if self._config_manager is None:
            return
        try:
            section = self._config_manager.get("crypto", default={})
        except Exception:
            section = {}
        if not isinstance(section, dict):
            return

        params = self._config.argon2_params
        self._config.argon2_params = Argon2Parameters(
            time_cost=int(section.get("argon2_time_cost", params.time_cost)),
            memory_cost_kib=int(section.get("argon2_memory_cost_kib", params.memory_cost_kib)),
            parallelism=int(section.get("argon2_parallelism", params.parallelism)),
            hash_len=params.hash_len,
        )
        self._config.compress = bool(section.get("compress", self._config.compress))

    async def initialize(self) -> None:
        logger.info("Initializing CryptoEngine.")
        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._config_manager is None:
            self._config_manager = self._services.resolve_optional(ConfigManager)
            if self._config_manager is not None:
                self._apply_config_overrides()
        if self._event_bus is None:
            self._event_bus = self._services.resolve_optional(EventBus)

        if self._key_source is None:
            logger.warning("No KeySource configured yet -- call set_key_source() before encrypting/decrypting.")

        logger.info("CryptoEngine initialized.", extra={"key_source_configured": self._key_source is not None})

    async def start(self) -> None:
        logger.info("CryptoEngine ready.")

    async def stop(self) -> None:
        logger.info("CryptoEngine stopped.")

    # ------------------------------------------------------------------
    # AAD construction
    # ------------------------------------------------------------------

    def _build_aad(self, aad_context: str, *, compressed: bool) -> bytes:
        # Binding version/context/compressed-flag into the authenticated
        # data means an attacker can't relabel a ciphertext as belonging
        # to a different context, or flip the compression flag, without
        # the GCM tag failing to verify.
        return f"{_AAD_PREFIX}|v{_ENVELOPE_VERSION}|{aad_context}|compressed={compressed}".encode("utf-8")

    # ------------------------------------------------------------------
    # Core: bytes in, envelope out (and back)
    # ------------------------------------------------------------------

    def encrypt_bytes(self, plaintext: bytes, *, aad_context: str = "") -> EncryptedEnvelope:
        if self._key_source is None:
            raise KeySourceNotConfiguredError("No KeySource configured -- call set_key_source() first.")

        payload = plaintext
        compressed = False
        if self._config.compress:
            candidate = zlib.compress(plaintext, level=self._config.compression_level)
            if len(candidate) < len(payload):
                payload = candidate
                compressed = True

        salt = secrets.token_bytes(16)
        try:
            kek = self._key_source.get_kek(salt)
        except CryptoEngineError:
            raise
        except Exception as exc:
            raise KeyDerivationError("Key derivation failed.") from exc
        if len(kek) != 32:
            raise KeyDerivationError(f"KeySource returned a {len(kek)}-byte key; AES-256 requires exactly 32 bytes.")

        dek = secrets.token_bytes(32)  # fresh, random DEK per encryption -- the core of envelope encryption
        aad = self._build_aad(aad_context, compressed=compressed)

        try:
            data_nonce = secrets.token_bytes(12)
            data_ciphertext = AESGCM(dek).encrypt(data_nonce, payload, aad)

            wrap_nonce = secrets.token_bytes(12)
            wrapped_dek = AESGCM(kek).encrypt(wrap_nonce, dek, aad)
        except Exception as exc:
            raise EncryptionError("AES-GCM encryption failed.") from exc

        envelope = EncryptedEnvelope(
            version=_ENVELOPE_VERSION,
            kdf=type(self._key_source).__name__,
            kdf_params=self._config.argon2_params.as_dict() if isinstance(self._key_source, PassphraseKeySource) else {},
            salt=salt,
            wrapped_dek_nonce=wrap_nonce,
            wrapped_dek_ciphertext=wrapped_dek,
            data_nonce=data_nonce,
            data_ciphertext=data_ciphertext,
            compressed=compressed,
            aad_context=aad_context,
        )

        logger.info(
            "Data encrypted.",
            extra={"plaintext_bytes": len(plaintext), "ciphertext_bytes": len(data_ciphertext), "compressed": compressed},
        )
        return envelope

    def decrypt_bytes(self, envelope: EncryptedEnvelope) -> bytes:
        if self._key_source is None:
            raise KeySourceNotConfiguredError("No KeySource configured -- call set_key_source() first.")

        try:
            kek = self._key_source.get_kek(envelope.salt)
        except CryptoEngineError:
            raise
        except Exception as exc:
            raise KeyDerivationError("Key derivation failed.") from exc
        if len(kek) != 32:
            raise KeyDerivationError(f"KeySource returned a {len(kek)}-byte key; AES-256 requires exactly 32 bytes.")

        aad = self._build_aad(envelope.aad_context, compressed=envelope.compressed)

        try:
            dek = AESGCM(kek).decrypt(envelope.wrapped_dek_nonce, envelope.wrapped_dek_ciphertext, aad)
        except InvalidTag as exc:
            raise DecryptionError(
                "Failed to unwrap the data key -- wrong key/passphrase, or the envelope was tampered with."
            ) from exc

        try:
            payload = AESGCM(dek).decrypt(envelope.data_nonce, envelope.data_ciphertext, aad)
        except InvalidTag as exc:
            raise DecryptionError("Failed to decrypt data -- the ciphertext was tampered with or is corrupt.") from exc

        if envelope.compressed:
            try:
                payload = zlib.decompress(payload)
            except zlib.error as exc:
                raise DecryptionError("Decompression failed after a successful decrypt -- envelope may be corrupt.") from exc

        logger.info("Data decrypted.", extra={"plaintext_bytes": len(payload)})
        return payload

    # ------------------------------------------------------------------
    # Key rotation (re-wrap the DEK only -- never touches the payload)
    # ------------------------------------------------------------------

    def rotate_key(self, envelope: EncryptedEnvelope, *, new_key_source: KeySource) -> EncryptedEnvelope:
        """
        Re-wrap this envelope's DEK under a new KEK, without decrypting
        or re-encrypting the (possibly large) payload -- the whole
        point of envelope encryption. Requires the CURRENT KeySource
        (already configured via `set_key_source`) to still be correct,
        since the existing wrapped DEK must be unwrapped first.
        """
        if self._key_source is None:
            raise KeySourceNotConfiguredError("No current KeySource configured -- set_key_source() first.")

        aad = self._build_aad(envelope.aad_context, compressed=envelope.compressed)

        try:
            current_kek = self._key_source.get_kek(envelope.salt)
            dek = AESGCM(current_kek).decrypt(envelope.wrapped_dek_nonce, envelope.wrapped_dek_ciphertext, aad)
        except InvalidTag as exc:
            raise DecryptionError("Failed to unwrap the data key with the current KeySource.") from exc
        except CryptoEngineError:
            raise
        except Exception as exc:
            raise KeyDerivationError("Key derivation failed for the current KeySource.") from exc

        new_salt = secrets.token_bytes(16)
        try:
            new_kek = new_key_source.get_kek(new_salt)
        except Exception as exc:
            raise KeyDerivationError("Key derivation failed for the new KeySource.") from exc
        if len(new_kek) != 32:
            raise KeyDerivationError(f"New KeySource returned a {len(new_kek)}-byte key; AES-256 requires 32 bytes.")

        try:
            new_wrap_nonce = secrets.token_bytes(12)
            new_wrapped_dek = AESGCM(new_kek).encrypt(new_wrap_nonce, dek, aad)
        except Exception as exc:
            raise EncryptionError("Failed to re-wrap the data key under the new KeySource.") from exc

        rotated = dataclasses.replace(
            envelope,
            kdf=type(new_key_source).__name__,
            kdf_params=self._config.argon2_params.as_dict() if isinstance(new_key_source, PassphraseKeySource) else {},
            salt=new_salt,
            wrapped_dek_nonce=new_wrap_nonce,
            wrapped_dek_ciphertext=new_wrapped_dek,
        )
        self._key_source = new_key_source
        logger.info("Envelope key rotated.", extra={"aad_context": envelope.aad_context, "new_kdf": type(new_key_source).__name__})
        return rotated

    # ------------------------------------------------------------------
    # Envelope serialization (JSON container, base64 for binary fields)
    # ------------------------------------------------------------------

    def _envelope_to_dict(self, envelope: EncryptedEnvelope) -> dict[str, Any]:
        return {
            "version": envelope.version,
            "kdf": envelope.kdf,
            "kdf_params": envelope.kdf_params,
            "salt": base64.b64encode(envelope.salt).decode("ascii"),
            "wrapped_dek_nonce": base64.b64encode(envelope.wrapped_dek_nonce).decode("ascii"),
            "wrapped_dek_ciphertext": base64.b64encode(envelope.wrapped_dek_ciphertext).decode("ascii"),
            "data_nonce": base64.b64encode(envelope.data_nonce).decode("ascii"),
            "data_ciphertext": base64.b64encode(envelope.data_ciphertext).decode("ascii"),
            "compressed": envelope.compressed,
            "aad_context": envelope.aad_context,
            "created_at": envelope.created_at.isoformat(),
        }

    def _envelope_from_dict(self, data: dict[str, Any]) -> EncryptedEnvelope:
        try:
            return EncryptedEnvelope(
                version=int(data["version"]),
                kdf=data["kdf"],
                kdf_params=data.get("kdf_params", {}),
                salt=base64.b64decode(data["salt"]),
                wrapped_dek_nonce=base64.b64decode(data["wrapped_dek_nonce"]),
                wrapped_dek_ciphertext=base64.b64decode(data["wrapped_dek_ciphertext"]),
                data_nonce=base64.b64decode(data["data_nonce"]),
                data_ciphertext=base64.b64decode(data["data_ciphertext"]),
                compressed=bool(data["compressed"]),
                aad_context=data.get("aad_context", ""),
                created_at=datetime.fromisoformat(data["created_at"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise EnvelopeFormatError("Malformed envelope data.") from exc

    def save_envelope(self, envelope: EncryptedEnvelope, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as stream:
                json.dump(self._envelope_to_dict(envelope), stream, ensure_ascii=False, indent=2)
            tmp_path.replace(path)
        except OSError as exc:
            raise CryptoEngineError(f"Unable to write envelope to '{path}'.") from exc

    def load_envelope(self, path: Path) -> EncryptedEnvelope:
        if not path.exists():
            raise CryptoEngineError(f"Envelope file not found: '{path}'.")
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvelopeFormatError(f"Unable to read envelope from '{path}'.") from exc
        return self._envelope_from_dict(data)

    # ------------------------------------------------------------------
    # File encryption
    # ------------------------------------------------------------------

    def encrypt_file(self, source_path: Path, destination_path: Path) -> EncryptedEnvelope:
        if not source_path.exists() or not source_path.is_file():
            raise CryptoEngineError(f"Source file not found: '{source_path}'.")
        try:
            plaintext = source_path.read_bytes()
        except OSError as exc:
            raise CryptoEngineError(f"Unable to read '{source_path}'.") from exc

        envelope = self.encrypt_bytes(plaintext, aad_context=f"file:{source_path.name}")
        self.save_envelope(envelope, destination_path)
        logger.info("File encrypted.", extra={"source": str(source_path), "destination": str(destination_path)})
        return envelope

    def decrypt_file(self, source_path: Path, destination_path: Path) -> None:
        envelope = self.load_envelope(source_path)
        plaintext = self.decrypt_bytes(envelope)
        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_bytes(plaintext)
        except OSError as exc:
            raise CryptoEngineError(f"Unable to write decrypted output to '{destination_path}'.") from exc
        logger.info("File decrypted.", extra={"source": str(source_path), "destination": str(destination_path)})

    # ------------------------------------------------------------------
    # Folder encryption (archive-then-encrypt)
    # ------------------------------------------------------------------

    def encrypt_folder(self, source_dir: Path, destination_path: Path) -> EncryptedEnvelope:
        if not source_dir.exists() or not source_dir.is_dir():
            raise CryptoEngineError(f"Source folder not found: '{source_dir}'.")

        buffer = BytesIO()
        try:
            with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
                # ZIP_STORED (no compression here) -- CryptoEngine's own
                # zlib compression step below already handles that, on
                # the whole archive at once rather than per-entry.
                for file_path in sorted(source_dir.rglob("*")):
                    if file_path.is_file():
                        archive.write(file_path, arcname=file_path.relative_to(source_dir))
        except OSError as exc:
            raise CryptoEngineError(f"Unable to archive folder '{source_dir}'.") from exc

        envelope = self.encrypt_bytes(buffer.getvalue(), aad_context=f"folder:{source_dir.name}")
        self.save_envelope(envelope, destination_path)
        logger.info("Folder encrypted.", extra={"source": str(source_dir), "destination": str(destination_path)})
        return envelope

    def decrypt_folder(self, source_path: Path, destination_dir: Path) -> None:
        envelope = self.load_envelope(source_path)
        archive_bytes = self.decrypt_bytes(envelope)

        destination_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
                self._safe_extract(archive, destination_dir)
        except zipfile.BadZipFile as exc:
            raise DecryptionError("Decrypted data is not a valid archive -- envelope may be corrupt.") from exc

        logger.info("Folder decrypted.", extra={"source": str(source_path), "destination": str(destination_dir)})

    def _safe_extract(self, archive: zipfile.ZipFile, destination_dir: Path) -> None:
        """
        Guards against "zip-slip": a malicious or corrupt archive entry
        like "../../etc/passwd" that would otherwise let extraction
        write outside `destination_dir`. Every member's resolved path
        is verified to stay within the destination *before* anything is
        extracted.
        """
        resolved_destination = destination_dir.resolve()
        for member in archive.namelist():
            target_path = (resolved_destination / member).resolve()
            if target_path != resolved_destination and resolved_destination not in target_path.parents:
                raise DecryptionError(f"Refusing to extract '{member}': path traversal (zip-slip) detected.")
        archive.extractall(resolved_destination)

    # ------------------------------------------------------------------
    # Async convenience wrappers (crypto work off the event loop thread,
    # with best-effort EventBus notification afterward)
    # ------------------------------------------------------------------

    async def encrypt_file_async(self, source_path: Path, destination_path: Path) -> EncryptedEnvelope:
        envelope = await asyncio.to_thread(self.encrypt_file, source_path, destination_path)
        await self._publish_event("crypto.encrypted", aad_context=envelope.aad_context, kind="file")
        return envelope

    async def decrypt_file_async(self, source_path: Path, destination_path: Path) -> None:
        await asyncio.to_thread(self.decrypt_file, source_path, destination_path)
        await self._publish_event("crypto.decrypted", aad_context=str(source_path), kind="file")

    async def encrypt_folder_async(self, source_dir: Path, destination_path: Path) -> EncryptedEnvelope:
        envelope = await asyncio.to_thread(self.encrypt_folder, source_dir, destination_path)
        await self._publish_event("crypto.encrypted", aad_context=envelope.aad_context, kind="folder")
        return envelope

    async def decrypt_folder_async(self, source_path: Path, destination_dir: Path) -> None:
        await asyncio.to_thread(self.decrypt_folder, source_path, destination_dir)
        await self._publish_event("crypto.decrypted", aad_context=str(source_path), kind="folder")

    async def _publish_event(self, topic: str, **payload: Any) -> None:
        if self._event_bus is None or not self._config.publish_events:
            return
        if self._identity is not None:
            payload.setdefault("principal", self._identity.persona.name)
        try:
            await self._event_bus.publish(Event(topic=topic, source="crypto_engine", payload=payload))
        except Exception:
            logger.exception("Failed to publish CryptoEngine event '%s'.", topic)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "key_source_configured": self._key_source is not None,
            "key_source_kind": type(self._key_source).__name__ if self._key_source else None,
            "compress": self._config.compress,
            "argon2_params": self._config.argon2_params.as_dict(),
            "identity_bound": self._identity is not None,
            "config_manager_bound": self._config_manager is not None,
            "event_bus_bound": self._event_bus is not None,
        }