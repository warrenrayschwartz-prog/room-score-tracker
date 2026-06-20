"""Database layer for Room Score Tracker.

Stores each user's children, scores, baseline photos, and settings
server-side so the data survives browser clears, new devices, and
reinstalls. Postgres on Railway via the DATABASE_URL env var, with a
SQLite fallback for local development.

Tables:
    users       id (uuid pk), created_at, email, email_norm, password_hash,
                display_name, google_sub, apple_sub, disabled, token_version
    app_state   user_id (uuid pk/fk), data (json) — children, scores,
                difficulty, maxAllowance
    images      id (uuid pk), user_id (fk, indexed), kind ('baseline' |
                'photo'), key (text), data (text), unique(user_id, kind, key)

The session token the client holds is a signed bearer credential (see
server.py). Never log it or put it in a URL.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Text,
    UniqueConstraint,
    Uuid,
    create_engine,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    scoped_session,
    sessionmaker,
)


# ── Engine ─────────────────────────────────────────────────────────────────

def _build_engine_url() -> str:
    """Read DATABASE_URL from env, normalize, fall back to local SQLite."""
    url = (os.environ.get('DATABASE_URL') or '').strip()
    if not url:
        local_path = Path(__file__).parent / 'roomscore.db'
        return f'sqlite:///{local_path.as_posix()}'
    # Railway/Heroku historically expose 'postgres://...'; SQLAlchemy 2.x
    # wants the modern 'postgresql+psycopg2://' prefix.
    if url.startswith('postgres://'):
        url = 'postgresql+psycopg2://' + url[len('postgres://'):]
    elif url.startswith('postgresql://') and '+psycopg2' not in url:
        url = 'postgresql+psycopg2://' + url[len('postgresql://'):]
    return url


_ENGINE_URL = _build_engine_url()
_IS_SQLITE = _ENGINE_URL.startswith('sqlite')

_engine_kwargs: dict[str, Any] = {'pool_pre_ping': True, 'future': True}
if _IS_SQLITE:
    _engine_kwargs['connect_args'] = {'check_same_thread': False}

engine = create_engine(_ENGINE_URL, **_engine_kwargs)

SessionLocal = scoped_session(
    sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
)


# ── Models ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'users'

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # email_norm (lowercased+trimmed) is the lookup/uniqueness key, enforced
    # in app code so the schema stays SQLite/Postgres portable. email keeps
    # original casing for display.
    email: Mapped[str | None] = mapped_column(Text, nullable=True)
    email_norm: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    # OAuth subject ids (the provider's stable unique user id).
    google_sub: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    apple_sub: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    disabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default='false'
    )
    # Bump to invalidate all existing tokens (logout-everywhere / reset).
    token_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default='0'
    )

    state: Mapped['AppState | None'] = relationship(
        'AppState', back_populates='user', cascade='all, delete-orphan',
        uselist=False,
    )
    images: Mapped[list['Image']] = relationship(
        'Image', back_populates='user', cascade='all, delete-orphan',
    )


class AppState(Base):
    """One row per user holding the small, non-image state as JSON:
    {children: [...], scores: {...}, difficulty: int, maxAllowance: number}.
    Images live in the images table because they're large."""
    __tablename__ = 'app_state'

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        primary_key=True,
    )
    data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )

    user: Mapped[User] = relationship('User', back_populates='state')


class Image(Base):
    """A stored image (or per-day photo bundle) for a user.

    kind='baseline': key is '<childOrShared>|<slotId>', data is a base64
        data URL string.
    kind='photo':    key is '<childName>|<day>', data is a JSON string
        mapping slotId -> base64 thumbnail (mirrors the old IndexedDB
        shape where each day held a dict of slot photos).
    """
    __tablename__ = 'images'
    __table_args__ = (
        UniqueConstraint('user_id', 'kind', 'key', name='uq_image_user_kind_key'),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False, index=True,
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    key: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )

    user: Mapped[User] = relationship('User', back_populates='images')


# ── Init ───────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every boot."""
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()


# ── Serializers ──────────────────────────────────────────────────────────────

def serialize_account(user: 'User') -> dict[str, Any]:
    """Account info for the client. NEVER includes password_hash,
    token_version, or the raw id (the client only holds a signed token)."""
    return {
        'email': user.email,
        'displayName': user.display_name,
        'hasPassword': bool(user.password_hash),
        'google': bool(user.google_sub),
        'apple': bool(user.apple_sub),
    }
