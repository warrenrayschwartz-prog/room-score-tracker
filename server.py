#!/usr/bin/env python3
"""Room Score Tracker — Flask server with accounts + server-side storage.

Auth mirrors DoggoTranslator: email/password (werkzeug hashing) plus
optional Google and Apple sign-in, with signed session tokens
(itsdangerous) carried in the X-User-Id header. Per-user data (children,
scores, baseline photos, settings) lives in Postgres on Railway (SQLite
fallback locally) so it survives browser clears and device changes.
"""

import functools
import hmac
import os
import secrets
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, make_response, g
import anthropic

from werkzeug.security import (
    generate_password_hash as _hash_pw,
    check_password_hash as _check_pw,
)
from itsdangerous import URLSafeTimedSerializer as _URLSerializer, BadData as _BadData

from db import (
    SessionLocal,
    User,
    AppState,
    Image,
    PushSub,
    init_db,
    serialize_account,
)

# ── Load .env ─────────────────────────────────────────────────────────────────
_env = Path(__file__).parent / '.env'
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB (images are big)

HERE = Path(__file__).parent

# ── Rate limiting ────────────────────────────────────────────────────────────
# Throttle auth endpoints to blunt password brute-force and reset-email
# bombing. Keyed on the real client IP (Railway puts it in X-Forwarded-For;
# request.remote_addr would otherwise be the shared proxy IP).
from flask_limiter import Limiter


def _client_ip():
    xff = request.headers.get('X-Forwarded-For', '')
    if xff:
        return xff.split(',')[0].strip()
    return request.remote_addr or '127.0.0.1'


limiter = Limiter(key_func=_client_ip, app=app, default_limits=[])

# Create tables on boot (no-op if they already exist).
init_db()


@app.teardown_appcontext
def _cleanup_session(exc=None):
    SessionLocal.remove()


def no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# ── Auth config ───────────────────────────────────────────────────────────────
_AUTH_SECRET = (os.environ.get('SECRET_KEY') or '').strip()
if not _AUTH_SECRET:
    _AUTH_SECRET = secrets.token_urlsafe(48)
    app.logger.error(
        'SECRET_KEY is not set. Auth tokens are signed with a random '
        'per-process key, so logins break across workers and restarts. '
        'Set SECRET_KEY in the environment.'
    )

_USER_SESSION_MAX_AGE = 60 * 60 * 24 * 60  # 60 days
_user_token_signer = _URLSerializer(_AUTH_SECRET, salt='rsc-user-session')

# Household (co-parent) invite tokens — short-lived, embed the inviting
# household's owner id.
_INVITE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days
_invite_signer = _URLSerializer(_AUTH_SECRET, salt='rsc-household-invite')

# Password-reset tokens: short-lived, separate salt. They embed the user's
# current token_version, and a successful reset bumps it — so a link is
# single-use and all other sessions are logged out on reset.
_RESET_MAX_AGE = 60 * 30  # 30 minutes
_reset_token_signer = _URLSerializer(_AUTH_SECRET, salt='rsc-pw-reset')

# Transactional email (Resend). Reset emails only send when RESEND_API_KEY
# is configured; until then /api/auth/forgot still returns a generic success
# (so it never reveals whether an email is registered).
_RESEND_API_KEY = (os.environ.get('RESEND_API_KEY') or '').strip()
# Use a real, friendly mailbox rather than "no-reply" — no-reply From
# addresses measurably hurt deliverability (Gmail/Resend flag them) and block
# legitimate replies. Per-message Reply-To (e.g. the inviting parent) is set by
# callers where it makes sense.
_MAIL_FROM = (os.environ.get('MAIL_FROM') or 'Room Score Tracker <hello@room-score-tracker.com>').strip()
_APP_BASE_URL = (os.environ.get('APP_BASE_URL') or 'https://room-score-tracker.com').strip().rstrip('/')


# ── Web Push (VAPID) ─────────────────────────────────────────────────────────
_VAPID_PUBLIC_KEY = (os.environ.get('VAPID_PUBLIC_KEY') or '').strip()
_VAPID_PRIVATE_KEY = (os.environ.get('VAPID_PRIVATE_KEY') or '').strip().replace('\\n', '\n')
_VAPID_SUBJECT = (os.environ.get('VAPID_SUBJECT') or 'mailto:warrenrayschwartz@gmail.com').strip()
_CRON_KEY = (os.environ.get('CRON_KEY') or '').strip()

_vapid = None  # loaded py_vapid instance, when configured
if _VAPID_PUBLIC_KEY and _VAPID_PRIVATE_KEY:
    try:
        from py_vapid import Vapid01
        try:
            _vapid = Vapid01.from_pem(_VAPID_PRIVATE_KEY.encode())
        except Exception:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            _key = load_pem_private_key(_VAPID_PRIVATE_KEY.encode(), password=None)
            _vapid = Vapid01(); _vapid.private_key = _key; _vapid.public_key = _key.public_key()
    except Exception as e:
        app.logger.exception('Could not load VAPID keys: %s', e)
        _vapid = None


def _push_enabled():
    return bool(_vapid and _VAPID_PUBLIC_KEY)


def _send_push(sess, sub, payload):
    """Send one push. Prunes the subscription if the endpoint is gone
    (404/410). Returns True on success."""
    import json as _json
    try:
        from pywebpush import webpush, WebPushException
    except Exception:
        return False
    try:
        webpush(
            subscription_info=_json.loads(sub.data),
            data=_json.dumps(payload),
            vapid_private_key=_vapid,
            vapid_claims={'sub': _VAPID_SUBJECT},
            timeout=10,
        )
        return True
    except WebPushException as e:
        code = getattr(getattr(e, 'response', None), 'status_code', None)
        if code in (404, 410):
            try:
                sess.delete(sub); sess.commit()
            except Exception:
                sess.rollback()
        else:
            app.logger.warning('push send failed (%s): %s', code, e)
        return False
    except Exception as e:
        app.logger.warning('push send error: %s', e)
        return False


def _send_email(to, subject, html, text=None, reply_to=None):
    """Send one email via Resend. Returns True on success. Never raises.

    Including a plain-text alternative (`text`) and a real `reply_to` markedly
    improves inbox placement — HTML-only mail from a young sending domain is a
    prime spam-filter target."""
    if not _RESEND_API_KEY:
        app.logger.error('RESEND_API_KEY not set — cannot send email to %s', to)
        return False
    try:
        import requests as _rq
        payload = {'from': _MAIL_FROM, 'to': [to], 'subject': subject, 'html': html}
        if text:
            payload['text'] = text
        if reply_to:
            payload['reply_to'] = reply_to
        r = _rq.post(
            'https://api.resend.com/emails',
            headers={
                'Authorization': f'Bearer {_RESEND_API_KEY}',
                'content-type': 'application/json',
            },
            json=payload,
            timeout=10,
        )
        if r.status_code not in (200, 201):
            app.logger.error('Resend send failed (%s): %s', r.status_code, r.text[:200])
            return False
        app.logger.info('Resend accepted email to %s (subject=%r)', to, subject)
        return True
    except Exception as e:
        app.logger.exception('send email failed: %s', e)
        return False

# OAuth providers activate only when configured. Client IDs are public.
_GOOGLE_CLIENT_ID = (os.environ.get('GOOGLE_CLIENT_ID') or '').strip()
_APPLE_CLIENT_ID = (os.environ.get('APPLE_CLIENT_ID') or '').strip()  # Services ID


def _google_auth_enabled():
    return bool(_GOOGLE_CLIENT_ID)


def _apple_auth_enabled():
    return bool(_APPLE_CLIENT_ID)


def _norm_email(raw):
    s = (raw or '').strip().lower()
    if s.count('@') != 1:
        return ''
    local, _, domain = s.partition('@')
    if not local or not domain or '.' not in domain:
        return ''
    return s[:254]


def _mint_user_token(user):
    return _user_token_signer.dumps({'u': str(user.id), 'v': int(user.token_version or 0)})


def _resolve_user_token(token, sess):
    if not token:
        return None
    try:
        data = _user_token_signer.loads(token, max_age=_USER_SESSION_MAX_AGE)
    except _BadData:
        return None
    try:
        uid = uuid.UUID(str(data.get('u')))
    except (ValueError, AttributeError, TypeError):
        return None
    user = sess.get(User, uid)
    if user is None or bool(getattr(user, 'disabled', False)):
        return None
    if int(getattr(user, 'token_version', 0) or 0) != int(data.get('v', -1)):
        return None
    return user


def _mint_reset_token(user):
    return _reset_token_signer.dumps({'u': str(user.id), 'v': int(user.token_version or 0)})


def _resolve_reset_token(token, sess):
    if not token:
        return None
    try:
        data = _reset_token_signer.loads(token, max_age=_RESET_MAX_AGE)
    except _BadData:
        return None
    try:
        uid = uuid.UUID(str(data.get('u')))
    except (ValueError, AttributeError, TypeError):
        return None
    user = sess.get(User, uid)
    if user is None or bool(getattr(user, 'disabled', False)):
        return None
    if int(getattr(user, 'token_version', 0) or 0) != int(data.get('v', -1)):
        return None
    return user


def _require_user(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        raw = (request.headers.get('X-User-Id') or '').strip()
        if not raw:
            return jsonify({'error': 'Sign in to continue.'}), 401
        sess = SessionLocal()
        user = _resolve_user_token(raw, sess)
        if user is None:
            return jsonify({'error': 'Sign in to continue.'}), 401
        g.current_user = user
        g.db = sess
        # The household whose data this request reads/writes: the co-parent's
        # owner if joined, otherwise the user themselves.
        g.owner_id = getattr(user, 'household_owner_id', None) or user.id
        return fn(*args, **kwargs)
    return wrapped


def _verify_google_id_token(credential):
    if not (_GOOGLE_CLIENT_ID and credential):
        return None
    try:
        import requests as _rq
        r = _rq.get(
            'https://oauth2.googleapis.com/tokeninfo',
            params={'id_token': credential}, timeout=8,
        )
        if r.status_code != 200:
            return None
        info = r.json()
    except Exception:
        return None
    if info.get('aud') != _GOOGLE_CLIENT_ID:
        return None
    if info.get('iss', '') not in ('accounts.google.com', 'https://accounts.google.com'):
        return None
    sub = info.get('sub')
    if not sub:
        return None
    email = (info.get('email') or '') if info.get('email_verified') in (True, 'true') else ''
    return {'sub': str(sub), 'email': _norm_email(email)}


def _verify_apple_id_token(identity_token):
    if not (_APPLE_CLIENT_ID and identity_token):
        return None
    try:
        import jwt
        from jwt import PyJWKClient
        signing_key = PyJWKClient('https://appleid.apple.com/auth/keys') \
            .get_signing_key_from_jwt(identity_token)
        claims = jwt.decode(
            identity_token,
            signing_key.key,
            algorithms=['RS256'],
            audience=_APPLE_CLIENT_ID,
            issuer='https://appleid.apple.com',
        )
    except Exception:
        return None
    sub = claims.get('sub')
    if not sub:
        return None
    email = _norm_email(claims.get('email')) if claims.get('email') else ''
    return {'sub': str(sub), 'email': email}


def _oauth_login_or_link(sess, provider, sub, email, claim):
    """(1) existing account with this provider sub -> log in; (2) matching
    verified-email account that isn't a password account -> link; (3) brand
    new account. (No anonymous claim row in this app; claim kept for parity
    but unused.)"""
    field = 'google_sub' if provider == 'google' else 'apple_sub'
    try:
        user = sess.query(User).filter(getattr(User, field) == sub).first()
        if user is None and email:
            cand = sess.query(User).filter(User.email_norm == email).first()
            if cand is not None and not cand.password_hash:
                user = cand
        created = user is None
        if user is None:
            user = User()
            sess.add(user)
        if bool(getattr(user, 'disabled', False)):
            return jsonify({'error': 'This account has been disabled.'}), 403
        setattr(user, field, sub)
        if email and not user.email_norm:
            clash = (
                sess.query(User)
                .filter(User.email_norm == email, User.id != user.id)
                .first()
            )
            if clash is None:
                user.email = email
                user.email_norm = email
        sess.commit()
        # 'created' lets the client decide whether to run the one-time
        # on-device data migration (only for brand-new accounts).
        return jsonify({'token': _mint_user_token(user),
                        'account': serialize_account(user), 'created': created})
    except Exception as e:
        sess.rollback()
        app.logger.exception('oauth login failed (%s): %s', provider, e)
        return jsonify({'error': 'Sign-in failed. Try again.'}), 500


# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return no_cache(make_response(send_from_directory(str(HERE), 'index.html')))


@app.route('/privacy')
def privacy():
    return send_from_directory(str(HERE), 'privacy.html')


@app.route('/sw.js')
def service_worker():
    resp = make_response(send_from_directory(str(HERE), 'sw.js', mimetype='application/javascript'))
    resp.headers['Service-Worker-Allowed'] = '/'
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


# ── Home-screen / PWA assets ────────────────────────────────────────────────────
@app.route('/manifest.webmanifest')
def manifest():
    return send_from_directory(str(HERE), 'manifest.webmanifest',
                               mimetype='application/manifest+json')


@app.route('/icon-180.png')
def icon_180():
    return send_from_directory(str(HERE), 'icon-180.png', mimetype='image/png')


@app.route('/icon-192.png')
def icon_192():
    return send_from_directory(str(HERE), 'icon-192.png', mimetype='image/png')


@app.route('/icon-512.png')
def icon_512():
    return send_from_directory(str(HERE), 'icon-512.png', mimetype='image/png')


@app.route('/apple-touch-icon.png')
@app.route('/apple-touch-icon-precomposed.png')
def apple_touch_icon():
    # iOS probes these default paths even with an explicit <link>.
    return send_from_directory(str(HERE), 'icon-180.png', mimetype='image/png')


# ── Auth routes ────────────────────────────────────────────────────────────────
@app.route('/api/auth/config')
def auth_config():
    """Lets the frontend know which sign-in buttons to show."""
    return jsonify({
        'google': _google_auth_enabled(),
        'googleClientId': _GOOGLE_CLIENT_ID,
        'apple': _apple_auth_enabled(),
        'appleClientId': _APPLE_CLIENT_ID,
    })


@app.route('/api/auth/signup', methods=['POST'])
@limiter.limit('10 per hour')
def auth_signup():
    sess = SessionLocal()
    data = request.get_json(silent=True) or {}
    email = _norm_email(data.get('email'))
    password = data.get('password') or ''
    if not email:
        return jsonify({'error': 'Enter a valid email address.'}), 400
    if not isinstance(password, str) or len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters.'}), 400
    if len(password) > 200:
        return jsonify({'error': 'Password is too long.'}), 400
    try:
        if sess.query(User).filter(User.email_norm == email).first() is not None:
            return jsonify({'error': 'That email is already registered. Try signing in.'}), 409
        user = User()
        sess.add(user)
        display = (data.get('displayName') or '').strip()[:80] or None
        user.email = (data.get('email') or '').strip()[:254]
        user.email_norm = email
        user.password_hash = _hash_pw(password)
        if display:
            user.display_name = display
        sess.commit()
        return jsonify({'token': _mint_user_token(user), 'account': serialize_account(user)}), 201
    except Exception as e:
        sess.rollback()
        from sqlalchemy.exc import IntegrityError as _IE
        if isinstance(e, _IE):
            return jsonify({'error': 'That email is already registered. Try signing in.'}), 409
        app.logger.exception('auth_signup failed: %s', e)
        return jsonify({'error': 'Could not create your account. Try again.'}), 500


@app.route('/api/auth/login', methods=['POST'])
@limiter.limit('20 per hour;10 per minute')
def auth_login():
    sess = SessionLocal()
    data = request.get_json(silent=True) or {}
    email = _norm_email(data.get('email'))
    password = data.get('password') or ''
    generic = jsonify({'error': 'Incorrect email or password.'})
    if not email or not isinstance(password, str) or not password:
        return generic, 401
    user = sess.query(User).filter(User.email_norm == email).first()
    if user is None or not user.password_hash:
        try:
            _check_pw('pbkdf2:sha256:600000$abcdefgh$' + ('0' * 64), password)
        except Exception:
            pass
        return generic, 401
    if not _check_pw(user.password_hash, password):
        return generic, 401
    if bool(getattr(user, 'disabled', False)):
        return jsonify({'error': 'This account has been disabled.'}), 403
    return jsonify({'token': _mint_user_token(user), 'account': serialize_account(user)})


@app.route('/api/auth/forgot', methods=['POST'])
@limiter.limit('5 per hour;3 per minute')
def auth_forgot():
    """Send a password-reset link. Always returns a generic success so it
    never reveals whether an email is registered."""
    sess = SessionLocal()
    data = request.get_json(silent=True) or {}
    email = _norm_email(data.get('email'))
    if email:
        try:
            user = sess.query(User).filter(User.email_norm == email).first()
            # Only password accounts can reset (OAuth-only accounts have no
            # password to reset).
            if user is not None and user.password_hash and not bool(getattr(user, 'disabled', False)):
                token = _mint_reset_token(user)
                link = f'{_APP_BASE_URL}/?reset={token}'
                html = (
                    '<div style="font-family:-apple-system,Helvetica,Arial,sans-serif;'
                    'max-width:480px;margin:0 auto;color:#111">'
                    '<h2 style="margin:0 0 12px">Reset your password</h2>'
                    '<p style="font-size:15px;line-height:1.5;color:#333">'
                    'We got a request to reset your Room Score Tracker password. '
                    'Click the button below to choose a new one. This link expires in 30 minutes.</p>'
                    f'<p style="margin:22px 0"><a href="{link}" '
                    'style="background:#007AFF;color:#fff;text-decoration:none;'
                    'padding:12px 22px;border-radius:10px;font-weight:600;font-size:15px;'
                    'display:inline-block">Reset password</a></p>'
                    '<p style="font-size:13px;color:#888;line-height:1.5">'
                    'If you didn\'t request this, you can safely ignore this email — '
                    'your password won\'t change. '
                    f'Or paste this link into your browser:<br><span style="color:#007AFF;word-break:break-all">{link}</span></p>'
                    '</div>'
                )
                _send_email(user.email or email, 'Reset your Room Score Tracker password', html)
        except Exception as e:
            app.logger.exception('auth_forgot failed: %s', e)
    return jsonify({'ok': True})


@app.route('/api/auth/reset', methods=['POST'])
@limiter.limit('20 per hour')
def auth_reset():
    """Complete a password reset. Verifies the token, sets the new password,
    bumps token_version (invalidating the link + all other sessions), and
    returns a fresh session token so the user is logged straight in."""
    sess = SessionLocal()
    data = request.get_json(silent=True) or {}
    token = data.get('token') or ''
    password = data.get('password') or ''
    if not isinstance(password, str) or len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters.'}), 400
    if len(password) > 200:
        return jsonify({'error': 'Password is too long.'}), 400
    user = _resolve_reset_token(token, sess)
    if user is None:
        return jsonify({'error': 'This reset link is invalid or has expired. Request a new one.'}), 400
    try:
        user.password_hash = _hash_pw(password)
        user.token_version = int(getattr(user, 'token_version', 0) or 0) + 1
        sess.commit()
    except Exception as e:
        sess.rollback()
        app.logger.exception('auth_reset failed: %s', e)
        return jsonify({'error': 'Could not reset your password. Try again.'}), 500
    return jsonify({'token': _mint_user_token(user), 'account': serialize_account(user)})


@app.route('/api/auth/me')
@_require_user
def auth_me():
    return jsonify({'account': serialize_account(g.current_user)})


@app.route('/api/auth/google', methods=['POST'])
@limiter.limit('30 per hour')
def auth_google():
    if not _google_auth_enabled():
        return jsonify({'error': 'Google sign-in is not available.'}), 503
    sess = SessionLocal()
    data = request.get_json(silent=True) or {}
    info = _verify_google_id_token(data.get('credential'))
    if not info:
        return jsonify({'error': 'Google sign-in failed. Try again.'}), 401
    return _oauth_login_or_link(sess, 'google', info['sub'], info['email'], data.get('claimUserId'))


@app.route('/api/auth/apple', methods=['POST'])
@limiter.limit('30 per hour')
def auth_apple():
    if not _apple_auth_enabled():
        return jsonify({'error': 'Apple sign-in is not available.'}), 503
    sess = SessionLocal()
    data = request.get_json(silent=True) or {}
    info = _verify_apple_id_token(data.get('identityToken') or data.get('credential'))
    if not info:
        return jsonify({'error': 'Apple sign-in failed. Try again.'}), 401
    return _oauth_login_or_link(sess, 'apple', info['sub'], info['email'], data.get('claimUserId'))


@app.route('/api/auth/change-password', methods=['POST'])
@limiter.limit('10 per hour')
@_require_user
def change_password():
    """Change the password for a signed-in user. If they already have a
    password, the current one must be supplied. (An OAuth-only account can
    set a password here without a current one.) Bumps token_version to log
    out other sessions, and returns a fresh token for this one."""
    sess = g.db
    user = g.current_user
    data = request.get_json(silent=True) or {}
    new_pw = data.get('newPassword') or ''
    if not isinstance(new_pw, str) or len(new_pw) < 8:
        return jsonify({'error': 'New password must be at least 8 characters.'}), 400
    if len(new_pw) > 200:
        return jsonify({'error': 'Password is too long.'}), 400
    if user.password_hash:
        if not _check_pw(user.password_hash, data.get('currentPassword') or ''):
            return jsonify({'error': 'Current password is incorrect.'}), 401
    try:
        user.password_hash = _hash_pw(new_pw)
        user.token_version = int(getattr(user, 'token_version', 0) or 0) + 1
        sess.commit()
    except Exception as e:
        sess.rollback()
        app.logger.exception('change_password failed: %s', e)
        return jsonify({'error': 'Could not change your password. Try again.'}), 500
    return jsonify({'token': _mint_user_token(user), 'account': serialize_account(user)})


# ── Data routes (per-user, auth required) ───────────────────────────────────────
def _empty_state():
    return {'children': [], 'scores': {}, 'difficulty': 3, 'maxAllowance': 50, 'rooms': []}


@app.route('/api/data')
@_require_user
def get_data():
    """State + baselines for the logged-in user. Daily photos are NOT
    included here — they can be many large blobs and are only needed when
    viewing a specific graded day, so the client fetches them lazily via
    GET /api/photo."""
    sess = g.db
    owner = g.owner_id
    st = sess.get(AppState, owner)
    state = (st.data if st and isinstance(st.data, dict) else None) or _empty_state()
    baselines = {}
    for img in (sess.query(Image)
                .filter(Image.user_id == owner, Image.kind == 'baseline').all()):
        baselines[img.key] = img.data
    return jsonify({'state': state, 'baselines': baselines})


@app.route('/api/photo')
@_require_user
def get_photo():
    """Fetch one day's saved photos on demand. Query: key=`<child>|<day>`.
    Returns {data: {slotId: thumbnail}} (empty if none saved)."""
    import json as _json
    sess = g.db
    owner = g.owner_id
    key = request.args.get('key') or ''
    if not key:
        return jsonify({'error': 'Missing key'}), 400
    img = (sess.query(Image)
           .filter(Image.user_id == owner, Image.kind == 'photo', Image.key == key)
           .first())
    if img is None:
        return jsonify({'data': {}})
    try:
        return jsonify({'data': _json.loads(img.data)})
    except Exception:
        return jsonify({'data': {}})


@app.route('/api/state', methods=['PUT'])
@_require_user
def put_state():
    sess = g.db
    owner = g.owner_id
    data = request.get_json(silent=True) or {}
    clean = {
        'children': data.get('children') or [],
        'scores': data.get('scores') or {},
        'difficulty': data.get('difficulty', 3),
        'maxAllowance': data.get('maxAllowance', 50),
        'rooms': data.get('rooms') or [],
    }
    st = sess.get(AppState, owner)
    if st is None:
        st = AppState(user_id=owner, data=clean)
        sess.add(st)
    else:
        st.data = clean
    sess.commit()
    return jsonify({'ok': True})


def _upsert_image(sess, user_id, kind, key, data):
    img = (
        sess.query(Image)
        .filter(Image.user_id == user_id, Image.kind == kind, Image.key == key)
        .first()
    )
    if img is None:
        img = Image(user_id=user_id, kind=kind, key=key, data=data)
        sess.add(img)
    else:
        img.data = data


@app.route('/api/image', methods=['PUT'])
@_require_user
def put_image():
    """Save one image. body: {kind:'baseline'|'photo', key, data}.
    For photos, data is a JSON string of {slotId: thumbnail}."""
    sess = g.db
    owner = g.owner_id
    data = request.get_json(silent=True) or {}
    kind = data.get('kind')
    key = data.get('key')
    payload = data.get('data')
    if kind not in ('baseline', 'photo') or not key or not isinstance(payload, str):
        return jsonify({'error': 'Bad image payload'}), 400
    _upsert_image(sess, owner, kind, key, payload)
    sess.commit()
    return jsonify({'ok': True})


@app.route('/api/image', methods=['DELETE'])
@_require_user
def delete_image():
    sess = g.db
    owner = g.owner_id
    data = request.get_json(silent=True) or {}
    kind = data.get('kind')
    key = data.get('key')
    if kind not in ('baseline', 'photo') or not key:
        return jsonify({'error': 'Bad request'}), 400
    sess.query(Image).filter(
        Image.user_id == owner, Image.kind == kind, Image.key == key
    ).delete()
    sess.commit()
    return jsonify({'ok': True})


@app.route('/api/baselines', methods=['PUT'])
@_require_user
def put_baselines():
    """Bulk replace all baseline images (used by Restore Baselines).
    body: {baselines: {key: dataURL}}."""
    sess = g.db
    owner = g.owner_id
    data = request.get_json(silent=True) or {}
    baselines = data.get('baselines') or {}
    if not isinstance(baselines, dict):
        return jsonify({'error': 'Bad request'}), 400
    sess.query(Image).filter(
        Image.user_id == owner, Image.kind == 'baseline'
    ).delete()
    for key, val in baselines.items():
        if isinstance(val, str) and val:
            sess.add(Image(user_id=owner, kind='baseline', key=key, data=val))
    sess.commit()
    return jsonify({'ok': True})


@app.route('/api/migrate', methods=['POST'])
@_require_user
def migrate():
    """One-time carry-over of a device's existing local data into a new
    account. Only runs if the account currently has NO state and NO images,
    so it can never clobber server data. body: {state, baselines:{key:data},
    photos:{key:{slot:thumb}}}."""
    import json as _json
    sess = g.db
    owner = g.owner_id
    has_state = sess.get(AppState, owner) is not None
    has_images = sess.query(Image).filter(Image.user_id == owner).first() is not None
    if has_state or has_images:
        return jsonify({'ok': True, 'migrated': False})
    data = request.get_json(silent=True) or {}
    state = data.get('state')
    if isinstance(state, dict):
        sess.add(AppState(user_id=owner, data={
            'children': state.get('children') or [],
            'scores': state.get('scores') or {},
            'difficulty': state.get('difficulty', 3),
            'maxAllowance': state.get('maxAllowance', 50),
            'rooms': state.get('rooms') or [],
        }))
    for key, val in (data.get('baselines') or {}).items():
        if isinstance(val, str) and val:
            sess.add(Image(user_id=owner, kind='baseline', key=key, data=val))
    for key, val in (data.get('photos') or {}).items():
        if isinstance(val, dict) and val:
            sess.add(Image(user_id=owner, kind='photo', key=key, data=_json.dumps(val)))
    sess.commit()
    return jsonify({'ok': True, 'migrated': True})


@app.route('/api/account', methods=['DELETE'])
@_require_user
def delete_account():
    """Permanently delete the signed-in user and all their data (App Store
    Guideline 5.1.1(v) requires in-app account deletion). Cascades to
    AppState + Image rows via the ORM relationship / FK ON DELETE CASCADE."""
    sess = g.db
    user = g.current_user
    try:
        # Any co-parents who joined this user's household lose access cleanly.
        sess.query(User).filter(User.household_owner_id == user.id).update(
            {User.household_owner_id: None}, synchronize_session=False)
        sess.delete(user)
        sess.commit()
    except Exception as e:
        sess.rollback()
        app.logger.exception('account deletion failed: %s', e)
        return jsonify({'error': 'Could not delete your account. Try again.'}), 500
    return jsonify({'ok': True})


# ── Household / co-parent sharing ────────────────────────────────────────────────
def _household_status(sess, user):
    """Describe the user's household for the UI."""
    if user.household_owner_id:
        owner = sess.get(User, user.household_owner_id)
        return {
            'role': 'member',
            'ownerEmail': (owner.email if owner else None),
            'members': [],
        }
    members = (sess.query(User)
               .filter(User.household_owner_id == user.id).all())
    return {
        'role': 'owner' if members else 'solo',
        'ownerEmail': user.email,
        'members': [m.email for m in members if m.email],
    }


@app.route('/api/household')
@_require_user
def household_status():
    return jsonify(_household_status(g.db, g.current_user))


@app.route('/api/household/invite', methods=['POST'])
@limiter.limit('20 per hour')
@_require_user
def household_invite():
    """Generate a co-parent invite for THIS user's household (the effective
    owner). Returns a shareable link + raw code."""
    owner_id = g.owner_id
    code = _invite_signer.dumps(str(owner_id))
    return jsonify({'code': code, 'link': f'{_APP_BASE_URL}/?join={code}'})


@app.route('/api/household/invite-email', methods=['POST'])
@limiter.limit('15 per hour')
@_require_user
def household_invite_email():
    """Email a co-parent invite directly to the given address."""
    if not _RESEND_API_KEY:
        return jsonify({'error': 'Email sending isn’t set up on this server.'}), 503
    data = request.get_json(silent=True) or {}
    to = _norm_email(data.get('email'))
    if not to:
        return jsonify({'error': 'Enter a valid email address.'}), 400
    code = _invite_signer.dumps(str(g.owner_id))
    link = f'{_APP_BASE_URL}/?join={code}'
    inviter = (getattr(g.current_user, 'email', None) or 'A parent')
    html = (
        '<div style="font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:480px;margin:0 auto;color:#111">'
        '<h2 style="margin:0 0 12px">You\'re invited to co-parent</h2>'
        f'<p style="font-size:15px;line-height:1.5;color:#333">{inviter} invited you to share children on '
        '<strong>Room Score Tracker</strong> — you\'ll both manage the same kids\' room scores from your own accounts.</p>'
        f'<p style="margin:22px 0"><a href="{link}" style="background:#1ed760;color:#06210f;text-decoration:none;'
        'padding:12px 22px;border-radius:10px;font-weight:700;font-size:15px;display:inline-block">Join their household</a></p>'
        '<p style="font-size:13px;color:#888;line-height:1.5">Sign in (or create a free account) and you\'ll be connected. '
        f'This link expires in 7 days. Or paste this into your browser:<br><span style="color:#1aa34a;word-break:break-all">{link}</span></p>'
        '</div>'
    )
    text = (
        f'{inviter} invited you to co-parent on Room Score Tracker — you\'ll both '
        'manage the same kids\' room scores from your own accounts.\n\n'
        f'Join here (sign in or create a free account): {link}\n\n'
        'This link expires in 7 days.'
    )
    reply_to = getattr(g.current_user, 'email', None) or None
    ok = _send_email(
        to, 'You\'re invited to co-parent on Room Score Tracker',
        html, text=text, reply_to=reply_to,
    )
    if not ok:
        return jsonify({'error': 'Could not send the email. Try again.'}), 502
    return jsonify({'ok': True})


@app.route('/api/household/join', methods=['POST'])
@limiter.limit('20 per hour')
@_require_user
def household_join():
    """Redeem an invite code to join another parent's household."""
    sess = g.db
    user = g.current_user
    data = request.get_json(silent=True) or {}
    code = (data.get('code') or '').strip()
    try:
        owner_id_str = _invite_signer.loads(code, max_age=_INVITE_MAX_AGE)
        owner_uuid = uuid.UUID(str(owner_id_str))
    except (_BadData, ValueError, AttributeError, TypeError):
        return jsonify({'error': 'This invite is invalid or has expired. Ask for a new one.'}), 400
    if owner_uuid == user.id:
        return jsonify({'error': "That's your own invite link."}), 400
    if user.household_owner_id:
        return jsonify({'error': "You're already sharing a household. Leave it first."}), 400
    # Don't let an owner-with-co-parents become a member (would orphan theirs).
    if sess.query(User).filter(User.household_owner_id == user.id).first() is not None:
        return jsonify({'error': "You're hosting a co-parent. Have them leave before you join another household."}), 400
    owner = sess.get(User, owner_uuid)
    if owner is None or bool(getattr(owner, 'disabled', False)):
        return jsonify({'error': 'That household no longer exists.'}), 400
    # Only join a root owner (no chains).
    if owner.household_owner_id:
        return jsonify({'error': 'This invite is invalid. Ask the main account holder for one.'}), 400
    user.household_owner_id = owner_uuid
    sess.commit()
    return jsonify({'ok': True, 'household': _household_status(sess, user)})


@app.route('/api/household/leave', methods=['POST'])
@_require_user
def household_leave():
    """Leave a household you joined (your own data, if any, becomes visible
    again). If you're the owner, removes a specific member by email."""
    sess = g.db
    user = g.current_user
    data = request.get_json(silent=True) or {}
    remove_email = _norm_email(data.get('memberEmail')) if data.get('memberEmail') else ''
    if remove_email:
        # Owner removing one of their co-parents.
        m = (sess.query(User)
             .filter(User.household_owner_id == user.id, User.email_norm == remove_email)
             .first())
        if m is not None:
            m.household_owner_id = None
            sess.commit()
        return jsonify({'ok': True, 'household': _household_status(sess, user)})
    # Member leaving their household.
    user.household_owner_id = None
    sess.commit()
    return jsonify({'ok': True, 'household': _household_status(sess, user)})


# ── Web Push reminders ───────────────────────────────────────────────────────
@app.route('/api/push/config')
def push_config():
    """Lets the client know whether push is available and the public key."""
    return jsonify({'enabled': _push_enabled(), 'publicKey': _VAPID_PUBLIC_KEY})


@app.route('/api/push/subscribe', methods=['POST'])
@_require_user
def push_subscribe():
    if not _push_enabled():
        return jsonify({'error': 'Reminders are not available yet.'}), 503
    import json as _json
    sess = g.db
    user = g.current_user
    sub = request.get_json(silent=True) or {}
    endpoint = sub.get('endpoint')
    if not endpoint:
        return jsonify({'error': 'Bad subscription'}), 400
    existing = sess.query(PushSub).filter(PushSub.endpoint == endpoint).first()
    if existing is not None:
        existing.user_id = user.id
        existing.data = _json.dumps(sub)
    else:
        sess.add(PushSub(user_id=user.id, endpoint=endpoint, data=_json.dumps(sub)))
    sess.commit()
    return jsonify({'ok': True})


@app.route('/api/push/unsubscribe', methods=['POST'])
@_require_user
def push_unsubscribe():
    sess = g.db
    data = request.get_json(silent=True) or {}
    endpoint = data.get('endpoint')
    if endpoint:
        sess.query(PushSub).filter(PushSub.endpoint == endpoint).delete()
    else:
        sess.query(PushSub).filter(PushSub.user_id == g.current_user.id).delete()
    sess.commit()
    return jsonify({'ok': True})


@app.route('/api/push/test', methods=['POST'])
@_require_user
def push_test():
    if not _push_enabled():
        return jsonify({'error': 'Reminders are not available yet.'}), 503
    sess = g.db
    subs = sess.query(PushSub).filter(PushSub.user_id == g.current_user.id).all()
    sent = sum(1 for s in subs if _send_push(sess, s, {
        'title': 'Room Score Tracker',
        'body': "Test reminder — you're all set! 🎉",
        'url': '/',
    }))
    return jsonify({'ok': True, 'sent': sent})


_DAY_KEYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def _clean_schedule(data):
    """Validate a {tz, days{Mon..Sun: 'HH:MM'|null}} schedule."""
    import re as _re
    tz = (str(data.get('tz') or 'UTC').strip() or 'UTC')[:64]
    days_in = data.get('days') or {}
    days = {}
    for d in _DAY_KEYS:
        v = days_in.get(d)
        ok = isinstance(v, str) and _re.match(r'^\d{2}:\d{2}$', v)
        if ok and 0 <= int(v[:2]) < 24 and 0 <= int(v[3:]) < 60:
            days[d] = f'{int(v[:2]):02d}:{int(v[3:]):02d}'
        else:
            days[d] = None
    return {'tz': tz, 'days': days}


@app.route('/api/push/schedule')
@_require_user
def push_get_schedule():
    import json as _json
    u = g.current_user
    sched = None
    if u.reminder:
        try:
            sched = _json.loads(u.reminder)
        except Exception:
            sched = None
    return jsonify({'schedule': sched})


@app.route('/api/push/schedule', methods=['PUT'])
@_require_user
def push_put_schedule():
    import json as _json
    sess = g.db
    u = g.current_user
    sched = _clean_schedule(request.get_json(silent=True) or {})
    u.reminder = _json.dumps(sched)
    sess.commit()
    return jsonify({'ok': True, 'schedule': sched})


@app.route('/api/cron/reminders', methods=['POST', 'GET'])
def cron_reminders():
    """Send the 'grade today's rooms' reminder to each user whose per-day
    schedule is due right now (in their own timezone). Run this frequently
    (every ~15 min). Protected by the CRON_KEY header."""
    if not _CRON_KEY or request.headers.get('X-Cron-Key', '') != _CRON_KEY:
        return jsonify({'error': 'Forbidden'}), 403
    if not _push_enabled():
        return jsonify({'error': 'Push not configured'}), 503
    import json as _json
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        ZoneInfo = None
    try:
        window = max(1, min(120, int(request.args.get('window', '20'))))
    except Exception:
        window = 20

    sess = SessionLocal()
    sent = 0
    due_users = 0
    for u in sess.query(User).filter(User.reminder.isnot(None)).all():
        try:
            sched = _json.loads(u.reminder) if u.reminder else None
        except Exception:
            sched = None
        if not sched:
            continue
        tz = None
        if ZoneInfo:
            try:
                tz = ZoneInfo(sched.get('tz') or 'UTC')
            except Exception:
                tz = None
        now_local = datetime.now(tz) if tz else datetime.utcnow()
        t = (sched.get('days') or {}).get(_DAY_KEYS[now_local.weekday()])
        if not t:
            continue
        scheduled = now_local.replace(hour=int(t[:2]), minute=int(t[3:]), second=0, microsecond=0)
        delta_min = (now_local - scheduled).total_seconds() / 60.0
        if not (0 <= delta_min < window):
            continue
        slot = now_local.strftime('%Y-%m-%d') + '|' + t
        if u.reminder_last_slot == slot:
            continue
        subs = sess.query(PushSub).filter(PushSub.user_id == u.id).all()
        if not subs:
            continue
        for s in subs:
            if _send_push(sess, s, {
                'title': 'Room Score Tracker',
                'body': "Time to grade today's rooms 📷",
                'url': '/',
            }):
                sent += 1
        u.reminder_last_slot = slot
        due_users += 1
    sess.commit()
    return jsonify({'ok': True, 'dueUsers': due_users, 'sent': sent})


# ── Grading (auth required) ─────────────────────────────────────────────────────
@app.route('/grade', methods=['POST'])
@_require_user
def grade():
    try:
        data = request.get_json(silent=True) or {}
        content = data.get('content')
        if not content:
            return jsonify({'error': 'No content provided'}), 400

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({'error': 'Server is not configured with an API key'}), 500

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model='claude-opus-4-7',
            max_tokens=8000,
            thinking={'type': 'adaptive'},
            messages=[{'role': 'user', 'content': content}],
        )

        text = next((b.text for b in response.content if b.type == 'text'), '')
        return jsonify({'text': text})

    except anthropic.AuthenticationError:
        return jsonify({'error': 'Invalid API key — check ANTHROPIC_API_KEY on Railway'}), 401
    except anthropic.RateLimitError:
        return jsonify({'error': 'Rate limit reached. Please wait a moment and try again.'}), 429
    except anthropic.BadRequestError as e:
        return jsonify({'error': f'Image could not be processed: {str(e)[:200]}'}), 400
    except Exception as e:
        app.logger.exception('Unexpected error in /grade')
        return jsonify({'error': f'Something went wrong: {str(e)[:200]}'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'claude-opus-4-7'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f'\n📋 Room Score Tracker on http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
