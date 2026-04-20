#!/usr/bin/env python3
"""Room Score Tracker — Flask server for Railway deployment."""

import os
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, make_response
import anthropic

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


def no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return no_cache(make_response(send_from_directory(str(HERE), 'index.html')))


@app.route('/grade', methods=['POST'])
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
