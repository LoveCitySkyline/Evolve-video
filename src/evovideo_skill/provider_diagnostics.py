"""Bounded, redacted provider error details; never logs full request/response bodies."""
import json
import os
import re
import urllib.error

BODY_LIMIT = 16384


def redact(value):
    text = str(value)
    for name, secret in os.environ.items():
        if secret and any(word in name.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'(?i)Bearer\s+\S+', 'Bearer [REDACTED]', text)
    text = re.sub(r'(?i)data:[^\s\"\']+', '[MEDIA REDACTED]', text)
    text = re.sub(r'https?://[^\s\"\']+', '[URL REDACTED]', text)
    text = re.sub(r'\b[A-Za-z0-9+/=_-]{96,}\b', '[LONG VALUE REDACTED]', text)
    return text[:2000]


def error_details(exc):
    """Read only whitelisted provider fields; never dump HTML or request bodies."""
    seen = set()
    cause = exc
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, urllib.error.HTTPError):
            cached = getattr(cause, '_evovideo_safe_details', None)
            if cached is not None:
                return dict(cached)
            row = {'error_type': type(cause).__name__, 'http_code': cause.code}
            cause._evovideo_safe_details = row
            try:
                body = cause.read(BODY_LIMIT + 1)
                row['response_truncated'] = len(body) > BODY_LIMIT
                if row['response_truncated']:
                    row['body_status'] = 'too_large_to_parse_safely'
                    return row
                payload = json.loads(body)
            except (ValueError, OSError, TypeError):
                row['body_status'] = 'empty_unreadable_or_non_json'
                return row
            if not isinstance(payload, dict):
                row['body_status'] = 'non_object_json'
                return row
            detail = payload.get('error', payload)
            if isinstance(detail, dict):
                row['provider_error'] = {k: redact(detail[k]) for k in
                    ('code', 'message', 'type', 'param', 'status')
                    if k in detail and isinstance(detail[k], (str, int))}
            # Request identifiers help the provider locate the failed request.
            for key in ('request_id', 'requestId'):
                if isinstance(payload.get(key), str):
                    row[key] = redact(payload[key])
            row['body_status'] = 'json_fields_only'
            return row
        if isinstance(cause, urllib.error.URLError):
            return {'error_type': type(cause).__name__, 'http_code': None,
                    'reason': redact(cause.reason)}
        cause = cause.__cause__ or cause.__context__
    return {'error_type': type(exc).__name__, 'http_code': None,
            'body_status': 'no_http_or_url_cause'}
