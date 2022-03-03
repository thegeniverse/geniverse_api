"""Authentication module"""
import jwt
from functools import wraps
from operator import itemgetter


def parse_tokens(read_token_fn):
    "Return all tokens given a token_reader"
    raw_tokens = read_token_fn()

    def line_is_valid(line):
        line_is_not_comment = not line.startswith("#")
        line_is_not_newline_char = line != "\n"
        return line_is_not_comment and line_is_not_newline_char

    tokens = {}

    for raw_token in raw_tokens:
        token, duration = raw_token.split(",")

        tokens[token] = duration

    return tokens


def create_file_token_reader(tokens_path="tokens.txt"):
    """Return a function that reads tokens given a path.

    Defaults to 'tokens.txt'.
    """

    def file_token_reader():
        with open(tokens_path, "r") as token_file:
            raw_tokens = token_file.readlines()

        return raw_tokens

    return file_token_reader


def create_api_key_guard(config):
    request = config.get("request")
    token_reader = config.get("token_reader")
    jwt_secret = config.get("jwt_secret")

    assert request is not None
    assert request.headers is not None
    assert token_reader is not None

    def api_key_required(endpoint_fn):
        """Decorator that guards an app route"""

        @wraps(endpoint_fn)
        def decorator():
            if "X-Access-Token" not in request.headers:
                return endpoint_fn(
                    {
                        "status": "ERROR",
                        "payload": {"reason": "Access token not in request."},
                    }
                )
            stored_tokens = parse_tokens(token_reader)
            token = request.headers["X-Access-Token"]
            if token in stored_tokens:
                assert jwt_secret is not None
                payload = jwt.decode(token, jwt_secret, "HS256")
                return endpoint_fn({"status": "OK", "payload": payload})

            return endpoint_fn(
                {"status": "ERROR", "payload": {"reason": "token not found in tokens"}}
            )

        return decorator

    return api_key_required
