import auth


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def test_token_parser():
    tokens = ["sample_token,sample_duration"]

    tokens = auth.parse_tokens(tokens)

    assert len(tokens) == 1
    assert tokens["sample_token"] == "sample_duration"


def test_error_when_no_access_token_header():
    config = {"request": AttrDict({"headers": {}}), "token_reader": lambda: []}
    api_guard = auth.create_api_key_guard(config)

    def unreachable_endpoint(data):
        assert "ERROR" in data["status"]

    response = api_guard(unreachable_endpoint)()


def test_access_token_header():
    # expected payload -> { "email": "test@email.com" }
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRlc3RAZW1haWwuY29tIn0.v64tKBhcV7yzH6LKOIGbZCQOOrwAl1xoyo6TWDTL3-o"
    config = {
        "request": AttrDict({"headers": {"X-Access-Token": token}}),
        "token_reader": lambda: [
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRlc3RAZW1haWwuY29tIn0.v64tKBhcV7yzH6LKOIGbZCQOOrwAl1xoyo6TWDTL3-o,infinite"
        ],
        "jwt_secret": "test_secret",
    }
    api_guard = auth.create_api_key_guard(config)

    def endpoint(data):
        assert "OK" in data["status"]
        assert "test@email.com" in data["payload"]["email"]

    response = api_guard(endpoint)()
