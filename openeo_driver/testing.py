"""
Reusable helpers and fixtures for testing
"""
import base64
import json
import re
import uuid
from pathlib import Path
from typing import Union, Callable, Pattern

from flask import Response
from flask.testing import FlaskClient
from werkzeug.datastructures import Headers

from openeo.capabilities import ComparableVersion
from openeo_driver.users.auth import HttpAuthHandler

TEST_USER = "Mr.Test"
TEST_USER_BEARER_TOKEN = "basic//" + HttpAuthHandler.build_basic_access_token(user_id=TEST_USER)
TEST_USER_AUTH_HEADER = {
    "Authorization": "Bearer " + TEST_USER_BEARER_TOKEN
}

TIFF_DUMMY_DATA = b'T1f7D6t6l0l' * 1000


def read_file(path: Union[Path, str], mode='r') -> str:
    """Get contents of given file as text string."""
    with Path(path).open(mode) as f:
        return f.read()


def load_json(path: Union[Path, str], preprocess: Callable[[str], str] = None) -> dict:
    """Parse data from JSON file"""
    data = read_file(path)
    if preprocess:
        data = preprocess(data)
    return json.loads(data)


def preprocess_check_and_replace(old: str, new: str) -> Callable[[str], str]:
    """
    Create a preprocess function that replaces `old` with `new` (and fails when there is no `old`).
    """

    def preprocess(orig: str) -> str:
        assert old in orig
        return orig.replace(old, new)

    return preprocess


def preprocess_regex_check_and_replace(pattern: str, replacement: str) -> Callable[[str], str]:
    """
    Create a regex based replacement preprocess function.
    """

    def preprocess(orig: str) -> str:
        new = re.sub(pattern, replacement, orig, flags=re.DOTALL)
        assert orig != new
        return new

    return preprocess


class ApiException(Exception):
    pass


class ApiResponse:
    """
    Thin wrapper around flask `Response` to simplify
    writing unit test (API status/error code) assertions
    """

    def __init__(self, response: Response):
        self.response = response

    @property
    def status_code(self) -> int:
        return self.response.status_code

    @property
    def data(self) -> bytes:
        return self.response.data

    @property
    def json(self) -> dict:
        return self.response.json

    @property
    def text(self) -> str:
        return self.response.get_data(as_text=True)

    @property
    def headers(self) -> Headers:
        return self.response.headers

    def assert_status_code(self, status_code: int) -> 'ApiResponse':
        """Check HTTP status code"""
        if self.status_code != status_code:
            message = f"Expected response with status code {status_code} but got {self.status_code}."
            if self.status_code >= 400:
                message += f" Error: {self.json}"
            raise ApiException(message)
        return self

    def assert_content(self) -> 'ApiResponse':
        """Assert that the response body is not empty"""
        # TODO: also check content type? also check (prefix of) data?
        assert self.response.content_length > 0
        return self

    def assert_error_code(self, code: str) -> 'ApiResponse':
        """Check OpenEO error code"""
        if self.status_code < 400:
            raise ApiException("Expected response with status < 400 but got {a}".format(a=self.status_code))
        error = self.json
        actual = error.get("code")
        if actual != code:
            raise ApiException("Expected response with error code {c!r} but got {a!r}. Error: {e!r}".format(
                c=code, a=actual, e=error
            ))
        return self

    def assert_substring(self, key, expected: Union[str, Pattern]):
        actual = self.json[key]
        if isinstance(expected, str):
            if expected in actual:
                return self
        elif expected.search(actual):
            return self
        raise ApiException("Expected {e!r} at {k!r}, but got {a!r}".format(e=expected, k=key, a=actual))

    def assert_error(self, status_code: int, error_code: str,
                     message: Union[str, Pattern] = None) -> 'ApiResponse':
        resp = self.assert_status_code(status_code).assert_error_code(error_code)
        if message:
            resp.assert_substring("message", message)
        return resp


class ApiTester:
    """
    Helper container class for compact writing of api version aware API tests
    """
    data_root = None

    def __init__(
            self, api_version: str, client: FlaskClient, data_root: Path = None,
            url_root: str = "/openeo/{api_version}"
    ):
        self.api_version = api_version
        self.api_version_compare = ComparableVersion(self.api_version)
        self.client = client
        if data_root:
            self.data_root = Path(data_root)
        self.default_request_headers = {}
        self.url_root = url_root.format(api_version=api_version)

    def url(self, path):
        """Build URL based on (possibly versioned) root URL."""
        return re.sub("/+", "/", f"/{self.url_root}/{path}")

    def _request_headers(self, headers: dict = None) -> dict:
        return {**self.default_request_headers, **(headers or {})}

    def set_auth_bearer_token(self, token: str = TEST_USER_BEARER_TOKEN):
        """Authentication: set bearer token header for all requests."""
        self.default_request_headers["Authorization"] = "Bearer " + token

    def get(self, path: str, headers: dict = None) -> ApiResponse:
        """Do versioned GET request, given non-versioned path"""
        return ApiResponse(self.client.get(path=self.url(path), headers=self._request_headers(headers)))

    def post(self, path: str, json: dict = None, headers: dict = None) -> ApiResponse:
        """Do versioned POST request, given non-versioned path"""
        return ApiResponse(self.client.post(
            path=self.url(path),
            json=json or {},
            content_type='application/json',
            headers=self._request_headers(headers),
        ))

    def delete(self, path: str, headers: dict = None) -> ApiResponse:
        return ApiResponse(self.client.delete(path=self.url(path), headers=self._request_headers(headers)))

    def put(self, path: str, json: dict = None, headers: dict = None) -> ApiResponse:
        """Do versioned PUT request, given non-versioned path"""
        return ApiResponse(self.client.put(
            path=self.url(path),
            json=json or {},
            content_type='application/json',
            headers=self._request_headers(headers),
        ))

    def patch(self, path: str, json: dict = None, headers: dict = None) -> ApiResponse:
        """Do versioned PATH request, given non-versioned path"""
        return ApiResponse(self.client.patch(
            path=self.url(path),
            json=json or {},
            content_type='application/json',
            headers=self._request_headers(headers),
        ))

    def data_path(self, filename: str) -> Path:
        """Get absolute pat to a test data file"""
        return self.data_root / filename

    def read_file(self, filename, mode='r') -> str:
        """Get contents of test file, given by relative path."""
        return read_file(self.data_path(filename), mode=mode)

    def load_json(self, filename, preprocess: Callable = None) -> dict:
        """Load test process graph from json file"""
        return load_json(path=self.data_path(filename), preprocess=preprocess)

    def get_process_graph_dict(self, process_graph: dict, title: str = None, description: str = None) -> dict:
        """
        Build dict containing process graph (e.g. to POST, or to expect in metadata),
        according to API version
        """
        if ComparableVersion("1.0.0").or_higher(self.api_version):
            data = {"process": {'process_graph': process_graph}}
        else:
            data = {'process_graph': process_graph}
        if title:
            data["title"] = title
        if description:
            data["description"] = description
        return data

    def result(
            self, process_graph: Union[dict, str], path="/result",
            preprocess: Callable = None
    ) -> ApiResponse:
        """Post a process_graph (as dict or by filename) and get response."""
        if isinstance(process_graph, str):
            # Assume it is a file name
            process_graph = self.load_json(process_graph, preprocess=preprocess)
        data = self.get_process_graph_dict(process_graph)
        self.set_auth_bearer_token()
        response = self.post(path=path, json=data)
        return response

    def check_result(
            self, process_graph: Union[dict, str], path="/result",
            preprocess: Callable = None
    ) -> ApiResponse:
        """Post a process_graph (as dict or by filename), get response and do basic checks."""
        response = self.result(process_graph=process_graph, path=path, preprocess=preprocess)
        return response.assert_status_code(200).assert_content()

    def validation(self, process_graph: Union[dict, str], preprocess: Callable = None, do_auth: bool = True) -> ApiResponse:
        """Post a process_graph (as dict or by filename) and run validation."""
        if isinstance(process_graph, str):
            # Assume it is a file name
            process_graph = self.load_json(process_graph, preprocess=preprocess)
        data = {'process_graph': process_graph}
        if do_auth:
            self.set_auth_bearer_token()
        response = self.post(path="/validation", json=data)
        # "Please note that a validation always returns with HTTP status code 200."
        response.assert_status_code(200)
        return response


class IgnoreOrder:
    """
    pytest helper to test equality of lists/tuples ignoring item order

    E.g., these asserts pass:
    >>> assert [1, 2, 3, 3] == IgnoreOrder([3, 1, 2, 3])
    >>> assert {"foo": [1, 2, 3]} == {"foo": IgnoreOrder([3, 2, 1])}
    """

    def __init__(self, items: Union[list, tuple], key=None):
        self.items = items
        self.key = key

    def __eq__(self, other):
        return type(other) == type(self.items) and sorted(other, key=self.key) == sorted(self.items, key=self.key)

    def __repr__(self):
        return '{c}({v!r})'.format(c=self.__class__.__name__, v=self.items)


class RegexMatcher:
    """
    pytest helper to check a string against a regex, especially in nested structures, e.g.:

        >>> assert {"foo": "baaaaa"} == {"foo": RegexMatcher("ba+")}
    """

    def __init__(self, pattern: str, flags=0):
        self.regex = re.compile(pattern=pattern, flags=flags)

    def __eq__(self, other):
        return isinstance(other, str) and bool(self.regex.match(other))

    def __repr__(self):
        return self.regex.pattern


class DictSubSet:
    """
    pytest helper to check if a dictionary contains a subset of items, e.g.:

    >> assert {"foo": "bar", "meh": 4} == DictSubSet({"foo": "bar"})
    """

    def __init__(self, items: dict = None, **kwargs):
        self.items = {**(items or {}), **kwargs}

    def __eq__(self, other):
        return isinstance(other, type(self.items)) and self.items == {k: other[k] for k in self.items if k in other}

    def __repr__(self):
        return repr(self.items)


class ListSubSet:
    """
    pytest helper to check if a list contains a subset of items, e.g.:

    >> assert [1, 2, 3, 666] == ListSubSet([1, 666])
    """
    # TODO: also take item counts into account?
    def __init__(self, items: list):
        self.items = items

    def __eq__(self, other):
        return isinstance(other, type(self.items)) and all(
            any(y == x for y in other)
            for x in self.items
        )

    def __repr__(self):
        return repr(self.items)


def generate_unique_test_process_id():
    # Because the process registries are global variables we can not mock easily
    # we'll add new test processes with a (random) unique name.
    return "_test_process_{u}".format(u=uuid.uuid4())


def build_basic_http_auth_header(username: str, password: str) -> str:
    """Build HTTP header for Basic HTTP authentication"""
    # Note: this is not the custom basic bearer token used in openEO API.
    return "Basic " + base64.b64encode("{u}:{p}".format(u=username, p=password).encode("utf-8")).decode('ascii')
