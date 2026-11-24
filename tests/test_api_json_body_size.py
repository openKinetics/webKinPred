import json

from django.test import RequestFactory

from api.views.v1_views import _parse_json_body, _parse_validate_json_body


def _request(rows):
    body = json.dumps({"data": rows, "targets": ["kcat"], "methods": {"kcat": "DLKcat"}})
    return RequestFactory().post("/api/v1/submit/", data=body, content_type="application/json")


def test_json_body_above_the_django_default_is_accepted():
    rows = [{"Protein Sequence": "M" + "A" * 999, "Substrate": "CCO"} for _ in range(3000)]
    request = _request(rows)
    assert len(request.body) > 2_621_440
    csv_file, params, error = _parse_json_body(request)
    assert error is None and params["targets"] == ["kcat"]
    assert csv_file.getvalue().count(b"\n") == 3001


def test_json_body_over_the_limit_returns_a_clear_413(settings):
    settings.DATA_UPLOAD_MAX_MEMORY_SIZE = 1024
    rows = [{"Protein Sequence": "M" + "A" * 99, "Substrate": "CCO"} for _ in range(50)]
    for parse in (_parse_json_body, _parse_validate_json_body):
        *_, error = parse(_request(rows))
        assert error.status_code == 413
        assert "CSV file upload" in json.loads(error.content)["error"]
