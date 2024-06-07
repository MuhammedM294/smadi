import pytest

from smadi.preprocess import validate_anomaly_method, validate_date_params
from smadi.anomaly_detectors import _Detectors


def test_validate_anomaly_method_valid_methods():
    # Test with valid methods
    valid_methods = list(_Detectors.keys())
    try:
        validate_anomaly_method(valid_methods, _Detectors)
    except ValueError:
        pytest.fail("validate_anomaly_method raised ValueError unexpectedly!")


def test_validate_anomaly_method_invalid_method():
    # Test with an invalid method
    invalid_methods = ["invalid_method"]
    with pytest.raises(ValueError) as exc_info:
        validate_anomaly_method(invalid_methods, _Detectors)
    assert "Anomaly method 'invalid_method' is not supported." in str(exc_info.value)


def test_validate_anomaly_method_partial_invalid_methods():
    # Test with a mix of valid and invalid methods
    mixed_methods = ["zscore", "invalid_method"]
    with pytest.raises(ValueError) as exc_info:
        validate_anomaly_method(mixed_methods, _Detectors)
    assert "Anomaly method 'invalid_method' is not supported." in str(exc_info.value)


def test_validate_anomaly_method_empty_list():
    # Test with an empty list
    empty_methods = []
    try:
        validate_anomaly_method(empty_methods, _Detectors)
    except ValueError:
        pytest.fail("validate_anomaly_method raised ValueError unexpectedly!")


def test_validate_date_params_month():
    result = validate_date_params(time_step="month", year=2023, month=5)
    assert result == {"year": [2023], "month": [5]}


def test_validate_date_params_dekad():
    result = validate_date_params(time_step="dekad", year=2023, month=5, dekad=2)
    assert result == {"year": [2023], "month": [5], "dekad": [2]}


def test_validate_date_params_week():
    result = validate_date_params(time_step="week", year=2023, week=20)
    assert result == {"year": [2023], "week": [20]}


def test_validate_date_params_bimonth():
    result = validate_date_params(time_step="bimonth", year=2023, month=5, bimonth=2)
    assert result == {"year": [2023], "month": [5], "bimonth": [2]}


def test_validate_date_params_day():
    result = validate_date_params(time_step="day", year=2023, month=5, day=15)
    assert result == {"year": [2023], "month": [5], "day": [15]}


def test_validate_date_params_invalid_time_step():
    with pytest.raises(
        ValueError,
        match="Unsupported time_step: invalid. Supported time_steps are month, dekad, week, bimonth, day",
    ):
        validate_date_params(time_step="invalid", year=2023)


def test_validate_date_params_missing_required_param():
    with pytest.raises(
        ValueError,
        match="For time_step 'month', the following parameters must be provided: month",
    ):
        validate_date_params(time_step="month", year=2023)


def test_validate_date_params_invalid_param_type():
    with pytest.raises(
        ValueError, match="The 'year' parameter must be an int of list of ints"
    ):
        validate_date_params(time_step="month", year="2023", month=5)


def test_validate_date_params_length_mismatch():
    with pytest.raises(
        ValueError,
        match="The length of the date parameters lists must be the same for multiple dates",
    ):
        validate_date_params(time_step="month", year=[2023, 2024], month=[5])


def test_validate_date_params_multiple_dates():
    result = validate_date_params(time_step="month", year=[2023, 2024], month=[5, 6])
    assert result == {"year": [2023, 2024], "month": [5, 6]}


if __name__ == "__main__":
    pytest.main()
