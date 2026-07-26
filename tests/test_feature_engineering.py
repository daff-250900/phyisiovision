from src.feature_engineering import calculate_angle


def test_right_angle() -> None:
    assert round(calculate_angle((1, 0), (0, 0), (0, 1))) == 90
