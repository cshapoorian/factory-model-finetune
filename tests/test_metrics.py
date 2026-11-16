import pytest

from factory_model_finetune.metrics import relative_change


def test_relative_change():
    assert relative_change(100, 110) == pytest.approx(0.10)


def test_relative_change_rejects_zero_baseline():
    with pytest.raises(ValueError):
        relative_change(0, 5)
