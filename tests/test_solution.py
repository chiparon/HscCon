import inspect

import pytest
import torch

import solution


PUBLIC_FUNCTIONS = {
    "hif4_calibration_and_quantize_weight": 3,
    "hif4_dynamic_quantize_activation": 3,
    "hif4_calibration_attention": 4,
    "hif4_dynamic_quantize_q": 5,
    "hif4_dynamic_quantize_k": 5,
    "hif4_dynamic_quantize_v": 5,
}


def test_submission_interface_signatures_are_stable() -> None:
    for name, parameter_count in PUBLIC_FUNCTIONS.items():
        function = getattr(solution, name)
        assert len(inspect.signature(function).parameters) == parameter_count


def test_dequantize_nvfp4_applies_one_scale_per_block() -> None:
    carrier = torch.arange(32, dtype=torch.float32).reshape(1, 32)
    scale = torch.tensor([[2.0, 0.5]], dtype=torch.float32)

    actual = solution.dequantize_nvfp4(carrier, scale)
    expected = torch.cat((carrier[:, :16] * 2.0, carrier[:, 16:] * 0.5), dim=-1)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


@pytest.mark.parametrize(
    ("carrier", "scale", "message"),
    [
        (torch.ones(15), torch.ones(1), "not divisible"),
        (torch.ones(16), torch.ones(2), "scale shape"),
        (torch.tensor(1.0), torch.tensor(1.0), "at least one dimension"),
    ],
)
def test_dequantize_nvfp4_rejects_invalid_shapes(
    carrier: torch.Tensor, scale: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        solution.dequantize_nvfp4(carrier, scale)
