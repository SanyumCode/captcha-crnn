import torch

from src.config import Config
from src.model import create_model
from src.utils import calculate_accuracy, decode_predictions


def test_model_output_shape_and_probabilities():
    model = create_model().eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH))
    assert output.shape == (27, 1, Config.NUM_CLASSES)
    assert torch.allclose(output.exp().sum(dim=2), torch.ones(27, 1), atol=1e-5)


def test_ctc_decoder_collapses_repeats_and_blank():
    output = torch.full((5, 1, Config.NUM_CLASSES), -20.0)
    for timestep, index in enumerate([0, 0, Config.NUM_CLASSES - 1, 1, 1]):
        output[timestep, 0, index] = 0.0
    assert decode_predictions(output) == ["01"]


def test_accuracy_empty_input():
    assert calculate_accuracy([], []) == 0.0
