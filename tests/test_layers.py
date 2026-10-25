import torch
from pytorch_physics.layers import linear

def test_linear_known_values_without_bias():
    # Test linear layer with known values without bias
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_output = torch.tensor([[5.0, 11.0], [11.0, 25.0]])

    output = linear(input_tensor, weight)
    assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"
    
    