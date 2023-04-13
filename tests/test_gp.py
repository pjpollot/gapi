import torch

from gpytorch.metrics import mean_squared_error

from gapi._models import GPRegressionModel

def test_gp_performance():
    torch.manual_seed(0)
    train_x = torch.rand(size=(100, 1))
    train_y = torch.sin(train_x * (2 * torch.pi)) + torch.randn_like(train_x) * 0.2
    train_y = train_y.squeeze()
    test_x = torch.linspace(0, 1, 100).unsqueeze(-1)
    test_y = torch.sin(test_x * (2 * torch.pi))
    test_y = test_y.squeeze()
    model = GPRegressionModel(train_x, train_y, training_auto=False)
    mse_before_training = mean_squared_error(model(test_x), test_y, squared=False)
    model._train_model()
    mse_after_training = mean_squared_error(model(test_x), test_y, squared=False)
    assert mse_before_training > mse_after_training