from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from torch import Tensor
from torch.optim import Adam


class GPRegressionModel(ExactGP):
    def __init__(
            self,
            train_x: Tensor,
            train_y: Tensor,
            training_auto: bool = True
    ) -> None:
        super().__init__(train_x, train_y, GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5))
        if training_auto:
            self._train_model()
        else: 
            self.eval()
            self.likelihood.eval()
    
    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def _train_model(self) -> None:
        self.train()
        self.likelihood.train()
        opt = Adam(self.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        for i in range(250):
            opt.zero_grad()
            output = self(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()
            opt.step()
        self.eval()
        self.likelihood.eval()