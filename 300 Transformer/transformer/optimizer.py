import torch
from torch.optim import Adam


class ScheduledAdam(Adam):
    """
    논문에서는 Adam Optimizer를 사용하였으며, hyper-parameters 는 다음과 같다
      B_1=0.9
      B_2-0.98
      e=10**-9

    Learning Rate는 training도중에 변경하도록 만들었음
    """

    def __init__(self, parameters, embed_dim: int,
                 betas=(0.9, 0.98), eps=1e-09, init_lr: float = 2.0,
                 warmup_steps: int = 4000, **kwargs):
        """
        Warm-up Steps 의 학습시 중요한 부분은 충분히 많은 데이터가 warm-up steps 동안 학습이 되어야 한다
        따라서 batch_size 가 2048 보다 작으면서 warmup_steps 도 4000 이하가 된다면 적은 데이터만
        warm-up steps만 적용이 된다. 따라서 batch_size를 크게 늘리던 warmup steps을 좀더 크게 가져가던 해야 한다

        :param parameters: transformer.parameters() <- Transformer Model weights
        :param embed_dim: it is used as "d_model" in paper and the default value  is 512
        :param warmup_steps: warm-up steps. the lr will linearly increase during warm-up steps
        """
        super().__init__(parameters, betas=betas, eps=eps, **kwargs)

        self.embed_dim = embed_dim
        self.init_lr = init_lr
        self.warmup_step = warmup_steps
        self.n_step = 0
        self.lr = 0

    def step(self, **kwargs):
        self._update_learning_rate()
        super().step()

    def _update_learning_rate(self):
        self.n_step += 1

        step, warmup_step = self.n_step, self.warmup_step
        init_lr = self.init_lr
        d_model = self.embed_dim

        self.lr = init_lr * (d_model ** -0.5) * min(step ** -0.5, step * warmup_step ** -1.5)

        for param_group in self.param_groups:
            param_group['lr'] = self.lr
