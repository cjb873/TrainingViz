from neuromancer.trainer import Trainer
from torch.utils.data import DataLoader
import torch


class VizTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self, best_model):
        assert isinstance(self.test_data, DataLoader)

        self.model.load_state_dict(best_model, strict=False)

        self.model.eval()

        self.callback.begin_test(self)

        nsteps = self.test_data.dataset.datadict['r'].shape[-2] - 1
        self.set_nsteps(nsteps)

        output = {}
        losses = []
        for batch in self.test_data:
            batch_output = self.model(batch)
            losses.append(batch_output[self.test_metric])
        output[f'mean_{self.test_metric}'] = torch.mean(torch.stack(losses))
        output = {**output, **batch_output}

        self.callback.end_test(self, output)

        nsteps = self.train_data.dataset.datadict['r'].shape[-2] - 1
        self.set_nsteps(nsteps)
        if self.logger is not None:
            self.logger.log_metrics({f"best_{k}": v for k, v in
                                     output.items()})

        return output

    def set_nsteps(self, nsteps):
        for node in self.model.nodes:
            node.nsteps = nsteps
