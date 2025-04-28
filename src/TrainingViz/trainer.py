from neuromancer.trainer import Trainer
from torch.utils.data import DataLoader
import torch

def move_batch_to_device(batch, device="cpu"):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class VizTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.callback.begin_train(self)
    
        try:
            for i in range(self.current_epoch, self.current_epoch+self.epochs):

                self.model.train()
                losses = []
                for t_batch in self.train_data:
                    t_batch['epoch'] = i
                    t_batch = move_batch_to_device(t_batch, self.device)
                    output = self.model(t_batch)

                    if self.multi_fidelity:
                        for node in self.model.nodes:
                            alpha_loss = node.callable.get_alpha_loss()
                            output[self.train_metric] += alpha_loss

                    self.optimizer.zero_grad()
                    output[self.train_metric].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    losses.append(output[self.train_metric])
                    self.callback.end_batch(self, output)

                output[f'mean_{self.train_metric}'] = torch.mean(torch.stack(losses))
                self.callback.begin_epoch(self, output)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(output[f'mean_{self.train_metric}'])

                with torch.set_grad_enabled(self.model.grad_inference):
                    self.model.eval()
                    if self.dev_data is not None:
                        losses = []
                        for d_batch in self.dev_data:
                            d_batch = move_batch_to_device(d_batch, self.device)
                            eval_output = self.model(d_batch)
                            losses.append(eval_output[self.dev_metric])
                        eval_output[f'mean_{self.dev_metric}'] = torch.mean(torch.stack(losses))
                        output = {**output, **eval_output}
                    self.callback.begin_eval(self, output)  # Used for alternate dev evaluation

                    if (self._eval_min and output[self.eval_metric] < self.best_devloss)\
                            or (not self._eval_min and output[self.eval_metric] > self.best_devloss):
                        self.best_model = deepcopy(self.model.state_dict())
                        self.best_devloss = output[self.eval_metric]
                        self.badcount = 0
                    else:
                        if i > self.warmup:
                            self.badcount += 1
                    if self.logger is not None:
                        self.logger.log_metrics(output, step=i)
                    else:
                        mean_loss = output[f'mean_{self.train_metric}']
                        if i % (self.epoch_verbose) == 0:
                            print(f'epoch: {i}  {self.train_metric}: {mean_loss}')

                    self.callback.end_eval(self, output, i)  # visualizations

                    self.callback.end_epoch(self, output)

                    if self.badcount > self.patience:
                        print('Early stopping!!!')
                        break
                    self.current_epoch = i + 1

        except KeyboardInterrupt:
            print("Interrupted training loop.")

        self.callback.end_train(self, output)  # write training visualizations

        # Assign best weights to the model
        self.model.load_state_dict(self.best_model)

        if self.logger is not None:
            self.logger.log_artifacts({
                "best_model_state_dict.pth": self.best_model,
                "best_model.pth": self.model,
            })
        return self.best_model


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


