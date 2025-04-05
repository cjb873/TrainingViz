from neuromancer.callbacks import Callback


class CallbackViz(Callback):
    def __init__(self):
        super().__init__()
        self.figsize = 25
        self.i = 0
        self.data = None

    def end_epoch(self, trainer, output):
        current_results = trainer.test(trainer.best_model)

        if self.data is None:
            self.data = {key: [val] for key, val in current_results.items()}
        else:
            {key: val.append(current_results[key]) for key, val in
             self.data.items()}
