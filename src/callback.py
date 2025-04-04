from neuromancer.callbacks import Callback
import matplotlib.pyplot as plt


class CallbackViz(Callback):
    def __init__(self):
        super().__init__()
        self.figsize = 25
        self.i = 0

    def end_epoch(self, trainer, output):
        output = trainer.test(trainer.best_model)
        nsteps_test = output['test_xn'].shape[-2] - 1
        nx = output['test_xn'].shape[-1]
        nu = output['test_u'].shape[-1]
        nref = output['test_r'].shape[-1]
        loss = output['test_loss']
        """
        xmin = output['test_xmin']
        xmax = output['test_xmax']
        umin = output['test_umin']
        umax = output['test_umax']

        Umin = umin * np.ones([nsteps_test, nu])
        Umax = umax * np.ones([nsteps_test, nu])
        Xmin = xmin * np.ones([nsteps_test+1, nx])
        Xmax = xmax * np.ones([nsteps_test+1, nx])
        """

        fig, ax = plt.subplots(2, figsize=(self.figsize, self.figsize))
        fig.suptitle(f"Test Trajectories at Epoch {self.i}", fontsize=48)
        ax[0].set_title("State Trajectories", fontsize=24)
        ax[0].plot(output['test_r'].detach().numpy().reshape(nsteps_test+1,
                   nref), '--', color="red", label="reference", linewidth=5)
        ax[0].plot(output['test_xn'].detach().numpy().reshape(nsteps_test+1,
                   nx), label='policy', linewidth=5)
        ax[1].set_title("Control Input", fontsize=24)
        ax[1].plot(output['test_u'].detach().numpy().reshape(nsteps_test, nu),
                   label='policy', linewidth=5)
        ax[1].text(-20, -1, f"Loss of Policy: {loss.item():.4f}", fontsize=18)
        plt.savefig(f"images/{self.i}.png")

        self.i += 1
