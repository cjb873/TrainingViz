import matplotlib.pyplot as plt


class Viz:
    def __init__(self, data: dict, y_key: str, r_key: str = None,
                 u_key: str = None, ymin_key: str = None, ymax_key: str = None,
                 umin_key: str = None, umax_key: str = None,
                 loss_key: str = None):

        self.data = data

        self.y_key = y_key

        self.r_key = r_key
        self.u_key = u_key
        self.ymin_key = ymin_key
        self.ymax_key = ymax_key
        self.umin_key = umin_key
        self.umax_key = umax_key
        self.loss_key = loss_key




    def animate(self):
        nsteps_test = self.data[self.y_key].shape[-2] - 1
        ny = self.data[self.y_key].shape[-1]
        nu = self.data[self.u_key].shape[-1]
        nref = self.data[self.r_key].shape[-1]
        loss = self.data[self.loss_key]

        xmin = self.data[self.ymin_key]
        xmax = self.data[self.ymax_key]
        umin = self.data[self.umin_key]
        umax = self.data[self.umax_key]

        Umin = umin * np.ones([nsteps_test, nu])
        Umax = umax * np.ones([nsteps_test, nu])
        Xmin = xmin * np.ones([nsteps_test+1, ny])
        Xmax = xmax * np.ones([nsteps_test+1, ny])

        fig, ax = plt.subplots(2, figsize=(self.figsize, self.figsize))
        fig.suptitle(f"Test Trajectories at Epoch {self.i}", fontsize=48)
        ax[0].set_title("State Trajectories", fontsize=24)
        ax[0].plot(self.data[self.r_key].detach().numpy().reshape(nsteps_test+1,
                   nref), '--', color="red", label="reference", linewidth=5)
        ax[0].plot(self.data[self.y_key].detach().numpy().reshape(nsteps_test+1,
                   ny), label='policy', linewidth=5)
        ax[1].set_title("Control Input", fontsize=24)
        ax[1].plot(self.data[self.u_key].detach().numpy().reshape(nsteps_test,
                   nu), label='policy', linewidth=5)
        ax[1].text(-20, -1, f"Loss of Policy: {loss.item():.4f}", fontsize=18)
        plt.savefig(f"images/{self.i}.png")


