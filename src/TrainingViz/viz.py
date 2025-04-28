import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


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

    def animate(self, fname):
        figsize = 10
        epochs = self.data[self.y_key].shape[0]
        nsteps_test = self.data[self.y_key].shape[-2] - 1
        ny = self.data[self.y_key].shape[-1]
        nu = self.data[self.u_key].shape[-1]
        # nref = self.data[self.r_key].shape[-1]
        loss = self.data[self.loss_key]
        y = self.data[self.y_key].detach().numpy()
        ref = self.data[self.r_key].detach().numpy()
        u = self.data[self.u_key].detach().numpy()

        xmin = self.data[self.ymin_key][0, 0, -1].item()
        xmax = self.data[self.ymax_key][0, 0, -1].item()
        umin = self.data[self.umin_key].unique()[0].item()
        umax = self.data[self.umax_key].unique()[0].item()

        Umin = umin * np.ones([nsteps_test, nu])
        Umax = umax * np.ones([nsteps_test, nu])
        Xmin = xmin * np.ones([nsteps_test+1, ny])
        Xmax = xmax * np.ones([nsteps_test+1, ny])

        fig, ax = plt.subplots(2, figsize=(figsize, figsize))
        title = fig.suptitle("Test Trajectories at Epoch 0", fontsize=48)
        ax[0].set_title("State Trajectories", fontsize=24)
        ax[0].plot(ref[0, :, :], '--', color="red", label="reference",
                   linewidth=5)
        line_y = ax[0].plot(y[0, :, :], label='policy', linewidth=5)
        ax[0].plot(Xmin, '--', color='black', linewidth=5)
        ax[0].plot(Xmax, '--', color='black', linewidth=5)
        ax[1].set_title("Control Input", fontsize=24)
        line_u = ax[1].plot(u[0, :], label='policy', linewidth=5)
        ax[1].plot(Umin, '--', color='black', linewidth=5)
        ax[1].plot(Umax, '--', color='black', linewidth=5)
        text = ax[1].text(-20, -5, f"Loss of Policy: {loss[0].item():.4f}",
                          fontsize=18)

        def update(frame):
            frame_y = y[frame, :, :]
            frame_u = u[frame, :]

            for i, col in enumerate(frame_y.T):
                line_y[i].set_ydata(col)

            if len(frame_u.shape) > 1:
                for col in frame_u.T:
                    line_u[0].set_ydata(col)
            else:
                line_u[0].set_ydata(frame_u)

            text.set_text(f"Loss of policy: {loss[frame].item():.4f}")
            title.set_text(f"Test Trajectories at epoch {frame}")

        ani = animation.FuncAnimation(fig=fig, func=update, frames=epochs,
                                      interval=500)
        ani.save(filename=f"{fname}.gif", writer="pillow")
