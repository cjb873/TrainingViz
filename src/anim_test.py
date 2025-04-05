

callb.data = torch.stack(callb.data)
callb.data = callb.data.numpy()
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

line = ax.plot(callb.data[0])
title = fig.suptitle("State Trajectories at epoch 0")
def update(frame):
    x = callb.data[frame]
    line[0].set_ydata(x[:,0])
    line[1].set_ydata(x[:,1])

    title.set_text(f"State Trajectories at epoch {frame}")

    return line

ani = animation.FuncAnimation(fig=fig, func=update, frames=callb.data.shape[0],
                              interval=500)

plt.show()
