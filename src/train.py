import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from trainer import VizTrainer
from neuromancer.dynamics import ode, integrators
from callback import CallbackViz


torch.manual_seed(0)
# ground truth system model
gt_model = psl.nonautonomous.VanDerPolControl()
# sampling rate
ts = gt_model.params[1]['ts']
# problem dimensions
nx = gt_model.nx    # number of states
nu = gt_model.nu    # number of control inputs
nref = nx           # number of references
# constraints bounds
umin = -5.
umax = 5.
xmin = -4.
xmax = 4.

# white-box ODE model with no-plant model mismatch
van_der_pol = ode.VanDerPolControl()
van_der_pol.mu = nn.Parameter(torch.tensor(gt_model.mu), requires_grad=False)

# integrate continuous time ODE
integrator = integrators.RK4(van_der_pol, h=torch.tensor(ts))
integrator_node = Node(integrator, ['xn', 'u'], ['xn'], name='model')


def get_policy_data(nsteps, n_samples, nsteps_test):
    # Training dataset generation
    train_d = DictDataset({'xn': torch.randn(n_samples, 1, nx),
                           'r': torch.zeros(n_samples, nsteps+1, nx)},
                          name='train')
    # Development dataset generation
    dev_d = DictDataset({'xn': torch.randn(n_samples, 1, nx),
                         'r': torch.zeros(n_samples, nsteps+1, nx)},
                        name='dev')

    test_d = DictDataset({'xn': torch.randn(1, 1, nx, dtype=torch.float32),
                         'r': torch.zeros(1, nsteps_test+1,
                                          nx, dtype=torch.float32)},
                         name='test')

    # torch dataloaders
    batch_size = 200
    train_loader = torch.utils.data.DataLoader(train_d,
                                               batch_size=batch_size,
                                               collate_fn=train_d.collate_fn,
                                               shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_d,
                                             batch_size=batch_size,
                                             collate_fn=dev_d.collate_fn,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_d,
                                              batch_size=1,
                                              collate_fn=test_d.collate_fn,
                                              shuffle=False)

    return train_loader, dev_loader, test_loader


nsteps = 50  # prediction horizon
n_samples = 2000    # number of sampled scenarios
nsteps_test = 100
train_loader, dev_loader, test_loader = \
        get_policy_data(nsteps, n_samples, nsteps_test)

# symbolic system model
model = Node(integrator, ['xn', 'u'], ['x'], name='model')

# neural net control policy with hard control action bounds
net = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                        nonlin=activations['gelu'], min=umin, max=umax)
policy = Node(net, ['xn', 'r'], ['u'], name='policy')

# closed-loop system model
cl_system = System([policy, integrator_node], nsteps=nsteps)


x = variable('xn')
ref = variable('r')
# objectives
regulation_loss = 5. * ((x == ref) ^ 2)  # target posistion
# constraints
state_lower_bound_penalty = 10.*(x > xmin)
state_upper_bound_penalty = 10.*(x < xmax)

# objectives and constraints names for nicer plot
regulation_loss.name = 'state_loss'
state_lower_bound_penalty.name = 'x_min'
state_upper_bound_penalty.name = 'x_max'

# list of constraints and objectives
objectives = [regulation_loss]
constraints = [
    state_lower_bound_penalty,
    state_upper_bound_penalty,
]

components = [cl_system]
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem(components, loss)

optimizer = torch.optim.AdamW(policy.parameters(), lr=0.002)


callb = CallbackViz()

#  Neuromancer trainer
trainer = VizTrainer(
    problem,
    train_loader,
    dev_loader,
    test_loader,
    optimizer=optimizer,
    epochs=100,
    train_metric='train_loss',
    eval_metric='dev_loss',
    test_metric='test_loss',
    warmup=50,
    callback=callb
)
# Train control policy
best_model = trainer.train()
# load best trained model
trainer.model.load_state_dict(best_model)

cl_system.nsteps = nsteps_test

trajectories = cl_system(test_loader.dataset.data)
