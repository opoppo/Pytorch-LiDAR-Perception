from test import *

precrecs_te=torch.load("./precrecs_te.pt")

plot_pr_curve(*precrecs_te);