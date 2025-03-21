# Differentiable-convex-opt-attention
 (inspired by https://locuslab.github.io/2019-10-28-cvxpylayers/)
An initial implementation of a visual attention mechanism that essentially crops the image using a convex polytope, where the shape and position of the polytope correspond to the solution of a convex optimization problem.
The goal is to allow the loss function in the optimization to be defined as a learnable function of the input. The work that inspired this project demonstrated how to efficiently compute the gradient of the solution to a convex optimization procedure with respect to the input variables/constraints.
