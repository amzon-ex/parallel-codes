## Parallel Jacobi iterative solver

Implements Jacobi iteration to solve the Laplace equation, in a rudimentary manner for now (doesn't use matrices as finite difference operators *yet*). The Jacobi iteration in this form goes as follows:
 - The Laplace equation (in 2D):
   $$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$$
 - We discretize this in a 2D spatial grid $(x, y) \in [0, 1] \times [0, 1] \in \mathbb{R}^2$ as
   $$\frac{u_{i+1, j} - 2 u_{i, j} + u_{i-1, j}}{\Delta x^2} + \frac{u_{i, j+1} - 2 u_{i, j} + u_{i, j-1}}{\Delta y^2} = 0$$
 - This we can use to get an iterative expression for the solution $u_{i, j}$ at the $k^{th}$ iteration:
   $$u^{k+1}_{i, j} = \frac{\Delta y^2(u^{k}_{i+1, j} + u^{k}_{i-1, j}) + \Delta x^2(u^{k}_{i, j+1} + u^{k}_{i, j-1})}{2(\Delta x^2 + \Delta y^2)}$$
 - We've kept the option of choosing $\Delta x$ and $\Delta y$ different, but while running the code, we set them to be the same. This can be easily tweaked.

The domain decomposition is along the x-direction - the simplest decomposition has been used.

**paralap.py** is the parallel implementation - it produces a plot of the evolution of the residual being monitored. Output goes to **paralap_out** and is a binary file readable by **paralap_read.py** which produces a plot of the output (switchable between a colormesh plot and a surface plot).
**paralap_uc.py** is the implementation without classes, which also uses numba. This improves serial speed significantly.

However, scaling is still horrible.
**TO-DO:** Profiling and scaling performance test.

**singalap.py** is the serial implementation which produces **singalap_out**. Again, the outfile can be read by **paralap_read.py** by passing the name of the outfile as an argument.
