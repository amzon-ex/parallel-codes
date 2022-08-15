from numpy import pi, zeros, linspace, sin, amax, abs, stack, repeat, argmax
from numba import njit
import matplotlib.pyplot as plt

class JacobiLaplace:

    def __init__(self, n):

        self.x_i = 0.0 
        self.x_f = 1.0
        self.y_i = 0.0 
        self.y_f = 1.0
        self.nx = n
        self.ny = n
        
        # self.max_iter = max_iter
        # self.tol = tol

        self.y = linspace(self.y_i, self.y_f, self.ny)
        self.x = linspace(self.x_i, self.x_f, self.nx)

        self.dx = (self.x_i - self.x_f) / self.nx
        self.dy = (self.y_i - self.y_f) / self.ny

        self.sol = zeros((self.nx, self.ny))
        self.newsol = zeros((self.nx, self.ny))
        
        self.res = []

        self.boundaries()

    # def supindx(self, i, j):
    #     return i + self.nx * j

    # Boundary conditions from the problem in https://barbagroup.github.io/essential_skills_RRC/laplace/1/
    def boundaries(self):
        # at the 'bottom' y-boundary y_i
        self.sol[:, 0] = 0.0

        # at the 'top' y-boundary y_f
        self.sol[:, -1] = sin(3*pi/2 * self.x)

        # at the 'left' physical x-boundary x_i
        self.sol[0, :] = 0.0

        # x_f: Neumann BC, nothing to do here, check update_sol()


    # Using njit to enable quick assignment via loops (which are otherwise very slow), since using vectorized  
    # numpy operations can lead to massive memory usage and slow down the operation instead. 
    # @njit
    def update_sol(self):
        # update solution at inner points
        dx2 = self.dx**2
        dy2 = self.dy**2
        self.newsol[:] = self.sol

        self.sol[1:-1, 1:-1] = (dy2 * (self.newsol[2:, 1:-1] + self.newsol[:-2, 1:-1]) + dx2 * (self.newsol[1:-1, 2:] + self.newsol[1:-1, :-2])) / (2.0 * (dx2 + dy2))
        
        # Neumann BC at x_f
        self.sol[-1, 1:-1] = (2*dy2 * self.newsol[-2, 1:-1] + dx2 * (self.newsol[-1, 2:] + self.newsol[-1, :-2])) / (2.0 * (dx2 + dy2))

    # Iterate until tolerance is reached
    def itersolve(self, max_iter, tol):
        # Carry out the iterative hunt
        for itr in range(1, max_iter + 1):
            # self.newsol[:] = self.sol
            # break
            self.update_sol()
            self.residual()

            # if(itr == 30):
            #     break

            if(itr == 1): 
                print(f'Initial residual = {self.res}')
            if(self.res[itr - 1] <= tol):
                print(f'Converged in {itr} iteration(s) with maximum point error {self.res[itr - 1]}.')
                break
        else:
            print(f'Maximum number of iterations ({max_iter}) exceeded.')
            print(f'Final residual = {self.res[itr - 1]}')

    # Calculate the point-wise residual over all individual domains and find the global maximum
    def residual(self):
        self.res.append(amax(abs(self.newsol - self.sol)))
        # loc = argmax(abs(self.sol - self.newsol))
        # print(f'maxpos:{loc//self.ny}, {loc%self.ny}, res:{self.res}, x:{self.x[loc//self.ny]}')
    
    # Gather pieces from all processors
    def get_solution(self):
        xxyyzz = stack([repeat(self.y.reshape(1, -1), self.nx, axis = 0), \
                        repeat(self.x.reshape(-1, 1), self.ny, axis = 1), \
                        self.newsol], axis = 2).ravel()
        xxyyzz.tofile('singalap_out')

        return self.res

# Instantiate and call JacobiLaplace
def main():
    n = 1000
    max_iter = 25000
    tol = 1e-3

    solver = JacobiLaplace(n)
    solver.itersolve(max_iter, tol)
    res = solver.get_solution()

    plt.figure(figsize=(12, 10))
    plt.grid(True, alpha = 0.3)
    plt.semilogy(res, '.-', ms=1.5)
    plt.show()

if __name__ == '__main__':
    main()