# Use 2nd-order Taylor operators to solve a Laplacian in parallel using Jacobi iteration, to scale by nodes
#
# 1) Explore alternate data transfer routine MPI_Sendrecv

from numpy import pi, zeros, linspace, sin, amax, abs, stack, repeat, argmax
from numba import njit
from mpi4py import MPI
import matplotlib.pyplot as plt

class JacobiLaplace:

    def __init__(self, n):
        self.parallel_setup()

        self.x_i = 0.0 
        self.x_f = 1.0
        self.y_i = 0.0 
        self.y_f = 1.0
        self.nx = n
        self.ny = n

        self.y = linspace(self.y_i, self.y_f, self.ny)
        self.x = linspace(self.x_i, self.x_f, self.nx)
        self.load_balance_2()
        self.x = linspace(self.x_i, self.x_f, self.nx)

        self.dx = (self.x_i - self.x_f) / self.nx
        self.dy = (self.y_i - self.y_f) / self.ny

        self.sol = zeros((self.nx, self.ny))
        self.newsol = zeros((self.nx, self.ny))

        self.res = []
        
        self.boundaries_2()

    # def supindx(self, i, j):
    #     return i + self.nx * j

    # Boundary conditions from the problem in https://barbagroup.github.io/essential_skills_RRC/laplace/1/
    def boundaries(self):
        # at the 'bottom' y-boundary y_i
        self.sol[:, 0] = 0.0

        # at the 'top' y-boundary y_f
        if (self.rank == 0):
            x = linspace(self.x_i, self.x_f, self.nx - 1)
            self.sol[:-1, -1] = sin(3*pi/2 * x)
        elif (self.rank == self.nprocs - 1):
            x = linspace(self.x_i, self.x_f, self.nx - 1)
            self.sol[1:, -1] = sin(3*pi/2 * x)
        else:
            x = linspace(self.x_i, self.x_f, self.nx - 2)
            self.sol[1:-1, -1] = sin(3*pi/2 * x)

        # at the 'left' physical x-boundary x_i
        if (self.rank == 0):
            self.sol[0, :] = 0.0
        elif (self.rank == self.nprocs - 1):
            # Neumann BC, nothing to do here, check update_sol()
            pass

    def boundaries_2(self):
        # at the 'bottom' y-boundary y_i
        self.sol[:, 0] = 0.0

        # at the 'top' y-boundary y_f
        self.sol[:, -1] = sin(3*pi/2 * self.x)

        # at the 'left' physical x-boundary x_i
        if (self.rank == 0):
            self.sol[0, :] = 0.0
        elif (self.rank == self.nprocs - 1):
            # Neumann BC, nothing to do here, check update_sol()
            pass

    # Using njit to enable quick assignment via loops (which are otherwise very slow), since using vectorized  
    # numpy operations can lead to massive memory usage and slow down the operation instead. 
    # @njit
    def update_sol(self):
        # update solution at inner points
        dx2 = self.dx**2
        dy2 = self.dy**2

        self.newsol[:] = self.sol

        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                self.sol[i, j] = (dy2 * (self.newsol[i+1, j] + self.newsol[i-1, j]) + dx2 * (self.newsol[i, j+1] + self.newsol[i, j-1])) / (2.0 * (dx2 + dy2))
        
        if (self.rank == self.nprocs - 1):
            for j in range(1, self.ny - 1):
                self.newsol[-1, j] = (2*dy2 * self.newsol[-2, j] + dx2 * (self.newsol[-1, j+1] + self.newsol[-1, j-1])) / (2.0 * (dx2 + dy2))
    
    def update_sol_2(self):
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
            self.interdom_transfer()
            self.residual()
            self.update_sol_2()
            # self.interdom_transfer()
            self.comm.Barrier()
            # self.residual()

            # if(self.rank == 0):
            #     print(f' itr:{itr}, res:{self.res[itr - 1]}\n')

            # if(itr == 250):
            #     break

            if(itr == 1 and self.rank == 0): 
                print(f'Initial residual = {self.res[itr - 1]}')
            if(self.res[itr - 1] <= tol):
                if(self.rank == 0): 
                    print(f'Converged in {itr} iteration(s) with maximum point error {self.res[itr - 1]}.')
                break
        else:
            if(self.rank == 0):
                print(f'Maximum number of iterations ({max_iter}) exceeded.')
                print(f'Final residual = {self.res[itr - 1]}')

    # Calculate the point-wise residual over all individual domains and find the global maximum
    def residual(self):
        locres = amax(abs(self.sol - self.newsol))
        # loc = argmax(abs(self.sol - self.newsol))
        # locx = loc//self.ny
        # locy = loc%self.ny
        # print(f'rank:{self.rank}, maxpos:{locx}, {locy}, res:{locres}, x:{self.x[locx]}, sol[{locx},{locy}]:{self.sol[locx,locy]}')
        self.res.append(self.comm.allreduce(locres, op = MPI.MAX))

    # Setup MPI
    def parallel_setup(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()

    # Domain decomposition
    def load_balance(self):
        nx_sub = self.nx // self.nprocs
        nx_rem = self.nx % self.nprocs
        
        self.nx = nx_sub
        if(self.rank < nx_rem):
            self.nx += 1
            indx = self.rank * self.nx
            self.x_i = self.x[indx]
            self.x_f = self.x[indx + self.nx - 1]
        else:
            indx = self.rank * self.nx + nx_rem
            self.x_i = self.x[indx]
            self.x_f = self.x[indx + self.nx - 1]
            
        # account for overlap (for each subdomain at each inter-domain boundary)
        if(self.rank == 0 or self.rank == self.nprocs - 1):
            self.nx += 1
        else:
            self.nx += 2

    def load_balance_2(self):
        nx_sub = self.nx // self.nprocs
        nx_rem = self.nx % self.nprocs
        
        self.nx = nx_sub
        if(self.rank < nx_rem):
            self.nx += 1
            indx_i = self.rank * self.nx
            indx_f = indx_i + self.nx - 1
        else:
            indx_i = self.rank * self.nx + nx_rem
            indx_f = indx_i + self.nx - 1
            
        # account for overlap (for each subdomain at each inter-domain boundary)
        if((self.rank == 0) != (self.rank == self.nprocs - 1)):
            self.nx += 1
        else:
            self.nx += 2
        
        if (self.rank == 0):
            self.x_i = self.x[indx_i]
            if (self.rank == self.nprocs - 1):
                self.x_f = self.x[indx_f]
            else:
                self.x_f = self.x[indx_f + 1]
        elif (self.rank == self.nprocs - 1):
            self.x_i = self.x[indx_i - 1]
            self.x_f = self.x[indx_f]
        else:
            self.x_i = self.x[indx_i - 1]
            self.x_f = self.x[indx_f + 1]
        
    # Transfer data from overlap regions
    def interdom_transfer(self):
        # Even procs send right, odd procs receive left
        if (self.rank % 2 == 0):
            if (self.rank != self.nprocs - 1):
                self.comm.Send(self.sol[-2, :], self.rank + 1, tag = 100)
        else:
            self.comm.Recv(self.sol[0, :], self.rank - 1, tag = 100)

        # Odd procs send right, even procs receive left
        if (self.rank % 2 != 0):
            if (self.rank != self.nprocs - 1):
                self.comm.Send(self.sol[-2, :], self.rank + 1, tag = 200)
        else:
            if (self.rank != 0):
                self.comm.Recv(self.sol[0, :], self.rank - 1, tag = 200)

        # Even procs send left, odd procs receive right
        if (self.rank % 2 == 0):
            if (self.rank != 0):
                self.comm.Send(self.sol[1, :], self.rank - 1, tag = 300)
        else:
            if (self.rank != self.nprocs - 1):
                self.comm.Recv(self.sol[-1, :], self.rank + 1, tag = 300)
            
        # Odd procs send left, even procs receive right
        if (self.rank % 2 != 0):
            self.comm.Send(self.sol[1, :], self.rank - 1, tag = 400)
        else:
            if (self.rank != self.nprocs - 1):
                self.comm.Recv(self.sol[-1, :], self.rank + 1, tag = 400)
    
    # Gather pieces from all processors
    def get_solution(self):
        # Normally, we wouldn't want to gather the arrays distributed on all the processes into one single
        # array; that would defeat the purpose of distributed-memory programming; since we weren't supposed
        # to be able to hold the entire grid on one device. In that case, we would write the results out to 
        # a file from every single processor (?)
        # In this case, we will allow some leeway, since this is primarily a shared-memory model anyway, being
        # run on a single personal computer.
        
        fh = MPI.File.Open(self.comm, "paralap_out", MPI.MODE_CREATE | MPI.MODE_WRONLY, MPI.INFO_NULL)
        fh.Set_size(0)

        if(self.rank == 0):
            xxyyzz = stack([repeat(self.y.reshape(1, -1), self.nx - 1, axis = 0), \
                            repeat(self.x[:-1].reshape(-1, 1), self.ny, axis = 1), \
                            self.sol[:-1]], axis = 2).ravel()
            offset = 3 * (self.nx - 1) * self.ny * self.x.dtype.itemsize * self.rank
        elif(self.rank == self.nprocs - 1):
            xxyyzz = stack([repeat(self.y.reshape(1, -1), self.nx - 1, axis = 0), \
                            repeat(self.x[1:].reshape(-1, 1), self.ny, axis = 1), \
                            self.sol[1:]], axis = 2).ravel()
            offset = 3 * (self.nx - 1) * self.ny * self.x.dtype.itemsize * self.rank
        else:
            xxyyzz = stack([repeat(self.y.reshape(1, -1), self.nx - 2, axis = 0), \
                            repeat(self.x[1:-1].reshape(-1, 1), self.ny, axis = 1), \
                            self.sol[1:-1]], axis = 2).ravel()
            offset = 3 * (self.nx - 2) * self.ny * self.x.dtype.itemsize * self.rank

        # Write all y-values first (all y-values are accessible to every process)
        # offset = 3 * self.nx * self.ny * self.x.dtype.itemsize * self.rank
        fh.Write_at(offset, xxyyzz)
        
        fh.Close()

        return self.rank, self.res


# Instantiate and call JacobiLaplace
def main():
    n = 500
    max_iter = 20000
    tol = 1e-5

    solver = JacobiLaplace(n)
    solver.itersolve(max_iter, tol)
    rank, res = solver.get_solution()

    if(rank == 0):
        plt.figure(figsize=(12, 10))
        plt.title('Maximum point error evolution')
        plt.xlabel('Iterations')
        plt.ylabel('Maximum point error')
        plt.grid(True, alpha = 0.3)
        plt.semilogy(res)
        plt.show()

if __name__ == '__main__':
    main()