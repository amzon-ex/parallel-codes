# Use 2nd-order Taylor operators to solve a Laplacian in parallel using Jacobi iteration, to scale by nodes
#
# 1) Explore alternate data transfer routine MPI_Sendrecv

from numpy import pi, zeros, linspace, sin, amax, abs, stack, repeat, argmax
from numba import njit
from mpi4py import MPI
# import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

n = 2000
max_iter = 25000
tol = 1e-3

x_i = 0.0 
x_f = 1.0
y_i = 0.0 
y_f = 1.0
nx = n
ny = n

y = linspace(y_i, y_f, ny)
x = linspace(x_i, x_f, nx)

# ----------------------------------------
# Load-balancing
# ----------------------------------------
nx_sub = nx // nprocs
nx_rem = nx % nprocs

nx = nx_sub
if(rank < nx_rem):
    nx += 1
    indx_i = rank * nx
    indx_f = indx_i + nx - 1
else:
    indx_i = rank * nx + nx_rem
    indx_f = indx_i + nx - 1
    
# account for overlap (for each subdomain at each inter-domain boundary)
if((rank == 0) != (rank == nprocs - 1)):
    nx += 1
else:
    nx += 2

if (rank == 0):
    x_i = x[indx_i]
    if (rank == nprocs - 1):
        x_f = x[indx_f]
    else:
        x_f = x[indx_f + 1]
elif (rank == nprocs - 1):
    x_i = x[indx_i - 1]
    x_f = x[indx_f]
else:
    x_i = x[indx_i - 1]
    x_f = x[indx_f + 1]

# Re-evaluate gridpoints specific to the processor
x = linspace(x_i, x_f, nx)
# print(rank, x_i, x_f, nx)

dx = (x_i - x_f) / nx
dy = (y_i - y_f) / ny

sol = zeros((nx, ny))
newsol = zeros((nx, ny))

res = zeros(max_iter)

# ----------------------------------------
# Boundary conditions from the problem in https://barbagroup.github.io/essential_skills_RRC/laplace/1/
# ----------------------------------------
# at the 'bottom' y-boundary y_i
sol[:, 0] = 0.0

# at the 'top' y-boundary y_f
sol[:, -1] = sin(3*pi/2 * x)

# at the 'left' physical x-boundary x_i
if (rank == 0):
    sol[0, :] = 0.0
elif (rank == nprocs - 1):
    # Neumann BC, nothing to do here, check update_sol()
    pass

# Using njit to enable quick assignment via loops (which are otherwise very slow), since using vectorized  
# numpy operations can lead to massive memory usage and slow down the operation instead.
# @profile
@njit
def update_sol(sol, newsol):
    # global sol, newsol
    # update solution at inner points
    dx2 = dx**2
    dy2 = dy**2
    print(nx, ny)
    for i in range(nx):
        for j in range(ny):
            newsol[i, j] = sol[i, j]

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            sol[i, j] = (dy2 * (newsol[i+1, j] + newsol[i-1, j]) + dx2 * (newsol[i, j+1] + newsol[i, j-1])) / (2.0 * (dx2 + dy2))
    
    if (rank == nprocs - 1):
        for j in range(1, ny - 1):
            sol[-1, j] = (2*dy2 * newsol[-2, j] + dx2 * (newsol[-1, j+1] + newsol[-1, j-1])) / (2.0 * (dx2 + dy2))


# Calculate the point-wise residual over all individual domains and find the global maximum
# @profile
@njit
def residual2(itr, res, sol, newsol):
    locres = 0.0
    for i in range(nx):
        for j in range(ny):
            diff = abs(sol[i, j] - newsol[i, j])
            if (locres < diff):
                locres = diff
    return locres

def residual(itr):
    locres = residual2(itr, res, sol, newsol)
    res[itr - 1] = comm.allreduce(locres, op = MPI.MAX)
    
    
# Transfer data from overlap regions
def interdom_transfer():
    # Even procs send right, odd procs receive left
    if (rank % 2 == 0):
        if (rank != nprocs - 1):
            comm.Send(sol[-2, :], rank + 1, tag = 100)
    else:
        comm.Recv(sol[0, :], rank - 1, tag = 100)

    # Odd procs send right, even procs receive left
    if (rank % 2 != 0):
        if (rank != nprocs - 1):
            comm.Send(sol[-2, :], rank + 1, tag = 200)
    else:
        if (rank != 0):
            comm.Recv(sol[0, :], rank - 1, tag = 200)

    # Even procs send left, odd procs receive right
    if (rank % 2 == 0):
        if (rank != 0):
            comm.Send(sol[1, :], rank - 1, tag = 300)
    else:
        if (rank != nprocs - 1):
            comm.Recv(sol[-1, :], rank + 1, tag = 300)
        
    # Odd procs send left, even procs receive right
    if (rank % 2 != 0):
        comm.Send(sol[1, :], rank - 1, tag = 400)
    else:
        if (rank != nprocs - 1):
            comm.Recv(sol[-1, :], rank + 1, tag = 400)

# ----------------------------------------
# Iterate until tolerance is reached
# ----------------------------------------
# Get start time
mytime = MPI.Wtime()
# Carry out the iterative hunt
for itr in range(1, max_iter + 1):
    interdom_transfer()
    residual(itr)
    update_sol(sol, newsol)
    # interdom_transfer()
    comm.Barrier()
    # residual()

    # if(rank == 0):
    #     print(f' itr:{itr}, res:{res[itr - 1]}\n')

    # if(itr == 250):
    #     break

    if(itr == 1 and rank == 0): 
        print(f'Initial residual = {res[itr - 1]}')
    if(res[itr - 1] <= tol):
        mytime = MPI.Wtime() - mytime
        if(rank == 0): 
            print(f'Converged in {itr} iteration(s) with maximum point error {res[itr - 1]}.')
            print(f'Time taken: {mytime:.3f} secs')
        break
else:
    mytime = MPI.Wtime() - mytime
    if(rank == 0):
        print(f'Maximum number of iterations ({max_iter}) exceeded.')
        print(f'Final residual = {res[itr - 1]}')
        print(f'Time taken: {mytime:.3f} secs')



# Gather pieces from all processors
def get_solution():
    # Normally, we wouldn't want to gather the arrays distributed on all the processes into one single
    # array; that would defeat the purpose of distributed-memory programming; since we weren't supposed
    # to be able to hold the entire grid on one device. In that case, we would write the results out to 
    # a file from every single processor (?)
    # In this case, we will allow some leeway, since this is primarily a shared-memory model anyway, being
    # run on a single personal computer.
    
    fh = MPI.File.Open(comm, "paralap_out", MPI.MODE_CREATE | MPI.MODE_WRONLY, MPI.INFO_NULL)
    fh.Set_size(0)

    if(rank == 0):
        xxyyzz = stack([repeat(y.reshape(1, -1), nx - 1, axis = 0), \
                        repeat(x[:-1].reshape(-1, 1), ny, axis = 1), \
                        sol[:-1]], axis = 2).ravel()
        offset = 3 * (nx - 1) * ny * x.dtype.itemsize * rank
    elif(rank == nprocs - 1):
        xxyyzz = stack([repeat(y.reshape(1, -1), nx - 1, axis = 0), \
                        repeat(x[1:].reshape(-1, 1), ny, axis = 1), \
                        sol[1:]], axis = 2).ravel()
        offset = 3 * (nx - 1) * ny * x.dtype.itemsize * rank
    else:
        xxyyzz = stack([repeat(y.reshape(1, -1), nx - 2, axis = 0), \
                        repeat(x[1:-1].reshape(-1, 1), ny, axis = 1), \
                        sol[1:-1]], axis = 2).ravel()
        offset = 3 * (nx - 2) * ny * x.dtype.itemsize * rank

    # Write all y-values first (all y-values are accessible to every process)
    # offset = 3 * nx * ny * x.dtype.itemsize * rank
    fh.Write_at(offset, xxyyzz)
    
    fh.Close()


get_solution()

# if(rank == 0):
#     plt.figure(figsize=(12, 10))
#     plt.title('Maximum point error evolution')
#     plt.xlabel('Iterations')
#     plt.ylabel('Maximum point error')
#     plt.grid(True, alpha = 0.3)
#     plt.semilogy(res[:itr])
#     plt.show()