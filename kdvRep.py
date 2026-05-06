import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import gmres, LinearOperator
import matplotlib.pyplot as plt

def p_d_o(order, x_min, x_max, N):
    """Periodic finite-difference derivative operator (circulant).

    Builds an Nx-by-Nx sparse CSR matrix for a centered finite-difference
    stencil with periodic wrap. Supported orders: 1 and 3 (same stencils
    as before).
    """
    h = (x_max - x_min) / N
    if order == 1:
        coeffs = np.array([1, -8, 0, 8, -1]) / (12 * h)
        offsets = np.array([-2, -1, 0, 1, 2])
    elif order == 3:
        # 7-point, fourth-order-accurate stencil for third derivative
        # coefficients for offsets [-3,-2,-1,0,1,2,3]
        coeffs = np.array([1, -8, 13, 0, -13, 8, -1]) / (8 * h ** 3)
        offsets = np.array([-3, -2, -1, 0, 1, 2, 3])
    else:
        raise ValueError(f"Unsupported derivative order: {order}")

    idx = np.arange(N)
    rows = []
    cols = []
    data = []

    for coeff, offset in zip(coeffs, offsets):
        col_idx = (idx + int(offset)) % N
        rows.append(idx)
        cols.append(col_idx)
        data.append(np.full(N, coeff, dtype=float))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    D = csr_matrix((data, (rows, cols)), shape=(N, N))
    return D

def resi(wk, un, gamma, dx):
    return (getEntropy(un + gamma*(wk - un), dx) - getEntropy(un, dx))
def r_prime(wk, un, gamma, dx):
    return np.dot(getEntropy_prime(un + gamma*(wk - un), dx),(wk - un))
    #return (np.dot(getEntropy_prime(un + gamma*(wk - un), dx),(wk - un)) + gamma*resi(wk,un,gamma,dx))/gamma**2



def kdvsolver(Nx, time, dt, N_opts, gmresOpts, relax):
    #Generate the grid
    x_range = (-10.0, 10.0)
    L = x_range[1] - x_range[0]
    xs = np.linspace(-10, 10, Nx, endpoint=False)
    dx = L/Nx

    D1 = p_d_o(1, x_range[0], x_range[1], Nx)
    D3 = p_d_o(3, x_range[0], x_range[1], Nx)

    Ix = diags(np.ones(Nx), 0)

    def u(x, t, x0, c):
        arg = (np.mod(x - c*t - x[0], L) - x0 + x[0])
        return 0.5*c*(1/np.cosh(0.5*np.sqrt(c)*arg))**2
    
    x0 = 0.0
    speed = 2.0

    #initial
    un = u(xs, 0.0, x0, speed)

    M = int(round(time / dt))
    ent = np.zeros(M + 1)
    err = np.zeros(M + 1)

    ent[0] = getEntropy(un, dx)

    def F(v, vn):
        return v - vn + dt*(v * (D1 @ v) + (D1@(v**2)) + 0.5*(D3 @ v))
    
    #def jac(v):
        #return Ix + dt*(diags(v,0) @ D1 + diags(D1 @ v,0) + 2*(D1 @ diags(v,0)) + 0.5*D3)

    def jac_matvec(v, y):
        return y + dt*(v*(D1@y) + (D1@v)*y + 2*(D1@(v*y)) + 0.5*(D3@y)) 
    
    t = np.zeros(M + 1)

    for k in range(M):

        if k % 100 == 0:
            print(f"time = {t[k]}")

        def f(u):
            return F(u,un)
        
        def jac(v):
            return LinearOperator((Nx, Nx), matvec=lambda y: jac_matvec(v, y))

        wk = un.copy()

        wk, resHist = newton(wk, f, jac, N_opts, gmresOpts)

        a = 1.0
        if relax:
            #a = (np.linalg.norm(un)**2 - np.dot(wk, un)) / np.linalg.norm(wk - un)**2
            '''
            def res(gamma):
                return resi(wk,un, gamma,dx)
            def res_prime(gamma):
                return r_prime(wk,un,gamma,dx)
            '''
            def res(gamma):
                return r_prime(wk, un, gamma, dx)          # <-- was resi

            def res_prime(gamma):
                # second derivative: dx * ||wk - un||^2
                d = wk - un
                return dx * np.dot(d, d)

            d = wk - un
            if np.linalg.norm(d) <= 1e-12:
                a = 1.0
            else:
                a = newton1D(1.0, res, res_prime)
            
            
            #print(f"current gamma : {a} and its shape {a.shape}")
            

        un = 2 * a * wk + (1 - 2 * a) * un

        t[k+1] = t[k] + a*dt
        
        ent[k + 1] = getEntropy(un, dx)
        err[k + 1] = getError(un, u(xs, t[k + 1], x0, speed), dx)
        #print("time", t[k+1], "entropy", ent[k+1], "error", err[k+1])

    print("initial error", err[:5])

    return xs, t, un, ent, err, resHist

def getEntropy(u, dx):
    return dx*np.linalg.norm(u)**2/2.0

def getEntropy_prime(u, dx):
    return dx*u

def getError(uex, u, dx):
    return np.sqrt(dx)*np.linalg.norm(u - uex)

def newton(x, f, j, N_opts, gmresopts):
    
    abs_tol_n = N_opts[0]
    r_tol_n = N_opts[1]
    max_n = N_opts[2]

    tolmax = gmresopts[0]
    max_G = gmresopts[1]
    gamma = gmresopts[2]

    Nx = len(x)
    it_count = 0
    res_hist =[]
    entropies = []

    f_val = f(x)
    f_val_norm = np.linalg.norm(f_val)/np.sqrt(Nx)
    res_hist.append(f_val_norm)
    stop_tol = abs_tol_n + r_tol_n*f_val_norm

    tol_G = 0.9

    for ns in range(max_n):
        if f_val_norm < stop_tol:
            break

        #def J(y):
            #eps = 1e-8 if np.linalg.norm(y) == 0 else 1e-8 * np.linalg.norm(x) / np.linalg.norm(y)
            #return (f(x + eps*y) - f(x - eps*y)) / (2*eps)  # MUST be (Nx,)
        #A = LinearOperator((Nx,Nx), matvec=j)

        A = j(x)


        f_val_norm_old = f_val_norm
        dx, info = gmres(A, -f_val, rtol=tol_G, maxiter=max_G, restart=max_G)

        x += dx

        f_val = f(x)
        f_val_norm = np.linalg.norm(f_val) / np.sqrt(Nx)
        res_hist.append(f_val_norm)
        tol_G = EisenstatWalker(
            tol_G,
            f_val_norm,
            f_val_norm_old,
            gamma,
            tolmax,
            stop_tol
        )

        it_count = ns + 1

    return x, res_hist

def EisenstatWalker(eta, fnorm_new, fnorm_old, gamma, etamax, stopTol):
    rat = fnorm_new/fnorm_old
    eta2 = eta*eta
    etanew = gamma*rat*rat
    if gamma*eta2 >0.1:
        etanew = max(etanew, gamma*eta2)
    etanew = min(etanew, etamax)
    eta = max(etanew, 0.5*stopTol/fnorm_new)
    return eta

def newton1D(x, f, j, tol=1e-12, maxIter=200):
    for i in range(maxIter):
        fx = f(x)
        jx = j(x)
        #print(f"  iter {i}: x={x:.6f}, f={fx:.6e}, j={jx:.6e}")
        if np.linalg.norm(fx) < tol:
            break
        x += -fx / jx
    return x

        
        
def kdv_main():
    N = 200
    time = 100
    dt = 0.05

    abstol = 0
    reltol = 1e-3
    maxiter = 10
    NewtonOpts = [abstol, reltol, maxiter]

    etamax = 0.9
    gamma = 0.9
    maxiter_G = 200
    gmresOpts = [etamax, maxiter_G, gamma]

    relax = True

    x, t, u, ent, err, resHist = kdvsolver(N, time, dt, NewtonOpts, gmresOpts, relax)

    plt.plot(t,ent)
    plt.title("Entropy over time")
    plt.savefig("Entropy_fig_resi")
    plt.show() 

    plt.plot(t,err)
    plt.title("Error over time")
    plt.savefig("Error_fig_resi")
    plt.show()

kdv_main()