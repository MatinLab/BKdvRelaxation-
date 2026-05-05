import numpy as np
from scipy.linalg import solve
import scipy.sparse as sp
import matplotlib.pyplot as plt

def newtonDirect(x, f, J, entropy, atol, rtol, maxIter, relax):
    #Set up saving data
    Nx = len(x)
    ut_count = 0
    res_hist = []
    entropies = []

    #for line search and relax
    un = x[:,None]

    #evaluate function at initial iterate
    F_val = f(x)
    F_val_norm = np.linalg.norm(F_val)/np.sqrt(200)
    res_hist.append(F_val_norm)
    entropies.append(entropy(x))
    stop_tol = atol + rtol*F_val_norm

    #solve non-linear system using Newton Direct solve!
    for ns in range(maxIter+1):
        if F_val_norm < stop_tol:
            print(f"Newton Converged on Iteration: {ns-1}")
            break
        dx = sp.linalg.spsolve(J(x), -F_val)
        x += dx

        if relax:
            un = np.column_stack((un,x))

        F_val = f(x)
        F_val_norm = np.linalg.norm(F_val)/np.sqrt(200)
        res_hist.append(F_val_norm)
        entropies.append(entropy(x))
        if ns == maxIter:
            print("Max Iterations Reached")
    res_hist = np.array(res_hist)
    entropies = np.array(entropies)
    return x, res_hist, entropies, un


def newton1D(x, f, j, tol=1e-12, maxIter=200):
    for i in range(maxIter):
        fx = f(x)
        jx = j(x)
        #print(f"  iter {i}: x={x:.6f}, f={fx:.6e}, j={jx:.6e}")
        if np.linalg.norm(fx) < tol:
            break
        x += -fx / jx
    return x

def brugersMain(relaxN):

    #Initial data
    xmin = -10.0
    xmax = 10.0
    N = 200

    L = xmax - xmin
    delta_x = L / N
    xs = np.linspace(xmin, xmax, N)


    #Build centered difference periodic (D1 in julia)
    coeffs = np.array([1, -8, 0, 8, -1])
    indexes = np.array([-2, -1, 0, 1, 2])

    D1 = sp.lil_matrix((N,N))
    D1 = D1 /(12*delta_x)
    for k, index in enumerate(indexes):
        D1[np.arange(N), (np.arange(N) + index) % N] = coeffs[k]
    D1 = D1.tocsr()

    c = 2
    dt = 0.5
    def u0(x):
        return 0.5*c*(1/np.cosh(0.5*np.sqrt(c)*x)**2)


    initial = np.array([u0(xi) for xi in xs])

    def F(v, vn, h):
        return v - vn + h *(v*(D1 @ v) + D1 @ (v**2))
    
    
    def J(v):
        lenghtN = len(v)
        I = sp.eye(N, format="csr")

        return (I + dt*((sp.diags(v,0) @ D1) + (sp.diags(D1 @ v,0)) + (D1 @ sp.diags(2.0*v, 0))))

    def entropy(U):
        return 0.5*delta_x*np.sum((2.0*U - initial)**2)
    
    def entropy_prime(U):
        return 2.0*delta_x*(2*U -initial)
    
    def f(u):
        return F(u, initial, dt)
    
    def r(u_new, u_old, gamma):
        return entropy(u_old + gamma*(u_new - u_old)) - entropy(u_old) #(- gamma*(entropy(u_new) - entropy(u_old)))
    
    def r_prime(u_new, u_old, gamma):
        return np.dot(entropy_prime(u_old + gamma*(u_new - u_old)),(u_new - u_old)) #- (entropy(u_new) - entropy(u_old))
    
    
    
    u = initial.copy()

    atol = 0.0
    rtol = 10**(-13)
    maxIter = 20

    u, residuals, entropies, U = newtonDirect(u, f, J, entropy, atol, rtol, maxIter, True)

    N = len(initial)
    print(len(residuals))

    for k in range(1, len(residuals)):
        Uk = U[:,k]
        def res(gamma):
            return r(Uk, initial, gamma)
        
        def res_prime(gamma):
            return r_prime(Uk, initial, gamma)
        

        if relaxN:
            gamma = newton1D(0.9, res, res_prime)
            print(f"current gamma : {gamma} and its shape {gamma.shape}")
        else:
            gamma = (np.linalg.norm(initial)**2 - np.dot(Uk, initial)) / np.linalg.norm(Uk - initial)**2
            print(f"current gamma : {gamma} and its shape {gamma.shape}")

        Ug = initial + gamma*(Uk - initial)

        entropies[k] = entropy(Ug)
        residuals[k] = np.linalg.norm(F(Ug, initial, gamma*dt))/np.sqrt(N)
    
    return residuals, entropies


resi, entro = brugersMain(False)
resis = []
entrops = []
print(resi)
print(entro)
iterations = np.arange(len(resi))


plt.plot(iterations, resi)
plt.xlabel("Newton Iteration")
plt.ylabel("Residual Norm")
plt.grid(True)
plt.title("Residual Norm at each Iteration")
plt.show()
plt.plot(iterations, entro)
plt.ylim(0.95, 0.93)
plt.plot(iterations, resi)
plt.xlabel("Newton Iteration")
plt.ylabel("Entropy")
plt.grid(True)
plt.title("Calculated Entropy at each Iteration")

plt.show()


    


    
    
