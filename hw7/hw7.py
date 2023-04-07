import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

def p1():
    ns = [5, 35]
    for n in ns:
        A = np.zeros((n, n))
        for i in range(1, n+1):
            for j in range(1, n+1):
                A[i-1, j-1] = int(2*np.sin(i*pi/2)*np.sin(j*pi/2))
                if i == j:
                    A[i-1, j-1] += i**2 * pi**2
        f = np.zeros(n)
        for k in range(1, n+1):
            f[k-1] = -np.sqrt(2)*(np.cos(k*pi))/(k*pi)

        alpha = np.linalg.solve(A, f)
        def u(x):
            summa = 0
            for j in range(1, n+1):
                summa += alpha[j-1]*np.sqrt(2)*(np.sin(j*pi*x))
            return summa
        
        xs = np.linspace(0, 1, 100)
        plt.plot(xs, u(xs))
        plt.title(f"U(x) appx with {n} nodes")
        plt.xlabel("x")
        plt.ylabel("U_h(x)")
        if n == 5:
            plt.savefig("hw7_p1a.pdf")
        else:
            plt.savefig("hw7_p1b.pdf")
        plt.show()

def p2():
    N = 32
    a, b = 0, 1

    h = (b-a)/(N-1)

    epss = [0.1, 0.25, 1]
    M = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(1, N+1):
        for j in range(1, N+1):
            if i == j:
                M[i-1, j-1] = 2*h/3
                K[i-1, j-1] = 2/h
            if i == j-1 or i == j+1:
                M[i-1, j-1] = h/6
                K[i-1, j-1] = -1/h

    f = np.zeros(N)
    
    # add ghost nodes to x
    xs = np.linspace(a-h, b+h, N+2)

    # print(xs)
    for k in range(1, N+1):
        f[k-1] = 1/h * (0.5*xs[k+1]**2 + xs[k]**2 + 0.5*xs[k-1] - xs[k]*xs[k-1] - xs[k]*xs[k+1])

    # print(f)

    for eps in epss:
        A = eps*K + M

        alpha = np.linalg.solve(A, f)
        def phi(k, x):
            if xs[k-1] < x and x < xs[k]:
                return (x - xs[k-1])/h
            elif xs[k] < x and x < xs[k+1]:
                return (xs[k+1] - x)/h
            else:
                return 0

        def u(x):
            summa = 0
            for j in range(1, N+1):
                summa += alpha[j-1]*phi(j, x)
            return summa
        
        x_plotting = np.linspace(0, 1, 100)
        u_plotting = [u(x) for x in x_plotting]

        plt.plot(x_plotting, u_plotting)
        plt.title(f"U(x) appx eps = {eps}")
        plt.xlabel("x")
        plt.ylabel("U(x)")
        if eps == 0.1:
            plt.savefig("hw7_p2a.pdf")
        elif eps == 0.25:
            plt.savefig("hw7_p2b.pdf")
        else:
            plt.savefig("hw7_p2c.pdf")
        plt.show()
        


if __name__ == '__main__':
    p2()