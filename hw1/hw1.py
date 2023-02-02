import matplotlib.pyplot as plt
import numpy as np

kap = 2.37
rho = 2.7
c = 0.897
theta = np.pi

def u(x, t):
    return np.exp((-kap * theta**2 * t) / (rho*c)) * np.sin(theta*x)

def p2():
    l = 10
    ts = np.linspace(0, 20, 100)
    #print(ts)
    xs = np.linspace(0, l, 100)

    X, Y = np.meshgrid(xs, ts)
    Z = u(X, Y)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 75)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x, t)')
    ax.set_title('3D contour of u(x, t)')

    plt.show()


def C1(f, x0, h):
    return (f(x0 + h) - 2*f(x0) + f(x0 - h)) / (h**2)

def C2(f, x0, h):
    return (-f(x0 + 2*h) + 16*f(x0 + h) - 30*f(x0) + 16*f(x0 - h) - f(x0 - 2*h)) / (12 * h**2)

def p4():
    f = lambda x: np.exp(x)
    x0 = np.pi * 7/8
    h = 10**(-np.linspace(0, 16, 17))
    print(h)

    fx0 = f(x0)

    C1s = C1(f, x0, h)
    print(C1s)
    err1 = np.array([abs(fx0 - c1) for c1 in C1s])
    plt.title(r"Error of C1 approximation for f''")
    plt.xlabel("h")
    plt.ylabel("|C1(h) - f''(x0)|")
    plt.loglog(h, err1)
    plt.show()

    C2s = C2(f, x0, h)
    err2 = np.array([abs(fx0 - c2) for c2 in C2s])
    plt.title(r"Error of C2 approximation for f''")
    plt.ylabel("|C2(h) - f''(x0)|")
    plt.loglog(h, err2)
    plt.show()

def main():
    p4()

if __name__ == "__main__":
    main()