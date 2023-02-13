import matplotlib.pyplot as plt
import numpy as np

def exp_euler(y0, yp, ts, h):
    ys = [y0]
    for i in range(1, len(ts)):
        yi = ys[i-1]
        ys.append(yi + h*yp(ts[i-1], yi))
    return ys

def f(t, y):
    return (1 + t) / (1 + y)

def y(t):
    return np.sqrt(t**2 + 2*t + 6) - 1

def p2():
    y0 = 2
    a, b = 1, 2
    h = 0.5
    ts = np.arange(a, b + h, h)
    y_appx = exp_euler(y0, f, ts, h)
    y_exact = y(ts)

    y_err = abs(y_appx - y_exact)
    print("The maximum error for this approximation is:", max(y_err))

    plt.plot(ts, y_appx, 'r-', label="Explicit Euler")
    plt.plot(ts, y_exact, 'b-', label="Exact")
    plt.title("Explicit Euler vs Analytic solution to IVP")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.savefig("hw2_p2b1.pdf")
    plt.show()

    plt.plot(ts, y_err)
    plt.title("Error of Euler Approx.")
    plt.xlabel("t")
    plt.ylabel("Absolute error")
    plt.savefig("hw2_p2b2.pdf")
    plt.show()

    return

def p3():
    y0 = 2
    a = 1
    hs = 10**np.linspace(-10, -5, 6)
    errs = []
    for h in hs:
        ts = np.arange(a, 1 + 1e-5, h)
        #print("h =", h)
        #print(ts)
        y_real = y(ts[-1])

        y_appx = exp_euler(y0, f, ts, h)[-1]
        errs.append(abs(y_real - y_appx))
    
    plt.loglog(hs, errs)
    plt.title("Error of Euler at t=1.00001")
    plt.xlabel("h")
    plt.ylabel("Absolute error")
    plt.savefig("hw2_p3.pdf")
    plt.show()
    return


if __name__ == '__main__':
    p3()