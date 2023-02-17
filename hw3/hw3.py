import matplotlib.pyplot as plt
import numpy as np

def f(t, y):
    return 1/t**2 - y/t - y**2

def dfdt(t, y):
    return -2/t**3 + y/t**2

def dfdt_be(t, y, dt):
    b = (1/t + 1/dt)
    return (-b + np.sqrt(b**2 - 4*(1/t**2 + y/dt))) / 2

def dfdt_cd(t, y, dt, f):
    return (f(t + dt, y) - f(t - dt, y)) / 2

def dfdy(t, y):
    return -1/t - 2*y

def y(t):
    return -1/t

def taylor2(y0, f, h, ts):
    ys = [y0]
    for i in range(1, len(ts)):
        yi = ys[i-1]
        ti = ts[i-1]
        fi = f(ti, yi)
        yi1 = yi + h*fi + h**2 * 0.5 * (dfdt(ti, yi) + dfdy(ti, yi)*fi)
        ys.append(yi1)
    return ys

def taylor2_be(y0, f, h, ts, dt):
    ys = [y0]
    for i in range(1, len(ts)):
        yi = ys[i-1]
        ti = ts[i-1]
        fi = f(ti, yi)
        yi1 = yi + h*fi + h**2 * 0.5 * (dfdt_be(ti, yi, dt) + dfdy(ti, yi)*fi)
        ys.append(yi1)
    return ys

def taylor2_cd(y0, f, h, ts, dt):
    ys = [y0]
    for i in range(1, len(ts)):
        yi = ys[i-1]
        ti = ts[i-1]
        fi = f(ti, yi)
        yi1 = yi + h*fi + h**2 * 0.5 * (dfdt_cd(ti, yi, dt, f) + dfdy(ti, yi)*fi)
        ys.append(yi1)
    return ys

def p1():
    y0 = -1
    a, b = 1, 2
    h = 0.05
    ts = np.arange(a, b + h, h)
    y_appx = taylor2(y0, f, h, ts)
    print(y_appx)

    y_exact = y(ts)
    print(y_exact)
    return

    y_err = abs(y_appx - y_exact)

    plt.plot(ts, y_appx, 'r-', label="Taylor")
    plt.plot(ts, y_exact, 'b-', label="Exact")
    plt.title("2nd order Taylor vs Analytic solution to IVP")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.savefig("hw3_p1a1.pdf")
    plt.show()
    
    plt.plot(ts, y_err)
    plt.title("Error of Taylor Approx.")
    plt.xlabel("t")
    plt.ylabel("Absolute error")
    plt.savefig("hw3_p1a2.pdf")
    plt.show()

    return

def p2():
    y0 = -1
    a, b = 1, 2
    hs = 2**(-1*np.linspace(1, 16, 16))
    y_apps = []
    y_bes = []
    y_cds = []
    for h in hs:
        dt = h / 2
        ts = np.arange(a, b + h, h)
        y_apps.append(taylor2(y0, f, h, ts)[-1])
        y_bes.append(taylor2_be(y0, f, h, ts, dt)[-1])
        y_cds.append(taylor2_cd(y0, f, h, ts, dt)[-1])
    #print(y_appx)

    y_exact = y(ts)[-1]
    #print(y_exact)

    # y_err = abs(y_appx - y_exact)

    plt.loglog(hs, abs(y_apps - y_exact), 'r-', label="Taylor")
    plt.loglog(hs, abs(y_bes - y_exact), 'g-', label="Taylor w Backwards Euler")
    plt.loglog(hs, abs(y_cds - y_exact), 'b-', label="Taylor w Centered Diff.")
    #plt.axvline(x = y_exact, label="Exact")
    plt.title("Error of appx at t=2 with respect to h")
    plt.suptitle("dt = h/2")
    plt.xlabel("h")
    plt.ylabel("absolute error")
    plt.legend()
    plt.savefig("hw3_p21.pdf")
    plt.show()
    
    # plt.plot(ts, y_err)
    # plt.title("Error of Taylor Approx.")
    # plt.xlabel("t")
    # plt.ylabel("Absolute error")
    # plt.savefig("hw3_p1a2.pdf")
    # plt.show()

    return




if __name__ == '__main__':
    p2()