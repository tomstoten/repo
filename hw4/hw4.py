import matplotlib.pyplot as plt
import numpy as np

def f(t, y):
    return 1 - y

def diff_method(w0, w1, f, h, ts):
    ws = [w0, w1]
    for i in range(2, len(ts)):
        wi1 = ws[i-1]
        wi = ws[i-2]

        ti = ts[i-2]
        fi = f(ti, wi)
        wi2 = 4*wi1 - 3*wi - 2*h*fi
        ws.append(wi2)
    return ws


def p3():
    y0 = 0
    y1 = 1 - np.exp(-0.1)
    a, b = 0, 1
    h = 0.1
    ts1 = np.arange(a, b + h, h)
    y_appx1 = diff_method(y0, y1, f, h, ts1)

    h = 0.01
    ts2 = np.arange(a, b + h, h)
    y1 = 1 - np.exp(-0.01)
    y_appx2 = diff_method(y0, y1, f, h, ts2)

    y = lambda t: 1 - np.exp(-t)

    y_exact = y(ts2)

    plt.plot(ts1, y_appx1, 'r-', label="h = 0.1")
    #plt.plot(ts2, y_appx2, 'b-', label="h = 0.01")
    plt.plot(ts2, y_exact, color='black', label="exact")
    plt.title("Difference method approximation for different h")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    #plt.savefig("hw4_p3c.pdf")
    plt.show()

    return


if __name__ == '__main__':
    p3()