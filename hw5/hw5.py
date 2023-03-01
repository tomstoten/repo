import numpy as np
import matplotlib.pyplot as plt

# this function from Amirhossein Rezaei on Stack Overflow
# NOT MY CODE
def rkf( f, a, b, x0, tol, hmax, hmin ):

    a2  =   2.500000000000000e-01  #  1/4
    a3  =   3.750000000000000e-01  #  3/8
    a4  =   9.230769230769231e-01  #  12/13
    a5  =   1.000000000000000e+00  #  1
    a6  =   5.000000000000000e-01  #  1/2

    b21 =   2.500000000000000e-01  #  1/4
    b31 =   9.375000000000000e-02  #  3/32
    b32 =   2.812500000000000e-01  #  9/32
    b41 =   8.793809740555303e-01  #  1932/2197
    b42 =  -3.277196176604461e+00  # -7200/2197
    b43 =   3.320892125625853e+00  #  7296/2197
    b51 =   2.032407407407407e+00  #  439/216
    b52 =  -8.000000000000000e+00  # -8
    b53 =   7.173489278752436e+00  #  3680/513
    b54 =  -2.058966861598441e-01  # -845/4104
    b61 =  -2.962962962962963e-01  # -8/27
    b62 =   2.000000000000000e+00  #  2
    b63 =  -1.381676413255361e+00  # -3544/2565
    b64 =   4.529727095516569e-01  #  1859/4104
    b65 =  -2.750000000000000e-01  # -11/40

    r1  =   2.777777777777778e-03  #  1/360
    r3  =  -2.994152046783626e-02  # -128/4275
    r4  =  -2.919989367357789e-02  # -2197/75240
    r5  =   2.000000000000000e-02  #  1/50
    r6  =   3.636363636363636e-02  #  2/55

    c1  =   1.157407407407407e-01  #  25/216
    c3  =   5.489278752436647e-01  #  1408/2565
    c4  =   5.353313840155945e-01  #  2197/4104
    c5  =  -2.000000000000000e-01  # -1/5

    t = a
    x = np.array(x0)
    h = hmax

    T = np.array( [t] )
    X = np.array( [x] )
    
    while t < b:

        if t + h > b:
            h = b - t

        k1 = h * f( x, t )
        k2 = h * f( x + b21 * k1, t + a2 * h )
        k3 = h * f( x + b31 * k1 + b32 * k2, t + a3 * h )
        k4 = h * f( x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h )
        k5 = h * f( x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h )
        k6 = h * f( x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, \
                    t + a6 * h )

        r = abs( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 ) / h
        if len( np.shape( r ) ) > 0:
            r = max( r )
        if r <= tol:
            t = t + h
            x = x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
            T = np.append( T, t )
            X = np.append( X, [x], 0 )

        h = h * min( max( 0.84 * ( tol / r )**0.25, 0.1 ), 4.0 )

        if h > hmax:
            h = hmax
        elif h < hmin:
            raise RuntimeError("Error: Could not converge to the required tolerance %e with minimum stepsize  %e." % (tol,hmin))
            break

    return ( T, X )

# Start of my code
def RK4(f, a, b, y0, h):
    ws = [y0]
    ts = np.arange(a, b + h, h) # get t values from a to b with step h
    for i in range(1, len(ts)):
        wi = ws[i-1]
        ti = ts[i-1]
        k1 = h*f(wi, ti)
        k2 = h*f(wi + k1/2, ti + h/2)
        k3 = h*f(wi + k2/2, ti + h/2)
        k4 = h*f(wi + k3, ti + h)

        wip1 = wi + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        ws.append(wip1)

    print(ti + h)
    return ts, ws

def p4():
    def f(y, t): 
        return -(y+1)*(y+3)

    def y_exact(t):
        return -3 + 2/(1 + np.exp(-2*t))

    a, b = 0, 3
    hmin = 0.05
    hmax = 0.5
    tol = 1e-6

    y0 = -2

    T, X = rkf(f, a, b, y0, tol, hmax, hmin)
    y = y_exact(T)

    t_min, w_min = RK4(f, a, b, y0, hmin)
    t_max, w_max = RK4(f, a, b, y0, hmax)

    print("RKF took", len(T), "steps")

    plt.plot(T, X, label="RKF")
    plt.plot(T, y, label="Exact")
    plt.plot(t_min, w_min, label="RK4, h=.05")
    # plt.plot(t_max, w_max, label="RK4, h=.5")
    plt.title("Solution to PDE")
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.savefig('hw5_p4b3.pdf')
    plt.show()

    y_err = abs(y - X)
    rk4_min_err = abs(y_exact(t_min) - w_min)
    rk4_max_err = abs(y_exact(t_max) - w_max)
    print("y(3.0) =", y[-1])
    print("rkf(3.0) =", X[-1])
    print("rk4(3.0) =", w_min[-1])
    print("The abs error for RKF at t=3 is:", y_err[-1])
    print("The abs error for RK4 h=.05 at t=3 is:", rk4_min_err[-1])
    print("The abs error for RK4 h=.5  at t=3 is:", rk4_max_err[-1])
    plt.plot(T, y_err, label="RKF err")
    plt.plot(t_min, rk4_min_err, label="RK4 err, h=.05")
    # plt.plot(t_max, rk4_max_err, label="RK4 err, h=.5")
    plt.title("Abs. error of RKF and RK4 solutions")
    plt.xlabel('t')
    plt.ylabel('abs. error')
    plt.legend()
    plt.savefig('hw5_p4b4.pdf')
    plt.show()

def exp_euler(u10, u20, f1, f2, ts, h):
    u1s = [u10]
    u2s = [u20]
    for i in range(1, len(ts)):
        u1i = u1s[i-1]
        u2i = u2s[i-1]
        u1s.append(u1i + h*f1(ts[i-1], u1i, u2i))  # Euler approx on u1
        u2s.append(u2i + h*f2(ts[i-1], u1i, u2i))  # Euler approx on u2
    return u1s, u2s

def RK4_soe(a, b, m, h, u10, u20, fs):
    t = a
    ws = [u10, u20]
    u1_appx = [u10]
    u2_appx = [u20]
    N = int((b - a)/h)
    for i in range(N):
        k1 = []
        for j in range(m):
            k1.append(h*fs[j](t, ws[0], ws[1]))
        k2 = []
        for j in range(m):
            k2.append(h*fs[j](t + h/2, ws[0] + k1[0]/2, ws[1] + k1[1]/2))
        k3 = []
        for j in range(m):
            k3.append(h*fs[j](t + h/2, ws[0] + k2[0]/2, ws[1] + k2[1]/2))
        k4 = []
        for j in range(m):
            k4.append(h*fs[j](t + h, ws[0] + k3[0], ws[1] + k3[1]))
        
        for j in range(m):
            ws[j] += (k1[j] + 2*k2[j] + 2*k3[j] + k4[j])/6

        u1_appx.append(ws[0])
        u2_appx.append(ws[1])
        t += h
    return u1_appx, u2_appx

def p5():
    def f1(t, u1, u2):
        return u2
    def f2(t, u1, u2):
        return 3*u2 - 2*u1 + 6*np.exp(-t)
    
    def y_exact(t):
        return 2*np.exp(2*t) - np.exp(t) + np.exp(-t)
    
    u10 = 2
    u20 = 2
    a, b = 0, 1
    h = 0.1

    ts = np.arange(a, b + h, h)

    u1_appx, u2appx = exp_euler(u10, u20, f1, f2, ts, h)
    rk4_appx, _ = RK4_soe(a, b, 2, h, u10, u20, [f1, f2])

    ys = y_exact(ts)

    #plt.plot(ts, u1_appx, label="Euler")
    plt.plot(ts, ys, label="Exact")
    plt.plot(ts, rk4_appx, label="RK4")
    plt.title("Solution to PDE")
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.savefig('hw5_p5c1.pdf')
    plt.show()

    y_err = abs(ys - u1_appx)
    y_err_rk4 = abs(ys - rk4_appx)
    plt.plot(ts, y_err_rk4)
    plt.title("Absolute error of Runge-Kutta appx")
    plt.xlabel('t')
    plt.ylabel('abs error')
    # plt.legend()
    plt.savefig('hw5_p5c2.pdf')
    plt.show()



if __name__ == '__main__':
    p5()