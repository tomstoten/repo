import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

def evaluateJ(yk, N, h):
    # Jacobian Matrix
    J = np.zeros((N,N))
    J[0][0] = 1
    J[N-1][N-1] = 1
    for i in range(1, N-1):
        J[i][i-1] = 1
        J[i][i] = -2 - 2*h**2 * np.exp(-2*yk[i])
        J[i][i+1] = 1
    return J

def nonlinearFD(a, b, ya, yb, f, evalJ, h, N, tol, nmax):
    y_l = np.array([ya + j * ((yb - ya)/(b - a)) * h for j in range(N)])
    # print(y_l)
    x_l = a
    def F(x, y_l, i):
        return (y_l[i - 1] - 2*y_l[i] + y_l[i+1]) - h**2 * f(y_l[i], x)
    
    for l in range(1, nmax):
        J_l = np.matrix(evalJ(y_l, N, h))
        
        F_l = [F(x_l, y_l, i) for i in range(1, len(y_l) - 1)]
        F_l.insert(0, 0)
        F_l.append(0)
        F_l = np.array(F_l)
        
        del_y = np.linalg.solve(J_l, -1*F_l)
        #print(del_y)
        y_l = y_l + del_y

        # check for convergence
        if np.linalg.norm(del_y) < tol:
            if(np.linalg.norm(F(x_l, y_l, l)) < tol):
                return y_l, 0
            return y_l, 1

        x_l += h
    return y_l, -1


def p1():
    def f(y, x):
        return -np.exp(-2*y)
    
    def y_exact(t):
        return np.log(t)
    
    a, b = 1, 2
    y1 = 0
    y2 = np.log(2)
    # print(y2)
    
    Ns = [9, 18]
    tol = 1e-8
    nmax = 100
    
    index = 'a'

    for N in Ns:
        h = (b - a)/(N-1)
        ts = np.linspace(a, b, N)
        ys = y_exact(ts)

        w, ier = nonlinearFD(a, b, y1, y2, f, evaluateJ, h, N, tol, nmax)
        print("ier:", ier)
        abs_err = abs(ys - w)
        rel_err = [abs_err[i] / ys[i] for i in range(len(ys))]

        plt.plot(ts, w, label="FD Appx")
        plt.plot(ts, ys, label="Exact")
        plt.title("FD appx of ODE ({} nodes)".format(N))
        plt.xlabel('t')
        plt.ylabel('y(t)')
        plt.legend()
        plt.show()

        plt.plot(ts, abs_err, label="Abs Err")
        plt.plot(ts, rel_err, label="Rel Err")
        plt.title("Error of FD appx to ODE ({} nodes, tol=1e-8)".format(N))
        plt.xlabel('t')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig(f'hw6_p1{index}_low_tol.pdf')
        plt.show()
        index = 'b'

# fd_bvp code copied here for problem 3
def eval_pqr(x):

     p = np.zeros(len(x))
     q = -np.pi*np.ones(len(x))
     r = np.zeros(len(x))
     
     return(p,q,r)     

def make_FDmatDir(f,x,h,N,alpha,beta,method=0):
#     # define method = 0 to be forward/backward diff,
#     # and method = 1 to be centered diff
#     p, q, r = eval_pqr(x)

# # create the finite difference matrix
#     # recasting to scipy sparse matrix storage

#     q_d = q[1:N]
#     q_d = np.insert(q_d, [0, 0], [0, 0])
#     q_d = np.append(q_d, [0, 0])
#     q_diag = sps.dia_matrix((q_d, [0]), shape=(N+3, N+3))
#     #print(q_diag.toarray())

    if method == 0:
        # forward/backward diff for boundary
        top_diag = 1/(h**2) * np.ones(N+1)
        bot_diag = top_diag.copy()
        main_diag = -2/(h**2) * np.ones(N+1)

        # set up first BC w forward diff
        top_diag = np.insert(top_diag, 0, -1/h)
        top_diag = np.insert(top_diag, 0, 0)

        # set up second BC w backward diff
        bot_diag = np.insert(bot_diag, len(bot_diag), 1/h)
        bot_diag = np.append(bot_diag, 0)

        main_diag = np.insert(main_diag, [0, len(main_diag)], [0, 0])

        toptop = np.zeros(N+2)
        toptop = np.insert(toptop, 2, 1/h)

        botbot = np.zeros(N+2)
        botbot = np.insert(botbot, len(botbot) - 2, -1/h)

        data = np.array([toptop, top_diag, main_diag, bot_diag, botbot])
        # print(data)
        offsets = [2, 1, 0, -1, -2]
        A = sps.dia_matrix((data, offsets), shape=(N+3, N+3))
        # print(A.toarray())
    elif method == 1:
        # forward/backward diff for boundary
        top_diag = 1/(h**2) * np.ones(N+1)
        bot_diag = top_diag.copy()
        main_diag = -2/(h**2) * np.ones(N+1)

        # set up first BC w forward diff
        top_diag = np.insert(top_diag, 0, 0)
        top_diag = np.insert(top_diag, 0, 0)

        # set up second BC w backward diff
        bot_diag = np.insert(bot_diag, len(bot_diag), 0)
        bot_diag = np.append(bot_diag, 0)

        main_diag = np.insert(main_diag, [0, len(main_diag)], [-1/(2*h), 1/(2*h)])

        toptop = np.zeros(N+2)
        toptop = np.insert(toptop, 2, 1/(2*h))

        botbot = np.zeros(N+2)
        botbot = np.insert(botbot, len(botbot) - 2, -1/(2*h))

        data = np.array([toptop, top_diag, main_diag, bot_diag, botbot])
        # print(data)
        offsets = [2, 1, 0, -1, -2]
        A = sps.dia_matrix((data, offsets), shape=(N+3, N+3))
        # print(A.toarray())


# create the right hand side rhs: (N+1) in size
    rhs = f(x)
# add BC in, N+3 in size now
    rhs = np.insert(rhs, 0, alpha)
    rhs = np.append(rhs, beta)
    
# solve for the approximate solution

    sol = sps.linalg.lsqr(A, rhs)[0]
    #print(sol)
    
    yapp = sol[1:N+2]

    return yapp

def p3():
    # neumann BVP
    def y_exact(x):
        return np.sin(np.pi*x)
    
    def f(x):
        return -np.pi**2 * np.sin(np.pi * x)
    
    a, b = 0, 1
    ypa, ypb = np.pi, -np.pi
    hs = 0.1 * 2**(-np.linspace(0,4,5))
    index = 1
    for h in hs:
        N = int((b - a) / h)
        x = np.linspace(a, b, N+1)

        yapp_fd = make_FDmatDir(f, x, h, N, ypa, ypb, method=0)
        yapp_cd = make_FDmatDir(f, x, h, N, ypa, ypb, method=1)

        yex = y_exact(x)

        # subtract const from endpoint

        diff_fd = yapp_fd[0] - yex[0]
        yapp_fd -= diff_fd

        diff_cd = yapp_cd[0] - yex[0]
        yapp_cd -= diff_cd

        # plt.plot(x,yapp,label = 'FD aprox')
        # plt.plot(x,yex,label = 'Exact')
        # plt.title("FD vs Exact soln to ODE (h = {})".format(h))
        # plt.xlabel('x')
        # plt.ylabel('y(x)')
        # plt.legend(loc = 'upper left')
        # plt.show()
     
        err_fd = abs(yex - yapp_fd)
        err_cd = abs(yex - yapp_cd)
        plt.plot(x,err_fd,label = 'Abs Err Forward Diff.')
        plt.plot(x,err_cd,label = 'Abs Err Centered Diff.')
        plt.title("Abs error of FD appx to ODE (h = {})".format(h))
        plt.xlabel('x')
        plt.ylabel('abs err')
        plt.legend(loc = 'upper left')
        plt.savefig(f"hw6_p3_{index}.pdf")
        plt.show()

        index += 1
        
    return 0

if __name__ == '__main__':
    p3()