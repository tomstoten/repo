import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.linalg as spl
import scipy.sparse as sps


def driver():

# this demo code considers the boundary value problem
# y'' = p(x)y'+q(x)y + r(x)
#y(a) = alpha, y(b) = beta

# boundary data
     a = 0
     b = 1
     alpha = 1
     beta = np.exp(2)

     #  exact solution 
     y = lambda x: np.exp(2*x)

     # step size
     hs = [10**(-i) for i in range(2, 6)]
     norms = []
     index = 1
     for h in hs:
          N = int((b-a)/h)
          
          x = np.linspace(a,b,N+1)
          
          yapp = make_FDmatDir(x,h,N,alpha,beta)
          
          yex = y(x)
     
          # plt.plot(x,yapp,label = 'FD aprox')
          # plt.plot(x,yex,label = 'Exact')
          # plt.title("FD vs Exact soln to ODE (h = {})".format(h))
          # plt.xlabel('x')
          # plt.ylabel('y(x)')
          # plt.legend(loc = 'upper left')
          # plt.show()
          
          err = np.zeros(N+1)
          for j in range(0,N+1):
               err[j] = abs(yapp[j]-yex[j])
               
          norms.append(np.linalg.norm(err))
          # print('err = ', err)
          
     plt.loglog(hs, norms)
     plt.title("Norm of Abs Error as a function of h")
     plt.xlabel('h')
     plt.ylabel('||err||')
     plt.savefig('hw6_p2.pdf')
     plt.show()
     return
     
def eval_pqr(x):

     p = np.zeros(len(x))
     q = np.zeros(len(x))
     r = np.array([4*np.exp(2*xi) for xi in x])
     
     return(p,q,r)     



def make_FDmatDir(x,h,N,alpha,beta):

# evaluate coefficients of differential equation
     (p,q,r) = eval_pqr(x)
     
 
# create the finite difference matrix     
     # recasting to scipy sparse matrix storage
     ex = 1/h**2 * np.ones(N-1)
     data = np.array([-ex, 2*ex, -ex])
     offsets = [-1, 0, 1]
     Matypp = sps.dia_matrix((data, offsets), shape=(N-1, N-1))
     #print(Matypp.toarray())
     '''
     p and q are 0 for this problem: scrub them from the calculation
     '''

     # ex = 1/(2*h) * np.ones(N-1)
     # data = np.array([-ex, ex])
     # offsets = [-1, 1]
     # Matyp = sps.dia_matrix((data, offsets), shape=(N-1, N-1))
     
     # q_diag = sps.dia_matrix((q[1:N], [0]), shape=(N-1, N-1))
     # #print(q_diag.toarray())
     # p_diag = sps.dia_matrix((p[1:N], [0]), shape=(N-1, N-1))
     # prod = sps.csr_array(np.matmul(p_diag.toarray(), Matyp.toarray()))
     #print(prod.toarray())
     
     #A = Matypp + np.matmul(np.diag(p[1:N],0),Matyp)+ np.diag(q[1:N])
     A = Matypp.tocsr()   # + prod + q_diag
     #print(A.toarray())
     # condN = spl.expm_cond(A.toarray())
     # print("The condition number is:", condN)

# create the right hand side rhs: (N-1) in size
     rhs = -r[1:N]
#  update with boundary data   
     rhs[0] = rhs[0] + (1/h**2-1/(2*h)*-p[1])*alpha
     rhs[N-2] = rhs[N-2] + (1/h**2+1/(2*h)*-p[N-1])*beta
     
# solve for the approximate solution

     sol = sps.linalg.spsolve(A, rhs)
     
     yapp = np.zeros(N+1)
     yapp[0] = alpha
     for j in range(1,N):
         yapp[j] = sol[j-1]
     yapp[N] = beta    

     return yapp
     
driver()     
