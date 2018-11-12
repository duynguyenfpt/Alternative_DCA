import numpy as np
import cplex
from scipy.sparse.linalg import eigs
import math

def genRandomMatrix(num_row, num_col):
    # calculate normal distribution
    np.random.seed(100)
    mu, sigma = 0, 1.0
    return np.matrix(np.random.random((num_row,num_col)))

# ============================= Start reading data from files ===================================
data = np.load('data/Testdata_Exp1_4_34.npz')
N = np.int(data['N'])
f1 = np.matrix(data['f1'])
f2 = np.matrix(data['f2'])
sigmaR = np.int(data['sigmaR'])
sigma1 = np.int(data['sigma1'])
sigma2 = np.int(data['sigma2'])
L = np.matrix(data['L'])
sigma1E = np.int(data['sigma1E'])
sigma2E = np.int(data['sigma2E'])
F1 = np.matrix(data['F1'])
F2 = np.matrix(data['F2'])
g1 = np.complex128(data['g1'])
g2 = np.complex128(data['g2'])
D1 = np.matrix(data['D1'])
D2 = np.matrix(data['D2'])
Z = np.matrix(data['Z'])
Pt = np.int(data['Pt'])
print_solving_process = None
e = 0.00001
tau = 100

# G(w) = (w.H)Dw +ln(C)
P1 = P2 = Pt/4
R1 = F1 * f2 * f2.H * F1.H
R2 = F2 * f1 * f1.H * F2.H

I = np.identity(N) #I is unit matrix

A1 = P2 * R1 + sigmaR * sigmaR * D1
A2 = P1 * R2 + sigmaR * sigmaR * D2

B1 = sigmaR * sigmaR * D1
B2 = sigmaR * sigmaR * D2



def get_largest_eigenvalue(aMatrix):
    values, vector = eigs(aMatrix,1,which="LM")
    return values[0]

def make_hat_matrix(Matrix):
    Mat_hat = np.vstack((np.hstack((Matrix.real, -Matrix.imag)),
                         np.hstack((Matrix.imag, Matrix.real))))
    return Mat_hat

A1_hat = make_hat_matrix(A1)
B1_hat = make_hat_matrix(B1)
A2_hat = make_hat_matrix(A2)
B2_hat = make_hat_matrix(B2)
D1_hat = make_hat_matrix(D1)
D2_hat = make_hat_matrix(D2)
ZH_hat = make_hat_matrix(Z.H)

ro = get_largest_eigenvalue(A1_hat/(2*sigma1*sigma1) + A2_hat/(2*sigma2*sigma2)      \
                            +2*(B1_hat/(sigma1*sigma1))+ 2*(B2_hat/(sigma1*sigma1)))
#ro = np.real(ro)

def compute_yk(aMatrix):
    aMatrix=np.matrix(aMatrix)
    return ro*aMatrix + (2*A1_hat*aMatrix) / (sigma1*sigma1+aMatrix.getT*A1_hat*aMatrix)\
                      + (2*A2_hat*aMatrix) / (sigma1*sigma1+aMatrix.getT*A2_hat*aMatrix)\
                      - (2*B1_hat*aMatrix) / (sigma1*sigma1+aMatrix.getT*B1_hat*aMatrix)\
                      - (2*B2_hat*aMatrix) / (sigma1*sigma1+aMatrix.getT*B2_hat*aMatrix)
def sqr_absolute_val(x):
    return np.real(x) * np.real(x) + np.imag(x) * np.imag(x)

def compute_F(aMatrix):
    aMatrix = np.matrix(aMatrix)
    tmp  =  1+(P1*sqr_absolute_val(g1) + P2*sqr_absolute_val(g2))/(sigma1E*sigma1E)
    tmp1 =   math.log(sigma1*sigma1 + (aMatrix.T*A1_hat*aMatrix).item()) \
           + math.log(sigma1*sigma1 + (aMatrix.T*A1_hat*aMatrix).item()) \
           - math.log(sigma1*sigma1 + (aMatrix.T*B1_hat*aMatrix).item()) \
           - math.log(sigma1*sigma1 + (aMatrix.T*B2_hat*aMatrix).item())
    return tmp - tmp1

#==================== DEFINE LIST OF VAR NAME ======================
list_var_name = []

# add x1, x2, ...,x2N of vector x
for i in range(1,2*N+1):
    list_var_name.append("x"+str(i))

#print(list_var_name) - Check Fine

# if(debug):
#     print("List of variable names:", list_var_name)
#
# ====================== DEFINE MODEL ======================

def def_model(aMatrix):
    # define model
    model = cplex.Cplex()

    if(not print_solving_process):
        model.set_results_stream(None)
    model.objective.set_sense(model.objective.sense.minimize)

    # lower bound
    lb = [-cplex.infinity]*(2*N)

    # qmat: quadratic term of objective function, must be symmetric
    obj_quad = []
    ro = get_largest_eigenvalue(A1_hat/2 + A2_hat/2 +2*B1_hat+ 2*B2_hat)
    print(ro)
    ##G(x) = 1/2 ||x|| ^2
    #||x|| ^2 = x^T. I . x
    # add elements in diagonal of Identity to a sparse list of cplex format
    for i in range(2*N):
        row = [[list_var_name[i]], [ro/2]]
        obj_quad.append(row)
    print(obj_quad)

    # set linear objective to model and lb
    model.variables.add(lb=lb, names=list_var_name)
    # set quadratic objective
    model.objective.set_quadratic(obj_quad)
    ##set linear Yk^T (x-xk) = Yk^T*x (linear) - Yk^T*xk(const)
    obj_linear= []
    yk = compute_yk(aMatrix)
    for i in range(2*N):
        model.objective.set_linear(str(i),yk[i])

    return model
    #print("Quadratic term of model:\n", model.objective.get_quadratic())

# =================================================================


def solving():
    # ============================= Finish reading data from files ===================================
    I = np.identity(N)  # I is unit matrix
    A1 = P2 * R1 + sigmaR * sigmaR * D1
    A2 = P1 * R2 + sigmaR * sigmaR * D2

    B1 = sigmaR * sigmaR * D1
    B2 = sigmaR * sigmaR * D2

    D = A1 / (4 * sigma1 * sigma1) + A2 / (4 * sigma2 * sigma2) + B1 / (sigma1 * sigma1) + B2 / (sigma2 * sigma2)
    C = 1 + (P1 * pow(np.abs(g1), 2) + P2 * pow(np.abs(g2), 2)) / pow(sigma1E, 2)

    # variables - vector w
    #w = cvx.Variable(shape=(N, 1), complex=True)

    xk = genRandomMatrix(2*N, 6)
    zk = xk[:,0]
    tk = 1
    tk_next = tk
    k = 0
    while (1):
        vk = zk
        tk_next = np.sqrt(1+4*tk*tk)/2
        if (k>=1):
            zk = xk[:,k]+ (xk[:,k]-xk[:,k-1]) *(tk-1)/(tk_next)
        check = False
        F_zk = compute_F(zk)
        for i in range(max(0,k-5),k):
            if (compute_F(xk[:,i-max(k-5,0)])>F_zk):
                check =True
                break
        if (check):
            vk = zk
        else:
            vk = xk[:,min(5,k)]

        problem_model = def_model(vk)
        #reset constrain
        problem_model.quadratic_constraints.delete()
        #power constraint
        list1 =[]
        list2 =[]
        list3 =[]
        quadratic_cs = P1*D1_hat+P2*D2_hat+np.identity(2*N)
        for i in range(0,2*N):
            for j in range(0,i+1):
                list1.append(list_var_name[i])
                list2.append(list_var_name[j])
                if (i==j):
                    list3.append(quadratic_cs[i][j])
                else:
                    list3.append(quadratic_cs[i][j]+quadratic_cs[j][i])


        quad = [cplex.SparseTriple(list1,list2,list3)]
        qsense = ["L"]
        qrhs = [Pt-P1-P2]
        #Z_hat. x = 0 constraint

        lin_expr =[]
        lin_senses = []
        lin_rhs = []
        for i in range(0,4):
            lin_list1=[]
            lin_list2=[]
            for j in range(2*N):
                lin_list1.append(list_var_name[j])
                lin_list2.append(ZH_hat[i][j])
            lin_expr.append(cplex.SparsePair(lin_list1,lin_list2))
            lin_rhs.append(0)
            lin_senses.append("E")

        problem_model.linear_constraints.add(lin_expr=lin_expr,senses=lin_senses,rhs=lin_rhs)
        problem_model.quadratic_constraints.add(quad_expr=quad,rhs=qrhs,sense=qsense)
        problem_model.solve()
        if (problem_model.solution.status.infeasible):
            break
        xk_next = problem_model.solution.get_values()
        if (k<5):
            xk[:,k+1] = xk_next
        else:
            for i in range(0,4):
                xk[:,k]=xk[:,k+1]
            xk[:,5]= xk_next

        if (abs(compute_F(xk_next)-compute_F(xk[:,math.min(4,k)]))/(1+compute_F(xk[:,math.min(4,k)])) < e):
            print("Stop at 2nd condional loop")
            print("Result: "+str(compute_F(xk_next)))
            break
        if (np.linalg.norm(xk_next-xk[:,math.min(4,k)])/(1+np.linalg.norm(xk[:,math.min(4,k)])*np.linalg.norm(xk[:,math.min(4,k)]))<e):
            print("Stop at 1nd condional loop")
            print("Result: " + str(compute_F(xk_next)))
            break

solving()