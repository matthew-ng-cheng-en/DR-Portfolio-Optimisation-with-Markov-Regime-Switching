from pulp import *
import numpy as np

def optimisePortfolio(M, K, L, N, conf, w, theta, p, targ, r):
    """
    Parameters:
    ==========
    
    M - dimension of returns (int)
    K - number of regimes (int)
    L - number of clusters in each regime (dict; k)
    N - number of returns in each cluster in each regime (dict; k,l)
    conf - confidence level for cVaR (float)
    w - probability of each regime (dict; k,l)
    theta - ambiguity level for each cluster in each regime (dict; k)
    p - probability of each cluster (dict; k,l)
    targ - target return (float)
    r - empirical returns grouped by cluster and regime (dict; k,l,n)
    
    """

    # variables
    v = LpVariable('v', None, None)
    x = {i: LpVariable('x({})'.format(i), 0, None) for i in range(1,M+1)} # portfolio weights, no shorting constraint
    alpha = {}
    beta = {}
    a = {}
    b = {}
    c = {}
    eta = {}
    nu = {}

    for k in range(1,K+1):
        for l in range(1,L[k]+1):
            beta[(k,l)] = LpVariable('beta({},{})'.format(k,l), 0, None)
            nu[(k,l)] = LpVariable('nu({},{})'.format(k,l), 0, None)

            for n in range(1,N[(k,l)]+1):
                alpha[(k,l,n)] = LpVariable('alpha({},{},{})'.format(k,l,n), None, None)
                a[(k,l,n)] = LpVariable('a({},{},{})'.format(k,l,n), None, None)
                c[(k,l,n)] = LpVariable('c({},{},{})'.format(k,l,n), 0, None)
                eta[(k,l,n)] = LpVariable('eta({},{},{})'.format(k,l,n), None, None)


                for i in range(1,M+1):
                    b[(k,l,n,i)] = LpVariable('b({},{},{},{})'.format(k,l,n,i), None, None)

    # problem
    prob = LpProblem('mean_cVaR', LpMinimize)

    # auxiliary lists used for objective function and constraints
    obj1 = []
    obj2 = []
    targ1 = []
    targ2 = []

    con1 = []
    con2 = []
    con3 = []
    con4 = []

    for k in range(1,K+1):
        for l in range(1,L[k]+1):
            obj1.append(w[k]*theta[(k,l)]*p[(k,l)]*beta[(k,l)])
            targ1.append(-w[k]*theta[(k,l)]*p[(k,l)]*nu[k,l])

            for n in range(1,N[(k,l)]+1):
                obj2.append(w[k]*p[(k,l)]*(1/N[(k,l)])*alpha[(k,l,n)])
                targ2.append(w[k]*p[(k,l)]*(1/N[(k,l)])*eta[(k,l,n)])

                for i in range(1,M+1):
                    aux1 = r[(k,l,n,i)]*b[(k,l,n,i)]
                    aux2 = r[(k,l,n,i)]*x[i]
                    con1.append(-aux1)
                    con2.append(aux1 + aux2)
                    con3.append(aux1)
                    con4.append(aux2)


    # objective function
    prob += v+(1/(1-conf))*(lpSum(obj1) + lpSum(obj2))

    # constraints
    prob += lpSum([x[i] for i in range(1,M+1)]) == 1, 'Weights Sum to 1'
    prob += lpSum(targ1) + lpSum(targ2) >= targ, 'Target Return'

    for k in range(1,K+1):
        for l in range(1,L[k]+1):
            for n in range(1,N[(k,l)]+1):
                prob += lpSum(con1) - a[(k,l,n)] + alpha[(k,l,n)] >= 0, 'Set 1 Constraint 1 ({},{},{})'.format(k,l,n)
                prob += beta[(k,l)] - c[(k,l,n)] >= 0, 'Set 1 Constraint 2 ({},{},{})'.format(k,l,n)

                prob += lpSum(con2) + a[(k,l,n)] + v >= 0, 'Set 2 Constraint 1 ({},{},{})'.format(k,l,n)
                # set 2 constraint 2 included in positivity condition of c

                prob += lpSum(con3) + a[(k,l,n)] >= 0, 'Set 3 Constraint 1 ({},{},{})'.format(k,l,n)
                # set 3 constraint 2 included in positivity condition of c

                prob += lpSum(con4) - eta[(k,l,n)] >= 0, 'Set 4 Constraint 1 ({},{},{})'.format(k,l,n)
                # set 4 constraint 2 included in positivity condition of nu

                for i in range(1,M+1):
                    prob += b[(k,l,n,i)] + beta[(k,l)] - c[(k,l,n)] >=0, 'Set 1 Constraint 3 ({},{},{},{})'.format(k,l,n,i)
                    prob += -b[(k,l,n,i)] + beta[(k,l)] - c[(k,l,n)] >=0, 'Set 1 Constraint 4 ({},{},{},{})'.format(k,l,n,i)

                    prob += -b[(k,l,n,i)] -x[i] + c[(k,l,n)] >=0, 'Set 2 Constraint 3 ({},{},{},{})'.format(k,l,n,i)
                    prob += b[(k,l,n,i)] + x[i] + c[(k,l,n)] >=0, 'Set 2 Constraint 4 ({},{},{},{})'.format(k,l,n,i)

                    prob += -b[(k,l,n,i)] + c[(k,l,n)] >=0, 'Set 3 Constraint 3 ({},{},{},{})'.format(k,l,n,i)
                    prob += b[(k,l,n,i)] + c[(k,l,n)] >=0, 'Set 3 Constraint 4 ({},{},{},{})'.format(k,l,n,i)

                    prob += -x[i] + nu[(k,l)] >=0, 'Set 4 Constraint 3 ({},{},{},{})'.format(k,l,n,i)
                    prob += x[i] + nu[(k,l)] >=0, 'Set 4 Constraint 4 ({},{},{},{})'.format(k,l,n,i)
                    
    prob.solve()
    
    return (np.array([x[i].value() for i in range(1,M+1)]), LpStatus[prob.status], value(prob.objective))