import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import joblib
import argparse
import time

from collections import defaultdict
from functools import reduce

import cvxpy as cp
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from torch.optim.lr_scheduler import ReduceLROnPlateau


root_dir = os.getcwd()
data_dir_100k = root_dir + "/ml-100k"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# data generating process
def effective_rank(mat):
    """calculate effective rank (expects a torch matrix)"""
    _U, sigmas, _V = mat.svd(compute_uv=False)
    x = sigmas / torch.sum(sigmas)
    return torch.exp(-torch.sum(x * torch.log(x + 1e-10)))

def extract_sv(W, name, T):
    U,D,V = W.svd()

def gen_lowrank(N, rank):
    """generating low rank matrices"""
    U = np.random.randn(N, rank).astype(np.float32)
    V = np.random.randn(N, rank).astype(np.float32)
    W = U.dot(V.T) / np.sqrt(rank)
    W = W / np.linalg.norm(W, 'fro') * N # normalization
    return torch.from_numpy(W).float()

# movielens
def data_ml_100k(data_dir=data_dir_100k, std_data=None):
    cols = ['User ID','Movie ID','Rating','Timestamp']
    df = pd.read_csv(data_dir + '/u.data', delimiter='\t', header=None, names=cols)
    W = np.zeros((943,1682), dtype=np.float32)
    u1 = df.to_numpy()
    
    if std_data == "minmax":
        mini = df["Rating"].min()
        maxi = df["Rating"].max()
        rnge = maxi - mini 
        for i in range(u1.shape[0]):            
            W[u1[i,0]-1, u1[i,1]-1] = (u1[i,2] - mini)/(rnge)
    elif std_data == "z":
        mu = df["Rating"].mean()
        sig = df["Rating"].std()
        for i in range(u1.shape[0]):
            W[u1[i,0]-1, u1[i,1]-1] = (u1[i,2] - mu)/sig
    else:
        for i in range(u1.shape[0]):
            W[u1[i,0]-1, u1[i,1]-1] = u1[i,2]
            
    return torch.from_numpy(W).float()


def setup_matcomplete(W, N, n_train):
    indices = torch.multinomial(torch.ones(N*N), n_train, replacement=False)
    X_x, X_y = indices // N, indices % N
    Y = W[X_x,X_y]
    train_pairs = np.array((X_x.numpy(), X_y.numpy())).T
    all_range = torch.arange(start=0, end=N)
    all_pairs = np.array(np.meshgrid(all_range, all_range)).T.reshape(-1,2)
    diff = np.array(list(set(map(tuple, all_pairs)) - set(map(tuple, train_pairs)))).T
    XT_x = torch.from_numpy(diff[0,:])
    XT_y = torch.from_numpy(diff[1,:])
    return W, X_x, X_y, XT_x, XT_y, Y


def setup_ml_100k_sample(W, trn_prop):
    cols = ['User ID','Movie ID','Rating','Timestamp']
    u1 = pd.read_csv(data_dir_100k+'/u.data', delimiter='\t', header=None, names=cols)
    n_train = int(trn_prop * u1.shape[0])    
    u1 = u1.to_numpy()    
    np.random.shuffle(u1)
    X_x = torch.from_numpy(u1[:n_train,0] - 1)
    X_y =  torch.from_numpy(u1[:n_train,1] - 1)
    Y = W[X_x, X_y]
    XT_x = torch.from_numpy(u1[(n_train+1):,0] - 1)
    XT_y =  torch.from_numpy(u1[(n_train+1):,1] - 1)
    return W, X_x, X_y, XT_x, XT_y, Y


def setup_data(args):
    if args.data == "gaussian":
        W = gen_lowrank(args.N, args.rank)
        return setup_matcomplete(W, args.N, args.sample_size)
    elif args.data == "ml-100k-sample":
        W = data_ml_100k()
        return setup_ml_100k_sample(W, args.trainprop)
    

def build_nn(shape, D, nonlinear, device, incl_bias=False):
    hidden_sizes = [shape[0]] + [shape[1]] * (D) 
    layers = zip(hidden_sizes, hidden_sizes[1:]) 
    if not incl_bias:
        if nonlinear:
            # stick activation layer between the linear components
            nn_list = [None, nn.ReLU()] * D
            nn_list[0::2] = [nn.Linear(f_in, f_out, bias=False) for (f_in, f_out) in layers]
            model = nn.Sequential(*nn_list[:-1]) 
        else:
            model = nn.Sequential(*[nn.Linear(f_in, f_out, bias=False) for (f_in, f_out) in layers])
    else:
        if nonlinear:
            nn_list = [None, nn.ReLU()] * D
            nn_list[0::2] = [nn.Linear(f_in, f_out, bias=True) for (f_in, f_out) in layers]
            model = nn.Sequential(*nn_list[:-1]) 
        else:
            model = nn.Sequential(*[nn.Linear(f_in, f_out, bias=True) for (f_in, f_out) in layers])        
    return model.to(device)


def W_e2e(model, args):
    """extract end-2-end W matrix from"""
    if args.incl_bias:
        return reduce( (lambda x, y: x @ y), [fc.weight.t() + fc.bias for fc in model[0::(1+args.nonlinear)]])        
    else:
        return reduce( (lambda x, y: x @ y), [fc.weight.t() for fc in model[0::(1+args.nonlinear)]])    


def init_nn(model, shape, D, initscale):
    # gaussian
    for param in model.parameters():
        scale = initscale ** (1. / D) / np.sqrt(max(shape))
        nn.init.normal_(param, std=scale)
    
def solve_nn(data, args, device):
    
    data = tuple(d.to(device) for d in data)
    W, X_x, X_y, XT_x, XT_y, Y = data
    if args.data == "gaussian":
        shape = (args.N, args.N)
    elif args.data == "ml-100k-sample":
        shape = (943,1682)
    
    model = build_nn(shape, args.depth, args.nonlinear, device, args.incl_bias)
    init_nn(model, shape, args.depth, args.initscale)

    
    if args.optim == "GD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "Adam_amsgrad":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)        
    elif args.optim == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)        
    elif args.optim == "NAdam":
        optimizer = optim.NAdam(model.parameters(), lr=args.lr)        
    elif args.optim == "RAdam":
        optimizer = optim.RAdam(model.parameters(), lr=args.lr)        
    elif args.optim == "Adadelta":
        optimizer = optim.Adadelta(model.parameters())  
    elif args.optim == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)        
    elif args.optim == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optim == "LBFGS":
        optimizer = optim.LBFGS(model.parameters())
    elif args.optim == "Momentum":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, nesterov=True, momentum=0.5)
    elif args.optim == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    
    criterion = nn.MSELoss()
    
    if args.lrsched:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, threshold=0.1, factor=0.5)
    

    Y_test = W[XT_x, XT_y]

        
    for T in range(args.niters):
        W_ = W_e2e(model, args)
        Y_ = W_[X_x, X_y]
                        
        loss = criterion(Y_, Y)

        if args.lam > 0:
            if args.reg_norm == "ratio":
                loss += args.lam  * (W_.norm('nuc') / W_.norm('fro'))
            elif args.reg_norm == "nuclear":
                loss += args.lam  * (W_.norm('nuc'))
            elif args.reg_norm == "sc_12":
                svs = extract_sv(W_)
                loss += (args.lam ) * torch.linalg.norm(svs, ord=1/2) 
            elif args.reg_norm == "sc_13":
                svs = extract_sv(W_)
                loss += (args.lam ) * torch.linalg.norm(svs, ord=1/3) 
            elif args.reg_norm == "sc_23":
                svs = extract_sv(W_)
                loss += (args.lam ) * torch.linalg.norm(svs, ord=2/3) 
            elif args.reg_norm == "sc_1223":
                svs = extract_sv(W_)
                loss += (args.lam ) * (torch.linalg.norm(svs, ord=1/2)/torch.linalg.norm(svs, ord=2/3))
            elif args.reg_norm == "sc_1323":
                svs = extract_sv(W_)
                loss += (args.lam ) * (torch.linalg.norm(svs, ord=1/3)/torch.linalg.norm(svs, ord=2/3))
            elif args.reg_norm == "sc_1312":
                svs = extract_sv(W_)
                loss += (args.lam ) * (torch.linalg.norm(svs, ord=1/3)/torch.linalg.norm(svs, ord=1/2))
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if T % args.log_interval == 0:
            with torch.no_grad():
                Y_test = torch.clamp(Y_test, min=1, max=5)
                test_loss = torch.mean((W_[XT_x, XT_y] - Y_test)**2)
                train_loss = torch.mean((Y - Y_)**2)
                effrank = effective_rank(W_).item()                
                test_rmse = torch.sqrt(test_loss).item()

                print(f"depth: {args.depth}, iteration: {T}, test_RMSE: {test_rmse}, erank: {effrank}")

                if args.lrsched:
                    scheduler.step(test_loss)          

    
    
# CP
def solve_cp(data, args):
    """nuclear norm minimization method"""
    W, X_x, X_y, XT_x, XT_y, Y = data # unpack data
    Z = cp.Variable([args.N,args.N])
    objective = cp.Minimize(cp.norm(Z, 'nuc'))
    # constraints
    mask = np.zeros([args.N,args.N])
    mask[X_x, X_y] = 1
    constraints = [cp.abs(cp.multiply(Z - W, mask)) <= 1e-3]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=True, use_indirect=False)

    W_cp = torch.from_numpy(Z.value).float()

# SI
def solve_si(data, args):
    """softimpute"""
    # moving library import here because it's just too slow
    from fancyimpute import SoftImpute, BiScaler
    W, X_x, X_y, XT_x, XT_y, Y = data # unpack data
    mask = torch.zeros([args.N,args.N])
    mask[X_x, X_y] = 1
    W_masked = mask * W
    W_masked = W_masked.numpy()
    W_masked[W_masked == 0] = np.nan
    X_incomplete_normalized = BiScaler().fit_transform(W_masked)
    W_SI = SoftImpute().fit_transform(X_incomplete_normalized)
    W_SI = torch.from_numpy(W_SI)

# Opt
def solve_opt(data, args):
    rpy2.robjects.numpy2ri.activate()
    W, X_x, X_y, XT_x, XT_y, Y = data # unpack data
    mask = torch.zeros([args.N,args.N])
    mask[X_x, X_y] = 1
    W_masked = mask * W
    W_m = W_masked.numpy()
    nr,nc = W_m.shape
    W_mr = ro.r.matrix(W_m, nrow=nr, ncol=nc)
    ro.r.assign("x", W_mr)
    W_opt = ro.r("""
    library(ROptSpace)
    x[x==0] <- NA
    res <- OptSpace(x)
    res$X %*% res$S %*% t(res$Y)
    """)
    W_opt = torch.from_numpy(np.asarray(W_opt))


def main():
    # parsing input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=["DLNN", "CP", "SI", "OPT"], default="DLNN", help="choice of algorithm")
    parser.add_argument('--data', choices=["gaussian", "ml-100k-sample"], default="gaussian")
    parser.add_argument('--rank', type=int, default=5, help="rank of the matrix to complete")
    parser.add_argument('--depth', type=int, default=3, help="depth of DLNN")
    parser.add_argument('--fold', type=int, default=1, help="MovieLens CV fold number")
    parser.add_argument('--optim', choices=["Adam", "LBFGS", "GD", "ASGD", "Adagrad", "RMSprop", "Momentum", "Adamax", "Adam_amsgrad", "AdamW", "NAdam", "RAdam", "Adadelta"], default="Adam")
    parser.add_argument('--N', type=int, default=100, help="size of W")
    parser.add_argument('--sample_size', type=int, default=2000, help="number of training samples")
    parser.add_argument('--test_size', type=int, default=100000, help="number of test samples")
    parser.add_argument('--niters', type=int, default=200000, help="number of max iterations")
    parser.add_argument('--nonlinear', action="store_true", help="nonlinear activation function?")
    parser.add_argument('--reg_norm', choices=["ratio", "nuclear", "sc_12", "sc_13", "sc_23", "sc_1223","sc_1323","sc_1312"], default="ratio", help="regularizer")    
    parser.add_argument('--lam', type=float, default=0.0, help="penalty parameter")
    parser.add_argument('--initscale', type=float, default=0.001, help="scale of initialization")
    parser.add_argument('--log_interval', type=int, default=1000, help="how often to calculate metrics during logging")
    parser.add_argument('--store_sv', action="store_true", help="keep singular values?")
    parser.add_argument('--lrsched', type=bool, default=False, help="schedule to keep LR small per gradient flow")
    parser.add_argument('--std', choices=["z", "minmax", None], default=None, help="std data")
    parser.add_argument('--incl_bias', type=bool, default=False, help="include bias")    
    parser.add_argument('--lr', type=float, default=0.01, help="init LR")    
    parser.add_argument('--trainprop', type=float, default=0.8, help="training proportion")    
    
    args = parser.parse_args()

    data = setup_data(args)

    if args.method == "DLNN":
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        solve_nn(data, args, device)
    elif args.method == "CP":
        solve_cp(data, args)
    elif args.method == "SI":
        solve_si(data, args)
    elif args.method == "OPT":
        solve_opt(data, args)
    
if __name__ == "__main__":
    main()