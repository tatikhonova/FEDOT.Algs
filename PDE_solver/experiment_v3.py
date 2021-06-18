# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:26:39 2020

@author: Sashka
"""
import functools
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import datetime

import numpy as np
import pandas as pd

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from scipy.optimize import minimize
import scipy as sp
# from scipy.optimize import differential_evolution
# from scipy.optimize import dual_annealing
import numba
import time
import matplotlib.pyplot as plt

import torch

from PDE_solver import string_reshape, plot_3D_surface, operator_norm, solution_interp_nn, solution_interp_RBF, \
    solution_interp, operator_norm_derivative

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["figure.max_open_warning"] = 1000

norm_bond = float(200)


# @ray.remote
def init_field_expetiment(nrun=1, grid_scenario=[[10, 10]], interp_type='random', diff_scheme=[1, 2]):
    np.random.seed()
    experiment = []

    x = np.linspace(0, 1, 10)
    t = np.linspace(0, 1, 10)

    old_grid = numba.typed.List()

    old_grid.append(x)
    old_grid.append(t)

    arr = np.random.random((10, 10))

    for grid_res in grid_scenario:

        x = np.linspace(0, 1, grid_res[0])
        t = np.linspace(0, 1, grid_res[1])

        new_grid = numba.typed.List()

        new_grid.append(x)
        new_grid.append(t)

        grid = np.meshgrid(*new_grid)
        grid = torch.tensor(grid, device=device)

        bcond = [{'boundary': 0, 'axis': 0, 'string': torch.zeros(len(new_grid[0]), device=device)},
                 {'boundary': len(x) - 1, 'axis': 0, 'string': torch.zeros(len(new_grid[0]), device=device)},
                 {'boundary': 0, 'axis': 1, 'string': torch.from_numpy(np.sin(np.pi * new_grid[1])).to(device)},
                 {'boundary': len(t) - 1, 'axis': 1,
                  'string': torch.from_numpy(np.sin(np.pi * new_grid[1])).to(device)}]

        if interp_type == 'scikit_interpn':
            arr = solution_interp(old_grid, arr, new_grid)
        elif interp_type == 'interp_RBF':
            arr = solution_interp_RBF(old_grid, arr, new_grid, method='linear', smooth=10)
        elif interp_type == 'interp_nn':
            arr = solution_interp_nn(old_grid, arr, new_grid)
        else:
            np.random.seed(grid_res)
            arr = np.random.random((len(x), len(t)))

        exact_sln = f'exact_sln/wave_{grid_res[0]}.csv'

        wolfram_interp = np.genfromtxt(exact_sln, delimiter=',')

        arr = torch.from_numpy(arr).to(device)

        operator = [[(1, 0, 2, 1)], [(-1 / 4, 1, 2, 1)]]

        def bfgs_method(f, fprime, params, x0, maxiter=None, epsi=10e-3):

            if maxiter is None:
                maxiter = 1000

            k = 0

            field_shape = x0.shape

            gfk = fprime(x0, *params)

            gfk = torch.flatten(gfk)
            x0 = torch.flatten(x0)

            N = int(x0.shape[0])

            I = torch.eye(N, dtype=torch.float64, device=device)
            Hk = I
            xk = x0

            while torch.linalg.norm(gfk).item() > epsi and k < maxiter:
                # pk - direction of search

                pk = -torch.matmul(Hk, gfk)

                # line_search = sp.optimize.line_search(lambda x: f(torch.from_numpy(x), *params).cpu().numpy(), lambda x: fprime(torch.from_numpy(x), *params).cpu().numpy(), xk.cpu().numpy(), pk.cpu().numpy())
                # alpha_k = line_search[0]

                alpha_k = 1

                xkp1 = xk + alpha_k * pk
                sk = xkp1 - xk
                xk = xkp1

                gfkp1 = fprime(xkp1.view(field_shape), *params)
                gfkp1 = torch.flatten(gfkp1)

                yk = gfkp1 - gfk
                gfk = gfkp1

                k += 1

                ro = 1.0 / (torch.matmul(yk, sk))

                A1 = I - ro * torch.outer(sk, yk)
                A2 = I - ro * torch.outer(yk, yk)
                Hk = torch.matmul(A1, torch.matmul(Hk, A2)) + (ro * torch.outer(sk, sk))

            return (xk.view(field_shape), k)

        start_time = time.time()
        operator_norm_1 = lambda x: operator_norm(x,
                                                  grid,
                                                  operator,
                                                  norm_bond,
                                                  bcond,
                                                  scheme_order=diff_scheme[0],
                                                  boundary_order=diff_scheme[1])
        # opt = minimize(operator_norm_1, arr.reshape(-1),options={'disp': True, 'maxiter': 3000}, tol=0.05)

        params = [grid, operator, norm_bond, bcond]
        result, k = bfgs_method(operator_norm_1, operator_norm_derivative, params, arr)

        print('Result of BFGS method:')
        print('Final Result (best point): %s' % (result))
        print('Iteration Count: %s' % (k))

        elapsed_time = time.time() - start_time

        print(operator_norm_1(result))
        print(f'[{datetime.datetime.now()}] grid_x = {grid_res[0]} grid_t={grid_res[1]} time = {elapsed_time}')

        full_sln_interp = string_reshape(result, new_grid)
        # print(full_sln_interp)
        error = np.abs(result.cpu().numpy() - wolfram_interp)
        max_error = np.max(error)
        wolfram_MAE = np.mean(error)
        print('Max error', max_error)
        print('Mean error', wolfram_MAE)
        arr = full_sln_interp

        plot_3D_surface(result.cpu().numpy(), wolfram_interp, new_grid)

        experiment.append(
            {'grid_x': len(x), 'grid_t': len(t), 'time': elapsed_time, 'MAE': wolfram_MAE, 'max_err': max_error,
             'interp_type': interp_type, 'scheme_order': diff_scheme[0], 'boundary_order': diff_scheme[1]})

        old_grid = new_grid

    return experiment


if os.path.isfile("interp_v3.csv"):
    df = pd.read_csv("interp_v3.csv", index_col=0)
else:
    df = pd.DataFrame(
        columns=['grid_x', 'grid_t', 'time', 'MAE', 'max_err', 'interp_type', 'scheme_order', 'boundary_order'])

# ray.init(ignore_reinit_error=True)

# warm up
# results = ray.get([init_field_expetiment.remote(run,interp_type='random') for run in range(1)])
results = init_field_expetiment(grid_scenario=[[10, 10]])

# for ds in [[1,1],[2,2],[1,2]]:
#     for interp_type in ['random','scikit_interpn','interp_RBF','interp_nn']:
#         results = ray.get([init_field_expetiment.remote(run,interp_type=interp_type,grid_scenario=[[10,10],[15,15],[20,20],[25,25],[30,30],[35,35],[40,40],[45,45],[50,50]],diff_scheme=ds) for run in range(20)])
#         for result in results:
#             df=df.append(result,ignore_index=True)
#         df.to_csv(f'interp_v3_{datetime.datetime.now().timestamp()}.csv',index=False)

