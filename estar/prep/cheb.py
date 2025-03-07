#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:16:50 2020

@author: mike_ubuntu
"""

import numpy as np
from prep.fin_der import FD_derivatives

def Get_cheb_for_point(matrix, axis, idx, grid, max_der_order = 3, points = 9, poly_order = None):
    if type(poly_order) == type(None):
        max_power = max_der_order + 1
    else:
        max_power = poly_order
    I = np.array([np.int(-(points-1)/2 + i) for i in np.arange(points)]) + idx[axis]
    F = matrix.take(I , axis = axis)
    x_raw = grid[axis].take(I, axis = axis)
    for i in range(idx.size):
        if i < axis:
            F = F.take(idx[i], axis = 0)
            x_raw = x_raw.take(idx[i], axis = 0)            
        elif i > axis:
            F = F.take(idx[i], axis = 1)
            x_raw = x_raw.take(idx[i], axis = 1)     
            

    poly = np.polynomial.chebyshev.Chebyshev.fit(x_raw, F, max_power)
    return x_raw[int(x_raw.size/2.)], poly


def Process_Point_Cheb(args):
    global PolyBoundary
    idx = np.array(args[0]); matrix = args[1]; grid = args[2]; points = args[3]; n_der = args[4]; poly_bound = args[5]; poly_order = args[6]
    print(args[0])
    poly_mask = [idx[dim] >= poly_bound and idx[dim] <= matrix.shape[dim] - poly_bound for dim in np.arange(matrix.ndim)]
    polynomials = np.empty(matrix.ndim, dtype = np.polynomial.chebyshev.Chebyshev)
    x = np.empty(idx.shape)
    for i in range(matrix.ndim):
        if poly_mask[i]:
            x_temp, poly_temp = Get_cheb_for_point(matrix, i, idx, grid, max_der_order=n_der, points = points, poly_order = poly_order)
            x[i] = x_temp
            polynomials[i] = poly_temp 

    derivatives = np.empty(matrix.ndim * (n_der))
    for var_idx in np.arange(matrix.ndim):
        if poly_mask[var_idx]:
            for der_idx in np.arange(1, n_der+1):
                derivatives[var_idx*(n_der) + (der_idx-1)] = polynomials[var_idx].deriv(m=der_idx)(x[var_idx])
        else:
#            print(derivatives[var_idx*(n_der) : (var_idx+1)*(n_der)].shape, FD_derivatives(matrix, 
#                        axis = var_idx, idx = idx, grid = grid, max_order = n_der, poly_bound = poly_bound).shape)
            derivatives[var_idx*(n_der) : (var_idx+1)*(n_der)] = FD_derivatives(matrix, 
                        axis = var_idx, idx = idx, grid = grid, max_order = n_der, poly_bound = poly_bound)
    
#    print(derivatives.shape)
    return(derivatives)