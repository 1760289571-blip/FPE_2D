9# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 18:55:28 2026

@author: 幽灵
"""

#This is for the example 2 in the paper
#I have modified Example 2 before.
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, grad, jacrev, hessian, jit, jacfwd
import optax
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import pickle
from jax.scipy import integrate

from functools import partial
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='TRBFN Training for FKP Equaions')

parser.add_argument('--test_case', type=str, default="FPE")
parser.add_argument('--Chosen_List', type=list, default= ["gaussian", "gaussian","gaussian"])
parser.add_argument('--rank', type = dict, default = 128)
parser.add_argument('--rbf_types', type = str, default = "three_one", help = "three_one/two")
parser.add_argument('--epochs', type = int, default = 50000)
parser.add_argument('--batches', type = int, default = 5000)
parser.add_argument('--Train', type = bool, default = True)
parser.add_argument('--m', type = int, default = 27)
parser.add_argument('--scale_r', type = float, default = 1.0)
parser.add_argument('--r_sde', type = float, default = 20)
parser.add_argument('--mu_ex', type = float, default = 0)
parser.add_argument('--mu_ffd', type = float, default = 0)
parser.add_argument('--V_thres', type = float, default = 20)
parser.add_argument('--tau_r', type = float, default = 0)
parser.add_argument('--tau_m', type = float, default = 0.02)
parser.add_argument('--tau_ee', type = float, default = 0.004)
parser.add_argument('--V_reset', type = float, default = 0)
parser.add_argument('--sigma_ex', type = float, default = 2)
parser.add_argument('--sigma_ffd', type = float, default = 2)
parser.add_argument('--c', type = float, default = 0.5)
parser.add_argument('--p_ee', type = float, default = 0.8)
parser.add_argument('--s_ee', type = float, default = 1)

args = parser.parse_args()

'''
OC: original code.
This is for the translation.
'''

'''
Part 1
Problem Definition
'''

Center_O = jnp.array([0,  0,  0, 0])

#Function which helps the computation of loss functional
'''
need to be modified for our version
'''

#DEBUG：not add data[2] and data[3]
func_into = (lambda data,mu_ex,mu_ffd,tau_m,tau_ee,p_ee,s_ee: jnp.asarray(
        [(-data[0]+mu_ex+data[2])/tau_m,  
         (-data[1]+mu_ex+data[3])/tau_m,
         (-data[2]+s_ee*mu_ffd*p_ee)/tau_ee,
         (-data[3]+s_ee*mu_ffd*p_ee)/tau_ee
        ]
    )
    )
'''
Part 2
This part group the parameters of the shape of parameters
'''
#dimension of the input
dim = 4

#N in the paper,  (OC: rank_A, rank_B = args.rank["A"], args.rank["B"] )
rank = args.rank

#m_use is one third/half of the number of k_{ij}^{\ell}, depends on different types of RBFs

#Shape this is the shape used to generate scale / coeff with thin the innerest layer
#OC: Shape = (1, 3, 1, rank_A, dim) if args.structure == "structure_A" else (1, 3, m_use, rank_B, dim)

if args.rbf_types == "three_one":
   m_use = args.m//3
   Shape = (3, m_use, rank, dim)
else:
   m_use = args.m//2
   Shape = (2, m_use, rank, dim)


#radius we generate from SDE
r_base = args.r_sde

#This part controls the scale of radius
scale_r = args.scale_r

#radius we use in generate sample points
r = r_base * scale_r

#random key for initialization
key_ini = jax.random.PRNGKey(1243)

#argument list for initialization
#OC: Boundary_Shift, Boundary_Width, args_init =[1.0, 1.0, 1.0], [0.7, 0.9, 1.0], [key_s, 4, 0.9 * r, r, Shape, jnp.sqrt(r)]

args_init = [key_ini, 0.9 * r, jnp.sqrt(r), r, Shape]


#code for generating the initial value of parameters name: def Initialization_Generation

def Initialization_Generation(key, width_ini, dist_ini_shift, constraint_ini_shift, shape):
    Param_Ini = {
    }
    
    num_points = 10 * (np.array(shape)).prod()
    num_shifts = (np.array(shape[:])).prod()

    #Generation of Shift from a certain distribution
    Shift_Points = (dist_ini_shift * jax.random.normal(key, shape = (num_points, )))
    #constrain the shift within the radius
    Param_Ini["shifts"] = (Shift_Points[jnp.abs(Shift_Points) < constraint_ini_shift])[:num_shifts].reshape(shape[:]) + Center_O

    #Generate the initialization of width:
    Param_Ini["width"] = jnp.zeros(shape) + width_ini
    #Generate the initialization of alpha_1:
    Param_Ini["alpha_1"] = jnp.zeros(shape) + 0.6
    
    #Generate the initialization of alpha_2 
    Param_Ini["alpha_2"] = jnp.zeros(shape[1:]) + 0.6
    
    #Generate the coefficients c:
    Param_Ini["coeff"] = jnp.zeros((shape[2],)) + 0.6

    return Param_Ini
    
initial_param = Initialization_Generation(*args_init)


'''
This is the group of implementation of RBF function and their corresponding derivatives
'''

#Types of RBF function list
Chosen_List = args.Chosen_List
mu_ex=args.mu_ex
mu_ffd=args.mu_ffd
tau_m=args.tau_m
tau_r=args.tau_r
tau_ee=args.tau_ee
V_thres=args.V_thres
V_reset=args.V_reset
sigma_ex=args.sigma_ex
sigma_ffd=args.sigma_ffd
c=args.c
p_ee=args.p_ee
s_ee=args.s_ee


#The lower bound of the scale/bandwidth for different kinds of kernels h_{ij}^{(\ell)}
#(OC: bd ={"wendland": 3e-02,"inverse_quadratic": 0.01} )
scale_bound = {"wendland": 3e-02,"inverse_quadratic": 0.01,'gaussian': 0.01}

#Basis Function Part
'''
version of explicit integral for three types of RBF function
'''
#gaussian integral
@jit
def gaussian_integral(sigma, shift):
    sigma_t =  1/ (scale_bound["gaussian"] + jnp.square(sigma) )
    results = 0.5 * ( jax.lax.erf( jnp.sqrt(sigma_t/2) * (r - shift) ) - jax.lax.erf( jnp.sqrt(sigma_t/2) * (-r - shift) ) )
    return results

#inverse quadratic integral:
@jit
def inverse_quadratic_integral(epsilon, shift):
    epsilon_sq = jnp.square(jnp.abs(epsilon) + scale_bound["inverse_quadratic"])
    func = lambda insi: ( insi * ( 3.0 * epsilon_sq + 2.0 * (insi**2) ) )/(4.0 * ((epsilon_sq + insi**2)**1.5))
    results = func(r - shift) - func(-r - shift)
    return results

#wendland kernel integral
def wendland_integral(h, shift):
    return 1.0

'''
version of RBF function for evaluation
'''
#gaussian kernel
def gaussian_1_test(sigma, data, shift):

    input = data - shift
    Integral_scaling = gaussian_integral(sigma, shift)
    sigma_t =  1/ (scale_bound["gaussian"] + jnp.square(sigma) )
    dim_each_result = (1/Integral_scaling) * ( jnp.sqrt(sigma_t)/jnp.sqrt(2 * np.pi) ) * jnp.exp( -0.5 * sigma_t * jnp.square(input))

    return dim_each_result

#inverse quadratic kernel
def inverse_quadratic_1_test(epsilon, data, shift):
    input = data - shift
    Integral_scaling = inverse_quadratic_integral(epsilon, shift)
    epsilon_sq = jnp.square(jnp.abs(epsilon) + scale_bound["inverse_quadratic"])
    dim_each_result = (1/Integral_scaling) * (0.75 * (epsilon_sq)**2) * ( 1/(epsilon_sq + jnp.square(input))**2.5 )
    return dim_each_result


def wendland_1_test(h, data, shift_w):
    shift_ = shift_w

    #h_deno = 3e-02 + jnp.abs( (r - jnp.abs(shift_)) * jnp.tanh(h) )

    h_deno = scale_bound["wendland"] + jnp.abs(h)

    h_sq = 1/h_deno

    input = data - shift_
    dim_each_result = 1.25 * h_sq * jnp.power(jax.nn.relu(1 - jnp.abs(h_sq * input)) , 3) * ( 3.0 * jnp.abs(h_sq * input) + 1 )
    return dim_each_result
    
def wendland_1_test2(h, data, shift_w):
    shift_ = shift_w

    #h_deno = 3e-02 + jnp.abs( (r - jnp.abs(shift_)) * jnp.tanh(h) )

    h_deno = jnp.abs(h)

    h_sq = 1/h_deno

    input = data - shift_
    dim_each_result = 1.25 * h_sq * jnp.power(jax.nn.relu(1 - jnp.abs(h_sq * input)) , 3) * ( 3.0 * jnp.abs(h_sq * input) + 1 )
    return dim_each_result

'''
RBF derivatives and functions for PINN loss function
'''

def gaussian_1(sigma, data, shift):
    input = data - shift
    Integral_scaling = gaussian_integral(sigma, shift)

    sigma_t =  1/ (scale_bound["gaussian"] + jnp.square(sigma) )

    dim_each_result = (1/Integral_scaling) * ( jnp.sqrt(sigma_t)/jnp.sqrt(2 * np.pi) ) * jnp.exp( -0.5 * sigma_t * jnp.square(input))
    #grad_dim_each_result = (1/Integral_scaling) * (-sigma_t * input * dim_each_result)
    #hessian_dim_each_result = (1/Integral_scaling) * (jnp.square(sigma_t) * jnp.square(input) - sigma_t) * dim_each_result
    grad_dim_each_result =  (-sigma_t * input * dim_each_result)
    hessian_dim_each_result = (jnp.square(sigma_t) * jnp.square(input) - sigma_t) * dim_each_result
    return dim_each_result, grad_dim_each_result, hessian_dim_each_result



def inverse_quadratic_1(epsilon, data, shift):
    input = data - shift
    Integral_scaling = inverse_quadratic_integral(epsilon, shift)

    epsilon_sq = jnp.square(jnp.abs(epsilon) + scale_bound["inverse_quadratic"])

    inver_ele = 1/(epsilon_sq + jnp.square(input))

    dim_each_result = (1/ Integral_scaling) * (0.75 * (epsilon_sq)**2 * ( inver_ele**2.5 ))
    grad_dim_each_result = (1/ Integral_scaling) * inver_ele * (-5 * input) * dim_each_result
    hessian_dim_each_result = (1/ Integral_scaling) * (grad_dim_each_result * (-7.0 * input) - dim_each_result * 5.0) * inver_ele
    return dim_each_result, grad_dim_each_result, hessian_dim_each_result



def wendland_1(h, data, shift_w):

    shift_ = shift_w

    h_deno = scale_bound["wendland"] + jnp.abs(h)

    h_sq = 1/h_deno
    input = data - shift_
    abs_in = jnp.abs(h_sq * input)
    ele = jax.nn.relu(1 - abs_in)
    dim_each_result = 1.25 * h_sq * (ele**3) * ( 3.0 * abs_in + 1 )
    grad_dim_each_result = 1.25 * h_sq * (-12.0 * (h_sq**2) * (input) ) * (ele**2)
    hessian_dim_each_result = (1.25 * h_sq) * (12 * (h_sq**2)) * ele * (3 * abs_in - 1)

    return dim_each_result, grad_dim_each_result, hessian_dim_each_result

def gaussian_conv(h,data,s,Sigma):
    os.environ['JAX_DEFAULT_DTYPE_BITS'] = '128'
    b1=data-mu_ex*(1-jnp.exp(-tau_r/tau_m))
    b2=-jnp.exp(-tau_r/tau_m)
    Integral_scaling = gaussian_integral(h, s)
    sigma_t =  1/ (scale_bound["gaussian"] + jnp.square(h) )
    simp_f = (jnp.sqrt(sigma_t)* jnp.exp(-(sigma_t * (b1 + b2 * s) ** 2) / (2 * b2**2 + 2 * Sigma * sigma_t))* (jax.lax.erf((jnp.sqrt(2)* (r * b2**2 - b1 * b2 + r * Sigma * sigma_t + Sigma * s * sigma_t)) / (2 * Sigma * jnp.sqrt((b2**2 + Sigma * sigma_t) / Sigma))) + jax.lax.erf((jnp.sqrt(2) * (r * b2**2 + b1 * b2 + r * Sigma * sigma_t - Sigma * s * sigma_t)) / (2 * Sigma * jnp.sqrt((b2**2 + Sigma * sigma_t) / Sigma)))) / (2 * jnp.sqrt((b2**2 + Sigma * sigma_t) / Sigma)))
    os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'
    return 1/Integral_scaling*simp_f

    
def wendland_conv(h,data,s,Sigma): # calculate integral involved in reset dynamics
    b1=data-mu_ex*(1-jnp.exp(-tau_r/tau_m))
    b2=-jnp.exp(-tau_r/tau_m)
    h = scale_bound["wendland"] + jnp.abs(h)
    simp_f= (Sigma*jnp.exp(-(b1 - b2*h + b2*s)**2/(2*Sigma))*(h - s)**2)/(b2**2*h**3) + (3*Sigma*jnp.exp(-(b1 - b2*h + b2*s)**2/(2*Sigma))*(h - s)**3)/(b2**2*h**4) + (3*Sigma*jnp.exp(-(b1 + b2*s)**2/(2*Sigma))*(b1**3 + 3*b1**2*b2*h + 4*b1**2*b2*s + 3*b1*b2**2*h**2 + 9*b1*b2**2*h*s + 6*b1*b2**2*s**2 + 5*Sigma*b1 + b2**3*h**3 + 6*b2**3*h**2*s + 9*b2**3*h*s**2 + 4*b2**3*s**3 + 6*Sigma*b2*h + 8*Sigma*b2*s))/(b2**5*h**4) - (3*Sigma*jnp.exp(-(b1 + b2*s)**2/(2*Sigma))*(b1**3 - 3*b1**2*b2*h + 4*b1**2*b2*s + 3*b1*b2**2*h**2 - 9*b1*b2**2*h*s + 6*b1*b2**2*s**2 + 5*Sigma*b1 - b2**3*h**3 + 6*b2**3*h**2*s - 9*b2**3*h*s**2 + 4*b2**3*s**3 - 6*Sigma*b2*h + 8*Sigma*b2*s))/(b2**5*h**4) + (Sigma*jnp.exp(-(b1 + b2*h + b2*s)**2/(2*Sigma))*(h + s)**2)/(b2**2*h**3) + (3*Sigma*jnp.exp(-(b1 + b2*h + b2*s)**2/(2*Sigma))*(h + s)**3)/(b2**2*h**4) - (Sigma*jnp.exp(-(b1 + b2*s)**2/(2*Sigma))*(b1**2 - 3*b1*b2*h + 3*b1*b2*s + 3*b2**2*h**2 - 6*b2**2*h*s + 3*b2**2*s**2 + 2*Sigma))/(b2**4*h**3) - (Sigma*jnp.exp(-(b1 + b2*s)**2/(2*Sigma))*(b1**2 + 3*b1*b2*h + 3*b1*b2*s + 3*b2**2*h**2 + 6*b2**2*h*s + 3*b2**2*s**2 + 2*Sigma))/(b2**4*h**3) - (2*Sigma*s**2*jnp.exp(-(b1 + b2*s)**2/(2*Sigma)))/(b2**2*h**3) - (3*Sigma*jnp.exp(-(b1 + b2*h + b2*s)**2/(2*Sigma))*(b1**3 + 3*b1**2*b2*h + 4*b1**2*b2*s + 3*b1*b2**2*h**2 + 9*b1*b2**2*h*s + 6*b1*b2**2*s**2 + 5*Sigma*b1 + b2**3*h**3 + 6*b2**3*h**2*s + 9*b2**3*h*s**2 + 4*b2**3*s**3 + 6*Sigma*b2*h + 8*Sigma*b2*s))/(b2**5*h**4) + (3*Sigma*jnp.exp(-(b1 - b2*h + b2*s)**2/(2*Sigma))*(b1**3 - 3*b1**2*b2*h + 4*b1**2*b2*s + 3*b1*b2**2*h**2 - 9*b1*b2**2*h*s + 6*b1*b2**2*s**2 + 5*Sigma*b1 - b2**3*h**3 + 6*b2**3*h**2*s - 9*b2**3*h*s**2 + 4*b2**3*s**3 - 6*Sigma*b2*h + 8*Sigma*b2*s))/(b2**5*h**4) + (Sigma*jnp.exp(-(b1 + b2*h + b2*s)**2/(2*Sigma))*(b1**2 + 3*b1*b2*h + 3*b1*b2*s + 3*b2**2*h**2 + 6*b2**2*h*s + 3*b2**2*s**2 + 2*Sigma))/(b2**4*h**3) + (Sigma*jnp.exp(-(b1 - b2*h + b2*s)**2/(2*Sigma))*(b1**2 - 3*b1*b2*h + 3*b1*b2*s + 3*b2**2*h**2 - 6*b2**2*h*s + 3*b2**2*s**2 + 2*Sigma))/(b2**4*h**3) + (3*Sigma*jnp.exp(-(b1 + b2*h + b2*s)**2/(2*Sigma))*(h + s)*(b1**2 + 3*b1*b2*h + 4*b1*b2*s + 3*b2**2*h**2 + 9*b2**2*h*s + 6*b2**2*s**2 + 3*Sigma))/(b2**4*h**4) + (3*Sigma*s*jnp.exp(-(b1 + b2*s)**2/(2*Sigma))*(b1**2 - 3*b1*b2*h + 4*b1*b2*s + 3*b2**2*h**2 - 9*b2**2*h*s + 6*b2**2*s**2 + 3*Sigma))/(b2**4*h**4) - (3*Sigma*s*jnp.exp(-(b1 + b2*s)**2/(2*Sigma))*(b1**2 + 3*b1*b2*h + 4*b1*b2*s + 3*b2**2*h**2 + 9*b2**2*h*s + 6*b2**2*s**2 + 3*Sigma))/(b2**4*h**4) + (3*Sigma*jnp.exp(-(b1 - b2*h + b2*s)**2/(2*Sigma))*(h - s)**2*(b1 - 3*b2*h + 4*b2*s))/(b2**3*h**4) - (Sigma*jnp.exp(-(b1 + b2*h + b2*s)**2/(2*Sigma))*(h + s)*(b1 + 3*b2*h + 3*b2*s))/(b2**3*h**3) + (3*Sigma*jnp.exp(-(b1 - b2*h + b2*s)**2/(2*Sigma))*(h - s)*(b1**2 - 3*b1*b2*h + 4*b1*b2*s + 3*b2**2*h**2 - 9*b2**2*h*s + 6*b2**2*s**2 + 3*Sigma))/(b2**4*h**4) + (Sigma*s*jnp.exp(-(b1 + b2*s)**2/(2*Sigma))*(b1 - 3*b2*h + 3*b2*s))/(b2**3*h**3) + (Sigma*s*jnp.exp(-(b1 + b2*s)**2/(2*Sigma))*(b1 + 3*b2*h + 3*b2*s))/(b2**3*h**3) + (Sigma*jnp.exp(-(b1 - b2*h + b2*s)**2/(2*Sigma))*(h - s)*(b1 - 3*b2*h + 3*b2*s))/(b2**3*h**3) - (3*Sigma*jnp.exp(-(b1 + b2*h + b2*s)**2/(2*Sigma))*(h + s)**2*(b1 + 3*b2*h + 4*b2*s))/(b2**3*h**4) - (3*Sigma*s**2*jnp.exp(-(b1 + b2*s)**2/(2*Sigma))*(b1 - 3*b2*h + 4*b2*s))/(b2**3*h**4) + (3*Sigma*s**2*jnp.exp(-(b1 + b2*s)**2/(2*Sigma))*(b1 + 3*b2*h + 4*b2*s))/(b2**3*h**4) + (3*jnp.sqrt(2*jnp.pi*Sigma)*jax.lax.erf((b1 + b2*s)/jnp.sqrt(2*Sigma))*(3*Sigma**2 + 6*Sigma*b1**2 + 9*Sigma*b1*b2*h + 12*Sigma*b1*b2*s + 3*Sigma*b2**2*h**2 + 9*Sigma*b2**2*h*s + 6*Sigma*b2**2*s**2 + b1**4 + 3*b1**3*b2*h + 4*b1**3*b2*s + 3*b1**2*b2**2*h**2 + 9*b1**2*b2**2*h*s + 6*b1**2*b2**2*s**2 + b1*b2**3*h**3 + 6*b1*b2**3*h**2*s + 9*b1*b2**3*h*s**2 + 4*b1*b2**3*s**3 + b2**4*h**3*s + 3*b2**4*h**2*s**2 + 3*b2**4*h*s**3 + b2**4*s**4))/(2*b2**5*h**4) - (3*jnp.sqrt(2*jnp.pi*Sigma)*jax.lax.erf((b1 + b2*s)/jnp.sqrt(2*Sigma))*(3*Sigma**2 + 6*Sigma*b1**2 - 9*Sigma*b1*b2*h + 12*Sigma*b1*b2*s + 3*Sigma*b2**2*h**2 - 9*Sigma*b2**2*h*s + 6*Sigma*b2**2*s**2 + b1**4 - 3*b1**3*b2*h + 4*b1**3*b2*s + 3*b1**2*b2**2*h**2 - 9*b1**2*b2**2*h*s + 6*b1**2*b2**2*s**2 - b1*b2**3*h**3 + 6*b1*b2**3*h**2*s - 9*b1*b2**3*h*s**2 + 4*b1*b2**3*s**3 - b2**4*h**3*s + 3*b2**4*h**2*s**2 - 3*b2**4*h*s**3 + b2**4*s**4))/(2*b2**5*h**4) - (3*jnp.sqrt(2*jnp.pi*Sigma)*jax.lax.erf((b1 + b2*h + b2*s)/jnp.sqrt(2*Sigma))*(3*Sigma**2 + 6*Sigma*b1**2 + 9*Sigma*b1*b2*h + 12*Sigma*b1*b2*s + 3*Sigma*b2**2*h**2 + 9*Sigma*b2**2*h*s + 6*Sigma*b2**2*s**2 + b1**4 + 3*b1**3*b2*h + 4*b1**3*b2*s + 3*b1**2*b2**2*h**2 + 9*b1**2*b2**2*h*s + 6*b1**2*b2**2*s**2 + b1*b2**3*h**3 + 6*b1*b2**3*h**2*s + 9*b1*b2**3*h*s**2 + 4*b1*b2**3*s**3 + b2**4*h**3*s + 3*b2**4*h**2*s**2 + 3*b2**4*h*s**3 + b2**4*s**4))/(2*b2**5*h**4) + (3*jnp.sqrt(2*jnp.pi*Sigma)*jax.lax.erf((b1 - b2*h + b2*s)/jnp.sqrt(2*Sigma))*(3*Sigma**2 + 6*Sigma*b1**2 - 9*Sigma*b1*b2*h + 12*Sigma*b1*b2*s + 3*Sigma*b2**2*h**2 - 9*Sigma*b2**2*h*s + 6*Sigma*b2**2*s**2 + b1**4 - 3*b1**3*b2*h + 4*b1**3*b2*s + 3*b1**2*b2**2*h**2 - 9*b1**2*b2**2*h*s + 6*b1**2*b2**2*s**2 - b1*b2**3*h**3 + 6*b1*b2**3*h**2*s - 9*b1*b2**3*h*s**2 + 4*b1*b2**3*s**3 - b2**4*h**3*s + 3*b2**4*h**2*s**2 - 3*b2**4*h*s**3 + b2**4*s**4))/(2*b2**5*h**4) + (jnp.sqrt(2*jnp.pi*Sigma)*jax.lax.erf((b1 + b2*h + b2*s)/jnp.sqrt(2*Sigma))*(b1 + b2*h + b2*s)*(b1**2 + 2*b1*b2*h + 2*b1*b2*s + b2**2*h**2 + 2*b2**2*h*s + b2**2*s**2 + 3*Sigma))/(2*b2**4*h**3) + (jnp.sqrt(2*jnp.pi*Sigma)*jax.lax.erf((b1 - b2*h + b2*s)/jnp.sqrt(2*Sigma))*(b1 - b2*h + b2*s)*(b1**2 - 2*b1*b2*h + 2*b1*b2*s + b2**2*h**2 - 2*b2**2*h*s + b2**2*s**2 + 3*Sigma))/(2*b2**4*h**3) - (jnp.sqrt(2*jnp.pi*Sigma)*jax.lax.erf((b1 + b2*s)/jnp.sqrt(2*Sigma))*(b1 + b2*h + b2*s)*(b1**2 + 2*b1*b2*h + 2*b1*b2*s + b2**2*h**2 + 2*b2**2*h*s + b2**2*s**2 + 3*Sigma))/(2*b2**4*h**3) - (jnp.sqrt(2*jnp.pi*Sigma)*jax.lax.erf((b1 + b2*s)/jnp.sqrt(2*Sigma))*(b1 - b2*h + b2*s)*(b1**2 - 2*b1*b2*h + 2*b1*b2*s + b2**2*h**2 - 2*b2**2*h*s + b2**2*s**2 + 3*Sigma))/(2*b2**4*h**3)
    return 1.25*simp_f
'''
def wendland_conv(h,data,s,Sigma): # calculate integral involved in reset dynamics
    b1=data-I*(1-jnp.exp(-tau_r/tau_m))
    b2=-jnp.exp(-tau_r/tau_m)
    N=1000
    result=0
    def f(x):
        return jnp.exp(-jnp.square(b1+b2*x)/(2*Sigma))*jnp.power(1-(x-s)/h,3)*(3*(x-s)/h+1)
    data_lab=jnp.linspace(s-h,s+h,N)
    for x in data_lab:
        result=result+f(x)
    return result/N
'''

'''
This is a dictionary which includes the RBF's function's integral, function for test, and function for pinn loss function
'''
Kernel_Dic = {
    "gaussian": (gaussian_integral, gaussian_1_test, gaussian_1,gaussian_conv),
    "inverse_quadratic":(inverse_quadratic_integral,  inverse_quadratic_1_test,  inverse_quadratic_1),
    "wendland":(wendland_integral, wendland_1_test, wendland_1,wendland_conv)
}

k1, k2, k3 = Kernel_Dic[Chosen_List[0]], Kernel_Dic[Chosen_List[1]], Kernel_Dic[Chosen_List[2]]

#This part is for evaluation
@jit 
def combine_k(bandwidth, dist_pa, data, shifts):
    result = (dist_pa[0] * k1[1](bandwidth[0], data, shifts[0]) +
              dist_pa[1] * k2[1](bandwidth[1], data, shifts[1]))

    if args.rbf_types == "three_one":
       result = result + dist_pa[2] * k3[1](bandwidth[2], data, shifts[2])

    return result


#This part is for computation of loss functional
@jit
def combine_k_no_bp(bandwidth, dist_pa, data, shift):
    r1, grad1, h1 = k1[2](bandwidth[0], data, shift[0])
    r2, grad2, h2 = k2[2](bandwidth[1], data, shift[1])

    if args.rbf_types == "three_one":
       r3, grad3, h3 = k3[2](bandwidth[2], data, shift[2])

    
    if args.rbf_types == "three_one":
       def func_comb(x1, x2, x3):
           result =dist_pa[0] * x1 + dist_pa[1] * x2 + dist_pa[2] * x3
           return result
       comb_r = func_comb(r1, r2, r3)
       comb_g = func_comb(grad1, grad2, grad3)
       comb_h = func_comb(h1, h2, h3)

    else:
       def func_comb(x1, x2):
           result =dist_pa[0] * x1 + dist_pa[1] * x2
           return result

       comb_r = func_comb(r1, r2)
       comb_g = func_comb(grad1, grad2)
       comb_h = func_comb(h1, h2)

    return comb_r, comb_g, comb_h

def combine_conv(bandwidth, dist_pa, data, shift,Sigma):
    c1=k1[3](bandwidth[0],data,shift[0],Sigma)
    c2=k2[3](bandwidth[1],data,shift[1],Sigma)
    if args.rbf_types == "three_one":
       c3 = k3[3](bandwidth[2], data, shift[2],Sigma)
    
    if args.rbf_types == "three_one":
       def func_comb(x1, x2, x3):
           result =dist_pa[0] * x1 + dist_pa[1] * x2 + dist_pa[2] * x3
           return result
       comb_c = func_comb(c1, c2, c3)
    else:
       def func_comb(x1, x2):
           result =dist_pa[0] * x1 + dist_pa[1] * x2
           return result

       comb_c = func_comb(c1, c2)

    return comb_c

'''
This part is for the evaluation function and training function
'''

#Factor which multiplied by the loss function (OC: Untrain_Param = {"normalizer": 10.0} )
Factor_Loss = 10.0 

def KDE(param, data):

    alpha_1 = jnp.square(param['alpha_1'])/(jnp.square(param['alpha_1'])).sum(axis = 0)
    alpha_2 = jnp.square(param['alpha_2']) / (jnp.square(param['alpha_2'])).sum(axis = 0)
    coeff = jnp.square(param["coeff"]) /(jnp.square(param["coeff"])).sum()

    output = (     (alpha_2 * combine_k(param["width"],
                                               alpha_1,
                                               data,
                                               param["shifts"])).sum(axis = 0)
                                                  ).prod(axis = -1)
    result = ( (coeff * output ) ).sum() * Factor_Loss
    return result

#training loss functional
def KDE_no_bp(param, data):

    alpha_1 = jnp.square(param['alpha_1'])/(jnp.square(param['alpha_1'])).sum(axis = 0)
    alpha_2 = jnp.square(param['alpha_2']) / (jnp.square(param['alpha_2'])).sum(axis = 0)
    coeff = jnp.square(param["coeff"]) /(jnp.square(param["coeff"])).sum()

    KDE_, Grad_, Hessian_ = combine_k_no_bp(param["width"], alpha_1, data, param["shifts"])
    func_alpha = lambda yi: (alpha_2 * yi).sum(axis = 0)
    output_KDE = func_alpha(KDE_)
    output_KDE_grad = func_alpha(Grad_)
    output_KDE_hessian = func_alpha(Hessian_)

    Lp = 0
    
    '''
    func_d_v is the drift function in F-P eqn, need to be changed.
    '''
    fun_d_v = -func_into(data,mu_ex,mu_ffd,tau_m,tau_ee,p_ee,s_ee)
    
    '''
    the part that compute D and the partial term t
    '''
    
    #calculate term of voltage
    for i in [0,1]:
        # calculate drift term
        List_choose = list(set(range(dim)) - set([i]))
        KDE_choose = ( (  output_KDE[:, List_choose]).prod(axis = -1)) * Factor_Loss
        grad_value = ( ( (  output_KDE_grad[:, i] * KDE_choose) * coeff)).sum()
        func_value=(coeff * KDE_choose * output_KDE[:, i]).sum() # need to be calculated
        Lp=1/tau_m*func_value+grad_value * fun_d_v[i]+Lp
        
        #calculate diffusion term (diagonal term)
        Laplacian_value = ( ( ( output_KDE_hessian[:, i] * KDE_choose) * coeff )).sum() 
        
        Lp = jnp.square(sigma_ex)/(2*jnp.square(tau_m))*Laplacian_value + Lp
    
    #calculate term of conductance
    for i in [2,3]:
        # calculate drift term
        List_choose = list(set(range(dim)) - set([i]))
        KDE_choose = ( (  output_KDE[:, List_choose]).prod(axis = -1)) * Factor_Loss
        grad_value = ( ( (  output_KDE_grad[:, i] * KDE_choose) * coeff)).sum()
        func_value=(coeff * KDE_choose * output_KDE[:, i]).sum() # need to be calculated
        Lp=1/tau_ee*func_value+grad_value * fun_d_v[i]+Lp
        
        #calculate diffusion term (diagonal term)
        Laplacian_value = ( ( ( output_KDE_hessian[:, i] * KDE_choose) * coeff )).sum() 
        
        Lp = jnp.square(sigma_ffd*s_ee)*p_ee/(2*jnp.square(tau_ee))*Laplacian_value + Lp
        
    #calculate diffusion term (non-diagonal term)
    KDE_derivative_v = ((output_KDE_grad[:,0:2]).prod(axis = -1)) * (output_KDE[:, 2:].prod(axis = -1)) * Factor_Loss
    KDE_derivative_g = ((output_KDE_grad[:,2:]).prod(axis = -1)) * (output_KDE[:, 0:2].prod(axis = -1)) * Factor_Loss
    
    Lp = c*jnp.square(sigma_ex/tau_m) *  ( ( KDE_derivative_v * coeff )).sum()   + Lp
    Lp = jnp.square(sigma_ffd*s_ee*p_ee/tau_ee) *  ( ( KDE_derivative_g * coeff )).sum()   + Lp
    
    #calculate Rp
    
    Rp = 0.0
    
    ep_delta=1
    for i in [0,1]:
        List_choose = list(set(range(dim)) - set([i]))
        # achive information about P(v_th,v2) or P(v1,v_th)
        d = data.copy()
        d = d.at[i].set(V_thres)
        KDE_, Grad_, Hessian_ = combine_k_no_bp(param["width"], alpha_1, d, param["shifts"])
        func_alpha = lambda yi: (alpha_2 * yi).sum(axis = 0)
        output_KDE = func_alpha(KDE_)
        output_KDE_grad = func_alpha(Grad_)
        output_KDE_hessian = func_alpha(Hessian_)
        #Calate K_i
        delta_ep=wendland_1_test2(ep_delta, data[i], V_reset)
        Sigma=sigma_ex**2/(2*tau_m)*(1-jnp.exp(-2*tau_r/tau_m))
    
        '''
        Integral_scaling = gaussian_integral(jnp.sqrt(Sigma), shift)
        '''
        '''
        K_i=1/jnp.sqrt(2*jnp.pi*Sigma)*(-sigma**2/(2*tau_m**2))*output_KDE_grad[:, i] # normalization need to be calculated
        # calculate integral
        integral=combine_conv(param["width"],alpha_1, d, param["shifts"], Sigma)
        output_integral = func_alpha(integral)
        '''
    
        K_i=(-sigma_ex**2/(2*tau_m**2))*output_KDE_grad[:, i]
        output_integral = output_KDE
        Rp=Rp+((Factor_Loss*(output_integral[:,List_choose]).prod(axis = -1)*K_i)*coeff).sum()*delta_ep
    return Lp+Rp
    
'''
        if i >1:
            Lp = Laplacian_value * V_store["V"] + 2.0 * grad_value * V_store["div_V"] + Lp

    #-divergence + part of diffusion:
    Lp = ( (coeff * KDE_choose * output_KDE[:, -1]).sum() ) * (V_store["laplace_V"] + 2.0 * V_store["V_3_4"] - div_into(data)) + Lp

    #compute second order partial derivative with V
    List_o, List_d =[0, 1], [2, 3]
    KDE_partial_choose = ( (  output_KDE[:, List_o]).prod(axis = -1)) * Factor_Loss
    KDE_derivative_choose = (output_KDE_grad[:, List_d]).prod(axis = -1)

    Lp = 2.0 * ( ( ( ( KDE_derivative_choose * KDE_partial_choose ) * coeff )).sum() ) * V_store["V"] + Lp
'''



#vectorized evaluation function
vec_KDE = jit( vmap(KDE, in_axes = (None, 0) , out_axes = 0) )

#vectorized Lp
vectorize_Lp = jit( vmap(KDE_no_bp, in_axes = (None, 0) , out_axes = 0) )

'''
This part is for the real loss functional
'''

'''
This is for the KDE_no_bp part
'''

def Monte_Functional(param, data_batch):
    L_p_result = vectorize_Lp(param, data_batch)
    Output = (jnp.square(L_p_result)).mean()
    return Output

'''
boundary condition
'''


'''
penalty of constraints of parameters
'''
#OC: def H_S(param)
def penalty_constraint_param(param):
    #r_j - |s_{ij}^{(\ell)} - O_j|
    r_1_constraint_result = jnp.abs(param["shifts"] - Center_O) - r

    #|s_{ij}^{(\ell)} - O_j| < r_j
    shifts_constraint_penalty = (jax.nn.relu(r_1_constraint_result)).mean()
    
    #|h_{ij}^{(\ell)}| < |r_j - |s_{ij}^{(\ell)} - O_j||
    width_constraint_penalty =  ( jax.nn.relu(  jnp.abs(param["width"])[0] + scale_bound[Chosen_List[0]] -  jnp.abs(r_1_constraint_result[0]) ).mean()
                                 + jax.nn.relu( jnp.abs(param["width"])[1] + scale_bound[Chosen_List[1]] -  jnp.abs(r_1_constraint_result[1]) ).mean()
                                 )
    if args.rbf_types == "three_one":
        width_constraint_penalty = width_constraint_penalty + jax.nn.relu( jnp.abs(param["width"])[2] + scale_bound[Chosen_List[2]] -  jnp.abs(r_1_constraint_result[2]) ).mean()
    
    Penalty = width_constraint_penalty + shifts_constraint_penalty

    return Penalty


'''
penalty of boundary condtion
'''

def Boundary_Control(param):

    left_end = Center_O - r * jnp.array([1.0] * dim)
    right_end = Center_O + r * jnp.array([1.0] * dim)

    alpha_1 = jnp.square(param['alpha_1'])/(jnp.square(param['alpha_1'])).sum(axis = 0)
    alpha_2 = jnp.square(param['alpha_2']) / (jnp.square(param['alpha_2'])).sum(axis = 0)

    output_left =  (alpha_2 * combine_k(param["width"],
                                               alpha_1,
                                               left_end,
                                               param["shifts"])  ).sum(axis = 0)
    
    output_right =  (alpha_2 * combine_k(param["width"],
                                               alpha_1,
                                               right_end,
                                               param["shifts"])  ).sum(axis = 0)
    
    result = (jnp.abs(output_left)).mean() + (jnp.abs(output_right)).mean()
    return result

'''
Loss function
'''
Loss_Params = {"reg_bound": 1e2,"reg_diff": 1.0, "penal_param":5e4}
def Loss_Func(param, data_batch):

    loss = ( Loss_Params["reg_diff"] * Monte_Functional(param, data_batch)
             + Loss_Params["reg_bound"] * Boundary_Control(param)
             + Loss_Params["penal_param"] * penalty_constraint_param(param)
             )
    return loss


#seed generator / sample generation
seeds = np.arange(220000, 2500000, dtype = int )#starting at 200000

'''
build the saving directories
'''

bw= "rk" + str(Shape[-2]) + "_" + "r" + str(r) + "_" + 'm_use' + str(m_use) + 'rbf_types' + args.rbf_types
kn = "kl"+ Chosen_List[0][0] + Chosen_List[1][0] + Chosen_List[2][0]

Path_Name = args.test_case[-3:] + "_" + "_" + bw + "_" + kn + "/"

Save_Path ='./experiment/example1/'+ Path_Name

if not os.path.isdir(Save_Path):
    os.makedirs(Save_Path,exist_ok=True)



'''
This part is for training function
'''


#Training Function
def fit(loss_func, optimizer, resampler, Para):

    losses = []
    Opt_State = optimizer.init(Para)

    @jit
    def step(param, opt_state, batch):
        loss_value, grads = jax.value_and_grad(loss_func)(param, batch)
        updates, opt_state = optimizer.update(grads, opt_state, param)
        param = optax.apply_updates(param, updates)
        return loss_value, param, opt_state
    Loss = []
    fh = open(Save_Path + 'output.txt', 'w')
    original_stderr = sys.stderr
    sys.stderr = fh

    for epoch in tqdm.tqdm(range(args.epochs)):

        Dataset = resampler(epoch)

        loss_value, Para, Opt_State = step(Para, Opt_State, Dataset)

        losses.append(loss_value)
        Loss.append(loss_value)

        if epoch % 500 == 0:
           print('loss value', np.mean(np.array(Loss)) )
           print("epoch", epoch)
           Loss = []

    sys.stderr = original_stderr
    fh.close()

    return losses, Para


seeds = np.arange(220000, 2500000, dtype = int ) #starting at 200000

def resample(i):
    key = jax.random.PRNGKey(seeds[i])
    Data_jax = jax.random.uniform(key, minval = Center_O -r, maxval = Center_O + r, shape = ((args.batches, dim)) )
    '''
    lab1=int(args.batches/16)
    lab2=int(args.batches/8)
    Data_jax=Data_jax.at[0:lab1,1].set(args.V_reset)
    Data_jax=Data_jax.at[lab1:lab2,0].set(args.V_reset)
    '''
    return Data_jax

Schedule = optax.polynomial_schedule(init_value = 9e-4, end_value=8e-6, power = 2, transition_steps = 70000, transition_begin=10000)

save_path_param = Save_Path + "Param.pkl"

def training():

    opt = optax.chain(
             optax.clip(100.0),
             optax.lion(learning_rate= Schedule))
    optimizer =optax.MultiSteps(opt, every_k_schedule= 1)
    losse_, Param = fit(Loss_Func, optimizer, resample, initial_param)
    f = open(save_path_param,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(Param,f)
    f.close()
    return Param, losse_

if args.Train == True:
   Final_Param, Loss = training()

else:
   Final_Param = pickle.load(open(save_path_param, "rb"))


'''
This part is for Evaluation
'''

#Number of test points
num_test = 100000
#test radius
test_radius = 1.0
#generate the testing points
np.random.seed(0)
Test_Points = np.random.uniform(low=-test_radius, high=test_radius, size=(num_test, dim))

#This three functions are for relative error computation
def Relative_Error_dist(Estimate, Accurate):
    Error = jnp.abs( (Estimate - Accurate))
    Error_result = Error/Accurate
    return Error_result

def data_filter_high_probability(my_array, Accurate, high_probability):
    result = my_array[Accurate > high_probability]
    return result


def batch_result(data, batch_size, param):
    num_batches = data.shape[0] // batch_size + (data.shape[0] % batch_size != 0)
    batches = np.array_split(data, num_batches)
    KDE_ESTIMATE = vec_KDE(param, batches[0])
    for batch in batches[1:]:
        KDE_ESTIMATE = np.concatenate((KDE_ESTIMATE, vec_KDE(param, batch)) )
    return KDE_ESTIMATE/10.0

def relative_error_high_prob(prob_area, test_points, vec_dens_accurate, param, error_description):
    kde_accurate = vec_dens_accurate(test_points)
    points_hpa = data_filter_high_probability(test_points, kde_accurate, prob_area)
    kde_hpa = batch_result(points_hpa, 5000, param)
    acc_hpa = vec_dens_accurate(points_hpa)
    rel_error_hpa = Relative_Error_dist(kde_hpa, acc_hpa)
    df_describe_error = pd.DataFrame(rel_error_hpa)
    error_data_frame = df_describe_error.describe(percentiles = error_description)
    return error_data_frame

Prob = [0.01, 0.05, 0.1]

Loss_Range = range(args.epochs)
plt.plot(Loss_Range, np.log10(np.array(Loss)), color='green')
plt.xlabel('Epochs')
plt.ylabel('Losses(log_10)')
plt.savefig(Save_Path + 'Losses_Graph')

np.save(Save_Path + 'Losses', np.array(Loss))

# 设置绘图参数
grid_size = 200  # 网格大小，可调整
r_plot = args.r_sde * 0.8  # 绘图范围，比训练范围小一些以聚焦重要区域

# 创建评估网格
print(f"´´½¨ {grid_size}x{grid_size} ÆÀ¹ÀÍø¸ñ...")
x = np.linspace(-r_plot, r_plot, grid_size)
y = np.linspace(-r_plot, r_plot, grid_size)
X, Y = np.meshgrid(x, y)
Z=np.full_like(X,0)
R=np.full_like(X,0)
grid_points = np.column_stack([X.ravel(), Y.ravel(),Z.ravel(),R.ravel()])

# 使用已有的vec_KDE函数评估概率密度
print("ÆÀ¹À¸ÅÂÊÃÜ¶È...")
batch_size = 500
Z_list = []

# 使用tqdm显示进度
for i in tqdm.tqdm(range(0, len(grid_points), batch_size), desc="ÆÀ¹À½ø¶È"):
    end_idx = min(i + batch_size, len(grid_points))
    batch = grid_points[i:end_idx]
    # 使用源代码中定义的vec_KDE函数
    densities = vec_KDE(Final_Param, batch)
    Z_list.append(densities)

Z = np.concatenate(Z_list).reshape(X.shape)

# 归一化概率密度（使其积分为1）
dx = x[1] - x[0]
dy = y[1] - y[0]
integral = np.sum(Z) * dx * dy
Z_normalized = Z / integral

print(f"¸概率密度积分: {integral:.6f}")
print(f"归一化后积分: {np.sum(Z_normalized) * dx * dy:.6f}")

# 创建热力图
print("生成热力图...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'稳态概率密度分布 - 参数: I={args.mu_ex}, tau_m={args.tau_m}, sigma={args.sigma_ex}, c={args.c}', 
             fontsize=16, fontweight='bold')

# 1. 标准热力图
ax1 = axes[0, 0]
im1 = ax1.imshow(Z_normalized, extent=[-r_plot, r_plot, -r_plot, r_plot], 
                origin='lower', cmap='hot', aspect='auto')
ax1.set_xlabel('v?', fontsize=12)
ax1.set_ylabel('v?', fontsize=12)
ax1.set_title('标准热力图', fontsize=14)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
plt.colorbar(im1, ax=ax1, label='概率密度')
plt.savefig(Save_Path + 'Distribution')
