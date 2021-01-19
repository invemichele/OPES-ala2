#! /usr/bin/env python3

### Kernel density estimation of the marginals over phi and energy  ###
### both the sampled one (biased) and the reweighted one (unbiased) ###

import sys
import numpy as np
pandas_on=True
try:
  import pandas as pd
except ModuleNotFoundError as err:
  print(err,file=sys.stderr) 
  print(" +++ WARNING: install pandas for faster I/O +++",file=sys.stderr)
  pandas_on=False
  pass

# Toggles
print_prob=True
print_fes=True
Kb=0.0083144621 #kj/mol
kbt=Kb*300
filename='Colvar.data'
if len(sys.argv)>1: #read filename if provided
  filename=sys.argv[1]
if not (print_prob or print_fes):
  sys.exit('  nothing to calculate')

# Read colvar
phi_col=1
ene_col=3
bias_col=4
if pandas_on:
  data=pd.read_table(filename,dtype=float,sep='\s+',comment='#',header=None,usecols=[phi_col,ene_col,bias_col])
  phi=np.array(data.iloc[:,0])
  ene=np.array(data.iloc[:,1])
  bias=np.array(data.iloc[:,2])
  del data
else:
  phi,ene,bias=np.loadtxt(filename,usecols=(phi_col,ene_col,bias_col),unpack=True);
weight=np.exp((bias-np.amax(bias))/kbt)
del bias

# KDE and weighted KDE
def build_marginal(cv,bandwidth,cv_grid,periodic=False):
  tot_bins=len(cv_grid)
  prob=np.zeros(tot_bins)
  rew_prob=np.zeros(tot_bins)
  if periodic:
    period=cv_grid[-1]-cv_grid[0]
  for i in range(tot_bins):
    if periodic:
      dx=np.absolute(cv_grid[i]-cv)
      arg2=(np.minimum(dx,period-dx)/bandwidth)**2
    else:
      arg2=np.power((cv_grid[i]-cv)/bandwidth,2)
    exp_arg2=np.exp(-0.5*arg2)
    prob[i]=np.sum(exp_arg2)
    rew_prob[i]=np.sum(weight*exp_arg2)
  prob/=np.trapz(prob,cv_grid)
  rew_prob/=np.trapz(rew_prob,cv_grid)
  return prob,rew_prob

# Calculate effective sampling size
Neff=np.sum(weight)**2/np.sum(weight**2)
head='cv  sampled_prob  reweighted_prob # Neff/N= %g'%(Neff/len(weight))
head_fes='cv  sampled_fes  reweighted_fes'

# Marginal over phi
sigma_phi=0.1
grid_min=-np.pi
grid_max=np.pi
tot_bins=300
phi_grid=np.linspace(grid_min,grid_max,tot_bins)
prob,rew_prob=build_marginal(phi,sigma_phi,phi_grid,True)
if print_prob:
  np.savetxt('Marginal-phi.data',np.c_[phi_grid,prob,rew_prob],header=head,fmt='%g')
if print_fes:
  fes=-kbt*np.log(prob/np.amax(prob))
  rew_fes=-kbt*np.log(rew_prob/np.amax(rew_prob))
  np.savetxt('FES-phi.data',np.c_[phi_grid,fes,rew_fes],header=head_fes,fmt='%g')

# Marginal over energy
sigma_ene=3
grid_min=-100
grid_max=300
tot_bins=300
ene_grid=np.linspace(grid_min,grid_max,tot_bins)
prob,rew_prob=build_marginal(ene,sigma_ene,ene_grid)
if print_prob:
  np.savetxt('Marginal-ene.data',np.c_[ene_grid,prob,rew_prob],header=head,fmt='%g')
if print_fes:
  fes=-kbt*np.log(prob/np.amax(prob))
  rew_fes=-kbt*np.log(rew_prob/np.amax(rew_prob))
  np.savetxt('FES-ene.data',np.c_[ene_grid,fes,rew_fes],header=head_fes,fmt='%g')

