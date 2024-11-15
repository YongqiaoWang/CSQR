import pystoned as pysd
import numpy as np
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
import matplotlib.pyplot as plt
import plotly

tau = 0.8

from pystoned.dataset import load_Tim_Coelli_frontier

# import all data
data = load_Tim_Coelli_frontier(x_select=['capital', 'labour'], y_select=['output'])
x, y = data.x, data.y


x1min=np.amin(x[:,0])
x1max=np.amax(x[:,0])
x2min=np.amin(x[:,1])
x2max=np.amax(x[:,1])
ymin=np.amin(y)
ymax=np.amax(y)

num_grid=300
draw_x1=np.linspace(x1min,x1max,num_grid)
draw_x2=np.linspace(x2min,x2max,num_grid)
mesh_x1,mesh_x2=np.meshgrid(draw_x1,draw_x2)

# CSQR        
CSQR = pysd.CSQR.CSQR(y=y, x=x, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
CSQR.optimize(OPT_LOCAL, solver='mosek')
#CSQR1_EstY = np.array(CSQR.get_frontier()).T
predicted_frontier_CSQR=np.zeros((num_grid,num_grid))
for ix1 in range(draw_x1.size):
    for ix2 in range(draw_x2.size):
        test_x=np.array([draw_x1[ix1],draw_x2[ix2]])
        test_x=test_x.reshape(1,2)
        predict_y=CSQR.get_predict(test_x)
        predicted_frontier_CSQR[ix1,ix2]=predict_y[0,0]
        
fig = plt.figure()
#fig = plt.gcf()
ax = fig.subplots()
ax.set_xlabel("capital")
ax.set_ylabel("labour")    
plt.scatter(x=x[:,0],y=x[:,1],c=y.flatten(),vmin=ymin,vmax=ymax,cmap='viridis',marker = 'o') 
plt.colorbar()
ylevels=np.linspace(8,35,10)
CS=ax.contour(mesh_x1,mesh_x2,predicted_frontier_CSQR,levels=ylevels,cmap='viridis',vmin=ymin,vmax=ymax)    #cmap='viridis',
ax.clabel(CS,inline=True,fontsize=10)
plt.savefig('Coelli_CSQR.pdf', format="pdf")   
plt.show()

# Translog
nSample = len(y)
nInput = len(x[0])
translog_x=np.zeros((nSample,nInput+nInput*nInput))
for ix in range(len(y)):
    obs=np.log(x[ix])
    obs_add1=np.append(obs, 1)
    translog_x[ix]=np.kron(obs_add1,obs)
        
LSQR_translog = pysd.LSQR.LSQR(y=np.log(y), x=translog_x, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
LSQR_translog.optimize(OPT_LOCAL, solver='mosek')
#LSQR_linear_EstY = np.array(LSQR_linear.get_frontier()).T
predicted_frontier_translog=np.zeros((num_grid,num_grid))
for ix1 in range(draw_x1.size):
    for ix2 in range(draw_x2.size):
        test_x=np.log(np.array([draw_x1[ix1],draw_x2[ix2]]))
        test_x_add1=np.append(test_x, 1)
        test_translog_x=np.kron(test_x_add1,test_x)
        test_translog_x=test_translog_x.reshape(1,6)
        predicted_frontier_translog[ix1,ix2]=np.exp(LSQR_translog.get_predict(test_translog_x))

fig = plt.figure()
#fig = plt.gcf()
ax = fig.subplots()
ax.set_xlabel("capital")
ax.set_ylabel("labour")    
plt.scatter(x=x[:,0],y=x[:,1],c=y.flatten(),vmin=ymin,vmax=ymax,cmap='viridis',marker = 'o') 
plt.colorbar()
ylevels=np.linspace(8,35,10)
CS=ax.contour(mesh_x1,mesh_x2,predicted_frontier_translog,levels=ylevels,cmap='viridis',vmin=ymin,vmax=ymax)    #cmap='viridis',
ax.clabel(CS,inline=True,fontsize=10)
plt.savefig('Coelli_translog.pdf', format="pdf")  
plt.show()

# Cobb-Douglas
LSQR_cobb = pysd.LSQR.LSQR(y=np.log(y), x=np.log(x), tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
LSQR_cobb.optimize(OPT_LOCAL, solver='mosek')
predicted_frontier_cobb=np.zeros((num_grid,num_grid))
for ix1 in range(draw_x1.size):
    for ix2 in range(draw_x2.size):
        test_x=np.array([np.log(draw_x1[ix1]),np.log(draw_x2[ix2])])
        test_x=test_x.reshape(1,2)
        predicted_frontier_cobb[ix1,ix2]=np.exp(LSQR_cobb.get_predict(test_x))

fig = plt.figure()
#fig = plt.gcf()
ax = fig.subplots()
ax.set_xlabel("capital")
ax.set_ylabel("labour")    
plt.scatter(x=x[:,0],y=x[:,1],c=y.flatten(),vmin=ymin,vmax=ymax,cmap='viridis',marker = 'o') 
plt.colorbar()
ylevels=np.linspace(8,35,10)
CS=ax.contour(mesh_x1,mesh_x2,predicted_frontier_cobb,levels=ylevels,cmap='viridis',vmin=ymin,vmax=ymax)    #cmap='viridis',
ax.clabel(CS,inline=True,fontsize=10)
plt.savefig('Coelli_cobb.pdf', format="pdf")  
plt.show()

# Linear
LSQR_linear = pysd.LSQR.LSQR(y=y, x=x, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
LSQR_linear.optimize(OPT_LOCAL, solver='mosek')
#LSQR_linear_EstY = np.array(LSQR_linear.get_frontier()).T
predicted_frontier_linear=np.zeros((num_grid,num_grid))
for ix1 in range(draw_x1.size):
    for ix2 in range(draw_x2.size):
        test_x=np.array([draw_x1[ix1],draw_x2[ix2]])
        test_x=test_x.reshape(1,2)
        predicted_frontier_linear[ix1,ix2]=LSQR_linear.get_predict(test_x)

fig = plt.figure()
#fig = plt.gcf()
ax = fig.subplots()
ax.set_xlabel("capital")
ax.set_ylabel("labour")    
plt.scatter(x=x[:,0],y=x[:,1],c=y.flatten(),vmin=ymin,vmax=ymax,cmap='viridis',marker = 'o') 
plt.colorbar()
ylevels=np.linspace(8,35,10)
CS=ax.contour(mesh_x1,mesh_x2,predicted_frontier_linear,levels=ylevels,cmap='viridis',vmin=ymin,vmax=ymax)    #cmap='viridis',
ax.clabel(CS,inline=True,fontsize=10)
plt.savefig('Coelli_linear.pdf', format="pdf")        
plt.show()

# # CSQR        
# CSQR = pysd.CSQR.CSQR(y=y, x=x, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# CSQR.optimize(OPT_LOCAL, solver='mosek')
# #CSQR1_EstY = np.array(CSQR.get_frontier()).T
# predicted_frontier_CSQR=np.zeros((num_grid,num_grid))
# for ix1 in range(draw_x1.size):
#     for ix2 in range(draw_x2.size):
#         test_x=np.array([draw_x1[ix1],draw_x2[ix2]])
#         test_x=test_x.reshape(1,2)
#         predict_y=CSQR.get_predict(test_x)
#         predicted_frontier_CSQR[ix1,ix2]=predict_y[0,0]
        
# fig = plt.figure()
# #fig = plt.gcf()
# ax = fig.subplots()
# ax.set_xlabel("capital")
# ax.set_ylabel("labour")    
# plt.scatter(x=x[:,0],y=x[:,1],c=y.flatten(),vmin=ymin,vmax=ymax,cmap='viridis',marker = 'o') 
# plt.colorbar()
# ylevels=np.linspace(8,35,10)
# CS=ax.contour(mesh_x1,mesh_x2,predicted_frontier_CSQR,levels=ylevels,cmap='viridis',vmin=ymin,vmax=ymax)    #cmap='viridis',
# ax.clabel(CS,inline=True,fontsize=10)
# plt.show()
# plt.savefig('Coelli_CSQR.pdf', format="pdf")   

  
  





# sSigmaV, sSigmaU, sDisturbSigma = 0.2, 0.2, 0.07

# sMinX, sMaxX = 1, 10
# ProbDisturb = 0.8
# TrueAlpha, TrueBeta  = 3, 1
# InputX = np.random.uniform(sMinX, sMaxX, (nSample, 1))
# compError = np.random.normal(0, sSigmaV, nSample)-np.abs(np.random.normal(0, sSigmaU, nSample))
# TrueY = TrueAlpha+TrueBeta*np.log(InputX)
# OutputY = (TrueAlpha+TrueBeta*np.log(InputX)).reshape(nSample)+compError



# CQR1 = pysd.CQER.CQR(y=OutputY, x=InputX, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# CQR1.optimize(OPT_LOCAL, solver='mosek')
# CQR1_EstY = np.array(CQR1.get_frontier()).T

# CER1 = pysd.CQER.CER(y=OutputY, x=InputX, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# CER1.optimize(OPT_LOCAL, solver='mosek')
# CER1_EstY = np.array(CER1.get_frontier()).T

# DEA_tau = 1-(1/(3*nSample))
# DEA1 = pysd.CQER.CQR(y=OutputY, x=InputX, tau=DEA_tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# DEA1.optimize(OPT_LOCAL, solver='mosek')
# DEA1_EstY = np.array(DEA1.get_frontier()).T


# DisturbY = OutputY*(1+abs(np.random.normal(0, sDisturbSigma, nSample))*(OutputY > CSQR1_EstY)
#                   *(OutputY > CQR1_EstY)*(OutputY > CER1_EstY)
#                   *np.random.binomial(1, ProbDisturb, nSample))

# CSQR2 = pysd.CSQR.CSQR(y=DisturbY, x=InputX, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# CSQR2.optimize(OPT_LOCAL, solver='mosek')
# CSQR2_EstY = np.array(CSQR2.get_frontier()).T

# CQR2 = pysd.CQER.CQR(y=DisturbY, x=InputX, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# CQR2.optimize(OPT_LOCAL, solver='mosek')
# CQR2_EstY = np.array(CQR2.get_frontier()).T

# CER2 = pysd.CQER.CER(y=DisturbY, x=InputX, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# CER2.optimize(OPT_LOCAL, solver='mosek')
# CER2_EstY = np.array(CER2.get_frontier()).T

# DEA2 = pysd.CQER.CQR(y=DisturbY, x=InputX, tau=DEA_tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# DEA2.optimize(OPT_LOCAL, solver='mosek')
# DEA2_EstY = np.array(DEA2.get_frontier()).T

# drawdf.draw_disturbed_frontier(InputX, OutputY, DisturbY, DEA1_EstY, DEA2_EstY, 'Full frontier', 'eg1_Full')
# drawdf.draw_disturbed_frontier(InputX, OutputY, DisturbY, CER1_EstY, CER2_EstY, 'Expectile frontier', 'eg1_CER')
# drawdf.draw_disturbed_frontier(InputX, OutputY, DisturbY, CQR1_EstY, CQR2_EstY, 'Quantile frontier', 'eg1_CQR')
# drawdf.draw_disturbed_frontier(InputX, OutputY, DisturbY, CSQR1_EstY, CSQR2_EstY, 'Superquantile frontier', 'eg1_CSQR')





