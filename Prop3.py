# sensitivity to positive disturbance
import draw_multiple_frontiers as drawmf
import pystoned as pysd
import numpy as np
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS

sSigmaV, sSigmaU = 0.1, 0.1
nSample = 40
sMinX, sMaxX = 0.01, 5
ProbDisturb = 0.8
TrueAlpha, TrueBeta  = -1, 0.6



InputX = np.random.uniform(sMinX, sMaxX, (nSample, 1))
compError = np.random.normal(0, sSigmaV, nSample)-np.abs(np.random.normal(0, sSigmaU, nSample))
OutputY = np.exp(TrueAlpha+(TrueBeta*np.log(InputX)).reshape(nSample)+compError)
TrueY = np.exp(TrueAlpha+TrueBeta*np.log(InputX))

vtau=[0.9,0.7,0.5,0.3,0.1]
#vtau=[0.9]

CSQR_EstY=np.zeros((nSample,len(vtau)))                  
for itau in range(len(vtau)):
    CSQR = pysd.CSQR.CSQR(y=OutputY, x=InputX, tau=vtau[itau], z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
    CSQR.optimize(OPT_LOCAL, solver='mosek')
    CSQR_EstY[:,itau] = np.array(CSQR.get_frontier()).T
    

CQR_EstY=np.zeros((nSample,len(vtau)))                  
for itau in range(len(vtau)):
    CQR = pysd.CQER.CQR(y=OutputY, x=InputX, tau=vtau[itau], z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
    CQR.optimize(OPT_LOCAL, solver='mosek')
    CQR_EstY[:,itau] = np.array(CQR.get_frontier()).T

CER_EstY=np.zeros((nSample,len(vtau)))                  
for itau in range(len(vtau)):
    CER = pysd.CQER.CER(y=OutputY, x=InputX, tau=vtau[itau], z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
    CER.optimize(OPT_LOCAL, solver='mosek')
    CER_EstY[:,itau] = np.array(CER.get_frontier()).T

DEAtau=[1]
Full_EstY=np.zeros((nSample,len(DEAtau)))                  
for itau in range(len(DEAtau)):
    DEA_tau = 1-(1/(3*nSample))
    Full = pysd.CQER.CQR(y=OutputY, x=InputX, tau=DEA_tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
    Full.optimize(OPT_LOCAL, solver='mosek')
    Full_EstY[:,itau]= np.array(Full.get_frontier()).T


drawmf.draw_multiple_frontiers(InputX, OutputY, Full_EstY, DEAtau, 'Full frontier', 'eg3_Full')  
drawmf.draw_multiple_frontiers(InputX, OutputY, CSQR_EstY, vtau, 'Expectile frontier', 'eg3_CER')    
drawmf.draw_multiple_frontiers(InputX, OutputY, CQR_EstY, vtau, 'Quantile frontier', 'eg3_CQR')
drawmf.draw_multiple_frontiers(InputX, OutputY, CER_EstY, vtau, 'Superquantile frontier', 'eg3_CSQR')


# CQR1 = pysd.CQER.CQR(y=OutputY, x=InputX, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# CQR1.optimize(OPT_LOCAL, solver='mosek')
# CQR1_EstY = np.array(CQR1.get_frontier()).T

# CER1 = pysd.CQER.CER(y=OutputY, x=InputX, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# CER1.optimize(OPT_LOCAL, solver='mosek')
# CER1_EstY = np.array(CER1.get_frontier()).T


# tau = 0.2

# CSQR2 = pysd.CSQR.CSQR(y=OutputY, x=InputX, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# CSQR2.optimize(OPT_LOCAL, solver='mosek')
# CSQR2_EstY = np.array(CSQR2.get_frontier()).T

# CQR2 = pysd.CQER.CQR(y=OutputY, x=InputX, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# CQR2.optimize(OPT_LOCAL, solver='mosek')
# CQR2_EstY = np.array(CQR2.get_frontier()).T

# CER2 = pysd.CQER.CER(y=OutputY, x=InputX, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
# CER2.optimize(OPT_LOCAL, solver='mosek')
# CER2_EstY = np.array(CER2.get_frontier()).T

# drawdf.draw_disturbed_frontier(InputX, OutputY, OutputY, CSQR1_EstY, CSQR2_EstY, 'Superquantile frontier', 'eg3_CSQR')
# drawdf.draw_disturbed_frontier(InputX, OutputY, OutputY, CQR1_EstY, CQR2_EstY, 'Quantile frontier', 'eg3_CQR')
# drawdf.draw_disturbed_frontier(InputX, OutputY, OutputY, CER1_EstY, CER2_EstY, 'Expectile frontier', 'eg3_CER')

