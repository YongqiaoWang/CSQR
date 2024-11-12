# sensitivity to negative disturbance

import draw_disturbed_frontier as drawdf
import pystoned as pysd
import numpy as np
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS

# tau = 0.5
# sSigmaV, sSigmaU, sDisturbSigma = 0.1, 0.1, 0.2
# nSample = 40
# sMinX, sMaxX = 0.01, 5
# ProbDisturb = 0.6
# TrueAlpha, TrueBeta  = -1, 0.7

# InputX = np.random.uniform(sMinX, sMaxX, (nSample, 1))
# compError = np.random.normal(0, sSigmaV, nSample)-np.abs(np.random.normal(0, sSigmaU, nSample))
# OutputY = np.exp(TrueAlpha+(TrueBeta*np.log(InputX)).reshape(nSample)+compError)
# TrueY = np.exp(TrueAlpha+TrueBeta*np.log(InputX))



sSigmaV, sSigmaU, sDisturbSigma = 0.5, 1, 0.2
nSample = 100
sMinX, sMaxX = 1, 9
ProbDisturb = 0.8
TrueAlpha, TrueBeta  = 3, 2
InputX = np.random.uniform(sMinX, sMaxX, (nSample, 1))
compError = np.random.normal(0, sSigmaV, nSample)-np.abs(np.random.normal(0, sSigmaU, nSample))
TrueY = TrueAlpha+TrueBeta*np.log(InputX)
OutputY = (TrueAlpha+TrueBeta*np.log(InputX)).reshape(nSample)+compError

SQ_tau = 0.5112
CSQR1 = pysd.CSQR.CSQR(y=OutputY, x=InputX, tau=SQ_tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
CSQR1.optimize(OPT_LOCAL, solver='mosek')
CSQR1_EstY = np.array(CSQR1.get_frontier()).T
CSQR1_EstY_By = np.array(CSQR1.get_frontierbyProduct()).T

Q_tau=0.8
CQR1 = pysd.CQER.CQR(y=OutputY, x=InputX, tau=Q_tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
CQR1.optimize(OPT_LOCAL, solver='mosek')
CQR1_EstY = np.array(CQR1.get_frontier()).T

E_tau=0.8745
CER1 = pysd.CQER.CER(y=OutputY, x=InputX, tau=E_tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
CER1.optimize(OPT_LOCAL, solver='mosek')
CER1_EstY = np.array(CER1.get_frontier()).T

DEA_tau = 1-(1/(3*nSample))
DEA1 = pysd.CQER.CQR(y=OutputY, x=InputX, tau=DEA_tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
DEA1.optimize(OPT_LOCAL, solver='mosek')
DEA1_EstY = np.array(DEA1.get_frontier()).T


DisturbY = OutputY*(1-abs(np.random.normal(0, sDisturbSigma, nSample))*(OutputY < CSQR1_EstY_By)
                  *(OutputY < CQR1_EstY)*(OutputY < CER1_EstY)
                  *np.random.binomial(1, ProbDisturb, nSample))

CSQR2 = pysd.CSQR.CSQR(y=DisturbY, x=InputX, tau=SQ_tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
CSQR2.optimize(OPT_LOCAL, solver='mosek')
CSQR2_EstY = np.array(CSQR2.get_frontier()).T

CQR2 = pysd.CQER.CQR(y=DisturbY, x=InputX, tau=Q_tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
CQR2.optimize(OPT_LOCAL, solver='mosek')
CQR2_EstY = np.array(CQR2.get_frontier()).T

CER2 = pysd.CQER.CER(y=DisturbY, x=InputX, tau=E_tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
CER2.optimize(OPT_LOCAL, solver='mosek')
CER2_EstY = np.array(CER2.get_frontier()).T

DEA2 = pysd.CQER.CQR(y=DisturbY, x=InputX, tau=DEA_tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
DEA2.optimize(OPT_LOCAL, solver='mosek')
DEA2_EstY = np.array(DEA2.get_frontier()).T

drawdf.draw_disturbed_frontier(InputX, OutputY, DisturbY, DEA1_EstY, DEA2_EstY, 'Full frontier', 'eg2_Full')
drawdf.draw_disturbed_frontier(InputX, OutputY, DisturbY, CER1_EstY, CER2_EstY, 'Expectile frontier', 'eg2_CER')
drawdf.draw_disturbed_frontier(InputX, OutputY, DisturbY, CQR1_EstY, CQR2_EstY, 'Quantile frontier', 'eg2_CQR')
drawdf.draw_disturbed_frontier(InputX, OutputY, DisturbY, CSQR1_EstY, CSQR2_EstY, 'Superquantile frontier', 'eg2_CSQR')





