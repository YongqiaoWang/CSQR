import pystoned as pysd
import numpy as np
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
import matplotlib.pyplot as plt
import plotly
import matplotlib.pyplot as plt


sSigmaV, sSigmaU = 0.5, 1
nSample = 100
sMinX, sMaxX = 1, 9
TrueAlpha, TrueBeta  = 3, 2
InputX = np.random.uniform(sMinX, sMaxX, (nSample, 1))
compError = np.random.normal(0, sSigmaV, nSample)-np.abs(np.random.normal(0, sSigmaU, nSample))
TrueY = TrueAlpha+TrueBeta*np.log(InputX)
OutputY = (TrueAlpha+TrueBeta*np.log(InputX)).reshape(nSample)+compError



#OptTau = 0.6523 #sSigmaV, sSigmaU = 0.5, 1
OptTau = 0.2208 #sSigmaV, sSigmaU = 1, 0.5

CSQR = pysd.CSQR.CSQR(y=OutputY, x=InputX, tau=OptTau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
CSQR.optimize(OPT_LOCAL, solver='mosek')
CSQR_EstY= np.array(CSQR.get_frontier()).T


fig = plt.figure()
ax = fig.subplots()
ax.scatter(InputX[:,0], OutputY, marker='o', color='white', edgecolor='k', s=12,label='Samples')
drawX = np.arange(sMinX,sMaxX,0.01)
drawY = TrueAlpha+TrueBeta*np.log(drawX)
ax.plot(drawX, drawY, color='k', label='True production function')
data = (np.stack([(InputX[:, 0]), CSQR_EstY], axis=1))
data = data[np.argsort(data[:, 0])]
x, f = data[:,0], data[:,1]   
ax.plot(x, f, color='r', label='Estimated superquantile frontier')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
legend = ax.legend(loc='lower right',
                        ncol=1,
                        fontsize=10,
                        frameon=False)
ax.set_xlabel("Input $x$")
ax.set_ylabel("Output $y$")
#ax.set_rasterized(True)
plt.savefig('optTau.pdf', format="pdf")
plt.show()
