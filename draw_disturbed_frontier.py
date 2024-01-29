import matplotlib.pyplot as plt
import numpy as np

def draw_disturbed_frontier(InputX, OutputY, DisturbY, EstY, EstDisturbY, frontier_name, file_name):
    #draw the figure of two disturbed frontiers

    fig = plt.figure()
    ax = fig.subplots()
    
    data = (np.stack([(InputX[:, 0]).T, OutputY, DisturbY, EstY, EstDisturbY], axis=0)).T
    data = data[np.argsort(data[:, 0])].T
    x, y, ty, f, tf = data[0], data[1], data[2], data[3], data[4]
    dptf = ax.plot(x, tf, color="b")
    dpf = ax.plot(x, f, color="b", linestyle='dashed')
    
    malpha = np.where(y == ty, 0.2, 1)
    dpty = ax.scatter(x, ty, marker='o', color='red', edgecolor='r', s=12) # alpha = malpha,
    dpy = ax.scatter(x, y, marker='o', color='white', edgecolor='k', s=12) # alpha = malpha,
    
    for i in range(len(OutputY)):
        if (y[i] != ty[i]):
            ax.plot((x[i], x[i]), (y[i], ty[i]), alpha =0.3, linestyle='solid', color='k') 

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    legend = plt.legend([dpf[0], dptf[0]],
                            [frontier_name+' before changes',
                             frontier_name+' after changes'],
                            loc='lower right',
                            ncol=1,
                            fontsize=10,
                            frameon=False)
    ax.set_xlabel("Input $x$")
    ax.set_ylabel("Output $y$")
    #ax.set_rasterized(True)
    plt.savefig(file_name+'.pdf', format="pdf")
    plt.show()
