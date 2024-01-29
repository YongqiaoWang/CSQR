import matplotlib.pyplot as plt
import numpy as np

def draw_multiple_frontiers(InputX, OutputY, EstY, vtau, frontier_name, file_name):
    #draw the figure of two disturbed frontiers
   
    fig = plt.figure()
    ax = fig.subplots()
    nSample, nFrontier = EstY.shape[0], EstY.shape[1]
    flist=[]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    
    for itau in range(nFrontier):
        data = (np.stack([(InputX[:, 0]), (EstY[:,itau])], axis=1))
        data = data[np.argsort(data[:, 0])]
        x, f = data[:,0], data[:,1]       
        #line =ax.plot(x, f, color="b", alpha=vtau[itau], label=str(vtau[itau]))
        line =ax.plot(x, f, color=colors[itau], label=str(vtau[itau]))
        flist.append(line)
    
    dpy = ax.scatter(InputX[:,0], OutputY, marker='o', color='white', edgecolor='k', s=12)
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    legend = ax.legend(loc='upper left',
                            ncol=1,
                            fontsize=10,
                            frameon=False)
    ax.set_xlabel("Input $x$")
    ax.set_ylabel("Output $y$")
    #ax.set_rasterized(True)
    plt.savefig(file_name+'.pdf', format="pdf")
    plt.show()
