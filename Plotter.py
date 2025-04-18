import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import numpy as np
import torch as th


class Plotter:

    """
    Plots:
        - Lyapunov Function and Lie Derivative heat map at each counter example check
        - additionally scatter counter examples
        - plus contour

        - Final Lyapunov function and Lie derivative and valid region
        - ROA (requires man tuning)
    
    """

    def __init__(self, device, lb=-6, ub=6, fidelity=100):

        self.device = device
        self.fidelity = fidelity
        self.lb = lb
        self.ub = ub

        self.X1 = th.linspace(lb, ub, fidelity, device=device) 
        self.X2 = th.linspace(lb, ub, fidelity, device=device)

        self.x1, self.x2 = np.meshgrid(self.X1.cpu().numpy(), self.X2.cpu().numpy())
        self.X_plot = th.tensor(np.stack([self.x1.flatten(), self.x2.flatten()]).T, device=device)


        self.Vs = []
        self.Ls = []
        self.ces1 = []
        self.ces2 = []
        self.iters = []


    def reset_plots(self):
        self.LFs = []
        self.VDs = []
        self.ces = []

        

    def update_plots(self, iter, V, L, ce1=None, ce2=None):
        self.iters.append(iter)
        self.Vs.append(V)
        self.Ls.append(L)

        if ce1:
            self.ces1.append(ce1[0])
            self.ces2.append(ce1[1])

        if ce2:
            self.ces1.append(ce2[0])
            self.ces2.append(ce2[1])

    
    def export_plots(self):

        # export Vs 
        for i, v in enumerate(self.Vs):
            
            plt.figure()
            plt.imshow(v, extent=[-6, 6, -6, 6], origin='lower')
            plt.scatter(self.ces1[:i], self.ces2[:i], color='red', s=30, marker='.')
            plt.colorbar(cmap=cm.coolwarm)
            plt.title(f'V @ iteration {self.iters[i]}')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.contour(self.x1, self.x2, v, 10, linewidths=0.4, colors='k')
            plt.savefig(f'V_{self.iters[i]}.png', dpi=300, bbox_inches='tight')


        # export Vs Video
        vmin = np.min(self.Vs)
        vmax = np.max(self.Vs)

        fig, ax = plt.subplots()

        im = ax.imshow(self.Vs[0], extent=[-6, 6, -6, 6], origin='lower', vmin=vmin, vmax=vmax, animated=True,)
        cs = ax.contour(self.x1, self.x2, self.Vs[0], 8, linewidths=0.4, colors='k')
        sc = ax.scatter(self.ces1[0], self.ces2[0], color='red', s=30, marker='.')

        cbar = fig.colorbar(im, ax=ax, cmap=cm.coolwarm) 
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Lyapunov Function')

        # Update function
        def _update(frame):
            im.set_array(self.Vs[frame])
            sc.set_offsets(np.c_[self.ces1[:frame], self.ces2[:frame]]) 
            
            for contour in cs.collections:
                contour.remove()

            cs = ax.contour(self.x1, self.x2, self.Vs[frame], 8, linewidths=0.4, colors='k')

            return [im, cs.collections, sc]

        # Create animation
        ani = animation.FuncAnimation(
            fig, _update, frames=len(self.Vs), interval=100, blit=False
        )
        ani.save('export/lyapunov_function.mp4', writer='ffmpeg')



        # export Ls
        for i, l in enumerate(self.Ls):
            
            plt.figure()
            plt.imshow(l, extent=[-6, 6, -6, 6], origin='lower')
            plt.scatter(self.ces1[:i], self.ces2[:i], color='red', s=30, marker='.')
            plt.colorbar(cmap=cm.coolwarm)
            plt.title(f'dV/dt @ iteration {self.iters[i]}')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.contour(self.x1, self.x2, l, 10, linewidths=0.4, colors='k')
            plt.savefig(f'L_{self.iters[i]}.png', dpi=300, bbox_inches='tight')


        # export Ls Video
        vmin = np.min(self.Ls)
        vmax = np.max(self.Ls)

        fig, ax = plt.subplots()

        im = ax.imshow(self.Ls[0], extent=[-6, 6, -6, 6], origin='lower', vmin=vmin, vmax=vmax, animated=True,)
        cs = ax.contour(self.x1, self.x2, self.Ls[0], 8, linewidths=0.4, colors='k')
        sc = ax.scatter(self.ces1[0], self.ces2[0], color='red', s=30, marker='.')

        cbar = fig.colorbar(im, ax=ax, cmap=cm.coolwarm) 
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Lie Derivative')

        # Update function
        def _update(frame):
            im.set_array(self.Ls[frame])
            sc.set_offsets(np.c_[self.ces1[:frame], self.ces2[:frame]]) 
            
            for contour in cs.collections:
                contour.remove()

            cs = ax.contour(self.x1, self.x2, self.Ls[frame], 8, linewidths=0.4, colors='k')

            return [im, cs.collections, sc]

        # Create animation
        ani = animation.FuncAnimation(
            fig, _update, frames=len(self.Ls), interval=100, blit=False
        )
        ani.save('export/lie_derivative.mp4', writer='ffmpeg')


        # export final lyapunov and lie derivative
        self.export_final_lyapunov_function(self.Vs[-1])
        self.export_final_lie_dervative(self.Ls[-1])



    def export_final_lyapunov_function(self, V):
        
        fig = plt.figure(figsize=(6, 6))
        plt.tight_layout()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X1.cpu(), self.X2.cpu(), V, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
        ax.contour(self.X1.cpu(), self.X2.cpu(), V, 10, zdir='z', offset=0, cmap=cm.coolwarm)

        # Plot Valid region computed by dReal
        r=6
        theta = np.linspace(0,2*np.pi,50)
        xc = r*np.cos(theta)
        yc = r*np.sin(theta)
        ax.plot(xc[:],yc[:],'r',linestyle='--', linewidth=2 ,label='Valid region')
        plt.legend(loc='upper right')
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.set_zlabel('V')
        
        plt.savefig('export/lyapunov_function.png', dpi=300, bbox_inches='tight')


    
    def export_final_lie_dervative(self, L):
        fig = plt.figure(figsize=(6, 6))
        plt.tight_layout()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X1.cpu(), self.X2.cpu(), L, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
        ax.contour(self.X1.cpu(), self.X2.cpu(), L, 10, zdir='z', offset=0, cmap=cm.coolwarm)

        # Plot Valid region computed by dReal
        r=6
        theta = np.linspace(0,2*np.pi,50)
        xc = r*np.cos(theta)
        yc = r*np.sin(theta)
        ax.plot(xc[:],yc[:],'r',linestyle='--', linewidth=2 ,label='Valid region')
        plt.legend(loc='upper right')
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.set_zlabel('dV/dt')
        
        plt.savefig('export/lie_derivative.png', dpi=300, bbox_inches='tight')


    def export_roa(self, V_multi, V_common):

        plt.figure()
        ax = plt.gca()

        # Vaild Region
        C = plt.Circle((0, 0),6, color='r', linewidth=1.5, fill=False)
        ax.add_artist(C)

        # # plot direction field
        # xd = np.linspace(-6, 6, 10) 
        # yd = np.linspace(-6, 6, 10)
        # Xd, Yd = np.meshgrid(xd,yd)
        # t = np.linspace(0,2,100)
        # Plotflow(Xd, Yd, t) 


        ax.contour(self.X1, self.X2, V_multi-1.2, 0, linewidths=2, colors='k', linestyles='-')
        ax.contour(self.X1, self.X2, V_common-2.6,0,linewidths=2, colors='m',linestyles='--')

        ax.contour(self.x1, self.x2, V_multi, 8, linewidths=0.4, colors='k')
        c1 = ax.contourf(self.x1, self.x2,V_multi, 8, alpha=0.4, cmap=cm.coolwarm)
        plt.colorbar(c1)

        plt.title('Region of Attraction')
        plt.legend([plt.Rectangle((0,0),1,2,color='k',fill=False,linewidth = 2), plt.Rectangle((0,0),1,2,color='m',fill=False,linewidth = 2, linestyle='--'), C],['Multiple Lyapunov Function','Common Lyapunov Function','Valid Region'],
                   loc='upper right', bbox_to_anchor=(1.4, 1.25))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.savefig('export/roa.png', dpi=300, bbox_inches='tight')



    


    def export_training_samples(self, samples):
        
        plt.figure()
        plt.scatter(samples[:, 0], samples[:, 1], marker='.')
        plt.grid()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Training Samples')
        plt.xlim(self.lb, self.ub)
        plt.ylim(self.lb, self.ub)
        plt.savefig("export/training_samples.png", dpi=300, bbox_inches='tight')


def export_learning_curve(L1, L2, out_iters):

    plt.figure()
    plt.plot(L1, label='1')
    plt.plot(L2, label='2')
    plt.title(f'Restart Num. {out_iters}')
    plt.xlabel('Iteration')
    plt.ylabel('Lyapunov Risk')
    plt.legend()
    plt.savefig("export/lyapunov_risk_curve.png", dpi=300, bbox_inches='tight')