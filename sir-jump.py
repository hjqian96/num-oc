import numpy as np
import numba
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

@jit(nopython=True)
def iteration(Iright,h,NI,Ns,tol,MaxC,NC,hc,d,alpha,beta,mu,rho,lam,sig1,sig2,sig3,qMainDiag,qSubDiag,la):
    #---- cost function ----
    cost=np.zeros((NI+1,NI+1,NI+1,Ns,NC))
    # only compute the cost in O
    for i in range(1,NI): # S
        for j in range(1,NI): # I
            for k in range(1,NI): #R
                for r in range(0,Ns):
                    for cit in range(0,NC):
                        S=i*h; I=j*h; R=k*h; c=cit*hc
                        cost[i,j,k,r,cit]=1+(I+S)*(r+1)+(r+1)*(I+S)*c*c
    #---- matrix of system coefficients -------
    F1=np.zeros((NI+1,NI+1,Ns))     # S
    F2=np.zeros((NI+1,NI+1,Ns,NC))  # I
    F3=np.zeros((NI+1,NI+1,Ns,NC))  # R
    G1=np.zeros((NI+1,Ns))
    G2=np.zeros((NI+1,Ns))
    G3=np.zeros((NI+1,Ns))
    for i in range(0,NI+1):
        for j in range(0,NI+1):
            for k in range(0,NI+1):
                for r in range(0,Ns):
                    for cit in range(0,NC):
                        S=i*h; I=j*h; R=k*h; c=cit*hc
                        F1[i,j,r]=alpha[r]-mu[r]*S-beta[r]*S*I
                        F2[i,j,r,cit]=beta[r]*S*I-(mu[r]+rho[r]+lam[r])*I-c*I
                        F3[j,k,r,cit]=lam[r]*I+c*I-mu[r]*R
                        G1[i]=sig1[r]*S
                        G2[j]=sig2[r]*I
                        G3[k]=sig3[r]*R
    #---- define the transition matrix -------
    PSforward=np.zeros((NI+1,NI+1,NI+1,Ns,NC))
    PSback=np.zeros((NI+1,NI+1,NI+1,Ns,NC))
    PIforward=np.zeros((NI+1,NI+1,NI+1,Ns,NC))
    PIback=np.zeros((NI+1,NI+1,NI+1,Ns,NC))
    PRforward=np.zeros((NI+1,NI+1,NI+1,Ns,NC))
    PRback=np.zeros((NI+1,NI+1,NI+1,Ns,NC))
    Pswitch=np.zeros((NI+1,NI+1,NI+1,Ns,NC))
    Pstay=np.zeros((NI+1,NI+1,NI+1,Ns,NC))
    Dlt=np.zeros((NI+1,NI+1,NI+1,Ns,NC))
    
    # only consider the inner of O
    for i in range(1,NI):
        for j in range(1,NI):
            for k in range(1,NI):
                for r in range(0,Ns):
                    for cit in range(0,NC):
                        s1=0.5*(G1[i,r]**2)+h*max(F1[i,j,r],0)
                        s2=0.5*(G1[i,r]**2)+h*max(-F1[i,j,r],0)
                        i1=0.5*(G2[j,r]**2)+h*max(F2[i,j,r,cit],0)
                        i2=0.5*(G2[j,r]**2)+h*max(-F2[i,j,r,cit],0)
                        r1=0.5*(G3[k,r]**2)+h*max(F3[j,k,r,cit],0)
                        r2=0.5*(G3[k,r]**2)+h*max(-F3[j,k,r,cit],0)
                        Qh=s1+s2+i1+i2+r1+r2+h-h*h*qMainDiag[r]
                        PSforward[i,j,k,r,cit]=s1/Qh
                        PSback[i,j,k,r,cit]=s2/Qh
                        PIforward[i,j,k,r,cit]=i1/Qh
                        PIback[i,j,k,r,cit]=i2/Qh
                        PRforward[i,j,k,r,cit]=r1/Qh
                        PRback[i,j,k,r,cit]=r2/Qh
                        Pswitch[i,j,k,r,cit]=h*h*qSubDiag[r]/Qh
                        Pstay[i,j,k,r,cit]=h/Qh
                        Dlt[i,j,k,r,cit]=h*h/Qh
    #====Initialization =========
    Vold=np.zeros((NI+1,NI+1,NI+1,Ns))
    Vnew=np.zeros((NI+1,NI+1,NI+1,Ns))
    Vsi=np.zeros((NI+1,NI+1,Ns))  # marginal for S and I
    Vir=np.zeros((NI+1,NI+1,Ns))  # marginal for I and R
    Vsr=np.zeros((NI+1,NI+1,Ns))  # marginal for S and R
    Copt=np.zeros((NI+1,NI+1,NI+1,Ns))
    Csi=np.zeros((NI+1,NI+1,Ns))  # marginal for S and I
    Cir=np.zeros((NI+1,NI+1,Ns))  # marginal for I and R
    Csr=np.zeros((NI+1,NI+1,Ns))  # marginal for S and R
    
    for i in range(0,NI+1):
        for j in range(0,NI+1):
            for k in range(0,NI+1):
                for r in range(0,Ns):
                    Vold[i,j,k,r]=1
    #==== start iteration ========
    maxitr=50000
    for n in range(0,maxitr):
        Vdiff=np.zeros((NI+1,NI+1,NI+1,Ns,NC))
        # only inner of O
        for i in range(1,NI):
            for j in range(1,NI):
                for k in range(1,NI):
                    for r in range(0,Ns):
                        S=i*h; I=j*h; R=k*h
                        # compute the jump size
                        vInt=0
                        for it in range(0,min(NI-i,min(NI-j,NI-k))+1):
                            vInt=vInt+Vold[i+it,j+it,k+it,r]*0.1*h*np.exp(-0.1*it*h)
                        for cit in range(0,NC):
                            A=Vold[i+1,j,k,r]*PSforward[i,j,k,r,cit]
                            B=Vold[i-1,j,k,r]*PSback[i,j,k,r,cit]
                            C=Vold[i,j+1,k,r]*PIforward[i,j,k,r,cit]
                            D=Vold[i,j-1,k,r]*PIback[i,j,k,r,cit]
                            E=Vold[i,j,k+1,r]*PRforward[i,j,k,r,cit]
                            F=Vold[i,j,k-1,r]*PRback[i,j,k,r,cit]
                            H=Vold[i,j,k,1-r]*Pswitch[i,j,k,r,cit]
                            J=Vold[i,j,k,r]*Pstay[i,j,k,r,cit]
                            Vdiff[i,j,k,r,cit]=(1-la*Dlt[i,j,k,r,cit])*(A+B+C+D+E+F+H+J)+la*Dlt[i,j,k,r,cit]*vInt+cost[i,j,k,r,cit]*Dlt[i,j,k,r,cit]
        # boundary condition
        for i in range(0,NI+1):
            for j in range(0,NI+1):
                for k in range(0,NI+1):
                    for r in range(0,Ns):
                        for cit in range(0,NC):
                            Vdiff[0,j,k,r,cit]=cost[0,j,k,r,cit]
                            Vdiff[NI,j,k,r,cit]=cost[NI,j,k,r,cit]
                            
                            Vdiff[i,0,k,r,cit]=cost[i,0,k,r,cit]
                            Vdiff[i,NI,k,r,cit]=cost[i,NI,k,r,cit]
                            
                            Vdiff[i,j,0,r,cit]=cost[i,j,0,r,cit]
                            Vdiff[i,j,NI,r,cit]=cost[i,j,NI,r,cit]
        #--- find the minimal value under control--------
        for i in range(1,NI):
            for j in range(1,NI):
                for k in range(1,NI):
                    for r in range(0,Ns):
                        Vnew[i,j,k,r]=np.min(Vdiff[i,j,k,r,:])
                        Vsi[i,j,r]=np.min(Vdiff[i,j,k,r,:])
                        Vir[j,k,r]=np.min(Vdiff[i,j,k,r,:])
                        Vsr[i,k,r]=np.min(Vdiff[i,j,k,r,:])
        #--- find the optimal control- ----
        for i in range(1,NI):
            for j in range(1,NI):
                for k in range(1,NI):
                    for r in range(0,Ns):
                        for cit in range(0,NC):
                            if(cit==0):
                                vmin=Vdiff[i,j,k,r,cit]
                                cmin=cit*hc
                            elif(Vdiff[i,j,k,r,cit]<vmin):
                                vmin=Vdiff[i,j,k,r,cit]
                                cmin=cit*hc
                        Copt[i,j,k,r]=cmin
                        Csi[i,j,r]=cmin
                        Cir[j,k,r]=cmin
                        Csr[i,k,r]=cmin
                        # boundary control
                        Copt[0,j,k,r]=Copt[1,j,k,r]; Copt[NI,j,k,r]=Copt[NI-1,j,k,r]
                        Copt[i,0,k,r]=Copt[i,1,k,r]; Copt[i,NI,k,r]=Copt[i,NI-1,k,r]
                        Copt[i,j,0,r]=Copt[i,j,1,r]; Copt[i,j,NI,r]=Copt[i,j,NI-1,r]
                        Csi[i,0,r]=Csi[i,1,r]; Csi[i,NI,r]=Csi[i,NI-1,r]
                        Csi[0,j,r]=Csi[1,j,r]; Csi[NI,j,r]=Csi[NI-1,j,r]
                        Cir[j,0,r]=Cir[j,1,r]; Cir[j,NI,r]=Cir[j,NI-1,r]
                        Cir[0,k,r]=Cir[1,k,r]; Cir[NI,k,r]=Cir[NI-1,k,r]
                        Csr[i,0,r]=Csr[i,k,r]; Csr[i,NI,r]=Csr[i,NI-1,r]
                        Csr[0,k,r]=Csr[1,k,r]; Csr[NI,k,r]=Csr[NI-1,k,r]
                        # corner
                        Copt[0,0,0,r]=Copt[1,1,1,r]; Copt[NI,NI,NI,r]=Copt[NI-1,NI-1,NI-1,r]
                        Copt[0,0,NI,r]=Copt[1,1,NI-1,r];Copt[NI,NI,0,r]=Copt[NI-1,NI-1,1,r]
                        Copt[0,NI,0,r]=Copt[1,NI-1,1,r]; Copt[NI,0,NI,r]=Copt[NI-1,1,NI-1,r]
                        Copt[0,NI,NI,r]=Copt[1,NI-1,NI-1,r]; Copt[NI,0,0,r]=Copt[NI-1,1,1,r]
                
                        Csi[0,0,r]=Csi[1,1,r]; Csi[NI,NI,r]=Csi[NI-1,NI-1,r]
                        Csi[0,NI,r]=Csi[1,NI-1,r]; Csi[NI,0]=Csi[NI-1,1,r]
                
                        Cir[0,0,r]=Csi[1,1,r]; Cir[NI,NI,r]=Csi[NI-1,NI-1,r]
                        Cir[0,NI,r]=Csi[1,NI-1,r]; Cir[NI,0,r]=Csi[NI-1,1,r]
                
                        Csr[0,0,r]=Csi[1,1,r]; Csr[NI,NI,r]=Csi[NI-1,NI-1,r]
                        Csr[0,NI,r]=Csi[1,NI-1,r]; Csr[NI,0,r]=Csi[NI-1,1,r]  
        #---- compute errormax
        errormax=np.max(np.abs(Vold-Vnew))
        print(n)
        print(errormax)
        if(errormax<tol):
            break
        else:
            Vold=Vnew

    return Vnew,Vsi,Vir,Vsr,Copt,Csi,Cir,Csr,errormax

#==== parameter setting ======
Iright=20; h=0.1
Sright=20; Rright=20
NI=int(Iright/h); Ns=2
tol=10**(-8)
MaxC=3; NC=16; hc=MaxC/(NC-1)
d=0.5
# mu is the disease-free death rate
# rho is the excess dealth rate due to the infectness
alpha=np.array([20,15])
beta=np.array([4,2])
mu=np.array([1,2])
rho=np.array([10,15])
lam=np.array([1,1])
sig1=np.array([1,2])
sig2=np.array([-1,1])
sig3=np.array([1,0.5])
qMainDiag=np.array([-1,-1])
qSubDiag=np.array([1,1])
la=0.1

Vnew,Vsi,Vir,Vsr,Copt,Csi,Cir,Csr,errormax=iteration(Iright,h,NI,Ns,tol,MaxC,NC,hc,d,alpha,beta,mu,rho,lam,sig1,sig2,sig3,qMainDiag,qSubDiag,la)
#===== Plotting =======
x1=np.linspace(0,Iright,NI+1)
y1=np.linspace(0,Iright,NI+1)
x1,y1=np.meshgrid(x1,y1)

fig1=plt.figure()
ax1=Axes3D(fig1)
vsi0=Vsi[:,:,0]
ax1.plot_surface(x1,y1,vsi0,cmap=cm.coolwarm)
ax1.set_xlabel("S(t)")
ax1.set_ylabel("I(t)")
ax1.set_zlabel("Value")
ax1.invert_xaxis()
# fig1.savefig("sir-vsi0.pdf",dpi=600,bbox_inches='tight')

fig2=plt.figure()
ax2=Axes3D(fig2)
vsi1=Vsi[:,:,1]
ax2.plot_surface(x1,y1,vsi1,cmap=cm.coolwarm)
ax2.set_xlabel("S(t)")
ax2.set_ylabel("I(t)")
ax2.set_zlabel("Value")
ax2.invert_xaxis()
# fig2.savefig("sir-vsi1.pdf",dpi=600,bbox_inches='tight')

fig3=plt.figure()
ax3=Axes3D(fig3)
vir0=Vir[:,:,0]
ax3.plot_surface(x1,y1,vir0,cmap=cm.coolwarm)
ax3.set_xlabel("I(t)")
ax3.set_ylabel("R(t)")
ax3.set_zlabel("Value")
ax3.invert_xaxis()
# fig3.savefig("sir-vir0.pdf",dpi=600,bbox_inches='tight')

fig4=plt.figure()
ax4=Axes3D(fig4)
vir1=Vir[:,:,1]
ax4.plot_surface(x1,y1,vir1,cmap=cm.coolwarm)
ax4.set_xlabel("I(t)")
ax4.set_ylabel("R(t)")
ax4.set_zlabel("value")
ax4.invert_xaxis()
# fig4.savefig("sir-vir1.pdf",dpi=600,bbox_inches='tight')

fig5=plt.figure()
ax5=Axes3D(fig5)
vsr0=Vsr[:,:,0]
ax5.plot_surface(x1,y1,vsr0,cmap=cm.coolwarm)
ax5.set_xlabel("S(t)")
ax5.set_ylabel("R(t)")
ax5.set_zlabel("value")
ax5.invert_xaxis()
# fig5.savefig("sir-vsr0.pdf",dpi=600,bbox_inches='tight')
fig6=plt.figure()
ax6=Axes3D(fig6)
vsr1=Vsr[:,:,1]
ax6.plot_surface(x1,y1,vsr1,cmap=cm.coolwarm)
ax6.set_xlabel("S(t)")
ax6.set_ylabel("R(t)")
ax6.set_zlabel("value")
ax6.invert_xaxis()
# fig6.savefig("sir-vsr1.pdf",dpi=600,bbox_inches='tight')

fig7=plt.figure()
ax7=Axes3D(fig7)
csi0=Csi[:,:,0]
ax7.plot_surface(x1,y1,csi0,cmap=cm.coolwarm)
ax7.set_xlabel("S(t)")
ax7.set_ylabel("I(t)")
ax7.set_zlabel("optimal control")
ax7.invert_xaxis()
# fig7.savefig("sir-csi0.pdf",dpi=600,bbox_inches='tight')

fig8=plt.figure()
ax8=Axes3D(fig8)
csi1=Csi[:,:,1]
ax8.plot_surface(x1,y1,csi1,cmap=cm.coolwarm)
ax8.set_xlabel("S(t)")
ax8.set_ylabel("I(t)")
ax8.set_zlabel("optimal control")
ax8.invert_xaxis()
# fig8.savefig("sir-csi1.pdf",dpi=600,bbox_inches='tight')

fig9=plt.figure()
ax9=Axes3D(fig9)
cir0=Cir[:,:,0]
ax9.plot_surface(x1,y1,cir0,cmap=cm.coolwarm)
ax9.set_xlabel("I(t)")
ax9.set_ylabel("R(t)")
ax9.set_zlabel("optimal control")
ax9.invert_xaxis()
# fig9.savefig("sir-cir0.pdf",dpi=600,bbox_inches='tight')

fig10=plt.figure()
ax10=Axes3D(fig10)
cir1=Cir[:,:,1]
ax10.plot_surface(x1,y1,cir1,cmap=cm.coolwarm)
ax10.set_xlabel("I(t)")
ax10.set_ylabel("R(t)")
ax10.set_zlabel("optimal control")
ax10.invert_xaxis()
# fig10.savefig("sir-cir1.pdf",dpi=600,bbox_inches='tight')

fig11=plt.figure()
ax11=Axes3D(fig11)
csr0=Csr[:,:,0]
ax11.plot_surface(x1,y1,csr0,cmap=cm.coolwarm)
ax11.set_xlabel("S(t)")
ax11.set_ylabel("R(t)")
ax11.set_zlabel("optimal control")
ax11.invert_xaxis()
# fig11.savefig("sir-csr0.pdf",dpi=600,bbox_inches='tight')

fig12=plt.figure()
ax12=Axes3D(fig12)
csr1=Csr[:,:,1]
ax12.plot_surface(x1,y1,csr1,cmap=cm.coolwarm)
ax12.set_xlabel("S(t)")
ax12.set_ylabel("R(t)")
ax12.set_zlabel("optimal control")
ax12.invert_xaxis()
# fig12.savefig("sir-csr1.pdf",dpi=600,bbox_inches='tight')