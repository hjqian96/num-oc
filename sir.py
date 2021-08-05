import numpy as np
import numba
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

@jit(nopython=True)
def iteration(Iright,h,NI,tol,MaxC,NC,hc,d,alpha,beta,mu,rho,lam,sig1,sig2,sig3):
    #---- cost function ----
    cost=np.zeros((NI+1,NI+1,NI+1,NC))
    # only compute the cost in O
    for i in range(1,NI): # S
        for j in range(1,NI): # I
            for k in range(1,NI): #R
                for cit in range(0,NC):
                    S=i*h; I=j*h; R=k*h; c=cit*hc
                    cost[i,j,k,cit]=1+I+2*I*(c**2)
    #---- matrix of system coefficients -------
    F1=np.zeros((NI+1,NI+1))     # S
    F2=np.zeros((NI+1,NI+1,NC))  # I
    F3=np.zeros((NI+1,NI+1,NC))  # R
    G1=np.zeros(NI+1)
    G2=np.zeros(NI+1)
    G3=np.zeros(NI+1)
    for i in range(0,NI+1):
        for j in range(0,NI+1):
            for k in range(0,NI+1):
                for cit in range(0,NC):
                    S=i*h; I=j*h; R=k*h; c=cit*hc
                    F1[i,j]=alpha-mu*S-beta*S*I
                    F2[i,j,cit]=beta*S*I-(mu+rho+lam)*I-c*I
                    F3[j,k,cit]=lam*I+c*I-mu*R
                    G1[i]=sig1*S
                    G2[j]=sig2*I
                    G3[k]=sig3*R
    #---- define the transition matrix -------
    PSforward=np.zeros((NI+1,NI+1,NI+1,NC))
    PSback=np.zeros((NI+1,NI+1,NI+1,NC))
    PIforward=np.zeros((NI+1,NI+1,NI+1,NC))
    PIback=np.zeros((NI+1,NI+1,NI+1,NC))
    PRforward=np.zeros((NI+1,NI+1,NI+1,NC))
    PRback=np.zeros((NI+1,NI+1,NI+1,NC))
    Dlt=np.zeros((NI+1,NI+1,NI+1,NC))
    
    # only consider the inner of O
    for i in range(1,NI):
        for j in range(1,NI):
            for k in range(1,NI):
                for cit in range(0,NC):
                    s1=0.5*(G1[i]**2)+h*max(F1[i,j],0)
                    s2=0.5*(G1[i]**2)+h*max(-F1[i,j],0)
                    i1=0.5*(G2[j]**2)+h*max(F2[i,j,cit],0)
                    i2=0.5*(G2[j]**2)+h*max(-F2[i,j,cit],0)
                    r1=0.5*(G3[k]**2)+h*max(F3[j,k,cit],0)
                    r2=0.5*(G3[k]**2)+h*max(-F3[j,k,cit],0)
                    Qh=s1+s2+i1+i2+r1+r2
                    PSforward[i,j,k,cit]=s1/Qh
                    PSback[i,j,k,cit]=s2/Qh
                    PIforward[i,j,k,cit]=i1/Qh
                    PIback[i,j,k,cit]=i2/Qh
                    PRforward[i,j,k,cit]=r1/Qh
                    PRback[i,j,k,cit]=r2/Qh
                    Dlt[i,j,k,cit]=h*h/Qh
    #====Initialization =========
    Vold=np.zeros((NI+1,NI+1,NI+1))
    Vnew=np.zeros((NI+1,NI+1,NI+1))
    Vsi=np.zeros((NI+1,NI+1))  # marginal for S and I
    Vir=np.zeros((NI+1,NI+1))  # marginal for I and R
    Vsr=np.zeros((NI+1,NI+1))  # marginal for S and R
    Copt=np.zeros((NI+1,NI+1,NI+1))
    Csi=np.zeros((NI+1,NI+1))  # marginal for S and I
    Cir=np.zeros((NI+1,NI+1))  # marginal for I and R
    Csr=np.zeros((NI+1,NI+1))  # marginal for S and R
    
    for i in range(0,NI+1):
        for j in range(0,NI+1):
            for k in range(0,NI+1):
                Vold[i,j,k]=(1+0+2*0*(MaxC**2))/d
    #==== start iteration ========
    maxitr=50000
    for n in range(0,maxitr):
        Vdiff=np.zeros((NI+1,NI+1,NI+1,NC))
        # only inner of O
        for i in range(1,NI):
            for j in range(1,NI):
                for k in range(1,NI):
                    S=i*h; I=j*h; R=k*h
                    for cit in range(0,NC):
                        df=np.exp(-d*Dlt[i,j,k,cit])
                        A=Vold[i+1,j,k]*PSforward[i,j,k,cit]
                        B=Vold[i-1,j,k]*PSback[i,j,k,cit]
                        C=Vold[i,j+1,k]*PIforward[i,j,k,cit]
                        D=Vold[i,j-1,k]*PIback[i,j,k,cit]
                        E=Vold[i,j,k+1]*PRforward[i,j,k,cit]
                        F=Vold[i,j,k-1]*PRback[i,j,k,cit]
                        Vdiff[i,j,k,cit]=(A+B+C+D+E+F)*df+cost[i,j,k,cit]*Dlt[i,j,k,cit]
        # boundary condition
        for i in range(0,NI+1):
            for j in range(0,NI+1):
                for k in range(0,NI+1):
                    for cit in range(0,NC):
                        Vdiff[0,0,k,cit]=cost[0,0,k,cit]
                        Vdiff[NI,NI,k,cit]=cost[NI,NI,k,cit]
                        Vdiff[0,NI,k,cit]=cost[0,NI,k,cit]
                        Vdiff[NI,0,k,cit]=cost[NI,0,k,cit]
                        
                        Vdiff[0,j,0,cit]=cost[0,j,0,cit]
                        Vdiff[NI,j,NI,cit]=cost[NI,j,NI,cit]
                        Vdiff[0,j,NI,cit]=cost[0,j,NI,cit]
                        Vdiff[NI,j,0,cit]=cost[NI,j,0,cit]
                        
                        Vdiff[i,0,0,cit]=cost[i,0,0,cit]
                        Vdiff[i,NI,NI,cit]=cost[i,NI,NI,cit]
                        Vdiff[i,0,NI,cit]=cost[i,0,NI,cit]
                        Vdiff[i,NI,0,cit]=cost[i,NI,0,cit] 
        #--- find the minimal value under control--------
        for i in range(1,NI):
            for j in range(1,NI):
                for k in range(1,NI):
                    Vnew[i,j,k]=np.min(Vdiff[i,j,k,:])
                    Vsi[i,j]=np.min(Vdiff[i,j,k,:])
                    Vir[j,k]=np.min(Vdiff[i,j,k,:])
                    Vsr[i,k]=np.min(Vdiff[i,j,k,:])
        #--- find the optimal control- ----
        for i in range(1,NI):
            for j in range(1,NI):
                for k in range(1,NI):
                    for cit in range(0,NC):
                        if(cit==0):
                            vmin=Vdiff[i,j,k,cit]
                            cmin=cit*hc
                        elif(Vdiff[i,j,k,cit]<vmin):
                            vmin=Vdiff[i,j,k,cit]
                            cmin=cit*hc
                    Copt[i,j,k]=cmin
                    Csi[i,j]=cmin
                    Cir[j,k]=cmin
                    Csr[i,k]=cmin
                    # boundary control
                    Copt[0,j,k]=Copt[1,j,k]; Copt[NI,j,k]=Copt[NI-1,j,k]
                    Copt[i,0,k]=Copt[i,1,k]; Copt[i,NI,k]=Copt[i,NI-1,k]
                    Copt[i,j,0]=Copt[i,j,1]; Copt[i,j,NI]=Copt[i,j,NI-1]
                    Csi[i,0]=Csi[i,1]; Csi[i,NI]=Csi[i,NI-1]
                    Csi[0,j]=Csi[1,j]; Csi[NI,j]=Csi[NI-1,j]
                    Cir[j,0]=Cir[j,1]; Cir[j,NI]=Cir[j,NI-1]
                    Cir[0,k]=Cir[1,k]; Cir[NI,k]=Cir[NI-1,k]
                    Csr[i,0]=Csr[i,k]; Csr[i,NI]=Csr[i,NI-1]
                    Csr[0,k]=Csr[1,k]; Csr[NI,k]=Csr[NI-1,k]
                    # corner
                    Copt[0,0,0]=Copt[1,1,1]; Copt[NI,NI,NI]=Copt[NI-1,NI-1,NI-1]
                    Copt[0,0,NI]=Copt[1,1,NI-1];Copt[NI,NI,0]=Copt[NI-1,NI-1,1]
                    Copt[0,NI,0]=Copt[1,NI-1,1]; Copt[NI,0,NI]=Copt[NI-1,1,NI-1]
                    Copt[0,NI,NI]=Copt[1,NI-1,NI-1]; Copt[NI,0,0]=Copt[NI-1,1,1]
                
                    Csi[0,0]=Csi[1,1]; Csi[NI,NI]=Csi[NI-1,NI-1]
                    Csi[0,NI]=Csi[1,NI-1]; Csi[NI,0]=Csi[NI-1,1]
                
                    Cir[0,0]=Csi[1,1]; Cir[NI,NI]=Csi[NI-1,NI-1]
                    Cir[0,NI]=Csi[1,NI-1]; Cir[NI,0]=Csi[NI-1,1]
                
                    Csr[0,0]=Csi[1,1]; Csr[NI,NI]=Csi[NI-1,NI-1]
                    Csr[0,NI]=Csi[1,NI-1]; Csr[NI,0]=Csi[NI-1,1]  
        #---- compute errormax
        errormax=np.max(np.abs(Vold-Vnew))
        print(n)
        print(errormax)
        if(errormax<tol):
            break
        else:
            Vold=Vnew

    return Vnew,Vsi,Vir,Vsr,Copt,Csi,Cir,Csr,errormax

#===== coefficient setting ========
Iright=20; h=0.2
Sright=20; Rright=20
NI=int(Iright/h)
tol=10**(-8)
MaxC=3; NC=16; hc=MaxC/(NC-1)
d=0.5
# mu is the disease-free death rate
# rho is the excess dealth rate due to the infectnessa
alpha=20; beta=4; mu=1; rho=10; lam=1
sig1=1; sig2=-1; sig3=1


Vnew,Vsi,Vir,Vsr,Copt,Csi,Cir,Csr,errormax=iteration(Iright,h,NI,tol,MaxC,NC,hc,d,alpha,beta,mu,rho,lam,sig1,sig2,sig3)

#==== Plotting ========
x1=np.linspace(0,Iright,NI+1)
y1=np.linspace(0,Iright,NI+1)
x1,y1=np.meshgrid(x1,y1)

fig1=plt.figure()
ax1=Axes3D(fig1)
vsi=Vsi[:,:]
ax1.plot_surface(x1,y1,vsi,cmap=cm.coolwarm)
ax1.set_xlabel("S(t)")
ax1.set_ylabel("I(t)")
ax1.set_zlabel("Value")
ax1.invert_xaxis()
# fig1.savefig("SIR_Vsi.pdf",dpi=600,bbox_inches='tight')
fig2=plt.figure()
ax2=Axes3D(fig2)
vir=Vir[:,:]
ax2.plot_surface(x1,y1,vir,cmap=cm.coolwarm)
ax2.set_xlabel("I(t)")
ax2.set_ylabel("R(t)")
ax2.set_zlabel("Value")
ax2.invert_xaxis()
# fig2.savefig("SIR_Vir.pdf",dpi=600,bbox_inches='tight')
fig3=plt.figure()
ax3=Axes3D(fig3)
vsr=Vsr[:,:]
ax3.plot_surface(x1,y1,vsr,cmap=cm.coolwarm)
ax3.set_xlabel("S(t)")
ax3.set_ylabel("R(t)")
ax3.set_zlabel("Value")
ax3.invert_xaxis()
# fig3.savefig("SIR_Vsr.pdf",dpi=600,bbox_inches='tight')
fig4=plt.figure()
ax4=Axes3D(fig4)
csi=Csi[:,:]
ax4.plot_surface(x1,y1,csi,cmap=cm.coolwarm)
ax4.set_xlabel("S(t)")
ax4.set_ylabel("I(t)")
ax4.set_zlabel("Optimal control")
ax4.invert_xaxis()
# fig4.savefig("SIR_Csi.pdf",dpi=600,bbox_inches='tight')
fig5=plt.figure()
ax5=Axes3D(fig5)
cir=Cir[:,:]
ax5.plot_surface(x1,y1,cir,cmap=cm.coolwarm)
ax5.set_xlabel("I(t)")
ax5.set_ylabel("R(t)")
ax5.set_zlabel("Optimal control")
ax5.invert_xaxis()
# fig5.savefig("SIR_Cir.pdf",dpi=600,bbox_inches='tight')
fig6=plt.figure()
ax6=Axes3D(fig6)
csr=Csr[:,:]
ax6.plot_surface(x1,y1,csr,cmap=cm.coolwarm)
ax6.set_xlabel("S(t)")
ax6.set_ylabel("R(t)")
ax6.set_zlabel("Optimal control")
ax6.invert_xaxis()
# fig6.savefig("SIR_Csr.pdf",dpi=600,bbox_inches='tight')