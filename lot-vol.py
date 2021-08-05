import numpy as np
import numba
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

@jit(nopython=True)
def iteration(tol,Nx,Ns,Nu,h,hu,la,alpha,beta,a,b,c,d,qMainDiag,qSubDiag):
    #---- cost function --------
    cost=np.zeros((Nx+1,Nx+1,Ns,Nu))
    for i in range(1,Nx):
        for j in range(1,Nx):
            for r in range(0,Ns):
                for uit in range(0,Nu):
                    x=i*h; y=j*h; u=uit*hu
                    cost[i,j,r,uit]=1+(r+1)*(x+y)*(1+u**2)
                    # boundary
                    cost[i,0,r,uit]=1+(i*h+0)*(r+1)
                    cost[i,Nx,r,uit]=1+(i*h+Nx*h)*(r+1)
                    cost[0,j,r,uit]=1+(0+j*h)*(r+1)
                    cost[Nx,j,r,uit]=1+(Nx*h+j*h)*(r+1)
                    # corner
                    cost[0,0,r,uit]=1
                    cost[0,Nx,r,uit]=1+(Nx*h)*(r+1)
                    cost[Nx,0,r,uit]=1+(Nx*h)*(r+1)
                    cost[Nx,Nx,r,uit]=1+(Nx*h+Nx*h)*(r+1)
    #---- matrix of system coefficients --------
    F1=np.zeros((Nx+1,Nx+1,Ns,Nu))
    F2=np.zeros((Nx+1,Nx+1,Ns,Nu))
    G1=np.zeros((Nx+1,Nx+1,Ns,Nu))
    G2=np.zeros((Nx+1,Nx+1,Ns,Nu))
    for i in range(0,Nx+1):
        for j in range(0,Nx+1):
            for r in range(0,Ns):
                for uit in range(0,Nu):
                    x=i*h; y=j*h; u=uit*hu
                    F1[i,j,r,uit]=x*(a[r]-b[r]*y+u)
                    F2[i,j,r,uit]=y*(-c[r]+d[r]*x+u)
                    G1[i,j,r,uit]=alpha[r]*x
                    G2[i,j,r,uit]=beta[r]*y
    #---- define the transition matrix ------
    Pforward=np.zeros((Nx+1,Nx+1,Ns,Nu))
    Pback=np.zeros((Nx+1,Nx+1,Ns,Nu))
    Pup=np.zeros((Nx+1,Nx+1,Ns,Nu))
    Pdown=np.zeros((Nx+1,Nx+1,Ns,Nu))
    Pswitch=np.zeros((Nx+1,Nx+1,Ns,Nu))
    Pstay=np.zeros((Nx+1,Nx+1,Ns,Nu))
    Dlt=np.zeros((Nx+1,Nx+1,Ns,Nu))
    for i in range(0,Nx+1):
        for j in range(0,Nx+1):
            for r in range(0,Ns):
                for uit in range(0,Nu):
                    a1=0.5*(G1[i,j,r,uit]**2)+max(F1[i,j,r,uit],0)
                    b1=0.5*(G1[i,j,r,uit]**2)+max(-F1[i,j,r,uit],0)
                    c1=0.5*(G2[i,j,r,uit]**2)+max(F2[i,j,r,uit],0)
                    d1=0.5*(G2[i,j,r,uit]**2)+max(-F2[i,j,r,uit],0)
                    Qh=a1+b1+c1+d1+h-h*h*qMainDiag[r]
                    Dlt[i,j,r,uit]=h*h/Qh
                    Pforward[i,j,r,uit]=a1/Qh
                    Pback[i,j,r,uit]=b1/Qh
                    Pup[i,j,r,uit]=c1/Qh
                    Pdown[i,j,r,uit]=d1/Qh
                    Pswitch[i,j,r,uit]=(h*h*qSubDiag[r])/Qh
                    Pstay[i,j,r,uit]=h/Qh                
    #==== initialization =======
    Vold=np.zeros((Nx+1,Nx+1,Ns)) # the value at Nth step
    Vnew=np.zeros((Nx+1,Nx+1,Ns)) # the value at (N+1)th step
    Uopt=np.zeros((Nx+1,Nx+1,Ns)) # the optiaml control when update
    # i: index of x-axis; j: index of y-axis;
    # r: state of switching (true state=r+1)
    for r in range(0,Ns):
        for i in range(0,Nx+1):
            for j in range(0,Nx+1):
                Vold[i,j,r]=1
    #====== Iteration ==========
    maxitr=50000
    for n in range(0,maxitr):
        Vdiff=np.zeros((Nx+1,Nx+1,Ns,Nu))
        for i in range(1,Nx):
            for j in range(1,Nx):
                for r in range(0,Ns):
                    # Compute the value of integral（jump size）in (29).
                    vInt=0
                    for it in range(0,min(Nx-i,Nx-j)+1):
                        vInt=vInt+Vold[i+it,j+it,r]*np.exp(-0.1*h*it)*0.1*h
                    for uit in range(0,Nu):
                        A=Vold[i+1,j,r]*Pforward[i,j,r,uit]
                        B=Vold[i-1,j,r]*Pback[i,j,r,uit]
                        C=Vold[i,j+1,r]*Pup[i,j,r,uit]
                        D=Vold[i,j-1,r]*Pdown[i,j,r,uit]
                        E=Vold[i,j,1-r]*Pswitch[i,j,r,uit]
                        F=Vold[i,j,r]*Pstay[i,j,r,uit]
                        Vdiff[i,j,r,uit]=(1-la*Dlt[i,j,r,uit])*(A+B+C+D+E+F)+la*Dlt[i,j,r,uit]*vInt+Dlt[i,j,r,uit]*cost[i,j,r,uit]
        # boundary condition
        for i in range(1,Nx):
            for j in range(1,Nx):
                for r in range(0,Ns):
                    for uit in range(0,Nu):
                        Vdiff[i,0,r,uit]=cost[i,0,r,uit]
                        Vdiff[i,Nx,r,uit]=cost[i,Nx,r,uit]
                        Vdiff[0,j,r,uit]=cost[0,j,r,uit]
                        Vdiff[Nx,j,r,uit]=cost[Nx,j,r,uit]
                        # corner
                        Vdiff[0,0,r,uit]=cost[0,0,r,uit]
                        Vdiff[0,Nx,r,uit]=cost[0,Nx,r,uit]
                        Vdiff[Nx,0,r,uit]=cost[Nx,0,r,uit]
                        Vdiff[Nx,Nx,r,uit]=cost[Nx,Nx,r,uit]
        #---- find the minimal value----------
        for i in range(0,Nx+1):
            for j in range(0,Nx+1):
                for r in range(0,Ns):
                    Vnew[i,j,r]=np.min(Vdiff[i,j,r,:])
        #=== find the control =====
        for i in range(1,Nx):
            for j in range(1,Nx):
                for r in range(0,Ns):
                    for uit in range(0,Nu):
                        if(uit==0):
                            vmin=Vdiff[i,j,r,uit]
                            umin=uit*hu
                        elif(Vdiff[i,j,r,uit]<vmin):
                            vmin=Vdiff[i,j,r,uit]
                            umin=uit*hu
                    Uopt[i,j,r]=umin
                    # control on the boundary
                    Uopt[i,0,r]=Uopt[i,1,r]
                    Uopt[i,Nx,r]=Uopt[i,Nx-1,r]
                    Uopt[0,j,r]=Uopt[1,j,r]
                    Uopt[Nx,j,r]=Uopt[Nx-1,j,r]
                    # corner points
                    Uopt[0,0,r]=Uopt[1,1,r]
                    Uopt[0,Nx,r]=Uopt[1,Nx-1,r]
                    Uopt[Nx,0,r]=Uopt[Nx-1,1,r]
                    Uopt[Nx,Nx,r]=Uopt[Nx-1, Nx-1,r]
        #---- find error ----
        errormax=np.max(np.abs(Vold-Vnew))
        print(n)
        print(errormax)
        if(errormax<tol):
            break
        else:
            Vold=Vnew
    return Vold,Vnew, errormax, Uopt


#---- Coefficients Setting-----
xleft=0; xright=10.0 # consider the bound set O=[0,10]x[0,10].
yleft=0; yright=10.0
h=0.01
Nx=int((xright-xleft)/h)
Ns=2
Nu=11; hu=0.2

la=0.1 # lambda
qMainDiag=np.array([-0.5,-0.5]) # (q_11,q_22)
qSubDiag=np.array([0.5,0.5])    # (q_12,q_21)
alpha=np.array([0.2,0.25])
beta=np.array([0.35,0.2])
a=np.array([0.6,0.8])
b=np.array([0.5,0.3])
c=np.array([0.45,0.5])
d=np.array([0.65,0.8])
tol=10**(-8)   # tolerance level

Vold,Vnew,errormax,Uopt=iteration(tol,Nx,Ns,Nu,h,hu,la,alpha,beta,a,b,c,d,qMainDiag,qSubDiag)
#=== Plotting ======
x1=np.linspace(xleft,xright,Nx+1); y1=np.linspace(yleft,yright,Nx+1)
x1=list(x1); y1=list(y1)
x1,y1=np.meshgrid(x1,y1)

fig1=plt.figure()
axes1=Axes3D(fig1)
v0=Vnew[:,:,0]
axes1.plot_surface(x1,y1,v0,cmap=cm.coolwarm)
axes1.set_xlabel("x")
axes1.set_ylabel("y")
axes1.set_zlabel("value V")
axes1.invert_xaxis()
#fig1.savefig("Lot-Vol-V1.pdf",dpi=600,bbox_inches='tight')

fig2=plt.figure()
axes2=Axes3D(fig2)
v1=Vnew[:,:,1]
axes2.plot_surface(x1,y1,v1,cmap=cm.coolwarm)
axes2.set_xlabel("x")
axes2.set_ylabel("y")
axes2.set_zlabel("value V")
axes2.invert_xaxis()
#fig2.savefig("Lot-Vol-V2.pdf",dpi=600,bbox_inches='tight')

fig3=plt.figure()
axes3=Axes3D(fig3)
c0=Uopt[:,:,0]
axes3.plot_surface(x1,y1,c0,cmap=cm.coolwarm)
axes3.set_xlabel("x")
axes3.set_ylabel("y")
axes3.set_zlabel("Optimal control u")
axes3.invert_yaxis()
#fig3.savefig("Lot-Vol-u1.pdf",dpi=600,bbox_inches='tight')

fig4=plt.figure()
axes4=Axes3D(fig4)
c1=Uopt[:,:,1]
axes4.plot_surface(x1,y1,c1,cmap=cm.coolwarm)
axes4.set_xlabel("x")
axes4.set_ylabel("y")
axes4.set_zlabel("Optimal control u")
axes4.invert_xaxis()

#fig4.savefig("Lot-Vol-u2.pdf",dpi=600,bbox_inches='tight')