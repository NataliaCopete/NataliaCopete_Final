
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


datos=np.loadtxt('datos_observacionales.dat')
tiempo=datos[:,0]
x=datos[:,1]
y=datos[:,2]
z=datos[:,3]
xyz=datos[:,1:]


def ecuaciones(var,t,sigma,ro,beta):
    x,y,z=var
    ecuaciones=[sigma*(y-x), x*(ro-z)-y, x*y-beta*z]
    return ecuaciones
def sol(t,sigma,ro,beta):
    yo=datos[0][:-1]
    solucion= odeint(ecuaciones,yo,t,args=(sigma,ro,beta))
    return solucion

def sol_red(t,sigma,ro,beta):
    solucion1=sol(t,sigma,ro,beta)
    lista=[]
    for i in range(0,310,+10):
        lista.append(solucion1[i])
    arr=np.array(lista)
    return arr
tiempos=np.linspace(0,10,1000)

def loglike(xyz,tiempo,param):
    sigma,ro,beta=param
    sigma_obs=1.0
    d= sol_red(tiempo,sigma,ro,beta)-xyz
    d = d/sigma_obs
    d = -0.5 * np.sum(d**2)
    return d

def divergence_loglikelihood(xyz,tiempo,param):
    n_param = len(param)
    div = np.ones(n_param)
    delta = 1E-4
    for i in range(n_param):
        delta_param = np.zeros(n_param)
        delta_param[i] = delta
        div[i] = loglike(xyz,tiempo, param + delta_param) 
        div[i] = div[i] - loglike(xyz,tiempo,param - delta_param)
        div[i] = div[i]/(2.0 * delta)
    return div


def energia(xyz,tiempo, param, param_momentum):
    m = 10.0
    K = 0.5 * np.sum(param_momentum**2)/m
    V = -loglike(xyz,tiempo,param)     
    return K + V




def leapfrog(xyz,tiempo, param, param_momentum):
    N = 7
    delta_t = 1E-2
    m = 10.0
    new_param = param.copy()
    new_param_momentum = param_momentum.copy()
    for i in range(N):
        new_param_momentum = new_param_momentum + divergence_loglikelihood(xyz,tiempo, param) * 0.5 * delta_t
        new_param = new_param + (new_param_momentum/m) * delta_t
        new_param_momentum = new_param_momentum + divergence_loglikelihood(xyz,tiempo, param) * 0.5 * delta_t
    new_param_momentum = -new_param_momentum
    return new_param, new_param_momentum



def montecarlo(xyz,tiempo,N=1000):
    lista_param=[np.random.random(3)]
    lista_momento=[np.random.random(3)]
    for i in range(1,N):
        propuesta_param,propuesta_momento=leapfrog(xyz,tiempo,lista_param[i-1],lista_momento[i-1])
        energia_nueva=energia(xyz,tiempo,propuesta_param,propuesta_momento)
        energia_vieja=energia(xyz,tiempo,lista_param[i-1],lista_momento[i-1])
        r=min(1,np.exp(-(energia_nueva-energia_vieja)))
        a=np.random.random()
        if(a<r):
            lista_param.append(propuesta_param)
            lista_momento.append(propuesta_momento)
        else:
            lista_param.append(lista_param[i-1])
            lista_momento.append(lista_momento[i-1])
    return lista_param
        
    


a=montecarlo(xyz,tiempos)


a=np.array(a)
sigma_dist=a[:,0]
ro_dist=a[:,1]
beta_dist=a[:,2]



plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
h1=plt.hist(sigma_dist)
#maxi1=h[0][h[1]==np.max(h[1])]
plt.subplot(3,1,2)
h2=plt.hist(ro_dist)
plt.subplot(3,1,3)
h3=plt.hist(beta_dist)
plt.savefig("parametros.pdf")