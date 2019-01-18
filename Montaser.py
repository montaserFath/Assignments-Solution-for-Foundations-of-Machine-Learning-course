import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import comb
############################# q1a     
def postD(nTotal, nWhite):
    likelihood=np.zeros(11)
    evidence=0
    for i in range (11):
        likelihood[i]=comb(nTotal,nWhite)*((i/10)**nWhite)*((1-(i/10))**(nTotal-nWhite))
        evidence+=((1/10)*likelihood[i])
        
    print("p(y)=",evidence)
    print ("p(y|theta)",likelihood)
    posterior=likelihood*(1/10)/(evidence)
    print("p(theta|y)",posterior)
    #print("summ",sum(theta))
    plt.plot(posterior)
    plt.show()
    #pass
    return (posterior)
############################# q1b      
def evidenceC(nTotal, nWhite):
    likelihood=np.zeros(11)
    evidence=0
    for i in range (0,11,2):
        likelihood[i]=comb(nTotal,nWhite)*((i/10)**nWhite)*((1-(i/10))**(nTotal-nWhite))
        evidence+=((1/6)*likelihood[i])
        
    print("p(y)=",evidence)
    print ("p(y|theta)",likelihood)
    posterior =likelihood*(1/6)/(evidence)
    print("p(theta|y)",posterior)
    print("summ",sum(posterior))
    plt.plot(evidence)
    plt.show()
    return (evidence)
