"""Losses and binary logit gradients."""
import math
def mse(predictions,targets): return sum((p-y)**2 for p,y in zip(predictions,targets))/len(predictions)
def mae(predictions,targets): return sum(abs(p-y) for p,y in zip(predictions,targets))/len(predictions)
def binary_cross_entropy(y,p):
    p=min(max(float(p),1e-12),1-1e-12); return -(y*math.log(p)+(1-y)*math.log(1-p))
def sigmoid(z): return 1/(1+math.exp(-max(-40,min(40,z))))
def bce_logit_gradient(z,y): return sigmoid(z)-y
def mse_logit_gradient(z,y):
    p=sigmoid(z); return (p-y)*p*(1-p)
def main(): print("BCE logit gradient:",bce_logit_gradient(0,1))
if __name__=="__main__": main()
