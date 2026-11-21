"""Sigmoid/tanh values and derivatives."""
import math
def sigmoid(x):
    if x>=0: return 1/(1+math.exp(-x))
    e=math.exp(x); return e/(1+e)
def sigmoid_derivative(x):
    p=sigmoid(x); return p*(1-p)
def tanh(x): return math.tanh(x)
def tanh_derivative(x): return 1-math.tanh(x)**2
def main():
    for x in (-6,-2,0,2,6): print(x,sigmoid(x),tanh(x))
if __name__=="__main__": main()
