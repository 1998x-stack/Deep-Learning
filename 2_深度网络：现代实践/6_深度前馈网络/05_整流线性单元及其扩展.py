"""ReLU-family activation functions."""
import math
def relu(x): return max(0.0,x)
def leaky_relu(x,alpha=.01): return x if x>=0 else alpha*x
def prelu(x,alpha): return x if x>=0 else alpha*x
def elu(x,alpha=1.0): return x if x>=0 else alpha*math.expm1(max(-40,x))
def selu(x): return 1.0507009873554805*(x if x>=0 else 1.6732632423543772*math.expm1(max(-40,x)))
def main(): print([relu(x) for x in (-2,-1,0,1,2)])
if __name__=="__main__": main()
