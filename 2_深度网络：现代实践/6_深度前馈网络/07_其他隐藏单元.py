"""Softplus, Swish, GELU, SELU, Mish and Maxout."""
import math
def softplus(x): return max(x,0)+math.log1p(math.exp(-abs(x)))
def swish(x,beta=1.0): return x/(1+math.exp(-max(-40,min(40,beta*x))))
def gelu(x): return .5*x*(1+math.erf(x/math.sqrt(2)))
def selu(x): return 1.0507009873554805*(x if x>0 else 1.6732632423543772*math.expm1(max(-40,x)))
def mish(x): return x*math.tanh(softplus(x))
def maxout(values): return max(values)
def main():
    for x in (-2,0,2): print(x,swish(x),gelu(x),mish(x))
if __name__=="__main__": main()
