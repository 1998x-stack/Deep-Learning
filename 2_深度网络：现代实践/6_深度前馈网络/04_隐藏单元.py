"""Affine transforms and nonlinear hidden representations."""
def affine(x,w,b): return sum(a*c for a,c in zip(x,w))+b
def compose_affine(x,w1,b1,w2,b2): return w2*(w1*x+b1)+b2
def activate(z,name):
    if name=="linear": return z
    if name=="relu": return max(0.0,z)
    if name=="tanh":
        import math; return math.tanh(z)
    raise ValueError(name)
def main(): print("two affine layers:",compose_affine(2,2,1,3,4))
if __name__=="__main__": main()
