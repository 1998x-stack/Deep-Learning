"""Exact two-unit ReLU construction for binary XOR."""
def relu(x): return max(0.0,float(x))
def xor_network(x1,x2): return int(round(relu(x1-x2)+relu(x2-x1)))
def main():
    for x1,x2 in ((0,0),(0,1),(1,0),(1,1)): print((x1,x2),xor_network(x1,x2))
if __name__=="__main__": main()
