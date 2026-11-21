"""Scalar gradient descent; gradient computation and update are separate."""
def objective(w): return (w-2.0)**2
def gradient(w): return 2.0*(w-2.0)
def gradient_descent(start,learning_rate,steps):
    w=float(start); history=[]
    for _ in range(steps): history.append(objective(w)); w-=learning_rate*gradient(w)
    history.append(objective(w)); return w,history
def main():
    for eta in (.1,.8,1.1):
        w,h=gradient_descent(-3,eta,8); print(eta,w,h[-1])
if __name__=="__main__": main()
