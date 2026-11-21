"""Stable output parameterizations."""
import math
def sigmoid(z):
    if z>=0: return 1/(1+math.exp(-z))
    e=math.exp(z); return e/(1+e)
def softmax(logits):
    m=max(logits); ex=[math.exp(x-m) for x in logits]; total=sum(ex); return [v/total for v in ex]
def multilabel_sigmoid(logits): return [sigmoid(x) for x in logits]
def main(): print(softmax([1.2,.2,-.5]))
if __name__=="__main__": main()
