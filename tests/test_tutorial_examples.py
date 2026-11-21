"""Regression tests for the first eight executable examples."""
import importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/"2_深度网络：现代实践"/"6_深度前馈网络"
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,BASE/file); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
class TutorialMathTests(unittest.TestCase):
 def test_xor(self):
  m=load("xor","00_实例：学习_XOR.py"); self.assertEqual([m.xor_network(a,b) for a,b in ((0,0),(0,1),(1,0),(1,1))],[0,1,1,0])
 def test_gradient(self):
  m=load("grad","01_基于梯度的学习.py"); w,h=m.gradient_descent(-3,.2,30); self.assertLess(abs(w-2),1e-5); self.assertLess(h[-1],h[0])
 def test_losses(self):
  m=load("loss","02_代价函数.py"); self.assertAlmostEqual(m.mse([1,2],[1,4]),2); self.assertAlmostEqual(m.bce_logit_gradient(0,1),-.5); self.assertGreater(m.binary_cross_entropy(1,.9),0)
 def test_outputs(self):
  m=load("out","03_输出单元.py"); p=m.softmax([1.2,.2,-.5]); self.assertAlmostEqual(sum(p),1,places=12); self.assertTrue(all(0<x<1 for x in p))
 def test_hidden(self):
  m=load("hidden","04_隐藏单元.py"); self.assertEqual(m.compose_affine(2,2,1,3,4),19); self.assertEqual(m.activate(-2,"relu"),0)
 def test_relu_family(self):
  m=load("relu","05_整流线性单元及其扩展.py"); self.assertEqual(m.relu(-1),0); self.assertAlmostEqual(m.leaky_relu(-2,.1),-.2); self.assertGreater(m.elu(-1),-1)
 def test_sigmoid_tanh(self):
  m=load("st","06_logistic_sigmoid_与双曲正切函数.py"); self.assertAlmostEqual(m.sigmoid(0),.5); self.assertAlmostEqual(m.tanh_derivative(0),1)
 def test_other(self):
  m=load("other","07_其他隐藏单元.py"); self.assertEqual(m.swish(0),0); self.assertEqual(m.mish(0),0); self.assertGreater(m.gelu(1),0)
if __name__=="__main__": unittest.main()
