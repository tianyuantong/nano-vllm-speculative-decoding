"""CPU control-flow/operation-order checks; substitutes do not execute tensors."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

class Module:
    def __call__(self,*args,**kwargs):return self.forward(*args,**kwargs)

class Trace:
    dtype='bf16'
    def __init__(self,log):self.log=log
    def float(self):self.log.append('float32');return self
    def pow(self,n):self.log.append(('pow',n));return self
    def mean(self,**kwargs):self.log.append(('mean',kwargs));return self
    def __add__(self,x):self.log.append('epsilon');return self
    def __mul__(self,x):self.log.append('multiply');return self
    def to(self,dtype):self.log.append(('cast',dtype));return self

class NormDispatch(unittest.TestCase):
    def setUp(self):
        fake=ModuleType('torch');fake.Tensor=Trace;fake.ones=lambda n:object();fake.compile=lambda f:f
        fake.nn=ModuleType('torch.nn');fake.nn.Module=Module;fake.nn.Parameter=lambda x:x
        fake.rsqrt=lambda x:(x.log.append('rsqrt') or x)
        self.p=patch.dict(sys.modules,{'torch':fake,'torch.nn':fake.nn});self.p.start()
        path=Path(__file__).resolve().parents[1]/'nanovllm/layers/layernorm.py'
        spec=importlib.util.spec_from_file_location('tested_norm',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);self.Norm=m.RMSNorm
    def tearDown(self):self.p.stop()
    def test_default_keeps_original_compiled_dispatch(self):
        n=self.Norm(128);n.rms_forward=lambda x:'original';n.eager_rms_forward=lambda x:self.fail('unexpected eager')
        self.assertEqual(n('x'),'original')
    def test_qk_flag_selects_eager_only(self):
        n=self.Norm(128,compile_rms=False);n.rms_forward=lambda x:self.fail('unexpected compile');n.eager_rms_forward=lambda x:'explicit'
        self.assertEqual(n('x'),'explicit')
    def test_residual_dispatch_unchanged(self):
        n=self.Norm(128,compile_rms=False);n.add_rms_forward=lambda x,r:('original residual',r)
        self.assertEqual(n('x','residual'),('original residual','residual'))
    def test_rounding_boundary_precedes_weight_multiplication(self):
        n=self.Norm(128,compile_rms=False);log=[];n(Trace(log))
        self.assertEqual(log,['float32',('pow',2),('mean',{'dim':-1,'keepdim':True}),'epsilon','rsqrt','multiply',('cast','bf16'),'multiply'])

if __name__=='__main__':unittest.main(verbosity=2)
