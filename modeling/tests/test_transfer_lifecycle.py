from __future__ import annotations
import unittest
from llm_infer_model.performance.transfer_lifecycle import PiecewiseCompletedResponse,FirstResponseImpulse,CompletedTransferLifecycle
from llm_infer_model.performance.parameters import InvalidModel,UnknownParameter

def model(first=True,window=True):
    e=('explicit-profile-evidence',)
    base=PiecewiseCompletedResponse((1024,1048576,8388608),(20000.,100000.,700000.),e)
    p=PiecewiseCompletedResponse(base.byte_knots,(40000.,50000.,60000.),e,True)
    return CompletedTransferLifecycle('a'*64,'b'*64,'dram->vram','registered',base,(FirstResponseImpulse(2000000,p),FirstResponseImpulse(10000000,p)),128,first,window,(1024,1048576),(8,64))

class TransferLifecycleTests(unittest.TestCase):
    def kw(self):return dict(context_sha256='a'*64,preparation_sha256='b'*64)
    def test_first_is_independent_of_future_window_and_window_preserves_impulse(self):
        m=model();first=m.first_response_ns(1048576,2000000,**self.kw())
        self.assertEqual(first,150000)
        self.assertEqual(m.window_ns(1048576,2000000,8,**self.kw()),first+7*100000)
        self.assertEqual(m.window_ns(1048576,2000000,64,**self.kw()),first+63*100000)
    def test_bytes_are_continuously_interpolated_but_not_implicitly_qualified(self):
        m=model();b=(1024+1048576)//2
        self.assertAlmostEqual(m.sustained.at(b),60000.,delta=.1)
        with self.assertRaises(UnknownParameter):m.first_response_ns(b,2000000,**self.kw())
        self.assertGreater(m.first_response_ns(b,2000000,**self.kw(),require_qualified=False),0)
    def test_unseen_idle_context_and_preparation_refuse(self):
        m=model()
        with self.assertRaises(UnknownParameter):m.window_ns(1024,3000000,8,**self.kw())
        with self.assertRaises(UnknownParameter):m.window_ns(1024,2000000,8,context_sha256='c'*64,preparation_sha256='b'*64)
        with self.assertRaises(UnknownParameter):m.window_ns(1024,2000000,8,context_sha256='a'*64,preparation_sha256='c'*64)
    def test_first_and_window_qualification_cannot_substitute_each_other(self):
        m=model(first=False,window=True)
        self.assertGreater(m.window_ns(1024,2000000,64,**self.kw()),0)
        with self.assertRaises(UnknownParameter):m.first_response_ns(1024,2000000,**self.kw())
    def test_invalid_knots_and_negative_total_first_refuse(self):
        with self.assertRaises(InvalidModel):PiecewiseCompletedResponse((1024,1024),(1.,2.),('e',))
        m=model();bad=PiecewiseCompletedResponse(m.sustained.byte_knots,(-30000.,-200000.,-800000.),('e',),True)
        with self.assertRaises(InvalidModel):CompletedTransferLifecycle('a'*64,'b'*64,'dram->vram','registered',m.sustained,(FirstResponseImpulse(0,bad),),128,False,False,(),())

if __name__=='__main__':unittest.main()
