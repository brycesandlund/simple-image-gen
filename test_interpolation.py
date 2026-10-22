import unittest
import torch
from train_interpolation import interpolate, train_interpolations


class InterpolationTest(unittest.TestCase):
    def test_endpoints_and_per_image_fraction(self):
        y=torch.randn(3,3,4,4)
        z=torch.randn_like(y)
        t=torch.tensor([0.,.2,1.]).view(3,1,1,1)
        x=interpolate(y,z,t)
        self.assertTrue(torch.equal(x[0],z[0]))
        self.assertTrue(torch.equal(x[2],y[2]))
        self.assertTrue(torch.allclose(x[1],.8*z[1]+.2*y[1]))

    def test_inputs_do_not_depend_on_model_outputs(self):
        y=torch.randn(4,3,4,4)
        histories=[]
        for bias in (0.,10.):
            model=torch.nn.Conv2d(3,3,1)
            torch.nn.init.zeros_(model.weight)
            torch.nn.init.constant_(model.bias,bias)
            history=[]
            model.register_forward_pre_hook(lambda module,args: history.append(args[0].detach().clone()))
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
            torch.manual_seed(123)
            losses,ts=train_interpolations(model,optimizer,y,3)
            self.assertEqual(len(losses),3)
            self.assertTrue(all(0<t<1 for t in ts))
            self.assertTrue(all(int(s['step'])==3 for s in optimizer.state.values()))
            histories.append(history)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(*histories)))
        self.assertFalse(torch.equal(histories[0][0],histories[0][1]))


if __name__=='__main__':
    unittest.main()
