import unittest
import torch
from train_denoising import replace_pixels, train_update, state_metrics

class DenoisingTest(unittest.TestCase):
    def test_endpoints_and_shared_rgb_mask(self):
        y=torch.ones(2,3,2,2)
        replacement=-torch.ones_like(y)
        u=torch.tensor([[[[.1,.4],[.7,.9]]]]).expand(2,1,2,2)
        self.assertTrue(torch.equal(replace_pixels(y,replacement,0,u),y))
        self.assertTrue(torch.equal(replace_pixels(y,replacement,1,u),replacement))
        r=torch.tensor([.2,.8]).reshape(2,1,1,1)
        x=replace_pixels(y,replacement,r,u)
        self.assertEqual((x[0,0]==-1).sum().item(),1)
        self.assertEqual((x[1,0]==-1).sum().item(),3)
        self.assertTrue(torch.equal(x[:,0],x[:,1]))
        self.assertTrue(torch.equal(x[:,1],x[:,2]))

    def test_single_update_and_target_dependent_input(self):
        model=torch.nn.Conv2d(3,3,1)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        y=torch.randn(2,3,4,4);noise=torch.randn_like(y)
        inputs=[]
        hook=model.register_forward_pre_hook(lambda m,a:inputs.append(a[0].clone()))
        train_update(model,optimizer,y,noise,torch.zeros(2,1,1,1))
        hook.remove()
        self.assertEqual(len(inputs),1)
        self.assertTrue(torch.equal(inputs[0],y))
        self.assertTrue(all(int(s['step'])==1 for s in optimizer.state.values()))

    def test_masked_metrics(self):
        y=torch.zeros(1,3,2,2);x=y.clone();x[:,:,0,:]=2
        mask=torch.tensor([[[[True,True],[False,False]]]])
        m=state_metrics(x,y,torch.zeros(3,2,2),mask)
        self.assertEqual(m['mse_to_target'],2)
        self.assertEqual(m['mse_replaced_pixels'],4)
        self.assertEqual(m['mse_retained_pixels'],0)
        self.assertIsNone(state_metrics(x,y,torch.zeros(3,2,2),torch.zeros_like(mask))['mse_replaced_pixels'])

if __name__=='__main__':unittest.main()
