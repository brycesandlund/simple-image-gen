import unittest
from unittest.mock import patch
import torch
import train_categorical as tc

class CategoricalTest(unittest.TestCase):
    def test_histograms_and_quantization(self):
        raw=torch.zeros(2,3,32,32,dtype=torch.uint8)
        raw[1]=255
        raw[:,0,0,0]=128
        p=tc.pixel_histograms(raw,200)
        self.assertTrue(torch.equal(p.sum(1),torch.ones(3072)))
        self.assertEqual(p[0,100].item(),1)
        self.assertEqual(p[1,0].item(),.5)
        self.assertEqual(p[1,199].item(),.5)
        samples=tc.sample_histograms(p,100,torch.Generator().manual_seed(1))
        self.assertTrue(torch.all(samples[:,0,0,0]==100))
        self.assertTrue(torch.all((samples[:,0,0,1]==0)|(samples[:,0,0,1]==199)))
        self.assertTrue(torch.equal(tc.decode(torch.tensor([0,199]),200),torch.tensor([-1.,1.])))

    def test_categorical_sampling_probabilities(self):
        torch.manual_seed(7)
        logits=torch.tensor([.1,.2,.7]).log().reshape(1,1,3,1,1).expand(12000,1,3,1,1)
        values=tc.sample_logits(logits).flatten()
        for x,p in [(-1,.1),(0,.2),(1,.7)]:
            self.assertAlmostEqual((values==x).float().mean().item(),p,delta=.02)

    def test_detached_pre_update_sample_and_fixed_target(self):
        torch.manual_seed(1)
        model=tc.CategoricalRefiner(width=4,bins=4)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        initial=torch.randn(2,3,8,8)
        target=torch.randint(4,(2,3,8,8))
        inputs=[];samples=[]
        hook=model.register_forward_pre_hook(lambda m,a: inputs.append(a[0].clone()))
        actual=tc.sample_logits
        def sampling(logits):
            result=actual(logits);samples.append(result.clone());return result
        with patch.object(tc,'sample_logits',side_effect=sampling):
            losses=tc.train_trajectory(model,optimizer,target,initial,3)
        hook.remove()
        self.assertEqual(len(losses),3)
        self.assertTrue(torch.equal(inputs[0],initial))
        for i in range(1,3): self.assertTrue(torch.equal(inputs[i],samples[i-1]))
        self.assertTrue(all(not x.requires_grad and x.grad_fn is None for x in inputs))
        self.assertTrue(all(int(s['step'])==3 for s in optimizer.state.values()))

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
