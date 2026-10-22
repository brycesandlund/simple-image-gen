import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from train import train_trajectory
from train_fixed import make_fixed_noise, noise_digest


class FixedPairsTest(unittest.TestCase):
    def test_pairing_survives_shuffle_and_training(self):
        noise = make_fixed_noise(9, 2042)
        original = noise.clone()
        targets = torch.randn(9, 3, 32, 32)
        loader = DataLoader(TensorDataset(torch.arange(9)), batch_size=4, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        model = torch.nn.Conv2d(3, 3, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        orders = []
        updates = 0
        for depth in (1, 2, 3):
            order = []
            for (ids,) in loader:
                order.extend(ids.tolist())
                for i in ids:
                    self.assertTrue(torch.equal(noise[i], original[i]))
                losses = train_trajectory(model, optimizer, targets[ids], depth, initial=noise[ids])
                self.assertEqual(len(losses), depth)
                updates += depth
            self.assertEqual(sorted(order), list(range(9)))
            orders.append(order)
        self.assertNotEqual(orders[0], orders[1])
        self.assertTrue(torch.equal(noise, original))
        self.assertEqual(noise_digest(noise), noise_digest(make_fixed_noise(9, 2042)))
        self.assertFalse(torch.equal(noise, make_fixed_noise(9, 1042)))
        self.assertEqual(updates, 18)
        self.assertTrue(all(int(state['step']) == 18 for state in optimizer.state.values()))


if __name__ == '__main__':
    unittest.main()
