
import jittor as jt
import random
import numpy as np
import unittest


class TestAdamw(unittest.TestCase):
    def test(self):
        import torch

        LR = 0.01
        BATCH_SIZE = 32
        EPOCH = 12
        WD = 0.1
        N = 1024

        # data
        x = []
        y = []
        for i in range(N):
            x.append(-1 + i * 2 / N)
        random.shuffle(x)
        x = np.array(x)
        y = x * x + np.random.randn(N) * 0.1

        class NetTorch(torch.nn.Module):
            def __init__(self):
                super(NetTorch, self).__init__()
                self.hidden = torch.nn.Linear(1, 20)   # hidden layer
                self.predict = torch.nn.Linear(20, 1)   # output layer

            def forward(self, x):
                x = torch.nn.functional.relu(self.hidden(x))      # activation function for hidden layer
                x = self.predict(x)             # linear output
                return x

        class NetJittor(jt.Module):
            def __init__(self):
                super(NetJittor, self).__init__()
                self.hidden = jt.nn.Linear(1, 20)   # hidden layer
                self.predict = jt.nn.Linear(20, 1)   # output layer

            def execute(self, x):
                x = jt.nn.relu(self.hidden(x))      # activation function for hidden layer
                x = self.predict(x)             # linear output
                return x

        net_torch = NetTorch()
        optim_torch = torch.optim.AdamW(net_torch.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay = WD)
        Loss_torch = torch.nn.MSELoss()

        net_jittor = NetJittor()
        net_jittor.hidden.weight = jt.array(net_torch.hidden.weight.detach().numpy())
        net_jittor.hidden.bias = jt.array(net_torch.hidden.bias.detach().numpy())
        net_jittor.predict.weight = jt.array(net_torch.predict.weight.detach().numpy())
        net_jittor.predict.bias = jt.array(net_torch.predict.bias.detach().numpy())
        optim_jittor = jt.optim.AdamW(net_jittor.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay = WD)
        Loss_jittor = jt.nn.MSELoss()

        for epoch in range(EPOCH):
            # print('Epoch: ', epoch)

            for i in range(N // BATCH_SIZE):
                bx = x[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, np.newaxis]
                by = y[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, np.newaxis]
                
                bx_torch = torch.Tensor(bx)
                by_torch = torch.Tensor(by)
                output_torch = net_torch(bx_torch)             
                loss_torch = Loss_torch(output_torch, by_torch)
                optim_torch.zero_grad()            
                loss_torch.backward()             
                optim_torch.step()                 

                bx_jittor = jt.array(bx)
                by_jittor = jt.array(by)
                output_jittor = net_jittor(bx_jittor)             
                loss_jittor = Loss_jittor(output_jittor, by_jittor)
                optim_jittor.step(loss_jittor)

                lt = float(loss_torch.detach().numpy())
                lj = float(loss_jittor.data)
                # print(abs(lt - lj))
                assert abs(lt - lj) < 1e-5

if __name__ == "__main__":
    unittest.main()
    