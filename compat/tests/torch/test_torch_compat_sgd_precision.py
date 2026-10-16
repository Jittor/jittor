"""Low-precision SGD golden trajectories from independent Torch 2.13 CPU/CUDA.

Four inputs are the first divergent parameters from the complete H3 training
probe. Every gradient below is produced by backward on a linear loss. CUDA low-precision goldens include every step and momentum buffer. CPU
low-precision scalar/vector rounding remains an explicitly unresolved boundary;
CPU tests preserve native arithmetic and check the independent float32 goldens.
"""
import unittest

import numpy as np
import torch
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks

_PARAMETERS = [0.0172119140625, -0.0023345947265625, -0.004791259765625, -0.0089111328125]
_GRADIENTS = [-0.3359375, -0.0220947265625, -0.019775390625, -0.052001953125]
_GOLDEN = {'cpu': {'bfloat16/plain/0/parameter': [0.0205078125, -0.002105712890625, -0.00457763671875, -0.0084228515625],
         'bfloat16/plain/1/parameter': [0.018798828125, -0.0022125244140625, -0.004669189453125, -0.0086669921875],
         'bfloat16/plain/2/parameter': [0.0238037109375,
                                        -0.0018768310546875,
                                        -0.004364013671875,
                                        -0.00787353515625],
         'bfloat16/momentum/0/parameter': [0.0205078125, -0.002105712890625, -0.00457763671875, -0.00836181640625],
         'bfloat16/momentum/0/momentum': [-0.3359375, -0.022216796875, -0.0198974609375, -0.05224609375],
         'bfloat16/momentum/1/parameter': [0.022216796875,
                                           -0.0019989013671875,
                                           -0.004486083984375,
                                           -0.00811767578125],
         'bfloat16/momentum/1/momentum': [-0.1669921875, -0.01123046875, -0.0101318359375, -0.0264892578125],
         'bfloat16/momentum/2/parameter': [0.02783203125, -0.0016326904296875, -0.004150390625, -0.00726318359375],
         'bfloat16/momentum/2/momentum': [-0.5546875, -0.03662109375, -0.032958984375, -0.08642578125],
         'bfloat16/nesterov/0/parameter': [0.023681640625,
                                           -0.0019073486328125,
                                           -0.004425048828125,
                                           -0.0079345703125],
         'bfloat16/nesterov/0/momentum': [-0.3359375, -0.022216796875, -0.0198974609375, -0.05224609375],
         'bfloat16/nesterov/1/parameter': [0.023193359375,
                                           -0.0019378662109375,
                                           -0.00445556640625,
                                           -0.00799560546875],
         'bfloat16/nesterov/1/momentum': [-0.1337890625, -0.009033203125, -0.0081787109375, -0.0213623046875],
         'bfloat16/nesterov/2/parameter': [0.03369140625, -0.0012359619140625, -0.003814697265625, -0.00634765625],
         'bfloat16/nesterov/2/momentum': [-0.625, -0.041259765625, -0.037109375, -0.09765625],
         'float16/plain/0/parameter': [0.02056884765625,
                                       -0.00211334228515625,
                                       -0.0045928955078125,
                                       -0.008392333984375],
         'float16/plain/1/parameter': [0.018890380859375,
                                       -0.002223968505859375,
                                       -0.00469207763671875,
                                       -0.0086517333984375],
         'float16/plain/2/parameter': [0.02392578125, -0.00189208984375, -0.00439453125, -0.00787353515625],
         'float16/momentum/0/parameter': [0.02056884765625,
                                          -0.00211334228515625,
                                          -0.0045928955078125,
                                          -0.00838470458984375],
         'float16/momentum/0/momentum': [-0.33544921875,
                                         -0.0221710205078125,
                                         -0.0199127197265625,
                                         -0.052276611328125],
         'float16/momentum/1/parameter': [0.022247314453125,
                                          -0.0020008087158203125,
                                          -0.004489898681640625,
                                          -0.00811767578125],
         'float16/momentum/1/momentum': [-0.1671142578125,
                                         -0.01116943359375,
                                         -0.0101165771484375,
                                         -0.026458740234375],
         'float16/momentum/2/parameter': [0.02777099609375,
                                          -0.0016345977783203125,
                                          -0.004161834716796875,
                                          -0.00725555419921875],
         'float16/momentum/2/momentum': [-0.552734375, -0.03662109375, -0.032928466796875, -0.08636474609375],
         'float16/nesterov/0/parameter': [0.023590087890625,
                                          -0.0019130706787109375,
                                          -0.004413604736328125,
                                          -0.0079193115234375],
         'float16/nesterov/0/momentum': [-0.33544921875,
                                         -0.0221710205078125,
                                         -0.0199127197265625,
                                         -0.052276611328125],
         'float16/nesterov/1/parameter': [0.023101806640625,
                                          -0.0019426345825195312,
                                          -0.004436492919921875,
                                          -0.00798797607421875],
         'float16/nesterov/1/momentum': [-0.13330078125,
                                         -0.00897216796875,
                                         -0.00815582275390625,
                                         -0.02130126953125],
         'float16/nesterov/2/parameter': [0.03375244140625,
                                          -0.001239776611328125,
                                          -0.003803253173828125,
                                          -0.006328582763671875],
         'float16/nesterov/2/momentum': [-0.62353515625, -0.041259765625, -0.037139892578125, -0.097412109375],
         'float32/plain/0/parameter': [0.020571289584040642,
                                       -0.0021136475261300802,
                                       -0.0045935059897601604,
                                       -0.008391113020479679],
         'float32/plain/1/parameter': [0.018891602754592896,
                                       -0.002224121242761612,
                                       -0.004692383110523224,
                                       -0.008651122450828552],
         'float32/plain/2/parameter': [0.023930665105581284,
                                       -0.0018927003256976604,
                                       -0.004395752213895321,
                                       -0.007871093228459358],
         'float32/momentum/0/parameter': [0.020566126331686974,
                                          -0.0021129471715539694,
                                          -0.0045920684933662415,
                                          -0.008388440124690533],
         'float32/momentum/0/momentum': [-0.3354211449623108,
                                         -0.022164763882756233,
                                         -0.019919129088521004,
                                         -0.052269287407398224],
         'float32/momentum/1/parameter': [0.022236229851841927,
                                          -0.0020013360772281885,
                                          -0.004490796010941267,
                                          -0.008124011568725109],
         'float32/momentum/1/momentum': [-0.16701042652130127,
                                         -0.011161106638610363,
                                         -0.01012726966291666,
                                         -0.02644289843738079],
         'float32/momentum/2/parameter': [0.027765236794948578,
                                          -0.0016352691454812884,
                                          -0.0041612680070102215,
                                          -0.0072600520215928555],
         'float32/momentum/2/momentum': [-0.5529007315635681,
                                         -0.03660669922828674,
                                         -0.03295278921723366,
                                         -0.08639593422412872],
         'float32/nesterov/0/parameter': [0.023584915325045586,
                                          -0.0019134642789140344,
                                          -0.004412796348333359,
                                          -0.00791801605373621],
         'float32/nesterov/0/momentum': [-0.3354211449623108,
                                         -0.022164763882756233,
                                         -0.019919129088521004,
                                         -0.052269287407398224],
         'float32/nesterov/1/parameter': [0.023096976801753044,
                                          -0.0019427388906478882,
                                          -0.004436802119016647,
                                          -0.007984139956533909],
         'float32/nesterov/1/momentum': [-0.1332027167081833,
                                         -0.00895832758396864,
                                         -0.00817190483212471,
                                         -0.021278923377394676],
         'float32/nesterov/2/parameter': [0.03373697027564049,
                                          -0.0012393693905323744,
                                          -0.00380448205396533,
                                          -0.006325174123048782],
         'float32/nesterov/2/momentum': [-0.6230958104133606,
                                         -0.04126286506652832,
                                         -0.03715090453624725,
                                         -0.09739348292350769]},
 'cuda': {'bfloat16/plain/0/parameter': [0.0206298828125,
                                         -0.0021209716796875,
                                         -0.004608154296875,
                                         -0.00836181640625],
          'bfloat16/plain/1/parameter': [0.0189208984375,
                                         -0.002227783203125,
                                         -0.00469970703125,
                                         -0.00860595703125],
          'bfloat16/plain/2/parameter': [0.02392578125, -0.00189208984375, -0.00439453125, -0.0078125],
          'bfloat16/momentum/0/parameter': [0.0206298828125,
                                            -0.002105712890625,
                                            -0.00457763671875,
                                            -0.00836181640625],
          'bfloat16/momentum/0/momentum': [-0.3359375, -0.022216796875, -0.0198974609375, -0.05224609375],
          'bfloat16/momentum/1/parameter': [0.0223388671875,
                                            -0.0019989013671875,
                                            -0.004486083984375,
                                            -0.00811767578125],
          'bfloat16/momentum/1/momentum': [-0.16796875, -0.01123046875, -0.0101318359375, -0.0264892578125],
          'bfloat16/momentum/2/parameter': [0.02783203125,
                                            -0.0016326904296875,
                                            -0.004150390625,
                                            -0.00726318359375],
          'bfloat16/momentum/2/momentum': [-0.5546875, -0.03662109375, -0.032958984375, -0.08642578125],
          'bfloat16/nesterov/0/parameter': [0.0235595703125,
                                            -0.00191497802734375,
                                            -0.004425048828125,
                                            -0.0079345703125],
          'bfloat16/nesterov/0/momentum': [-0.3359375, -0.022216796875, -0.0198974609375, -0.05224609375],
          'bfloat16/nesterov/1/parameter': [0.0230712890625,
                                            -0.00194549560546875,
                                            -0.00445556640625,
                                            -0.00799560546875],
          'bfloat16/nesterov/1/momentum': [-0.1337890625, -0.009033203125, -0.0081787109375, -0.0213623046875],
          'bfloat16/nesterov/2/parameter': [0.03369140625,
                                            -0.00124359130859375,
                                            -0.0038299560546875,
                                            -0.00634765625],
          'bfloat16/nesterov/2/momentum': [-0.625, -0.041259765625, -0.037109375, -0.09765625],
          'float16/plain/0/parameter': [0.02056884765625,
                                        -0.00211334228515625,
                                        -0.0045928955078125,
                                        -0.008392333984375],
          'float16/plain/1/parameter': [0.018890380859375,
                                        -0.002223968505859375,
                                        -0.00469207763671875,
                                        -0.0086517333984375],
          'float16/plain/2/parameter': [0.02392578125, -0.00189208984375, -0.00439453125, -0.00787353515625],
          'float16/momentum/0/parameter': [0.02056884765625,
                                           -0.00211334228515625,
                                           -0.0045928955078125,
                                           -0.00838470458984375],
          'float16/momentum/0/momentum': [-0.33544921875,
                                          -0.0221710205078125,
                                          -0.0199127197265625,
                                          -0.052276611328125],
          'float16/momentum/1/parameter': [0.022247314453125,
                                           -0.0020008087158203125,
                                           -0.004489898681640625,
                                           -0.00811767578125],
          'float16/momentum/1/momentum': [-0.1671142578125,
                                          -0.01116943359375,
                                          -0.0101165771484375,
                                          -0.026458740234375],
          'float16/momentum/2/parameter': [0.0277862548828125,
                                           -0.0016345977783203125,
                                           -0.004161834716796875,
                                           -0.007251739501953125],
          'float16/momentum/2/momentum': [-0.55322265625, -0.03662109375, -0.032928466796875, -0.08642578125],
          'float16/nesterov/0/parameter': [0.023590087890625,
                                           -0.0019130706787109375,
                                           -0.004413604736328125,
                                           -0.0079193115234375],
          'float16/nesterov/0/momentum': [-0.33544921875,
                                          -0.0221710205078125,
                                          -0.0199127197265625,
                                          -0.052276611328125],
          'float16/nesterov/1/parameter': [0.023101806640625,
                                           -0.0019426345825195312,
                                           -0.004436492919921875,
                                           -0.00798797607421875],
          'float16/nesterov/1/momentum': [-0.13330078125,
                                          -0.00897216796875,
                                          -0.00815582275390625,
                                          -0.02130126953125],
          'float16/nesterov/2/parameter': [0.03375244140625,
                                           -0.001239776611328125,
                                           -0.003803253173828125,
                                           -0.006328582763671875],
          'float16/nesterov/2/momentum': [-0.62353515625, -0.041259765625, -0.037139892578125, -0.097412109375],
          'float32/plain/0/parameter': [0.020571289584040642,
                                        -0.0021136475261300802,
                                        -0.0045935059897601604,
                                        -0.008391113020479679],
          'float32/plain/1/parameter': [0.018891602754592896,
                                        -0.002224121242761612,
                                        -0.004692383110523224,
                                        -0.008651122450828552],
          'float32/plain/2/parameter': [0.023930665105581284,
                                        -0.0018927003256976604,
                                        -0.004395752213895321,
                                        -0.007871093228459358],
          'float32/momentum/0/parameter': [0.020566126331686974,
                                           -0.0021129471715539694,
                                           -0.0045920684933662415,
                                           -0.008388440124690533],
          'float32/momentum/0/momentum': [-0.3354211449623108,
                                          -0.022164763882756233,
                                          -0.019919129088521004,
                                          -0.052269287407398224],
          'float32/momentum/1/parameter': [0.022236229851841927,
                                           -0.0020013360772281885,
                                           -0.004490796010941267,
                                           -0.008124011568725109],
          'float32/momentum/1/momentum': [-0.16701042652130127,
                                          -0.011161106638610363,
                                          -0.01012726966291666,
                                          -0.02644289843738079],
          'float32/momentum/2/parameter': [0.027765236794948578,
                                           -0.0016352691454812884,
                                           -0.0041612680070102215,
                                           -0.0072600520215928555],
          'float32/momentum/2/momentum': [-0.5529007315635681,
                                          -0.03660669922828674,
                                          -0.03295278921723366,
                                          -0.08639593422412872],
          'float32/nesterov/0/parameter': [0.023584915325045586,
                                           -0.0019134642789140344,
                                           -0.004412796348333359,
                                           -0.00791801605373621],
          'float32/nesterov/0/momentum': [-0.3354211449623108,
                                          -0.022164763882756233,
                                          -0.019919129088521004,
                                          -0.052269287407398224],
          'float32/nesterov/1/parameter': [0.023096976801753044,
                                           -0.0019427388906478882,
                                           -0.004436802119016647,
                                           -0.007984139956533909],
          'float32/nesterov/1/momentum': [-0.1332027167081833,
                                          -0.00895832758396864,
                                          -0.00817190483212471,
                                          -0.021278923377394676],
          'float32/nesterov/2/parameter': [0.03373697027564049,
                                           -0.0012393693905323744,
                                           -0.00380448205396533,
                                           -0.006325174123048782],
          'float32/nesterov/2/momentum': [-0.6230958104133606,
                                          -0.04126286506652832,
                                          -0.03715090453624725,
                                          -0.09739348292350769]}}

class TestSGDPrecision(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if jt.flags.use_cuda else "cpu"
        self.count = jt.core.backend_fallback_count()
        self.guard = forbid_backend_fallbacks()
        self.guard.__enter__()

    def tearDown(self):
        self.guard.__exit__(None, None, None)
        self.assertEqual(jt.core.backend_fallback_count(), self.count)

    def _check(self, dtype, configuration):
        if self.device == "cpu" and dtype != "float32":
            self.skipTest("CUDA-only low-precision SGD policy; Torch CPU scalar/vector rounding differs")
        options = {
            "plain": {},
            "momentum": {"momentum": .9, "weight_decay": .03, "dampening": .2},
            "nesterov": {"momentum": .9, "weight_decay": .03, "nesterov": True},
        }[configuration]
        parameter = torch.tensor(_PARAMETERS, dtype=getattr(torch, dtype),
                                 device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([parameter], lr=.01, **options)
        for step, factor in enumerate((1., -.5, 1.5)):
            optimizer.zero_grad()
            gradient = torch.tensor([v * factor for v in _GRADIENTS],
                                    dtype=getattr(torch, dtype), device=self.device)
            (parameter * gradient).sum().backward()
            published = parameter.grad
            np.testing.assert_array_equal(published.float().cpu().numpy(),
                                          gradient.float().cpu().numpy())
            optimizer.step()
            self.assertIs(parameter.grad, published)
            parameter.sync()
            self.assertEqual(parameter.location(), "device" if self.device == "cuda" else "cpu")
            self.assertEqual(parameter.dtype, getattr(torch, dtype))
            stem = f"{dtype}/{configuration}/{step}"
            states = {"parameter": parameter}
            if options.get("momentum"):
                states["momentum"] = optimizer.state[parameter]["momentum_buffer"]
            for name, value in states.items():
                reference = _GOLDEN[self.device][stem + "/" + name]
                got = value.detach().float().cpu().numpy()
                if dtype == "float32":
                    np.testing.assert_allclose(got, reference, atol=2e-9, rtol=2e-6,
                                               err_msg=stem + "/" + name)
                else:
                    np.testing.assert_array_equal(got, reference, err_msg=stem + "/" + name)

    def test_bfloat16_plain(self):
        self._check("bfloat16", "plain")

    def test_bfloat16_momentum(self):
        self._check("bfloat16", "momentum")

    def test_bfloat16_nesterov(self):
        self._check("bfloat16", "nesterov")

    def test_float16_plain(self):
        self._check("float16", "plain")

    def test_float16_momentum(self):
        self._check("float16", "momentum")

    def test_float16_nesterov(self):
        self._check("float16", "nesterov")

    def test_float32_plain(self):
        self._check("float32", "plain")

    def test_float32_momentum(self):
        self._check("float32", "momentum")

    def test_float32_nesterov(self):
        self._check("float32", "nesterov")

    def test_native_sgd_default_float32_update(self):
        parameter = jt.array(_PARAMETERS, dtype="float32")
        gradient = jt.array(_GRADIENTS, dtype="float32").stop_grad()
        optimizer = jt.optim.SGD([parameter], lr=.01, fused=False)
        self.assertFalse(optimizer.torch_math)
        optimizer.step((parameter * gradient).sum())
        reference = _GOLDEN[self.device]["float32/plain/0/parameter"]
        np.testing.assert_allclose(parameter.numpy(), reference, atol=2e-9, rtol=2e-6)

    def test_cpu_low_precision_original_plain_rounding(self):
        if self.device != "cpu":
            self.skipTest("CPU-only original native path is excluded by this CUDA process")
        references = {
            "bfloat16": [0.0206298828125, -0.0021209716796875, -0.004608154296875, -0.00836181640625],
            "float16": _GOLDEN["cpu"]["float16/plain/0/parameter"],
        }
        for dtype, reference in references.items():
            parameter = torch.tensor(_PARAMETERS, dtype=getattr(torch, dtype), requires_grad=True)
            gradient = torch.tensor(_GRADIENTS, dtype=getattr(torch, dtype))
            optimizer = torch.optim.SGD([parameter], lr=.01)
            (parameter * gradient).sum().backward()
            optimizer.step()
            np.testing.assert_array_equal(parameter.float().numpy(), reference)

    def test_cuda_fsdp_shard_forwards_the_frontend_precision_policy(self):
        if self.device != "cuda":
            self.skipTest("CUDA-only SGD precision policy for FSDP shards")
        from jittor.compat.fsdp2.optimizer import _sgd_update_for_param
        for dtype in ("bfloat16", "float16"):
            parameter = torch.tensor(_PARAMETERS, dtype=getattr(torch, dtype), device=self.device,
                                     requires_grad=True)
            gradient = torch.tensor(_GRADIENTS, dtype=getattr(torch, dtype), device=self.device)
            optimizer = torch.optim.SGD([parameter], lr=.01)
            group = optimizer.param_groups[0]
            with jt.no_grad():
                updated, _ = _sgd_update_for_param(optimizer, group, None, None, parameter,
                                                  gradient, group["values"][0], index=0)
            updated.sync()
            self.assertEqual(updated.location(), "device")
            np.testing.assert_array_equal(updated.float().cpu().numpy(),
                                          _GOLDEN["cuda"][dtype + "/plain/0/parameter"])
