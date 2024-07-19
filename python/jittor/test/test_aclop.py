import unittest
import jittor as jt
from .test_core import expect_error
import numpy as np
from jittor import init, Module
import numpy as np
@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestACL(unittest.TestCase):
    @jt.flag_scope(use_acl=1)
    def test_getitem(self):
        a = jt.ones(100, 2)
        b = a[0:2, 0:2]
        np.testing.assert_allclose(b.numpy(), [[1,1],[1,1]])
        print("test getitem success")

    @jt.flag_scope(use_acl=1)
    def test_setitem(self):
        a = jt.ones(2, 2)
        b = jt.Var(0)
        a[0:1, 0:1] = b
        np.testing.assert_allclose(a.numpy(), [[0,1],[1,1]])
        print("test setitem success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_grad(self): 
        a = jt.ones(2, 2)
        b = a[0:1, 0:1]
        optimizer = jt.optim.SGD([a], 0.1)
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res=a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[1,0],[0,0]])
        print("test getitem grad success")
        
    @jt.flag_scope(use_acl=1)
    def test_setitem_grad(self):
        a = jt.ones(3, 3)
        b = jt.ones(2, 2)
        a[0:2, 0:2] = b * 2
        optimizer = jt.optim.SGD([a, b], 0.1)
        loss = a.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res_a = a.opt_grad(optimizer)
        res_b = b.opt_grad(optimizer)
        np.testing.assert_allclose(res_a.numpy(), [[1,1,1],[1,1,1],[1,1,1]])
        np.testing.assert_allclose(res_b.numpy(), [[2,2],[2,2]])
        print("test setitem grad success")
    
    @jt.flag_scope(use_acl=1)
    def test_concat(self):    
        a = jt.ones(2, 2)
        b = jt.ones(2, 2)
        c = jt.concat([a, b], 0)
        np.testing.assert_allclose(c.numpy(), [[1,1],[1,1],[1,1],[1,1]])
        print("test concat success")
    
    @jt.flag_scope(use_acl=1)
    def test_maxpool_grad(self):  
        a = jt.ones(1, 1, 4, 4)
        max_pool = jt.nn.Pool(2, op='maximum')
        optimizer = jt.optim.SGD([a], 0.1)
        b = max_pool(a)
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res=a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[[[1,0,1,0],[0,0,0,0],[1,0,1,0],[0,0,0,0]]]])
        print("test maxpool grad success")
        
    @jt.flag_scope(use_acl=1)
    def test_triu(self):
        a = jt.ones(3, 3)
        b = jt.triu_(a, 0)
        c = jt.triu_(a, 1)
        np.testing.assert_allclose(b.numpy(), [[1,1,1],[0,1,1],[0,0,1]])
        np.testing.assert_allclose(c.numpy(), [[0,1,1],[0,0,1],[0,0,0]])
        print("test triu success")
    
    @jt.flag_scope(use_acl=1)
    def test_bmm(self):
        a = jt.ones(3, 2, 2).float32()
        b = jt.bmm(a, a)
        np.testing.assert_allclose(b.numpy(), [[[2,2],[2,2]],[[2,2],[2,2]],[[2,2],[2,2]]])
        print("test bmm success")
    
    @jt.flag_scope(use_acl=1)
    def test_matmul(self):
        a = jt.ones(1, 4, 4)
        b = jt.ones(4, 2)
        c = jt.matmul(a, b)
        np.testing.assert_allclose(c.numpy(), [[[4,4],[4,4],[4,4],[4,4]]])
        print("test matmul success")
        
    @jt.flag_scope(use_acl=1)
    def test_maxpool(self):
        a = jt.ones(1, 1, 4, 4)
        max_pool = jt.nn.Pool(2, op='maximum') 
        np.testing.assert_allclose(max_pool(a).numpy(), [[[[1,1],[1,1]]]])
        print("test maxpool success")
         
    @jt.flag_scope(use_acl=1)
    def test_transpose(self):
        a = jt.ones(1, 2, 2)
        b = a.transpose(0, 2)
        np.testing.assert_allclose(b.numpy(), [[[1],[1]],[[1],[1]]])
        print("test transpose success")

    @jt.flag_scope(use_acl=1)
    def test_matmul_grad(self):
        a = jt.ones(1, 2, 2)
        b = jt.ones(2, 2)
        optimizer = jt.optim.SGD([a, b], 0.1)
        loss = jt.matmul(a, b).sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res_a = a.opt_grad(optimizer)
        res_b = b.opt_grad(optimizer)
        np.testing.assert_allclose(res_a.numpy(), [[[2,2],[2,2]]])
        np.testing.assert_allclose(res_b.numpy(), [[2,2],[2,2]])
        print("test matmul grad success")
        
    @jt.flag_scope(use_acl=1)
    def test_bmm_grad(self): 
        a = jt.ones(3, 2, 2).float32()
        optimizer = jt.optim.SGD([a], 0.1)
        c = jt.bmm(a, a)
        loss = c.sum()

        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[[4,4],[4,4]],[[4,4],[4,4]],[[4,4],[4,4]]])
        print("test bmm grad success")
    
    @jt.flag_scope(use_acl=1)
    def test_avgpool(self):
        a = jt.ones(1, 1, 4, 4)
        avg_pool = jt.nn.Pool(2, op='mean')
        b = avg_pool(a)
        np.testing.assert_allclose(b.numpy(), [[[[1,1],[1,1]]]]) 
        print("test avgpool success")
    
    @jt.flag_scope(use_acl=1)
    def test_adaptive_maxpool2d(self):
        a = jt.float32([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                          [13, 14, 15, 16]]]])
        pool = jt.nn.AdaptiveMaxPool2d((2, 2))
        b = pool(a)
        np.testing.assert_allclose(b.numpy(), [[[[6, 8], [14, 16]]]])
        print("test adaptive_maxpool2d success")
    
    @jt.flag_scope(use_acl=1)
    def test_adaptive_maxpool2d_grad(self):  
        a = jt.float32([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                          [13, 14, 15, 16]]]])
        max_pool = jt.nn.AdaptiveMaxPool2d((2, 2))
        optimizer = jt.optim.SGD([a], 0.1)
        b = max_pool(a)
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res =  a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[[[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]]]])
        print("test adaptive_maxpool2d grad success")
    
    @jt.flag_scope(use_acl=1)
    def test_adaptive_avgpool2d(self):
        a = jt.float32([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                          [13, 14, 15, 16]]]])
        pool = jt.nn.AdaptiveAvgPool2d((2, 2))
        b = pool(a)
        np.testing.assert_allclose(b.numpy(), [[[[3.5, 5.5], [11.5, 13.5]]]]) 
        print("test adaptive_avgpool2d success")
        
    @jt.flag_scope(use_acl=1)
    def test_adaptive_avgpool2d_grad(self):  
        a = jt.float32([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                          [13, 14, 15, 16]]]])
        avg_pool = jt.nn.AdaptiveAvgPool2d((2, 2))
        optimizer = jt.optim.SGD([a], 0.1)
        b = avg_pool(a)
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res =  a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]]])
        print("test adaptive_avgpool2d grad success")
    @jt.flag_scope(use_acl=1)
    def test_index(self):
        a = jt.ones(2, 3)
        [s1,s2] = jt.index(a.shape)
        np.testing.assert_allclose(s1.numpy(), [[0,0,0],[1,1,1]])
        np.testing.assert_allclose(s2.numpy(), [[0,1,2],[0,1,2]])
        print("test index success")

    @jt.flag_scope(use_acl=1)
    def test_gather(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.gather(a, 1, jt.array([[0, 0], [1, 0]]))
        np.testing.assert_allclose(b.numpy(), [[1, 1], [4, 3]])
        print("test gather success")
    
    @jt.flag_scope(use_acl=1)
    def test_gather_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        optimizer = jt.optim.SGD([a], 0.1)
        b = jt.gather(a, 1, jt.array([[0, 0], [1, 0]]))
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[2, 0], [1, 1]])
        print("test gather grad success")
    
    @jt.flag_scope(use_acl=1)
    def test_scatter(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.array([[0, 0], [0, 0]]) 
        b = jt.scatter(b, 1, jt.array([[0, 0], [1, 0]]), a, reduce="add")
        np.testing.assert_allclose(b.numpy(), [[3, 0], [4, 3]])
        print("test scatter success")
    
    @jt.flag_scope(use_acl=1)
    def test_scatter_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        b = jt.float32([[0, 0], [0, 0]]) 
        optimizer = jt.optim.SGD([a,b], 0.1)
        c = jt.scatter(b, 1, jt.array([[0, 0], [1, 0]]), a, reduce="add")
        loss = c.max()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res_a = a.opt_grad(optimizer)
        res_b = b.opt_grad(optimizer)
        np.testing.assert_allclose(res_a.numpy(), [[0, 0], [0, 1]]) 
        np.testing.assert_allclose(res_b.numpy(), [[0, 0], [1, 0]]) 
        print("test scatter grad success")

    @jt.flag_scope(use_acl=1)
    def test_where(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.ones(2, 2)
        c = jt.where(a > 2, a, b)
        np.testing.assert_allclose(c.numpy(), [[1,1],[3,4]]) 
        print("test where success")
        
    @jt.flag_scope(use_acl=1)
    def test_where_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        b = jt.array([[2., 2.], [2., 2.]])
        c = jt.where(a > 2, a, b)
        optimizer = jt.optim.SGD([a,b], 0.1)
        loss = c.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res_a = a.opt_grad(optimizer)
        res_b = b.opt_grad(optimizer)
        np.testing.assert_allclose(res_a.numpy(), [[0, 0], [1, 1]])
        np.testing.assert_allclose(res_b.numpy(), [[1, 1], [0, 0]])
        print("test where grad success")
    
    @jt.flag_scope(use_acl=1)
    def test_flip(self):
        a = jt.array([[1., 2.], [3., 4.]]) 
        b = a.flip((0,1))
        np.testing.assert_allclose(b.numpy(), [[4,3],[2,1]])
        print("test flip success")
    
    @jt.flag_scope(use_acl=1)
    def test_flip_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        optimizer = jt.optim.SGD([a], 0.1)
        b = a.flip((0,1))
        loss = b.max()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[0, 0], [0, 1]])
        print("test flip grad success")
    
    
     
if __name__ == "__main__":
    unittest.main()