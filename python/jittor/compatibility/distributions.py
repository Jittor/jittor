import jittor as jt

class RelaxedBernoulli:
    def __init__(self, temperature, probs=None, logits=None):
        self.temperature = temperature
        self.probs = probs
        self.logits = logits
    
    def rsample(self):
        noise = jt.rand_like(self.logits)
        eps = 1e-20
        noise = jt.clamp(noise, eps, 1.0 - eps)
        logit_noise = jt.log(noise) - jt.log(1 - noise)
        sample = (self.logits + logit_noise) / self.temperature
        return jt.sigmoid(sample)
