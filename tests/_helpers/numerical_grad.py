"""Small finite-difference helper retained for legacy operator tests."""

import numpy as np


def ngrad(func, variables, eps):
    output = func(variables)
    gradients = []
    for index in range(len(variables)):
        original = variables[index].astype("float64")
        if type(original) == np.ndarray and original.size > 1:
            gradient = []
            for element in range(len(original.flatten())):
                value = original.flatten()
                value[element] += eps
                variables[index] = value.reshape(original.shape)
                perturbed = func(variables)
                gradient.append((perturbed - output) / eps)
            gradients.append(np.array(gradient).reshape(original.shape))
        else:
            variables[index] = variables[index] + eps
            perturbed = func(variables)
            gradients.append((perturbed - output) / eps)
        variables[index] = original
    return output, gradients
