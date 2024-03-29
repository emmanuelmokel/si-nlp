import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef

class GaussianLikelihood:
    """
    Minus Gaussian likelihood for regression problems.

    Mean squared error (MSE) divided by `2 * noise_var`.
    """
    
    def __init__(self, noise_var = 0.5):
        self.noise_var = noise_var

        self.mse = torch.nn.functional.mse_loss
    
    def __call__(self, model, input, target):

        output = model(input)
        
        if self.noise_var is not None:
            mse = self.mse(output, target)
            loss = mse / (2 * self.noise_var)
            return loss, output, {"mse": mse}
        
        else:
            mean = output[:,0].view_as(target)
            var = output[:,1].view_as(target)

            mse = self.mse(mean, target, reduction='none')
            mean_portion = mse / (2 * var)
            var_portion = 0.5 * torch.log(var)
            loss = mean_portion + var_portion

            return loss.mean(), output[:,0], {'mse': torch.mean((mean - target)**2.0)}


def cross_entropy(model, batch):
    # standard cross-entropy loss function
    
    output = model(**batch)

    loss = output.loss

    return loss #, output, {}


def cross_entropy_output(output, target):
    # standard cross-entropy loss function

    loss = F.cross_entropy(output, target)

    return loss, {}

def matthews_loss(model, input, target):
     # Matthews correlation coefficent
     output = model(**input)
     loss = matthews_corrcoef(target, output)

     return loss, output, {}


def adversarial_cross_entropy(model, input, target, lossfn = F.cross_entropy, epsilon = 0.01):
    # loss function based on algorithm 1 of "simple and scalable uncertainty estimation using
    # deep ensembles," lakshminaraynan, pritzel, and blundell, nips 2017, 
    # https://arxiv.org/pdf/1612.01474.pdf
    # note: the small difference bw this paper is that here the loss is only backpropped
    # through the adversarial loss rather than both due to memory constraints on preresnets
    # we can change back if we want to restrict ourselves to VGG-like networks (where it's fine).

    #scale epsilon by min and max (should be [0,1] for all experiments)
    #see algorithm 1 of paper
    scaled_epsilon = epsilon * (input.max() - input.min())

    #force inputs to require gradient
    input.requires_grad = True

    #standard forwards pass
    output = model(input)
    loss = lossfn(output, target)

    #now compute gradients wrt input
    loss.backward(retain_graph = True)
        
    #now compute sign of gradients
    inputs_grad = torch.sign(input.grad)
    
    #perturb inputs and use clamped output
    inputs_perturbed = torch.clamp(input + scaled_epsilon * inputs_grad, 0.0, 1.0).detach()
    #inputs_perturbed.requires_grad = False

    input.grad.zero_()
    #model.zero_grad()

    outputs_perturbed = model(inputs_perturbed)
    
    #compute adversarial version of loss
    adv_loss = lossfn(outputs_perturbed, target)

    #return mean of loss for reasonable scalings
    return (loss + adv_loss)/2.0, output, {}

def masked_loss(y_pred, y_true, void_class = 11., weight=None, reduce = True):
    # masked version of crossentropy loss

    el = torch.ones_like(y_true) * void_class
    mask = torch.ne(y_true, el).long()

    y_true_tmp = y_true * mask

    loss = F.cross_entropy(y_pred, y_true_tmp, weight=weight, reduction='none')
    loss = mask.float() * loss

    if reduce:
        return loss.sum()/mask.sum()
    else:
        return loss, mask

def seg_cross_entropy(model, input, target, weight = None):
    output = model(input)

    # use masked loss function
    loss = masked_loss(output, target, weight=weight)

    return {'loss': loss, 'output': output}

def seg_ale_cross_entropy(model, input, target, num_samples = 50, weight = None):
        #requires two outputs for model(input)

        output = model(input)
        mean = output[:, 0, :, :, :]
        scale = output[:, 1, :, :, :].abs()

        output_distribution = torch.distributions.Normal(mean, scale)

        total_loss = 0

        for _ in range(num_samples):
                sample = output_distribution.rsample()

                current_loss, mask = masked_loss(sample, target, weight=weight, reduce=False)
                total_loss = total_loss + current_loss.exp()
        mean_loss = total_loss / num_samples

        return {'loss': mean_loss.log().sum() / mask.sum(), 'output': mean, 'scale': scale}
