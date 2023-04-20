
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import csv
import copy
import decimal
import torch
import math
from torch.optim.optimizer import Optimizer
import numpy as np
import torch.nn as nn



class Ours(Optimizer):
  def __init__(self, params, steps, h,alpha, beta,loss_func,plr):
    self.params = params
    self.radius = 0
    self.alpha = alpha
    self.beta = beta

    self.avg_p1_u = 0
    self.avg_p2_u = 0
    self.p1 = None
    self.p2 = None
    self.p1_grad = None
    self.p2_grad = None
    self.steps = steps
    self.h = h
    self.loss_func = loss_func
    self.plr = plr
    self.iter = 0
    self.avg_theta = 0
    self.h_backup = h
    self.p_moment = 0
    self.beta = beta
    self.avg_depth = 0
    self.iter = 0
    self.flag = 1
    self.avg_sum=0
    self.avg_sum = 0
    self.avg_angle = 0
    self.avg_radius = torch.tensor([0], device='cuda')
    self.plane_radius = torch.tensor([0], device='cuda')
    self.set_locality_flag = 1
    defaults = dict(lr=h, beta=beta, alpha=alpha)
    super(Ours, self).__init__(params, defaults)

  def __setstate__(self, state):
      super(Ours, self).__setstate__(state)

  def make_rand_vector(self, dims):
      vec = torch.rand(dims, requires_grad=True)

      return vec

  def angle(self, a, b):
      a = a.detach()
      b = b.detach()
      a = a/torch.norm(a)
      b = b/torch.norm(b)

      term = torch.tensor([float(decimal.Decimal(torch.norm(a-b).item())/decimal.Decimal(torch.norm(a+b).item()))], device='cuda')
      ##a = torch.cat([a,torch.tensor([0])])
      ##b = torch.cat([b,torch.tensor([0])])
      ##cross = torch.cross(a,b)
      ##dot = torch.dot(a,b)
      ##angle = torch.atan(torch.norm(cross)/dot)
      angle = 2 * torch.atan(term)
      return angle #* 180/torch.pi

  def projection(self,p):
    v = p - self.plane_point
    projected_normal = torch.dot(v, self.plane_normal)
    projected_point = p - projected_normal
    return projected_point


  def step(self, loss):
    self.iter += 1
    flag = 0
    self.h = self.param_groups[0]['lr']
    self.p1 = copy.deepcopy(self.params)
    grad_x = torch.autograd.grad(loss, self.params,  )
    grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])#.reshape(-1,1)
    grad_x_vec_d = grad_x_vec.detach()
    norm_grad = torch.norm(grad_x_vec_d, 2)
    #del grad_x
    #del grad_x_vec
    if(norm_grad <= 1e-6):
      #print("Converged")
      return
    self.p1_grad = grad_x_vec_d
    self.p1_test = 0

    ## GOING PERPENDICULAR

    rand_vec = (self.make_rand_vector(self.p1_grad.shape).to('cuda')).detach()
    perp_vec = (rand_vec - ((rand_vec.T @ self.p1_grad)/(torch.pow(torch.norm(self.p1_grad,2),2))) * self.p1_grad).detach()
    self.rand_vec = rand_vec

    norm_len = torch.norm(perp_vec, 2)

    #if(norm_len >= 1):
    perp_vec = (perp_vec / (torch.norm(perp_vec, 2)))
    ##while(flag == 0 and self.h >= 1e-50):
    ##    #self.params = copy.deepcopy(self.p1)
    for (opt_param, p1_param) in zip(self.params, self.p1):
        opt_param.data = p1_param.data



    step_length = 0



    self.perp_vec = perp_vec
    self.grad_normal = grad_x_vec_d


    #self.p_moment =  self.beta * self.p_moment + (1-self.beta) * torch.pow(torch.norm(perp_vec.detach(),2),2)

    step_length += torch.norm(perp_vec.detach(), 2) * self.h #/ (torch.sqrt(self.p_moment) + 1e-7)
    index = 0
    for p in self.params:
        term = perp_vec[index: index + p.numel()].detach().reshape(p.shape)
        #print("UPDATING, adding term", torch.norm(term) * self.h)
        p.data = p.data + ((-1) * (term * self.h))# / (torch.sqrt(self.p_moment) + 1e-7) )
        index += p.numel()
    ##param_vec = torch.cat([g.contiguous().view(-1) for g in self.params])
    ##print(torch.norm(param_vec))



    loss_val = self.loss_func()
    #print(loss_val)
    grad_x = torch.autograd.grad(loss_val, self.params,)
    grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])#.reshape(-1,1)
    grad_x_vec_d = grad_x_vec.detach()
    #del grad_x
    #del grad_x_vec
    self.p2_grad = grad_x_vec_d.clone()
    #print(torch.norm(self.p1_grad - self.p2_grad))
    ang_g1g2 = math.ceil(self.angle(self.p1_grad, self.p2_grad)* 180/torch.pi)
    ang_p1g2 = math.ceil(self.angle(self.perp_vec, self.p2_grad) * 180/torch.pi)

    if(ang_p1g2 < 90):
        self.p_moment = 0
        #self.avg_locality =0
        #self.iter = 0
        self.flag = 1

    #if(ang_g1g2 <= 10 and ang_p1g2 > 90 ): #
    #    #print("OKAY")
    #    self.h = self.h_backup
    #    flag = 1
    #else:
        #print("G1_G2", ang_g1g2, "P1 G2", ang_p1g2, " H ", self.h, "DISTANCE ", torch.norm(self.p1_grad - self.p2_grad, 2).item())
    #    self.h = self.h / 10


    #    #print(torch.norm(p3_vec))
    #    #print(" TRAPPED ", torch.norm(param_vec))
    #    flag = 0

    # DONE PERPENDICULAR

    #print(self.p1)
    self.p2 = copy.deepcopy(self.params)
    p1_vec = torch.cat([g.contiguous().view(-1) for g in self.p1])

    self.theta = (self.angle(self.p1_grad, self.p2_grad) + 1e-2).detach()
    #print("THETA " , self.theta.item() * 180/torch.pi)
    import time
    #time.sleep(0.5)

    #step_length = self.h * self.steps
    self.radius = step_length * (torch.cos(self.theta)/torch.sin(self.theta)).to('cuda')

    index = 0
    #self.params = copy.deepcopy(self.p1)
    for param,p1_param in zip(self.params, self.p1):
        param.data = p1_param.data

    if(self.radius < self.avg_radius):
        #print("\nJUMPING ",self.flag, "\n")
        self.radius = 2.0 * self.radius
    self.normalized_grad = ((self.p1_grad/(torch.norm(self.p1_grad,2))) * self.radius.detach()).detach()

    for p in self.params:

        term = self.normalized_grad[index: index + p.numel()].detach().reshape(p.shape)
        #print("TERM ", term)
        p.data.add_((-1)*term)
        index += p.numel()
    p2_vec = torch.cat([g.contiguous().view(-1) for g in self.p2])
    #print(" DISTANCE     ", torch.norm(p1_vec-p2_vec))
    self.avg_theta += self.theta.detach() * 180/torch.pi
    self.avg_theta = self.avg_theta/self.iter
    self.avg_radius = self.beta * self.avg_radius.detach() + (1-self.beta) * self.radius.detach()
    
    self.avg_p1_u = self.angle(self.p1_grad, self.normalized_grad)
    self.avg_p1_u = self.avg_p1_u/self.iter
    
    self.avg_p2_u += self.angle(self.normalized_grad, self.p2_grad)
    self.avg_p2_u = self.avg_p2_u/self.iter
    
    self.avg_angle += self.theta.detach()
    self.avg_angle = self.avg_angle/self.iter
    
    del grad_x_vec
    del self.p1
    del self.p2
    del self.rand_vec
    del self.p1_grad
    del self.p2_grad
    #self.h = self.decayed_learning_rate(self.iter).item()
    #print("MEMORY ALLOATED ",torch.cuda.memory_allocated(0))
    ##if( self.h <= self.alpha ):
    ##    self.h = h
    ##    self.iter = 0
