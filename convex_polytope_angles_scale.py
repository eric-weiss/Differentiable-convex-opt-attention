import numpy as np
import numpy.random as npr

import torch
from torch import nn
import torch.nn.functional as F

import os
import sys
import shutil

import matplotlib.pyplot as pp
import matplotlib as mpl
from matplotlib import cm


import cv2
from torchmin import minimize


def to_np(x):
    return x.detach().numpy()


def plotConstraints(angles, scale, xmin=-1, xmax=1, ymin=-1, ymax=1):
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 512),
                         np.linspace(ymin, ymax, 512))
    xxFlat = xx.ravel()
    yyFlat = yy.ravel()
    XYgrid = np.vstack((xxFlat, yyFlat)).T

    verts = np.vstack((np.cos(angles), np.sin(angles))).T
    verts -= np.mean(verts,axis=0)[np.newaxis,:]
    verts *= scale/np.sqrt(np.sum(verts**2,axis=1))[:,np.newaxis]
    vert_diffs = verts-np.roll(verts,1,0)
    midpoints = (verts + np.roll(verts,1,0))/2.0
    normals = np.dot(vert_diffs,np.asarray([[0.0,1.0],[-1.0,0.0]]))/np.sqrt(np.sum(vert_diffs**2,axis=1))[:,np.newaxis]
    offsets = np.sum(normals*midpoints,axis=1)
    dots = -np.dot(XYgrid, normals.T)+offsets[np.newaxis,:]
    mask = 1.0*np.all(dots<0.0, axis=1)
    # out = np.sum(dots, axis=1)
    out = logsumexp(10.0*dots/scale, axis=1)
    out = -np.array(out).reshape(xx.shape)
    return out


def compute_rock_mask(angles, scale, center, roundness, tile_size=512, xmin=-1, xmax=1, ymin=-1, ymax=1):
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, tile_size),
                         np.linspace(ymin, ymax, tile_size))
    xxFlat = xx.ravel()
    yyFlat = yy.ravel()
    XYgrid = torch.as_tensor(np.vstack((xxFlat, yyFlat)).T, dtype=torch.float32)
    XYgrid = XYgrid - center[None,:]

    scale = torch.sigmoid(scale)
    angles = angles[:-1] + angles[-1]

    verts = torch.vstack((torch.cos(angles), torch.sin(angles))).t()
    verts = verts - torch.mean(verts, dim=0)[None, :]
    verts = verts*scale/torch.sqrt(torch.sum(verts**2, dim=1))[:, None]
    vert_diffs = verts-torch.roll(verts, 1, 0)
    midpoints = (verts + torch.roll(verts, 1, 0))/2.0
    rotate_90 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)
    normals = torch.mm(vert_diffs, rotate_90)/torch.sqrt(torch.sum(vert_diffs**2, dim=1))[:, None]
    offsets = torch.sum(normals*midpoints, dim=1)
    dots = -torch.mm(XYgrid, normals.t())+offsets[None, :]
    # mask = 1.0*torch.all(dots<0.0, dim=1)
    # mask = -torch.max(dots, dim=1)[0]
    mask = torch.logsumexp(roundness*dots/scale, dim=1)
    out = torch.reshape(mask, (tile_size, tile_size))
    out = torch.sigmoid(4.0 * out)
    # pp.matshow(out.detach().numpy())
    # pp.colorbar()
    # pp.show()
    return out

def compute_rock_mask(angles, scale, center, roundness, tile_size=512, xmin=-1, xmax=1, ymin=-1, ymax=1):
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, tile_size),
                         np.linspace(ymin, ymax, tile_size))
    xxFlat = xx.ravel()
    yyFlat = yy.ravel()
    XYgrid = torch.as_tensor(np.vstack((xxFlat, yyFlat)).T, dtype=torch.float32)
    XYgrid = XYgrid - center[None,:]

    scale = torch.sigmoid(scale)
    angles = angles[:-1] + angles[-1]

    verts = torch.vstack((torch.cos(angles), torch.sin(angles))).t()
    verts = verts - torch.mean(verts, dim=0)[None, :]
    verts = verts*scale/torch.sqrt(torch.sum(verts**2, dim=1))[:, None]
    vert_diffs = verts-torch.roll(verts, 1, 0)
    midpoints = (verts + torch.roll(verts, 1, 0))/2.0
    rotate_90 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)
    normals = torch.mm(vert_diffs, rotate_90)/torch.sqrt(torch.sum(vert_diffs**2, dim=1))[:, None]
    offsets = torch.sum(normals*midpoints, dim=1)
    dots = -torch.mm(XYgrid, normals.t())+offsets[None, :]
    # mask = 1.0*torch.all(dots<0.0, dim=1)
    # mask = -torch.max(dots, dim=1)[0]
    mask = torch.logsumexp(roundness*dots/scale, dim=1)
    out = torch.reshape(mask, (tile_size, tile_size))
    out = torch.sigmoid(4.0 * out)
    # pp.matshow(out.detach().numpy())
    # pp.colorbar()
    # pp.show()
    return out


n_sides = 8

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def logsumexp(x,axis=-1):
    return np.log(np.sum(np.exp(x), axis=axis))



image = cv2.imread('/home/ubuntu/PycharmProjects/pythonProject2/rock.png')
img = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),dtype=np.float32)
img -= np.min(img)
img /= np.max(img)
tile_shape = img.shape
img_t = torch.as_tensor(img, dtype=torch.float32)


weights = np.ones(n_sides)*4.0
angles = np.cumsum(np.random.dirichlet(weights*10.0))*2.0*np.pi + np.random.uniform(0.0,2.0*np.pi)
angles = np.concatenate([angles, np.array([0.0])])
scale_0 = torch.tensor([1.0], dtype=torch.float32)
angles_0 = torch.as_tensor(angles, dtype=torch.float32)
means_0 = torch.tensor([0.0,1.0], dtype=torch.float32)
roundness_0 = torch.tensor([20.0], dtype=torch.float32)
center_0 = torch.tensor([0.0,0.0], dtype=torch.float32)

def neg_log_likelihood(img_data, means, mask):
    diffs = torch.sqrt(1e-6+(img_data[:, :, None] - means[None, None, :])**2)
    return torch.sum(mask*diffs[:,:,0]) + torch.sum((1.0-mask)*diffs[:,:,1])

def center_loss(center):
    mask = compute_rock_mask(angles_0, scale_0, center, roundness_0, tile_size=tile_shape[0])
    return neg_log_likelihood(img_t, means_0, mask)

def scale_loss(scale):
    mask = compute_rock_mask(angles_0, scale, center_0, roundness_0, tile_size=tile_shape[0])
    return neg_log_likelihood(img_t, means_0, mask)

def angle_loss(angles):
    mask = compute_rock_mask(angles, scale_0, center_0, roundness_0, tile_size=tile_shape[0])
    return neg_log_likelihood(img_t, means_0, mask) - 1e-3*torch.sum(torch.log((angles[:-1] - torch.roll(angles[:-1],1,0))**2))

def mean_loss(means):
    mask = compute_rock_mask(angles_0, scale_0, center_0, roundness_0, tile_size=tile_shape[0])
    return neg_log_likelihood(img_t, means, mask) + 1.0e-3/((means[0]-means[1])**2)

def roundness_loss(roundness):
    mask = compute_rock_mask(angles_0, scale_0, center_0, roundness, tile_size=tile_shape[0])
    return neg_log_likelihood(img_t, means_0, mask)

def reconstruct(means, angles, scale, roundness):
    mask = compute_rock_mask(angles, scale, roundness, tile_size=tile_shape[0])
    return mask*means[0] + (1.0-mask)*means[1]


def render_border(img_data, mask):
    xgrad = (np.roll(mask,-1,0)-np.roll(mask,1,0))
    ygrad = (np.roll(mask,-1,1)-np.roll(mask,1,1))
    mag = np.sqrt(xgrad ** 2 + ygrad ** 2)
    mag /= np.max(mag)
    color_img = np.ones((img_data.shape[0],img_data.shape[1],3))*img_data[:,:,np.newaxis]/2.0
    color_img[:,:,0] += mag/2.0
    return color_img

def plot_params():
    mask = compute_rock_mask(angles_0, scale_0, center_0, roundness_0, tile_size=tile_shape[0]).detach().numpy()
    pp.imshow(render_border(img, mask))
    pp.show()


for i in range(1000):

    if i%50==0:
        plot_params()

    res = minimize(mean_loss, means_0.detach(), method='newton-cg')
    means_0 = res['x']
    # print(means_0)
    # plot_params()

    res = minimize(scale_loss, scale_0.detach(), method='newton-cg')
    scale_0 = res['x']
    # print(scale_0)
    # plot_params()

    res = minimize(center_loss, center_0.detach(), method='newton-cg')
    center_0 = res['x']
    # print(center_0)
    # plot_params()

    res = minimize(angle_loss, angles_0.detach(), method='newton-cg')
    angles_0 = res['x']
    # print(angles_0)


    res = minimize(roundness_loss, roundness_0.detach(), method='newton-cg')
    roundness_0 = res['x']
    # print(roundness_0)
    #plot_params()




for i in range(10):
    weights = np.ones(n_sides)*4.0
    angles = np.cumsum(np.random.dirichlet(weights*10.0))*2.0*np.pi + np.random.uniform(0.0,2.0*np.pi)
    scale = torch.tensor([0.5], dtype=torch.float32)
    angles = torch.as_tensor(angles,dtype=torch.float32)

    zz0 = plotConstraints(angles, scale)
    zz0 = compute_rock_mask(torch.as_tensor(angles,dtype=torch.float32), torch.tensor(scale,dtype=torch.float32)).numpy()
    roundness = np.random.gamma(10,1)
    zz0 = sigmoid(zz0*roundness)
    xgrad = (np.roll(zz0,-1,0)-np.roll(zz0,1,0))
    ygrad = (np.roll(zz0,-1,1)-np.roll(zz0,1,1))
    sun_angle = np.random.uniform(0.0,2.0*np.pi)
    border_shading = xgrad*np.cos(sun_angle) + ygrad*np.sin(sun_angle)
    # xgrad = xgrad*(zz0<0.25)
    # pp.matshow(1.0/(1.0+np.exp(-xgrad*1000.0)))
    pp.matshow(border_shading)
    pp.matshow(zz0)
    pp.colorbar()
    z=xgrad+1.j*ygrad
    # plt = cplot.plot(xgrad+1.j*ygrad,(-1,1,512),(-1,1,512))
    # pp.imshow(cplot.get_srgb1(z))
    pp.show()




import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


_G = cp.Parameter((ncon, nx))
_h = cp.Parameter(ncon)
_x = cp.Parameter(nx)
_y = cp.Variable(nx)
obj = cp.Minimize(0.5 * cp.sum_squares(_x - _y))
cons = [_G @ _y <= _h]
prob = cp.Problem(obj, cons)

layer = CvxpyLayer(prob, parameters=[_G, _h, _x], variables=[_y])

torch.manual_seed(6)
G = torch.FloatTensor(ncon, nx).uniform_(-4, 4)
z0 = torch.full([nx], 0.5)
s0 = torch.full([ncon], 0.5)
h = G.mv(z0) + s0






def layerbatch(Gin, hin, xin):
    return layer(Gin.repeat(nb, 1, 1), hin.repeat(nb, 1), xin)


torch.manual_seed(0)
x = torch.randn(nx)
y, = layer(G, h, x)
print(f'Input: {to_np(x)}\nOutput: {to_np(y)}')

torch.manual_seed(22)
G_hat = nn.Parameter(torch.FloatTensor(ncon, nx).uniform_(-4, 4).requires_grad_())
h_hat = G_hat.mv(z0) + s0

x = nn.Parameter(torch.FloatTensor(nb, nx).uniform_(0, 1))

opt = torch.optim.NAdam([G_hat], lr=1e-1)
xopt = torch.optim.NAdam([x], lr=1e-2)
losses = []

from matplotlib import collections as mc

d = 'polytope_images'
if os.path.exists(d):
    shutil.rmtree(d)
os.makedirs(d)

for i in range(2500):

    y, = layerbatch(G, h, x)
    h_hat = G_hat.mv(z0) + s0
    yhat, = layer(G_hat, h_hat, x)
    ydiffs = (yhat - y)
    xnorms = torch.sqrt(torch.sum((x - 0.5) ** 2, dim=1))
    loss = torch.sum(xnorms[:, None] * ydiffs ** 2)
    ydf = ydiffs.detach()

    losses.append(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        weights = torch.sum(ydf ** 2, dim=1)
        x = torch.gather(x, 0, torch.multinomial(weights, nb, replacement=True)[:, None]) + torch.normal(0.0, 0.125,
                                                                                                         size=(nb, 2))
        x *= 0.5

    if i % 10 == 0:
        yp = y.detach().numpy()
        yph = yhat.detach().numpy()
        xp = x.detach().numpy()
        plt.scatter(xp[:, 0], xp[:, 1])
        plt.scatter(yp[:, 0], yp[:, 1])
        plt.scatter(yph[:, 0], yph[:, 1])
        plt.savefig(f"{i}.png")
        plt.clf()
        print(i)

    if i % 20 == 0:
        fig, ax = plotConstraints(to_np(G), to_np(h), to_np(G_hat), to_np(h_hat))
        fig.tight_layout()
        plt.show()
        plt.close(fig)
