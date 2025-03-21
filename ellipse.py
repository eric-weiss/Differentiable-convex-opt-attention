import numpy as np
import numpy.random as npr

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import os
import sys
import shutil

import matplotlib.pyplot as pp
import matplotlib as mpl
from matplotlib import cm

import cplot

import cv2
from torchmin import minimize


def to_np(x):
    return x.detach().numpy()

def construct_log_polar_matrix(input_shape, n_angles, n_radii, min_radius, max_radius):
    xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, input_shape[0]),
                         np.linspace(-1.0, 1.0, input_shape[1]))
    xxFlat = xx.ravel()
    yyFlat = yy.ravel()
    input_XYgrid = np.vstack((xxFlat, yyFlat)).T

    rr, aa = np.meshgrid(np.exp(np.linspace(np.log(min_radius), np.log(max_radius), n_radii)),
                         np.linspace(0, 2.0*np.pi, n_angles, endpoint=False))

    rrFlat = rr.ravel()
    aaFlat = aa.ravel()
    xx_transformed = (rrFlat*np.cos(aaFlat))
    yy_transformed = (rrFlat*np.sin(aaFlat))
    output_XYgrid = np.vstack((xx_transformed, yy_transformed)).T

    diffs = np.sum((input_XYgrid[:,np.newaxis,:] - output_XYgrid[np.newaxis,:,:])**2,axis=2)
    weights = np.exp(-diffs/rrFlat[np.newaxis,:])
    # weights = weights*(weights==np.max(weights,axis=1,keepdims=True))
    weights = weights*np.exp(200.*weights)/np.sum(np.exp(200.*weights),axis=1,keepdims=True)
    weights = weights/np.sum(weights+1e-9,axis=0,keepdims=True)
    return weights.T







def compute_rock_mask(verts, sharpness, roundness=80.0, tile_size=512, xmin=-1, xmax=1, ymin=-1, ymax=1):
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, tile_size),
                         np.linspace(ymin, ymax, tile_size))
    xxFlat = xx.ravel()
    yyFlat = yy.ravel()
    XYgrid = torch.as_tensor(np.vstack((xxFlat, yyFlat)).T, dtype=torch.float32)

    vert_diffs = verts-torch.roll(verts, 1, 0)
    vert_sums = (verts + torch.roll(verts, 1, 0))
    rotate_90 = -torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)
    normals = torch.mm(vert_diffs, rotate_90)/torch.sqrt(torch.sum(vert_diffs**2, dim=1))[:, None]
    offsets = -torch.sum(normals*vert_sums, dim=1)/2.0
    dots = (torch.mm(XYgrid, normals.t())+offsets[None, :])/((xmax-xmin)/float(tile_size))
    # mask = 1.0*torch.all(dots<0.0, dim=1)
    # mask = torch.max(1.0*dots, dim=1)[0]
    mask = torch.sum(dots*torch.softmax(dots,dim=1),dim=1)
    # mask = torch.logsumexp(dots, dim=1)
    out = torch.reshape(mask, (tile_size, tile_size))
    out = torch.sigmoid(-out*sharpness)
    # pp.matshow(out.detach().numpy())
    # v = verts.detach().numpy() * tile_shape[0] / 2.0 + tile_shape[0] / 2.0
    # pp.scatter(v[:, 0], v[:, 1])
    # # pp.colorbar()
    # pp.show()
    return out


img = cv2.imread('/home/ubuntu/PycharmProjects/pythonProject2/rock2.png')
rock_img = cv2.imread('/home/ubuntu/PycharmProjects/pythonProject2/onlyrock.png')
dirt_img = cv2.imread('/home/ubuntu/PycharmProjects/pythonProject2/onlydirt.png')
shadow_img = cv2.imread('/home/ubuntu/PycharmProjects/pythonProject2/onlyshadow.png')
# img = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.float32)
rock_img=np.asarray(rock_img,dtype=np.float32)
dirt_img=np.asarray(dirt_img,dtype=np.float32)
shadow_img=np.asarray(shadow_img,dtype=np.float32)
img=np.asarray(img,dtype=np.float32)



n_sides = 4

weights = np.ones(n_sides)*1.0
angles = np.cumsum(np.random.dirichlet(weights*1.0))*2.0*np.pi + np.random.uniform(0.0,2.0*np.pi)
verts_0 = nn.Parameter(torch.as_tensor(0.5*np.stack((np.cos(angles), np.sin(angles))).T, dtype=torch.float32))
roundness_0 = nn.Parameter(torch.tensor([20.0], dtype=torch.float32))

img = compute_rock_mask(verts_0, roundness_0, tile_size=128).detach().numpy()

n_angles = 64
n_radii = 16
# img = np.zeros((128,128)).astype(np.float32)
# img[60:62,:] = 1.0
# logpolar_img = np.zeros((n_angles,n_radii)).astype(np.float32)
# logpolar_img[2,:] = 1.0
# logpolar_img[:,4] = 1.0
logpolar_mat = construct_log_polar_matrix(img.shape,n_angles,n_radii,.1,.95)
inv_logpolar_mat = np.linalg.pinv(logpolar_mat)
logpolar_img = np.dot(logpolar_mat, img.ravel()).reshape((n_angles,n_radii))
reconstructed_img = np.dot(inv_logpolar_mat, logpolar_img.ravel()).reshape(img.shape)
# reconstructed_img = np.dot(logpolar_mat.T, logpolar_img.ravel()).reshape(img.shape)

ax = pp.subplot(131)
ax.imshow(img)
ax = pp.subplot(132)
ax.imshow(logpolar_img)
ax = pp.subplot(133)
ax.imshow(reconstructed_img)
pp.show()


img = np.log(img)
img = img-np.roll(0.25*(np.roll(img,1,1) + np.roll(img,-1,1) + np.roll(img,1,0) + np.roll(img,-1,0)),1,2)

# img = img-np.roll(img,1,2)
# img = np.exp(img)
pp.hist(img.flatten(),32)
pp.show()
# img -= np.min(img,axis=2,keepdims=True)
# img /= np.max(img,axis=2,keepdims=True)
img -= np.min(img)
img /= np.max(img)
pp.imshow(img)
pp.show()

max_vals = np.max(rock_img, axis=(0,1))
rock_img = rock_img / max_vals[np.newaxis,np.newaxis,:]
dirt_img = dirt_img / max_vals[np.newaxis,np.newaxis,:]

dirt_reflection = np.array([np.mean(channel[channel!=0]) for channel in np.swapaxes(dirt_img,0,2)])
print(dirt_reflection)

dirt_img = (dirt_img + rock_img)/(dirt_reflection[np.newaxis,np.newaxis,:])
norms = np.max(dirt_img,axis=2)
dirt_img /= norms[:,:,np.newaxis]
pp.imshow(dirt_img)
pp.show()
# img = img/norms[:,:,np.newaxis]
# pp.matshow(norms)
# pp.colorbar()
# pp.show()
# img = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),dtype=np.float32)
dirt_img -= np.min(dirt_img)
dirt_img /= np.max(dirt_img)
dirt_img = np.asarray(cv2.cvtColor(dirt_img, cv2.COLOR_BGR2RGB), dtype=np.uint8)
pp.imshow(dirt_img)
pp.show()
tile_shape = img.shape
img_t = torch.as_tensor(img, dtype=torch.float32)


weights = np.ones(n_sides)*4.0
angles = np.cumsum(np.random.dirichlet(weights*10.0))*2.0*np.pi + np.random.uniform(0.0,2.0*np.pi)

verts_0 = nn.Parameter(torch.as_tensor(0.25*np.stack((np.cos(angles), np.sin(angles))).T, dtype=torch.float32))
means_0 = nn.Parameter(torch.tensor([0.5,0.5], dtype=torch.float32))
roundness_0 = nn.Parameter(torch.tensor([20.0], dtype=torch.float32))
scales_0 = nn.Parameter(torch.tensor([-1.0, -1.0], dtype=torch.float32))

def plot(x):
    pp.matshow(x.detach().numpy())
    pp.colorbar()
    pp.show()

def polygon_area(verts):
    return torch.sum(verts[:,0]*torch.roll(verts[:,1], -1, 0) - verts[:,1]*torch.roll(verts[:,1], -1, 0))/2.0

def polygon_perimeter(verts):
    dists = torch.sqrt(torch.sum((verts - torch.roll(verts, 1, 0))**2, dim=1))
    return torch.sum(dists)

# def convexity_loss(`)

def expected_log_likelihood(img_data, means, scales, mask):
    diffs = (img_data[:, :, None] - means[None, None, :])**2
    log_probs = -(diffs/torch.exp(2.0*scales[None, None, :]) + scales[None, None, :])
    probs = torch.exp(log_probs)
    rock_post = probs[:, :, 0] * mask
    dirt_post = probs[:, :, 1] * (1.0-mask)
    Z = rock_post + dirt_post
    rock_probs = rock_post/Z
    # m = torch.distributions.Bernoulli(rock_post/Z)
    # rock_probs = m.sample()
    # plot(mask)
    # plot(rock_post / Z)
    return torch.sum(rock_probs*(log_probs[:,:,0]) + (1.0-rock_probs)*(log_probs[:,:,1]))

def rock_posterior(img_data, means, scales, mask):
    diffs = (img_data[:, :, None] - means[None, None, :])**2
    log_probs = -(torch.sqrt(1e-9+diffs)/torch.exp(1.0*scales[None, None, :]) + scales[None, None, :])
    probs = torch.exp(log_probs)
    rock_post = probs[:, :, 0] * mask
    dirt_post = probs[:, :, 1] * (1.0-mask)
    Z = rock_post + dirt_post
    # m = torch.distributions.Bernoulli(rock_post / Z)
    # rock_probs = m.sample()
    return rock_post / Z

def vert_loss(verts):
    mask = compute_rock_mask(verts, roundness_0, tile_size=tile_shape[0])
    return -expected_log_likelihood(img_t, means_0, scales_0, mask) + polygon_area(verts)/polygon_perimeter(verts)**2

def mean_loss(means):
    mask = compute_rock_mask(verts_0, roundness_0, tile_size=tile_shape[0])
    return -expected_log_likelihood(img_t, means, scales_0, mask)

def scale_loss(scales):
    mask = compute_rock_mask(verts_0, roundness_0, tile_size=tile_shape[0])
    return -expected_log_likelihood(img_t, means_0, scales, mask)

def roundness_loss(roundness):
    mask = compute_rock_mask(verts_0, roundness, tile_size=tile_shape[0])
    return -expected_log_likelihood(img_t, means_0, scales_0, mask)


def render_border(img_data, mask):
    xgrad = (np.roll(mask,-1,0)-np.roll(mask,1,0))
    ygrad = (np.roll(mask,-1,1)-np.roll(mask,1,1))
    mag = np.sqrt(xgrad ** 2 + ygrad ** 2)
    mag /= np.max(mag)
    color_img = np.ones((img_data.shape[0],img_data.shape[1],3))*img_data[:,:,np.newaxis]/2.0
    color_img[:,:,0] += mag/2.0
    return color_img

def plot_params():
    mask = compute_rock_mask(verts_0, roundness_0, tile_size=tile_shape[0])
    post = rock_posterior(img_t, means_0, scales_0, mask).detach().numpy()
    pp.matshow(post)
    v = verts_0.detach().numpy()*tile_shape[0]/2.0 + tile_shape[0]/2.0
    pp.scatter(v[:,0],v[:,1])
    pp.show()
    pp.imshow(render_border(img, mask.detach().numpy()))
    pp.show()


opt = optim.SGD([means_0, scales_0], lr=0.000001)
vert_opt = optim.SGD([verts_0], lr=0.000001)

for i in range(10000):

    if i%20==0:
        print(verts_0)
        print(means_0)
        print(scales_0)
        print(roundness_0)
        print(polygon_area(verts_0))
        plot_params()

    mask = compute_rock_mask(verts_0, roundness_0, tile_size=tile_shape[0])
    loss = -expected_log_likelihood(img_t, means_0, scales_0, mask) - polygon_area(verts_0)/(polygon_perimeter(verts_0))

    opt.zero_grad()
    vert_opt.zero_grad()
    loss.backward()
    opt.step()
    vert_opt.step()

    # res = minimize(mean_loss, means_0.detach(), method='newton-cg')
    # means_0 = res['x']
    # print(means_0)
    # plot_params()

    # res = minimize(vert_loss, verts_0.detach(), method='newton-cg')
    # verts_0 = res['x']
    # print(scale_0)
    # plot_params()

    # res = minimize(scale_loss, scales_0.detach(), method='newton-cg')
    # scales_0 = res['x']


    # roundness_0 = roundness_0*1.01

    # res = minimize(roundness_loss, roundness_0.detach(), method='newton-cg')
    # roundness_0 = res['x']
    # print(roundness_0)
    # plot_params()




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

# for i in range(2500):
#
#     y, = layerbatch(G, h, x)
#     h_hat = G_hat.mv(z0) + s0
#     yhat, = layer(G_hat, h_hat, x)
#     ydiffs = (yhat - y)
#     xnorms = torch.sqrt(torch.sum((x - 0.5) ** 2, dim=1))
#     loss = torch.sum(xnorms[:, None] * ydiffs ** 2)
#     ydf = ydiffs.detach()
#
#     losses.append(loss)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#
# with torch.no_grad():
#         weights = torch.sum(ydf ** 2, dim=1)
#         x = torch.gather(x, 0, torch.multinomial(weights, nb, replacement=True)[:, None]) + torch.normal(0.0, 0.125,size=(nb, 2))
#         x *= 0.5
#
#     if i % 10 == 0:
#         yp = y.detach().numpy()
#         yph = yhat.detach().numpy()
#         xp = x.detach().numpy()
#         plt.scatter(xp[:, 0], xp[:, 1])
#         plt.scatter(yp[:, 0], yp[:, 1])
#         plt.scatter(yph[:, 0], yph[:, 1])
#         plt.savefig(f"{i}.png")
#         plt.clf()
#         print(i)
#
#     if i % 20 == 0:
#         fig, ax = plotConstraints(to_np(G), to_np(h), to_np(G_hat), to_np(h_hat))
#         fig.tight_layout()
#         plt.show()
#         plt.close(fig)
