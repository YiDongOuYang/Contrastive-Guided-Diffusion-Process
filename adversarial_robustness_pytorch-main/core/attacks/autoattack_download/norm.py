"""Functions for computing vector Lp norms and dual norms."""

import numpy as np
import scipy.linalg
import torch
import torch.fft


def norm_f(x, norm_type):
  """Differentiable implementation of norm handling any Lp norm."""
  if norm_type == 'l2':
    return np.float_power(np.maximum(1e-7, np.sum(x*x)), 0.5)
  if norm_type == 'l1':
    return np.sum(np.abs(x))
  if norm_type == 'linf':
    return np.max(np.abs(x))
  if norm_type == 'dft1':
    dft = scipy.linalg.dft(x.shape[0]) / np.sqrt(x.shape[0])
    return np.sum(np.abs(dft @ x))
  if norm_type == 'dftinf1d':
    dft = scipy.linalg.dft(x.shape[0]) / np.sqrt(x.shape[0])
    return np.max(np.abs(dft @ x))
  if norm_type == 'dftinf':
    shape = x.shape
    dftx = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
    dftx = dftx.view(shape[0], -1)
    return dftx.abs().max(dim=-1)[0]
  return 0


def norm_projection(delta, norm_type, eps=1.):
  """Projects to a norm-ball centered at 0.

  Args:
    delta: An array of size dim x num containing vectors to be projected.
    norm_type: A string denoting the type of the norm-ball.
    eps: A float denoting the radius of the norm-ball.

  Returns:
    An array of size dim x num, the projection of delta to the norm-ball.
  """
  shape = delta.shape
  if norm_type == 'l2':
    # Euclidean projection: divide all elements by a constant factor
    avoid_zero_div = 1e-12
    norm2 = np.sum(delta**2, axis=0, keepdims=True)
    norm = np.sqrt(np.maximum(avoid_zero_div, norm2))
    # only decrease the norm, never increase
    delta = delta * np.clip(eps / norm, a_min=None, a_max=1)
  elif norm_type == 'dftinf':
    # transform to DFT, project using known projections, then transform back
    # n x d x h x w
    dftxdelta = torch.fft.fftn(delta, dim=(-2, -1), norm='ortho')
    # L2 projection of each coordinate to the L2-ball in the complex plane
    dftz = dftxdelta.reshape(1, -1)
    dftz = torch.cat((torch.real(dftz), torch.imag(dftz)), dim=0)
    def l2_proj(delta, eps):
      avoid_zero_div = 1e-15
      norm2 = torch.sum(delta**2, dim=0, keepdim=True)
      norm = torch.sqrt(torch.clamp(norm2, min=avoid_zero_div))
      # only decrease the norm, never increase
      delta = delta * torch.clamp(eps / norm, max=1)
      return delta
    dftz = l2_proj(dftz, eps)
    dftz = (dftz[0, :] + 1j * dftz[1, :]).reshape(delta.shape)
    # project back from DFT
    delta = torch.fft.ifftn(dftz, dim=(-2, -1), norm='ortho')
    # Projected vector can have an imaginary part
    delta = torch.real(delta)
  return delta.reshape(shape)


def steepest_ascent_direction(grad, norm_type, eps_tot):
  shape = grad.shape
  if norm_type == 'dftinf':
    dftxgrad = torch.fft.fftn(grad, dim=(-2, -1), norm='ortho')
    dftz = dftxgrad.reshape(1, -1)
    dftz = torch.cat((torch.real(dftz), torch.imag(dftz)), dim=0)
    def l2_normalize(delta, eps):
      avoid_zero_div = 1e-15
      norm2 = torch.sum(delta**2, dim=0, keepdim=True)
      norm = torch.sqrt(torch.clamp(norm2, min=avoid_zero_div))
      delta = delta * eps / norm
      return delta
    dftz = l2_normalize(dftz, eps_tot)
    dftz = (dftz[0, :] + 1j * dftz[1, :]).reshape(shape)
    delta = torch.fft.ifftn(dftz, dim=(-2, -1), norm='ortho')
    adv_step = torch.real(delta)
  return adv_step
