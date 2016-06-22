--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Utility functions for models
------------------------------------------------------------------------------]]

local utils = {}

--------------------------------------------------------------------------------
-- SpatialConstDiagonal module
-- all BN modules in ResNet to be transformed into SpatialConstDiagonal
if not nn.SpatialConstDiagonal then
  local module, parent = torch.class('nn.SpatialConstDiagonal', 'nn.Module')

  function module:__init(nOutputPlane, inplace)
    parent.__init(self)
    self.a = torch.Tensor(1,nOutputPlane,1,1)
    self.b = torch.Tensor(1,nOutputPlane,1,1)
    self.inplace = inplace
    self:reset()
  end

  function module:reset()
    self.a:fill(1)
    self.b:zero()
  end

  function module:updateOutput(input)
    if self.inplace then
      self.output:set(input)
    else
      self.output:resizeAs(input):copy(input)
    end
    self.output:cmul(self.a:expandAs(input))
    self.output:add(self.b:expandAs(input))
    return self.output
  end

  function module:updateGradInput(input, gradOutput)
    if self.inplace then
      self.gradInput:set(gradOutput)
    else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    end
    self.gradInput:cmul(self.a:expandAs(gradOutput))
    return self.gradInput
  end
end

--------------------------------------------------------------------------------
-- function: goes over a net and recursively replaces modules
-- using callback function
local function replace(self, callback)
  local out = callback(self)
  if self.modules then
    for i=#self.modules,1,-1 do
      local m = self.modules[i]
      local mm = replace(m, callback)
      if mm then self.modules[i] = mm else self:remove(i) end
    end
  end
  return out
end

--------------------------------------------------------------------------------
-- function: replace BN layer to SpatialConstDiagonal
function utils.BNtoFixed(net, ip)
  return replace(
    net,
    function(x)
    if torch.typename(x):find'SpatialBatchNormalization' then
      local no = x.running_mean:numel()
      local y = nn.SpatialConstDiagonal(no, ip):type(x._type)
      if x.running_var then
        x.running_std = x.running_var:pow(-0.5)
      end
      y.a:copy(x.running_std)
      y.b:add(-1,x.running_mean):cmul(x.running_std)
      if x.affine then
        y.a:cmul(x.weight)
        y.b:cmul(x.weight):add(x.bias)
      end
      return y
    else
      return x
    end
  end
  )
end

--------------------------------------------------------------------------------
-- function: linear2convTrunk
function utils.linear2convTrunk(net,fSz)
  return replace(
  net,
  function(x)
    if torch.typename(x):find('Linear') then
      local nInp,nOut = x.weight:size(2)/(fSz*fSz),x.weight:size(1)
      local w = torch.reshape(x.weight,nOut,nInp,fSz,fSz)
      local y = cudnn.SpatialConvolution(nInp,nOut,fSz,fSz,1,1)
      y.weight:copy(w)
      y.gradWeight:copy(w)
      y.bias:copy(x.bias)
      return y
    elseif torch.typename(x):find('Threshold') then
      return cudnn.ReLU()
    elseif not torch.typename(x):find('View') and
      not torch.typename(x):find('SpatialZeroPadding') then
      return x
    end
  end
  )
end

--------------------------------------------------------------------------------
-- function: linear2convHeads
function utils.linear2convHead(net)
  return replace(
  net,
  function(x)
    if torch.typename(x):find('Linear') then
      local nInp,nOut = x.weight:size(2),x.weight:size(1)
      local w = torch.reshape(x.weight,nOut,nInp,1,1)
      local y = cudnn.SpatialConvolution(nInp,nOut,1,1,1,1)
      y.weight:copy(w)
      y.gradWeight:copy(w)
      y.bias:copy(x.bias)
      return y
    elseif torch.typename(x):find('Threshold') then
      return cudnn.ReLU()
    elseif not torch.typename(x):find('View') and
      not torch.typename(x):find('Copy') then
      return x
    end
  end
  )
end

--------------------------------------------------------------------------------
-- function: replace 0-padding of 3x3 conv into mirror-padding
function utils.updatePadding(net, nn_padding)
  if torch.typename(net) == "nn.Sequential" or
    torch.typename(net) == "nn.ConcatTable" then
    for i = #net.modules,1,-1 do
      local out = utils.updatePadding(net:get(i), nn_padding)
      if out ~= -1 then
        local pw, ph = out[1], out[2]
        net.modules[i] = nn.Sequential():add(nn_padding(pw,pw,ph,ph))
          :add(net.modules[i]):cuda()
      end
    end
  else
    if torch.typename(net) == "nn.SpatialConvolution" or
      torch.typename(net) == "cudnn.SpatialConvolution" then
      if (net.kW == 3 and net.kH == 3) or (net.kW==7 and net.kH==7) then
        local pw, ph = net.padW, net.padH
        net.padW, net.padH = 0, 0
        return {pw,ph}
      end
    end
  end
  return -1
end

return utils
