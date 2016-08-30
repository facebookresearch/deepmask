--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Utility functions for models
------------------------------------------------------------------------------]]

local utils = {}

--------------------------------------------------------------------------------
-- all BN modules in ResNet to be transformed into SpatialConstDiagonal
-- (with inn routines)
local inn = require 'inn'
local innutils = require 'inn.utils'
if not nn.SpatialConstDiagonal then
  torch.class('nn.SpatialConstDiagonal', 'inn.ConstAffine')
end
utils.BNtoFixed = innutils.BNtoFixed

--------------------------------------------------------------------------------
-- function: linear2convTrunk
function utils.linear2convTrunk(net,fSz)
  return net:replace(function(x)
    if torch.typename(x):find('Linear') then
      local nInp,nOut = x.weight:size(2)/(fSz*fSz),x.weight:size(1)
      local w = torch.reshape(x.weight,nOut,nInp,fSz,fSz)
      local y = cudnn.SpatialConvolution(nInp,nOut,fSz,fSz,1,1)
      y.weight:copy(w); y.gradWeight:copy(w); y.bias:copy(x.bias)
      return y
    elseif torch.typename(x):find('Threshold') then
      return cudnn.ReLU()
    elseif torch.typename(x):find('View') or
       torch.typename(x):find('SpatialZeroPadding') then
      return nn.Identity()
    else
      return x
    end
  end
  )
end

--------------------------------------------------------------------------------
-- function: linear2convHeads
function utils.linear2convHead(net)
  return net:replace(function(x)
    if torch.typename(x):find('Linear') then
      local nInp,nOut = x.weight:size(2),x.weight:size(1)
      local w = torch.reshape(x.weight,nOut,nInp,1,1)
      local y = cudnn.SpatialConvolution(nInp,nOut,1,1,1,1)
      y.weight:copy(w); y.gradWeight:copy(w); y.bias:copy(x.bias)
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
