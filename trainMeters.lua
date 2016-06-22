--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Contains the tree metrics used during training/evaluation:
  - lossmeter: measure the average loss.
  - binarymeter: measure error of predicted objectness score and ground truth
                objectness annotation.
  - ioumeter: measure iou between infered and ground truth masks.
------------------------------------------------------------------------------]]

--------------------------------------------------------------------------------
-- loss meter
do
  local LossMeter = torch.class('LossMeter')
  -- init
  function LossMeter:__init()
    self:reset()
  end

  -- function: reset
  function LossMeter:reset()
    self.sum = 0; self.n = 0
  end

  -- function: add
  function LossMeter:add(value,n)
    n = n or 1
    self.sum = self.sum + value
    self.n = self.n + n
  end

  -- function: value
  function LossMeter:value()
    return self.sum / self.n
  end
end

--------------------------------------------------------------------------------
-- binary meter
do
  local BinaryMeter = torch.class('BinaryMeter')
  -- init
  function BinaryMeter:__init()
    self:reset()
  end
  -- function: reset
  function BinaryMeter:reset()
    self.acc = 0; self.n = 0
  end

  -- function: add
  function BinaryMeter:add(output, target)
    target, output = target:squeeze(), output:squeeze()
    assert(output:nElement() == target:nElement(),
      'target and output do not match')

    local acc = torch.cmul(output,target)
    self.acc = self.acc + acc:ge(0):sum()
    self.n = self.n + output:size(1)
  end

  -- function: value
  function BinaryMeter:value()
    local res = self.acc/self.n
    return res*100
  end
end

--------------------------------------------------------------------------------
-- iou meter
do
  local IouMeter = torch.class('IouMeter')
  -- init
  function IouMeter:__init(thr,sz)
    self.sz = sz
    self.iou = torch.Tensor(sz)
    self.thr = math.log(thr/(1-thr))
    self:reset()
  end

  -- function: reset
  function IouMeter:reset()
    self.iou:zero(); self.n = 0
  end

  -- function: add
  function IouMeter:add(output, target)
    target, output = target:squeeze():float(), output:squeeze():float()
    assert(output:nElement() == target:nElement(),
      'target and output do not match')

    local batch,h,w = output:size(1),output:size(2),output:size(3)
    local nOuts = h*w
    local iouptr = self.iou:data()

    local int,uni
    local pred = output:ge(self.thr)
    local pPtr,tPtr = pred:data(), target:data()
    for b = 0,batch-1 do
      int,uni = 0,0
      for i = 0,nOuts-1 do
        local id = b*nOuts+i
        if pPtr[id] == 1 and tPtr[id] == 1 then int = int + 1 end
        if pPtr[id] == 1 or tPtr[id] == 1 then uni = uni + 1 end
      end
      if uni > 0 then iouptr[self.n+b] = int/uni end
    end
    self.n = self.n + batch
  end

  -- function: value
  function IouMeter:value(s)
    if s then
      local res
      local nb = math.max(self.iou:ne(0):sum(),1)
      local iou = self.iou:narrow(1,1,nb)
      if s == 'mean' then
        res = iou:mean()
      elseif s == 'median' then
        res = iou:median():squeeze()
      elseif tonumber(s) then
        local iouSort, _ = iou:sort()
        res = iouSort:ge(tonumber(s)):sum()/nb
      elseif s == 'hist' then
        res = torch.histc(iou,20)/nb
      end

      return res*100
    else
      local value = {}
      for _,s in ipairs(self.stats) do
        value[s] = self:value(s)
      end
      return value
    end
  end
end
