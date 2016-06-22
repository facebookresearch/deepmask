--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Inference module for DeepMask
------------------------------------------------------------------------------]]

require 'image'
local argcheck = require 'argcheck'

local Infer = torch.class('Infer')

--------------------------------------------------------------------------------
-- function: unfold the mask output into a matrix of masks
local function unfoldMasksMatrix(masks)
  local umasks = {}
  local oSz = math.sqrt(masks[1]:size(1))
  for _,mask in pairs(masks) do
    local umask = mask:reshape(oSz,oSz,mask:size(2),mask:size(3))
    umask=umask:transpose(1,3):transpose(2,3):transpose(2,4):transpose(3,4)
    table.insert(umasks,umask)
  end
  return umasks
end

--------------------------------------------------------------------------------
-- function: init
Infer.__init = argcheck{
  noordered = true,
  {name="self", type="Infer"},
  {name="np", type="number",default=500},
  {name="scales", type="table"},
  {name="meanstd", type="table"},
  {name="model", type="nn.Container"},
  {name="iSz", type="number", default=160},
  {name="dm", type="boolean", default=true},
  {name="timer", type="boolean", default=false},
  call =
    function(self, np, scales, meanstd, model, iSz, dm, timer)
      --model
      self.trunk = model.trunk
      self.mHead = model.maskBranch
      self.sHead = model.scoreBranch

      -- number of proposals
      self.np = np

      --mean/std
      self.mean, self.std = meanstd.mean, meanstd.std

      -- input size and border width
      self.iSz, self.bw = iSz, iSz/2

      -- timer
      if timer then self.timer = torch.Tensor(6):zero() end

      -- create  scale pyramid
      self.scales = scales
      self.pyramid = nn.ConcatTable()
      for i = 1,#scales do
        self.pyramid:add(nn.SpatialReSamplingEx{rwidth=scales[i],
          rheight=scales[i], mode='bilinear'})
      end

      -- allocate topScores and topMasks
      self.topScores = torch.Tensor()
      self.topMasks = torch.ByteTensor()
    end
}

--------------------------------------------------------------------------------
-- function: forward
local inpPad = torch.CudaTensor()
function Infer:forward(input)
  if input:type() == 'torch.CudaTensor' then input = input:float() end

  -- forward pyramid
  if self.timer then sys.tic() end
  local inpPyramid = self.pyramid:forward(input)
  if self.timer then self.timer:narrow(1,1,1):add(sys.toc()) end

  -- forward all scales through network
  local outPyramidMask,outPyramidScore = {},{}
  for i,_ in pairs(inpPyramid) do
    local inp = inpPyramid[i]:cuda()
    local h,w = inp:size(2),inp:size(3)

    -- padding/normalize
    if self.timer then sys.tic() end
    inpPad:resize(1,3,h+2*self.bw,w+2*self.bw):fill(.5)
    inpPad:narrow(1,1,1):narrow(3,self.bw+1,h):narrow(4,self.bw+1,w):copy(inp)
    for i=1,3 do inpPad[1][i]:add(-self.mean[i]):div(self.std[i]) end
    cutorch.synchronize()
    if self.timer then self.timer:narrow(1,2,1):add(sys.toc()) end

    -- forward trunk
    if self.timer then sys.tic() end
    local outTrunk = self.trunk:forward(inpPad):squeeze()
    cutorch.synchronize()
    if self.timer then self.timer:narrow(1,3,1):add(sys.toc()) end

    -- forward score branch
    if self.timer then sys.tic() end
    local outScore = self.sHead:forward(outTrunk)
    cutorch.synchronize()
    if self.timer then self.timer:narrow(1,4,1):add(sys.toc()) end
    table.insert(outPyramidScore,outScore:clone():squeeze())

    -- forward mask branch
    if self.timer then sys.tic() end
    local outMask = self.mHead:forward(outTrunk)
    cutorch.synchronize()
    if self.timer then self.timer:narrow(1,5,1):add(sys.toc()) end
    table.insert(outPyramidMask,outMask:float():squeeze())
  end

  self.mask = unfoldMasksMatrix(outPyramidMask)
  self.score = outPyramidScore

  if self.timer then self.timer:narrow(1,6,1):add(1) end
end

--------------------------------------------------------------------------------
-- function: get top scores
-- return a tensor k x 4, where k is the number of top scores.
-- each line contains: the score value, the scaleNb and position(of M(:))
local sortedScores = torch.Tensor()
local sortedIds = torch.Tensor()
local pos = torch.Tensor()
function Infer:getTopScores()
  local topScores = self.topScores

  -- sort scores/ids for each scale
  local nScales=#self.scales
  local rowN=self.score[nScales]:size(1)*self.score[nScales]:size(2)
  sortedScores:resize(rowN,nScales):zero()
  sortedIds:resize(rowN,nScales):zero()
  for s = 1,nScales do
    self.score[s]:mul(-1):exp():add(1):pow(-1) -- scores2prob

    local sc = self.score[s]
    local h,w = sc:size(1),sc:size(2)

    local sc=sc:view(h*w)
    local sS,sIds=torch.sort(sc,true)
    local sz = sS:size(1)
    sortedScores:narrow(2,s,1):narrow(1,1,sz):copy(sS)
    sortedIds:narrow(2,s,1):narrow(1,1,sz):copy(sIds)
  end

  -- get top scores
  local np = self.np
  pos:resize(nScales):fill(1)
  topScores:resize(np,4):fill(1)
  np=math.min(np,rowN)

  for i = 1,np do
    local scale,score = 0,0
    for k = 1,nScales do
      if sortedScores[pos[k]][k] > score then
        score = sortedScores[pos[k]][k]
        scale = k
      end
    end
    local temp=sortedIds[pos[scale]][scale]
    local x=math.floor(temp/self.score[scale]:size(2))
    local y=temp%self.score[scale]:size(2)+1
    x,y=math.max(1,x),math.max(1,y)

    pos[scale]=pos[scale]+1
    topScores:narrow(1,i,1):copy(torch.Tensor({score,scale,x,y}))
  end

  return topScores
end

--------------------------------------------------------------------------------
-- function: get top masks.
local imgMask = torch.ByteTensor()
function Infer:getTopMasks(thr,h,w)
  local topMasks = self.topMasks

  thr = math.log(thr/(1-thr)) -- 1/(1+e^-s) > th => s > log(1-th)

  local masks,topScores,np = self.mask,self.topScores,self.np
  topMasks:resize(np,h,w):zero()
  imgMask:resize(h,w)
  local imgMaskPtr = imgMask:data()

  for i = 1,np do
    imgMask:zero()
    local scale,x,y=topScores[i][2], topScores[i][3], topScores[i][4]
    local s=self.scales[scale]
    local sz = math.floor(self.iSz/s)
    local mask = masks[scale]
    x,y = math.min(x,mask:size(1)),math.min(y,mask:size(2))
    mask = mask[x][y]:float()
    local mask = image.scale(mask,sz,sz,'bilinear')
    local mask_ptr = mask:data()

    local t = 16/s
    local delta = self.iSz/2/s
    for im =0, sz-1 do
      local ii = math.floor((x-1)*t-delta+im)
      for jm = 0,sz- 1 do
        local jj=math.floor((y-1)*t-delta+jm)
        if  mask_ptr[sz*im + jm] > thr and
          ii >= 0 and ii <= h-1 and jj >= 0 and jj <= w-1 then
          imgMaskPtr[jj+ w*ii]=1
        end
      end
    end
    topMasks:narrow(1,i,1):copy(imgMask)
  end

  return topMasks
end

--------------------------------------------------------------------------------
-- function: get top proposals
function Infer:getTopProps(thr,h,w)
  self:getTopScores()
  self:getTopMasks(thr,h,w)
  return self.topMasks, self.topScores
end

--------------------------------------------------------------------------------
-- function: display timer
function Infer:printTiming()
  local t = self.timer
  t:div(t[t:size(1)])

  print('| time pyramid:',t[1])
  print('| time pre-process:',t[2])
  print('| time trunk:',t[3])
  print('| time score branch:',t[4])
  print('| time mask branch:',t[5])
  print('| time total:',t:narrow(1,1,t:size(1)-1):sum())
end

return Infer
