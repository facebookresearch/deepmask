--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Inference module for SharpMask
------------------------------------------------------------------------------]]

require 'image'
local argcheck = require 'argcheck'

local Infer = torch.class('Infer')

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
  {name="dm", type="boolean", default=false},
  {name="timer", type="boolean", default=false},
  call =
    function(self, np, scales, meanstd, model, iSz, dm, timer)
      --model
      self.trunk = model.trunk
      self.mBranch = model.maskBranchDM
      self.sBranch = model.scoreBranch
      self.refs = model.refs
      self.neths = model.neths
      self.skpos = model.skpos
      self.fSz = model.fSz
      self.dm = dm -- flag to use deepmask instead of sharpmask

      -- number of proposals
      self.np = np

      --mean/std
      self.mean, self.std = meanstd.mean, meanstd.std

      -- input size and border width
      self.iSz, self.bw = iSz, iSz/2

      -- timer
      if timer then self.timer = torch.Tensor(8):zero() end

      -- create  scale pyramid
      self.scales = scales
      self.pyramid = nn.ConcatTable()
      for i = 1,#scales do
        self.pyramid:add(nn.SpatialReSamplingEx{rwidth=scales[i],
          rheight=scales[i], mode='bilinear'})
      end

      -- allocate topScores, topMasks and topPatches
      self.topScores, self.topMasks = torch.Tensor(), torch.ByteTensor()
      local topPatches
      if self.dm then
        topPatches = torch.CudaTensor(self.np,512):zero()
      else
        topPatches = {}
        topPatches[1] = torch.CudaTensor(self.np,512):zero()
        for j = 1, #model.refs do
          local sz = model.fSz*2^(j-1)
          topPatches[j+1] = torch.CudaTensor(self.np,model.ks/2^(j),sz,sz)
        end
      end
      self.topPatches = topPatches
    end
}

--------------------------------------------------------------------------------
-- function: forward
local inpPad = torch.CudaTensor()
function Infer:forward(input,id)
  if input:type() == 'torch.CudaTensor' then input = input:float() end

  -- forward pyramid
  if self.timer then sys.tic() end
  local inpPyramid = self.pyramid:forward(input)
  if self.timer then self.timer:narrow(1,1,1):add(sys.toc()) end

  -- forward all scales through network
  local outPyramidTrunk,outPyramidScore,outPyramidSkip = {},{},{}
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
    local outTrunk = self.trunk:forward(inpPad)
    cutorch.synchronize()
    if self.timer then self.timer:narrow(1,3,1):add(sys.toc()) end
    table.insert(outPyramidTrunk,outTrunk:clone():squeeze())

    -- forward score branch
    if self.timer then sys.tic() end
    local outScore = self.sBranch:forward(outTrunk)
    cutorch.synchronize()
    if self.timer then self.timer:narrow(1,4,1):add(sys.toc()) end
    table.insert(outPyramidScore,outScore:clone():squeeze())

    -- forward horizontal nets
    if not self.dm then
      local hOuts = {}
      for k,neth in pairs(self.neths) do
        if self.timer then sys.tic() end
        neth:forward(self.trunk.modules[self.skpos[k]].output)
        cutorch.synchronize()
        if self.timer then self.timer:narrow(1,5,1):add(sys.toc()) end
        hOuts[k] = neth.output:clone()
      end
      outPyramidSkip[i] = hOuts
    end
  end

  -- get top scores
  self:getTopScores(outPyramidScore)

  -- get top patches and top masks, depending on mode
  local topMasks0
  if self.dm then
    if self.timer then sys.tic() end
    self:getTopPatchesDM(outPyramidTrunk)
    if self.timer then self.timer:narrow(1,6,1):add(sys.toc()) end

    if self.timer then sys.tic() end
    topMasks0 = self.mBranch:forward(self.topPatches)
    local osz = math.sqrt(topMasks0:size(2))
    topMasks0 = topMasks0:view(self.np,osz,osz)
    if self.timer then self.timer:narrow(1,7,1):add(sys.toc()) end
  else
    if self.timer then sys.tic() end
    self:getTopPatches(outPyramidTrunk,outPyramidSkip)
    if self.timer then self.timer:narrow(1,6,1):add(sys.toc()) end

    if self.timer then sys.tic() end
    topMasks0 = self:forwardRefinement(self.topPatches)
    if self.timer then self.timer:narrow(1,7,1):add(sys.toc()) end
  end
  self.topMasks0 = topMasks0:float():squeeze()

  collectgarbage()

  if self.timer then self.timer:narrow(1,8,1):add(1) end
end

--------------------------------------------------------------------------------
-- function: forward refinement inference
-- input is a table containing the output of bottom-up and the output of all
-- horizontal layers
function Infer:forwardRefinement(input)
  local currentOutput = self.refs[0]:forward(input[1])
  for i = 1,#self.refs do
    currentOutput = self.refs[i]:forward({input[i+1],currentOutput})
  end
  cutorch.synchronize()
  self.output = currentOutput
  return self.output
end

--------------------------------------------------------------------------------
-- function: get top patches
function Infer:getTopPatchesDM(outPyramidTrunk)
  local topscores = self.topScores
  local ts_ptr = topscores:data()
  for i = 1, topscores:size(1) do
    local pos = (i-1)*4
    local s,x,y = ts_ptr[pos+1], ts_ptr[pos+2], ts_ptr[pos+3]
    local patch = outPyramidTrunk[s]:narrow(2,x,1):narrow(3,y,1)
    self.topPatches:narrow(1,i,1):copy(patch)
  end
end

--------------------------------------------------------------------------------
-- function: get top patches
local t
function Infer:getTopPatches(outPyramidTrunk,outPyramidSkip)
  local topscores = self.topScores
  local ts_ptr = topscores:data()

  if not t then t={}; for j = 1, #self.skpos do t[j]=2^(j-1) end end

  for i = 1, #self.topPatches do self.topPatches[i]:zero() end
  for i = 1, self.np do
    local pos = (i-1)*4
    local s,x,y = ts_ptr[pos+1], ts_ptr[pos+2], ts_ptr[pos+3]

    -- get patches from output outPyramidTrunk
    local patch = outPyramidTrunk[s]:narrow(2,x,1):narrow(3,y,1)
    self.topPatches[1]:narrow(1,i,1):copy(patch)

    for j = 1, #self.skpos do
      local isz =(self.fSz)*t[j]
      local xx,yy = (x-1)*t[j]+1 , (y-1)*t[j]+1
      local o = outPyramidSkip[s][j]
      local dx=math.min(isz,o:size(3)-xx+1)
      local dy=math.min(isz,o:size(4)-yy+1)
      local patch = o:narrow(3,xx,dx):narrow(4,yy,dy)
      self.topPatches[j+1]:narrow(1,i,1):narrow(3,1,dx):narrow(4,1,dy)
      :copy(patch)
    end
  end
  cutorch.synchronize()
  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: get top scores
-- return a tensor k x 4, where k is the number of top scores.
-- each line contains: the score value, the scaleNb and position(of M(:))
local sortedScores = torch.Tensor()
local sortedIds = torch.Tensor()
local pos = torch.Tensor()
function Infer:getTopScores(outPyramidScore)
  local topScores = self.topScores

  self.score = outPyramidScore
  local np = self.np
  -- sort scores/ids for each scale
  local nScales=#self.score
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
local topMasks = torch.ByteTensor()
local imgMask = torch.ByteTensor()
function Infer:getTopMasks(thr,h,w)
  thr = math.log(thr/(1-thr)) -- 1/(1+e^-s) > th => s > log(1-th)

  local topMasks0,topScores,np = self.topMasks0,self.topScores,self.np
  topMasks:resize(np,h,w):zero()
  imgMask:resize(h,w)
  local imgMaskPtr = imgMask:data()

  for i = 1,np do
    imgMask:zero()
    local scale,x,y = topScores[i][2],topScores[i][3],topScores[i][4]
    local s = self.scales[scale]
    local sz = math.floor(self.iSz/s)
    local mask = topMasks0[i]
    local x,y = math.min(x,mask:size(1)),math.min(y,mask:size(2))
    local mask = image.scale(mask,sz,sz,'bilinear')
    local maskPtr = mask:data()

    local t,delta = 16/s, self.iSz/2/s
    for im =0, sz-1 do
      local ii = math.floor((x-1)*t-delta+im)
      for jm = 0,sz- 1 do
        local jj=math.floor((y-1)*t-delta+jm)
        if  maskPtr[sz*im + jm] > thr and
        ii >= 0 and ii <= h-1 and jj >= 0 and jj <= w-1 then
          imgMaskPtr[jj+ w*ii]=1
        end
      end
    end

    topMasks:narrow(1,i,1):copy(imgMask)
  end

  self.topMasks = topMasks
  return topMasks
end

--------------------------------------------------------------------------------
-- function: get top proposals
function Infer:getTopProps(thr,h,w)
  self:getTopMasks(thr,h,w)
  return self.topMasks, self.topScores
end

--------------------------------------------------------------------------------
-- function: display timer
function Infer:printTiming()
  local t = self.timer
  t:div(t[t:size(1)])

  print('\n| timing:')
  print('| time pyramid:',t[1])
  print('| time pre-process:',t[2])
  print('| time trunk:',t[3])
  print('| time score branch:',t[4])
  print('| time skip connections:',t[5])
  print('| time topPatches:',t[6])
  print('| time refinement:',t[7])
  print('| time total:',t:narrow(1,1,t:size(1)-1):sum())
end

return Infer
