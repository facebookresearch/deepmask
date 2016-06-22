--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Training and testing loop for DeepMask
------------------------------------------------------------------------------]]

local optim = require 'optim'
paths.dofile('trainMeters.lua')

local Trainer = torch.class('Trainer')

--------------------------------------------------------------------------------
-- function: init
function Trainer:__init(model, criterion, config)
  -- training params
  self.config = config
  self.model = model
  self.maskNet = nn.Sequential():add(model.trunk):add(model.maskBranch)
  self.scoreNet = nn.Sequential():add(model.trunk):add(model.scoreBranch)
  self.criterion = criterion
  self.lr = config.lr
  self.optimState ={}
  for k,v in pairs({'trunk','mask','score'}) do
    self.optimState[v] = {
      learningRate = config.lr,
      learningRateDecay = 0,
      momentum = config.momentum,
      dampening = 0,
      weightDecay = config.wd,
    }
  end

  -- params and gradparams
  self.pt,self.gt = model.trunk:getParameters()
  self.pm,self.gm = model.maskBranch:getParameters()
  self.ps,self.gs = model.scoreBranch:getParameters()

  -- allocate cuda tensors
  self.inputs, self.labels = torch.CudaTensor(), torch.CudaTensor()

  -- meters
  self.lossmeter  = LossMeter()
  self.maskmeter  = IouMeter(0.5,config.testmaxload*config.batch)
  self.scoremeter = BinaryMeter()

  -- log
  self.modelsv = {model=model:clone('weight', 'bias'),config=config}
  self.rundir = config.rundir
  self.log = torch.DiskFile(self.rundir .. '/log', 'rw'); self.log:seekEnd()
end

--------------------------------------------------------------------------------
-- function: train
function Trainer:train(epoch, dataloader)
  self.model:training()
  self:updateScheduler(epoch)
  self.lossmeter:reset()

  local timer = torch.Timer()

  local fevaltrunk = function() return self.model.trunk.output, self.gt end
  local fevalmask  = function() return self.criterion.output,   self.gm end
  local fevalscore = function() return self.criterion.output,   self.gs end

  for n, sample in dataloader:run() do
    -- copy samples to the GPU
    self:copySamples(sample)

    -- forward/backward
    local model, params, feval, optimState
    if sample.head == 1 then
      model, params = self.maskNet, self.pm
      feval,optimState = fevalmask, self.optimState.mask
    else
      model, params = self.scoreNet, self.ps
      feval,optimState = fevalscore, self.optimState.score
    end

    local outputs = model:forward(self.inputs)
    local lossbatch = self.criterion:forward(outputs, self.labels)

    model:zeroGradParameters()
    local gradOutputs = self.criterion:backward(outputs, self.labels)
    if sample.head == 1 then gradOutputs:mul(self.inputs:size(1)) end
    model:backward(self.inputs, gradOutputs)

    -- optimize
    optim.sgd(fevaltrunk, self.pt, self.optimState.trunk)
    optim.sgd(feval, params, optimState)

    -- update loss
    self.lossmeter:add(lossbatch)
  end

  -- write log
  local logepoch =
    string.format('[train] | epoch %05d | s/batch %04.2f | loss: %07.5f ',
      epoch, timer:time().real/dataloader:size(),self.lossmeter:value())
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  --save model
  torch.save(string.format('%s/model.t7', self.rundir),self.modelsv)
  if epoch%50 == 0 then
    torch.save(string.format('%s/model_%d.t7', self.rundir, epoch),
      self.modelsv)
  end

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: test
local maxacc = 0
function Trainer:test(epoch, dataloader)
  self.model:evaluate()
  self.maskmeter:reset()
  self.scoremeter:reset()

  for n, sample in dataloader:run() do
    -- copy input and target to the GPU
    self:copySamples(sample)

    if sample.head == 1 then
      local outputs = self.maskNet:forward(self.inputs)
      self.maskmeter:add(outputs:view(self.labels:size()),self.labels)
    else
      local outputs = self.scoreNet:forward(self.inputs)
      self.scoremeter:add(outputs, self.labels)
    end
    cutorch.synchronize()

  end
  self.model:training()

  -- check if bestmodel so far
  local z,bestmodel = self.maskmeter:value('0.7')
  if z > maxacc then
    torch.save(string.format('%s/bestmodel.t7', self.rundir),self.modelsv)
    maxacc = z
    bestmodel = true
  end

  -- write log
  local logepoch =
    string.format('[test]  | epoch %05d '..
      '| IoU: mean %06.2f median %06.2f suc@.5 %06.2f suc@.7 %06.2f '..
      '| acc %06.2f | bestmodel %s',
      epoch,
      self.maskmeter:value('mean'),self.maskmeter:value('median'),
      self.maskmeter:value('0.5'), self.maskmeter:value('0.7'),
      self.scoremeter:value(), bestmodel and '*' or 'x')
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: copy inputs/labels to CUDA tensor
function Trainer:copySamples(sample)
  self.inputs:resize(sample.inputs:size()):copy(sample.inputs)
  self.labels:resize(sample.labels:size()):copy(sample.labels)
end

--------------------------------------------------------------------------------
-- function: update training schedule according to epoch
function Trainer:updateScheduler(epoch)
  if self.lr == 0 then
    local regimes = {
      {   1,  50, 1e-3, 5e-4},
      {  51, 120, 5e-4, 5e-4},
      { 121, 1e8, 1e-4, 5e-4}
    }

    for _, row in ipairs(regimes) do
      if epoch >= row[1] and epoch <= row[2] then
        for k,v in pairs(self.optimState) do
          v.learningRate=row[3]; v.weightDecay=row[4]
        end
      end
    end
  end
end

return Trainer
