--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Per patch evaluation of DeepMask/SharpMask
------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'

paths.dofile('DeepMask.lua')
paths.dofile('SharpMask.lua')

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('per patch evaluation of DeepMask/SharpMask')
cmd:text()
cmd:argument('-model', 'model to load')
cmd:text('Options:')
cmd:option('-seed', 1, 'Manually set RNG seed')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-testmaxload', 200, 'max number of testing batches')
cmd:option('-save', false, 'save output')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)
torch.manualSeed(config.seed)
math.randomseed(config.seed)

local inputs = torch.CudaTensor()

--------------------------------------------------------------------------------
-- loading model and config
print('| loading model file...' .. config.model)
local m = torch.load(config.model..'/model.t7')
local c = m.config
for k,v in pairs(c) do if config[k] == nil then config[k] = v end end
local epoch = 0
if paths.filep(config.model..'/log') then
  for line in io.lines(config.model..'/log') do
    if string.find(line,'train') then epoch = epoch + 1 end
  end
  print(string.format('| number of examples seen until now: %d (%d epochs)',
    epoch*config.maxload*config.batch,epoch))
end
config.hfreq = 0 -- only evaluate masks

local model = m.model
if torch.type(model)=='nn.DeepMask' then
  model=nn.Sequential():add(model.trunk):add(model.maskBranch)
end
model:evaluate()

--------------------------------------------------------------------------------
-- directory to save results
local savedir
if config.save then
  require 'image'
  savedir = string.format('%s/epoch=%d/res-patch/',config.model,epoch)
  os.execute(string.format('mkdir -p %s',savedir))
end

--------------------------------------------------------------------------------
-- initialize data provider and mask meter
local DataLoader = paths.dofile('DataLoader.lua')
local _, valLoader = DataLoader.create(config)

paths.dofile('trainMeters.lua')
local maskmeter = IouMeter(0.5,config.testmaxload*config.batch)

--------------------------------------------------------------------------------
-- function display output
local function saveRes(input,target,output,savedir,n)
  local batch,h,w = target:size(1),config.gSz,config.gSz

  local input,target,output = input:float(),target:float(),output:float()
  input = input:narrow(3,16,config.iSz):narrow(4,16,config.iSz)
  output:mul(-1):exp():add(1):pow(-1) -- transform outs in probability
  output = output:view(batch,h,w)

  local imgRGB = torch.Tensor(batch,3,h,w):zero()
  local outJet = torch.Tensor(batch,3,h,w):zero()

  for b = 1, batch do
    imgRGB:narrow(1,b,1):copy(image.scale(input[b],w,h))
    local oj = torch.floor(output[b]*100):add(1):double()
    oj = image.scale(oj,w,h); oj = image.y2jet(oj)
    outJet:narrow(1,b,1):copy(oj)
    local mask = image.scale(target[b],w,h):ge(0):double()
    local me = image.erode(mask,torch.DoubleTensor(3,3):fill(1))
    local md = image.dilate(mask,torch.DoubleTensor(3,3):fill(1))
    local maskf = md - me
    maskf = maskf:eq(1)
    imgRGB:narrow(1,b,1):add(-imgRGB:min()):mul(1/imgRGB:max())
    imgRGB[b][1][maskf]=1; imgRGB[b][2][maskf]=0; imgRGB[b][3][maskf]=0
  end

  -- concatenate
  local res = torch.Tensor(3,h*batch,w*2):zero()
  for b = 1, batch do
    res:narrow(2,(b-1)*h+1,h):narrow(3,1,w):copy(imgRGB[b])
    res:narrow(2,(b-1)*h+1,h):narrow(3,w+1,w):copy(outJet[b])
  end

  image.save(string.format('%s/%d.jpg',savedir,n),res)
end

--------------------------------------------------------------------------------
-- start evaluation
print('| start per batch evaluation')
maskmeter:reset()
sys.tic()
for n, sample in valLoader:run() do
  xlua.progress(n,config.testmaxload)

  -- copy input and target to the GPU
  inputs:resize(sample.inputs:size()):copy(sample.inputs)

  -- infer mask in batch
  local output = model:forward(inputs):float()
  cutorch.synchronize()
  output = output:view(sample.labels:size())

  -- compute IoU
  maskmeter:add(output,sample.labels)

  -- save?
  if config.save then
    saveRes(sample.inputs, sample.labels, output, savedir, n)
  end

  collectgarbage()
end
cutorch.synchronize()
print('| finish')

--------------------------------------------------------------------------------
-- log
print('----------------------------------------------')
local log = string.format('| model: %s\n',config.model)
log = log..string.format('| # epochs: %s\n',epoch)
log = log..string.format(
  '| # samples: %d\n'..
  '| samples/s %7d '..
  '| mean %06.2f median %06.2f '..
  'iou@.5 %06.2f  iou@.7 %06.2f ',
  maskmeter.n,config.batch*config.testmaxload/sys.toc(),
  maskmeter:value('mean'),maskmeter:value('median'),
  maskmeter:value('0.5'), maskmeter:value('0.7')
  )
print(log)
