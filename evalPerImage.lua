--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Full scene evaluation of DeepMask/SharpMask
------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'
require 'image'

local cjson = require 'cjson'
local tds = require 'tds'
local coco = require 'coco'

paths.dofile('DeepMask.lua')
paths.dofile('SharpMask.lua')

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('full scene evaluation of DeepMask/SharpMask')
cmd:text()
cmd:argument('-model', 'model to load')
cmd:text('Options:')
cmd:option('-datadir', 'data/', 'data directory')
cmd:option('-seed', 1, 'manually set RNG seed')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-split', 'val', 'dataset split to be used (train/val)')
cmd:option('-np', 500,'number of proposals')
cmd:option('-thr', .2, 'mask binary threshold')
cmd:option('-save', false, 'save top proposals')
cmd:option('-startAt', 1, 'start image id')
cmd:option('-endAt', 5000, 'end image id')
cmd:option('-smin', -2.5, 'min scale')
cmd:option('-smax', .5, 'max scale')
cmd:option('-sstep', .5, 'scale step')
cmd:option('-timer', false, 'breakdown timer')
cmd:option('-dm', false, 'use DeepMask version of SharpMask')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)
torch.manualSeed(config.seed)
math.randomseed(config.seed)
local maskApi = coco.MaskApi
local meanstd = {mean={ 0.485, 0.456, 0.406 }, std={ 0.229, 0.224, 0.225 }}

--------------------------------------------------------------------------------
-- load model and config
print('| loading model file... ' .. config.model)
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

local model = m.model
model:inference(config.np)
model:cuda()

--------------------------------------------------------------------------------
-- directory to save results
local savedir = string.format('%s/epoch=%d/',config.model,epoch)
print(string.format('| saving results results in %s',savedir))
os.execute(string.format('mkdir -p %s',savedir))
os.execute(string.format('mkdir -p %s/t7',savedir))
os.execute(string.format('mkdir -p %s/jsons',savedir))
if config.save then os.execute(string.format('mkdir -p %s/res',savedir)) end

--------------------------------------------------------------------------------
-- create inference module
local scales = {}
for i = config.smin,config.smax,config.sstep do table.insert(scales,2^i) end

if torch.type(model)=='nn.DeepMask' then
  paths.dofile('InferDeepMask.lua')
elseif torch.type(model)=='nn.SharpMask' then
  paths.dofile('InferSharpMask.lua')
end

local infer = Infer{
  np = config.np,
  scales = scales,
  meanstd = meanstd,
  model = model,
  iSz = config.iSz,
  dm = config.dm,
  timer = config.timer,
}

--------------------------------------------------------------------------------
-- get list of eval images
local annFile = string.format('%s/annotations/instances_%s2014.json',
  config.datadir,config.split)
local coco = coco.CocoApi(annFile)
local imgIds = coco:getImgIds()
imgIds,_ = imgIds:sort()

--------------------------------------------------------------------------------
-- function: encode proposals
local function encodeProps(props,np,img,k,masks,scores)
  local t = (k-1)*np
  local enc = maskApi.encode(masks)

  for i = 1, np do
    local elem = tds.Hash()
    elem.segmentation = tds.Hash(enc[i])
    elem.image_id=img.id
    elem.category_id=1
    elem.score=scores[i][1]

    props[t+i] = elem
  end
end

--------------------------------------------------------------------------------
-- function: convert props to json and save
local function saveProps(props,savedir,s,e)
  --t7
  local pathsvt7 = string.format('%s/t7/props-%d-%d.t7', savedir,s,e)
  torch.save(pathsvt7,props)
  --json
  local pathsvjson = string.format('%s/jsons/props-%d-%d.json', savedir,s,e)
  local propsjson = {}
  for _,prop in pairs(props) do -- hash2table
    local elem = {}
    elem.category_id = prop.category_id
    elem.image_id = prop.image_id
    elem.score = prop.score
    elem.segmentation={
      size={prop.segmentation.size[1],prop.segmentation.size[2]},
      counts = prop.segmentation.counts or prop.segmentation.count
    }
    table.insert(propsjson,elem)
  end
  local jsonText = cjson.encode(propsjson)
  local f = io.open(pathsvjson,'w'); f:write(jsonText); f:close()
  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: read image
local function readImg(datadir,split,fileName)
  local pathImg = string.format('%s/%s2014/%s',datadir,split,fileName)
  local inp = image.load(pathImg,3)
  return inp
end

--------------------------------------------------------------------------------
-- run
print('| start eval')
local props, svcount = tds.Hash(), config.startAt
for k = config.startAt,config.endAt do
  xlua.progress(k,config.endAt)

  -- load image
  local img = coco:loadImgs(imgIds[k])[1]
  local input = readImg(config.datadir,config.split,img.file_name)
  local h,w = img.height,img.width

  -- forward all scales
  infer:forward(input)

  -- get top proposals
  local masks,scores = infer:getTopProps(config.thr,h,w)

  -- encode proposals
  encodeProps(props,config.np,img,k,masks,scores)

  -- save top masks?
  if config.save then
    local res = input:clone()
    maskApi.drawMasks(res, masks, 10)
    image.save(string.format('%s/res/%d.jpg',savedir,k),res)
  end

  -- save proposals
  if k%500 == 0 then
    saveProps(props,savedir,svcount,k); props = tds.Hash(); collectgarbage()
    svcount = svcount + 500
  end

  collectgarbage()
end

if config.timer then infer:printTiming() end
collectgarbage()
print('| finish')
