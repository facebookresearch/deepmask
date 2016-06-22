--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Dataset sampler for for training/evaluation of DeepMask and SharpMask
------------------------------------------------------------------------------]]

require 'torch'
require 'image'
local tds = require 'tds'
local coco = require 'coco'

local DataSampler = torch.class('DataSampler')

--------------------------------------------------------------------------------
-- function: init
function DataSampler:__init(config,split)
  assert(split == 'train' or split == 'val')

  -- coco api
  local annFile = string.format('%s/annotations/instances_%s2014.json',
  config.datadir,split)
  self.coco = coco.CocoApi(annFile)

  -- mask api
  self.maskApi = coco.MaskApi

  -- mean/std computed from random subset of ImageNet training images
  self.mean, self.std = {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}

  -- class members
  self.datadir = config.datadir
  self.split = split

  self.iSz = config.iSz
  self.objSz = math.ceil(config.iSz*128/224)
  self.wSz = config.iSz + 32
  self.gSz = config.gSz
  self.scale = config.scale
  self.shift = config.shift

  self.imgIds = self.coco:getImgIds()
  self.annIds = self.coco:getAnnIds()
  self.catIds = self.coco:getCatIds()
  self.nImages = self.imgIds:size(1)

  if split == 'train' then self.__size  = config.maxload*config.batch
  elseif split == 'val' then self.__size = config.testmaxload*config.batch end

  if config.hfreq > 0 then
    self.scales = {} -- scale range for score sampling
    for scale = -3,2,.25 do table.insert(self.scales,scale) end
    self:createBBstruct(self.objSz,config.scale)
  end

  collectgarbage()
end
local function log2(x) return math.log(x)/math.log(2) end

--------------------------------------------------------------------------------
-- function: create BB struct of objects for score sampling
-- each key k contain the scale and bb information of all annotations of
-- image k
function DataSampler:createBBstruct(objSz,scale)
  local bbStruct = tds.Vec()

  for i = 1, self.nImages do
    local annIds = self.coco:getAnnIds({imgId=self.imgIds[i]})
    local bbs = {scales = {}}
    if annIds:dim() ~= 0 then
      for i = 1,annIds:size(1) do
        local annId = annIds[i]
        local ann = self.coco:loadAnns(annId)[1]
        local bbGt = ann.bbox
        local x0,y0,w,h = bbGt[1],bbGt[2],bbGt[3],bbGt[4]
        local xc,yc, maxDim = x0+w/2,y0+h/2, math.max(w,h)

        for s = -32,32,1 do
          if maxDim > objSz*2^((s-1)*scale) and
            maxDim <= objSz*2^((s+1)*(scale)) then
            local ss = -s*scale
            local xcS,ycS = xc*2^ss,yc*2^ss
            if not bbs[ss] then
              bbs[ss] = {}; table.insert(bbs.scales,ss)
            end
            table.insert(bbs[ss],{xcS,ycS,category_id=ann.category})
            break
          end
        end
      end
    end
    bbStruct:insert(tds.Hash(bbs))
  end
  collectgarbage()
  self.bbStruct = bbStruct
end

--------------------------------------------------------------------------------
-- function: get size of epoch
function DataSampler:size()
  return self.__size
end

--------------------------------------------------------------------------------
-- function: get a sample
function DataSampler:get(headSampling)
  local input,label
  if headSampling == 1 then -- sample masks
    input, label = self:maskSampling()
  else -- sample score
    input,label = self:scoreSampling()
  end

  if torch.uniform() > .5 then
    input = image.hflip(input)
    if headSampling == 1 then label = image.hflip(label) end
  end

  -- normalize input
  for i=1,3 do input:narrow(1,i,1):add(-self.mean[i]):div(self.std[i]) end

  return input,label
end

--------------------------------------------------------------------------------
-- function: mask sampling
function DataSampler:maskSampling()
  local iSz,wSz,gSz = self.iSz,self.wSz,self.gSz

  local cat,ann = torch.random(80)
  while not ann or ann.iscrowd == 1 or ann.area < 100 or ann.bbox[3] < 5
    or ann.bbox[4] < 5 do
      local catId = self.catIds[cat]
      local annIds = self.coco:getAnnIds({catId=catId})
      local annid = annIds[torch.random(annIds:size(1))]
      ann = self.coco:loadAnns(annid)[1]
  end
  local bbox = self:jitterBox(ann.bbox)
  local imgName = self.coco:loadImgs(ann.image_id)[1].file_name

  -- input
  local pathImg = string.format('%s/%s2014/%s',self.datadir,self.split,imgName)
  local inp = image.load(pathImg,3)
  local h, w = inp:size(2), inp:size(3)
  inp = self:cropTensor(inp, bbox, 0.5)
  inp = image.scale(inp, wSz, wSz)

  -- label
  local iSzR = iSz*(bbox[3]/wSz)
  local xc, yc = bbox[1]+bbox[3]/2, bbox[2]+bbox[4]/2
  local bboxInpSz = {xc-iSzR/2,yc-iSzR/2,iSzR,iSzR}
  local lbl = self:cropMask(ann, bboxInpSz, h, w, gSz)
  lbl:mul(2):add(-1)

  return inp, lbl
end

--------------------------------------------------------------------------------
-- function: score head sampler
local imgPad = torch.Tensor()
function DataSampler:scoreSampling(cat,imgId)
  local idx,bb
  repeat
    idx = torch.random(1,self.nImages)
    bb = self.bbStruct[idx]
  until #bb.scales ~= 0

  local imgId = self.imgIds[idx]
  local imgName = self.coco:loadImgs(imgId)[1].file_name
  local pathImg = string.format('%s/%s2014/%s',self.datadir,self.split,imgName)
  local img = image.load(pathImg,3)
  local h,w = img:size(2),img:size(3)

  -- sample central pixel of BB to be used
  local x,y,scale
  local lbl = torch.Tensor(1)
  if torch.uniform() > .5 then
    x,y,scale = self:posSamplingBB(bb)
    lbl:fill(1)
  else
    x,y,scale = self:negSamplingBB(bb,w,h)
    lbl:fill(-1)
  end

  local s = 2^-scale
  x,y  = math.min(math.max(x*s,1),w), math.min(math.max(y*s,1),h)
  local isz = math.max(self.wSz*s,10)
  local bw =isz/2

  --pad/crop/rescale
  imgPad:resize(3,h+2*bw,w+2*bw):fill(.5)
  imgPad:narrow(2,bw+1,h):narrow(3,bw+1,w):copy(img)
  local inp = imgPad:narrow(2,y,isz):narrow(3,x,isz)
  inp = image.scale(inp,self.wSz,self.wSz)

  return inp,lbl
end

--------------------------------------------------------------------------------
-- function: crop bbox b from inp tensor
function DataSampler:cropTensor(inp, b, pad)
  pad = pad or 0
  b[1], b[2] = torch.round(b[1])+1, torch.round(b[2])+1 -- 0 to 1 index
  b[3], b[4] = torch.round(b[3]), torch.round(b[4])

  local out, h, w, ind
  if #inp:size() == 3 then
    ind, out = 2, torch.Tensor(inp:size(1), b[3], b[4]):fill(pad)
  elseif #inp:size() == 2 then
    ind, out = 1, torch.Tensor(b[3], b[4]):fill(pad)
  end
  h, w = inp:size(ind), inp:size(ind+1)

  local xo1,yo1,xo2,yo2 = b[1],b[2],b[3]+b[1]-1,b[4]+b[2]-1
  local xc1,yc1,xc2,yc2 = 1,1,b[3],b[4]

  -- compute box on binary mask inp and cropped mask out
  if b[1] < 1 then xo1=1; xc1=1+(1-b[1]) end
  if b[2] < 1 then yo1=1; yc1=1+(1-b[2]) end
  if b[1]+b[3]-1 > w then xo2=w; xc2=xc2-(b[1]+b[3]-1-w) end
  if b[2]+b[4]-1 > h then yo2=h; yc2=yc2-(b[2]+b[4]-1-h) end
  local xo, yo, wo, ho = xo1, yo1, xo2-xo1+1, yo2-yo1+1
  local xc, yc, wc, hc = xc1, yc1, xc2-xc1+1, yc2-yc1+1
  if yc+hc-1 > out:size(ind)   then hc = out:size(ind  )-yc+1 end
  if xc+wc-1 > out:size(ind+1) then wc = out:size(ind+1)-xc+1 end
  if yo+ho-1 > inp:size(ind)   then ho = inp:size(ind  )-yo+1 end
  if xo+wo-1 > inp:size(ind+1) then wo = inp:size(ind+1)-xo+1 end
  out:narrow(ind,yc,hc); out:narrow(ind+1,xc,wc)
  inp:narrow(ind,yo,ho); inp:narrow(ind+1,xo,wo)
  out:narrow(ind,yc,hc):narrow(ind+1,xc,wc):copy(
  inp:narrow(ind,yo,ho):narrow(ind+1,xo,wo))

  return out
end

--------------------------------------------------------------------------------
-- function: crop bbox from mask
function DataSampler:cropMask(ann, bbox, h, w, sz)
  local mask = torch.FloatTensor(sz,sz)
  local seg = ann.segmentation

  local scale = sz / bbox[3]
  local polS = {}
  for m, segm in pairs(seg) do
    polS[m] = torch.DoubleTensor():resizeAs(segm):copy(segm); polS[m]:mul(scale)
  end
  local bboxS = {}
  for m = 1,#bbox do bboxS[m] = bbox[m]*scale end

  local Rs = self.maskApi.frPoly(polS, h*scale, w*scale)
  local mo = self.maskApi.decode(Rs)
  local mc = self:cropTensor(mo, bboxS)
  mask:copy(image.scale(mc,sz,sz):gt(0.5))

  return mask
end

--------------------------------------------------------------------------------
-- function: jitter bbox
function DataSampler:jitterBox(box)
  local x, y, w, h = box[1], box[2], box[3], box[4]
  local xc, yc = x+w/2, y+h/2
  local maxDim = math.max(w,h)
  local scale = log2(maxDim/self.objSz)
  local s = scale + torch.uniform(-self.scale,self.scale)
  xc = xc + torch.uniform(-self.shift,self.shift)*2^s
  yc = yc + torch.uniform(-self.shift,self.shift)*2^s
  w, h = self.wSz*2^s, self.wSz*2^s
  return {xc-w/2, yc-h/2,w,h}
end

--------------------------------------------------------------------------------
--function: posSampling: do positive sampling
function DataSampler:posSamplingBB(bb)
  local r = math.random(1,#bb.scales)
  local scale = bb.scales[r]
  r=torch.random(1,#bb[scale])
  local x,y = bb[scale][r][1], bb[scale][r][2]
  return x,y,scale
end

--------------------------------------------------------------------------------
--function: negSampling: do negative sampling
function DataSampler:negSamplingBB(bb,w0,h0)
  local x,y,scale
  local negSample,c = false,0
  while not negSample and c < 100 do
    local r = math.random(1,#self.scales)
    scale = self.scales[r]
    x,y = math.random(1,w0*2^scale),math.random(1,h0*2^scale)
    negSample = true
    for s = -10,10 do
      local ss = scale+s*self.scale
      if bb[ss] then
        for _,c in pairs(bb[ss]) do
          local dist = math.sqrt(math.pow(x-c[1],2)+math.pow(y-c[2],2))
          if dist < 3*self.shift then
            negSample = false
            break
          end
        end
      end
      if negSample == false then break end
    end
    c=c+1
  end
   return x,y,scale
end

return DataSampler
