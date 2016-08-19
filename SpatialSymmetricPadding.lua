--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

SpatialSymmetricPadding module

The forward(A) pads input tensor A with mirror reflections of itself
It is the same function as Matlab padarray(A, padsize, 'symmetric' )
The padding is of the form: cba[abcd...]
While nn.SpatialReflectionPadding does: dcb[abcd...]
And nn.SpatialReplicationPadding does: aaa[abcd...]
(where [abcd...] is a tensor)
The updateGradInput(input, gradOutput) is inherited from nn.SpatialZeroPadding,
where the padded region is treated as constant and
the gradients are accumulated in the backward pass
------------------------------------------------------------------------------]]

local SpatialSymmetricPadding, parent =
  torch.class('nn.SpatialSymmetricPadding', 'nn.SpatialZeroPadding')

function SpatialSymmetricPadding:__init(pad_l, pad_r, pad_t, pad_b)
   parent.__init(self, pad_l, pad_r, pad_t, pad_b)
end

function SpatialSymmetricPadding:updateOutput(input)
  assert(input:dim()==4, "only Dimension=4 implemented")
  -- sizes
  local h = input:size(3) + self.pad_t + self.pad_b
  local w = input:size(4) + self.pad_l + self.pad_r
  if w < 1 or h < 1 then error('input is too small') end
  self.output:resize(input:size(1), input:size(2), h, w)
  self.output:zero()
  -- crop input if necessary
  local c_input = input
  if self.pad_t < 0 then
    c_input = c_input:narrow(3, 1 - self.pad_t, c_input:size(3) + self.pad_t)
  end
  if self.pad_b < 0 then
    c_input = c_input:narrow(3, 1, c_input:size(3) + self.pad_b)
  end
  if self.pad_l < 0 then
    c_input = c_input:narrow(4, 1 - self.pad_l, c_input:size(4) + self.pad_l)
  end
  if self.pad_r < 0 then
    c_input = c_input:narrow(4, 1, c_input:size(4) + self.pad_r)
  end
  -- crop outout if necessary
  local c_output = self.output
  if self.pad_t > 0 then
    c_output = c_output:narrow(3, 1 + self.pad_t, c_output:size(3) - self.pad_t)
  end
  if self.pad_b > 0 then
    c_output = c_output:narrow(3, 1, c_output:size(3) - self.pad_b)
  end
  if self.pad_l > 0 then
    c_output = c_output:narrow(4, 1 + self.pad_l, c_output:size(4) - self.pad_l)
  end
  if self.pad_r > 0 then
    c_output = c_output:narrow(4, 1, c_output:size(4) - self.pad_r)
  end
  -- copy input to output
  c_output:copy(c_input)
  -- symmetric padding that fills in values on the padded region
  if w<2*self.pad_l or w<2*self.pad_r or h<2*self.pad_t or h<2*self.pad_b then
    error('input is too small')
  end
  for i=1,self.pad_t do
    self.output:narrow(3,self.pad_t-i+1,1):copy(
    self.output:narrow(3,i+self.pad_t,1))
  end
  for i=1,self.pad_b do
    self.output:narrow(3,self.output:size(3)-self.pad_b+i,1):copy(
    self.output:narrow(3,self.output:size(3)-self.pad_b-i+1,1))
  end
  for i=1,self.pad_l do
    self.output:narrow(4,self.pad_l-i+1,1):copy(
    self.output:narrow(4,i+self.pad_l,1))
  end
  for i=1,self.pad_r do
    self.output:narrow(4,self.output:size(4)-self.pad_r+i,1):copy(
    self.output:narrow(4,self.output:size(4)-self.pad_r-i+1,1))
  end
  return self.output
end
