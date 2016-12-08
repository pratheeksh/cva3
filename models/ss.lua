local nn = require 'nn'



local Convolution = nn.SpatialConvolutionMM
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local SubSampling = nn.SpatialSubSampling
local model  = nn.Sequential()

model:add(Convolution(3, 12, 5, 5, 1, 1, 2))
model:add(SubSampling(12, 2, 2, 2, 2))
model:add(ReLU())
model:add(Convolution(12, 48, 5, 5, 1,1, 2))
model:add(SubSampling(48, 2, 2, 2, 2))
model:add(ReLU())
model:add(View(48*8*8))
model:add(Linear(48*8*8, 100))
model:add(ReLU())
model:add(Linear(100, 100))
model:add(ReLU())

model:add(Linear(100, 43))

return model
