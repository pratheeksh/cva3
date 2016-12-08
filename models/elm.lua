local nn = require 'nn'



local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local SubSampling = nn.SpatialSubSampling
local model  = nn.Sequential()

model:add(Convolution(3, 100, 3, 3))
model:add(Max(2, 2, 2, 2))
model:add(Convolution(100, 150, 4, 4))
model:add(Max(2, 2, 2, 2))
model:add(Convolution(150, 250, 3, 3))
model:add(Max(2, 2, 2, 2))
model:add(Convolution(250, 200, 4, 4))
model:add(View(200))
model:add(Linear(200, 12000))
model:add(Tanh())
model:add(Linear(12000, 43))


return model
