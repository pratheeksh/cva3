local nn = require 'nn'
--- : 3x48x48-100C7-MP2-150C4-150MP2-250C4-250MP2-300N-43N


--  2x48x48-100C5-MP2-100C5-MP2-100C4-MP2-300N-100N-6N represents a net with 2
-- input images of size 48x48, a convolutional layer with 100 maps and 5x5 filters, a max-pooling layer over
-- non overlapping regions of size 2x2, a convolutional layer with 100 maps and 4x4 filters, a max-pooling
-- layer over non overlapping regions of size 2x2, a fully connected layer with 300 hidden units, a fully
-- connected layer with 100 hidden units and a fully connected output layer with 6 neurons (one per class).


local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear

local model  = nn.Sequential()

model:add(Convolution(3, 100, 7, 7))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(Convolution(100, 150, 4, 4))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(Convolution(150, 250, 4, 4))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(View(2250))
model:add(Linear(2250, 300))
model:add(ReLU())
model:add(Linear(300, 43))

return model
