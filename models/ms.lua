local nn = require 'nn'
--- : 3x48x48-100C7-MP2-150C4-150MP2-250C4-250MP2-300N-43N


--  2x48x48-100C5-MP2-100C5-MP2-100C4-MP2-300N-100N-6N represents a net with 2
-- input images of size 48x48, a convolutional layer with 100 maps and 5x5 filters, a max-pooling layer over
-- non overlapping regions of size 2x2, a convolutional layer with 100 maps and 4x4 filters, a max-pooling
-- layer over non overlapping regions of size 2x2, a fully connected layer with 300 hidden units, a fully
-- connected layer with 100 hidden units and a fully connected output layer with 6 neurons (one per class).


local Convolution = nn.SpatialConvolutionMM
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear

local normkernel = image.gaussian1D(7)
-- function getconv(nin, nout)

-- 	local first  = nn.Sequential()
-- 	first:add(Convolution(nin, nout, 5, 5, 1, 1, 2, 2))
-- 	first:add(ReLU())
-- 	first:add(Max(2,2,2,2))    
-- 	--model:add(nn.SpatialContrastiveNormalization(100,normkernel))

-- 	conv = nn.Sequential()
-- 	local second = Max(2,2,2,2)
-- 	local parallel = nn.ConcatTable()
-- 	parallel:add(first)
-- 	parallel:add(second)
-- 	conv:add(parallel)
-- 	conv:add(nn.JoinTable(1,3))

-- 	return conv
-- end


-- local get_network_multiscale = function()
local cnn = nn.Sequential()

  -- TODO use one of the improved architectures

  cnn:add(nn.SpatialConvolutionMM(3,100,5,5,1,1,2))
    cnn:add(nn.ReLU())

  cnn:add(nn.SpatialMaxPooling( 2, 2, 2, 2))
  -- cnn:add(nn.SpatialContrastiveNormalization(12, torch.Tensor(4,4):fill(1)))

  local branch = nn.ConcatTable()
  local branch_1 = nn.Sequential()
  branch_1:add(nn.SpatialConvolutionMM(100,150,5,5,1,1,2))
    cnn:add(nn.ReLU())

  branch_1:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  local branch_2 = nn.SpatialMaxPooling( 2, 2, 2, 2)

  branch:add(branch_1)
  branch:add(branch_2)

  cnn:add(branch)

  cnn:add(nn.JoinTable(1,3))

  cnn:add(nn.Reshape(250*12*12))
  cnn:add(nn.Linear(250*12*12, 10000))
  cnn:add(nn.ReLU())
  cnn:add(nn.Linear(10000, 100))
  cnn:add(nn.ReLU())
  cnn:add(nn.Linear(100, 43))

-- -- end

-- local conv1 = getconv(3, 100)
-- local conv2 = getconv(100, 150)
-- local model = nn.Sequential()
-- print(conv1)
-- print(conv2)
-- model:add(Max(2,2,2,2))
-- model:add(conv1)
-- -- model:add(conv2)

-- -- model:add(View(2250))
-- -- model:add(Linear(2250, 300))
-- -- model:add(ReLU())
-- -- model:add(Linear(300, 43))
-- print(cnn)
input = torch.Tensor(3,48,48)                        
out = cnn:forward(input)                        
print(out:size())
return cnn
