local nn = require 'nn'
local Convolution = nn.SpatialConvolutionMM
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear

local cnn = nn.Sequential()
  cnn:add(nn.SpatialConvolutionMM(3,50,5,5,1,1,2))
  cnn:add(nn.ReLU())
  cnn:add(nn.SpatialMaxPooling( 2, 2, 2, 2))
  local branch = nn.ConcatTable()
  local branch_1 = nn.Sequential()
  branch_1:add(nn.SpatialConvolutionMM(50,100,5,5,1,1,2))
  branch_1:add(nn.ReLU())
  branch_1:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  local branch_2 = nn.SpatialMaxPooling( 2, 2, 2, 2)
  branch:add(branch_1)
  branch:add(branch_2)
  cnn:add(branch)
  cnn:add(nn.JoinTable(1,3))
  cnn:add(nn.SpatialConvolutionMM(150,200,5,5,1,1,2))
  cnn:add(nn.ReLU())
  cnn:add(nn.SpatialMaxPooling( 2, 2, 2, 2))
  cnn:add(nn.SpatialConvolutionMM(200,400,5,5,1,1,2))
  cnn:add(nn.ReLU())
  cnn:add(nn.SpatialMaxPooling( 2, 2, 2, 2))
  cnn:add(nn.Reshape(400*3*3))
  cnn:add(nn.ReLU())
  cnn:add(nn.Linear(400*3*3, 1000))
  cnn:add(nn.ReLU())
  cnn:add(nn.Dropout(0.5))  
  cnn:add(nn.Linear(1000, 100))
  cnn:add(nn.ReLU())
  cnn:add(nn.Linear(100, 43))
  cnn:add(nn.LogSoftMax())
print(cnn)
input = torch.Tensor(3,48,48)                        
out = cnn:forward(input)                        
print(out:size())
return cnn
