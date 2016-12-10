require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
require 'cunn'
require 'cudnn' -- faster convolutions
require 'cutorch'

--[[
--  Hint:  Plot as much as you can.
--  Look into torch wiki for packages that can help you plot.
--]]

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 32, 32
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

torch.setdefaulttensortype('torch.DoubleTensor')

-- torch.setnumthreads(1)
torch.manualSeed(opt.manualSeed)
-- cutorch.manualSeedAll(opt.manualSeed)

function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

--[[
-- Hint:  Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing
-- information by resizing bigger images to a smaller size?
--]]
function subtractMean(img)
    return img - torch.mean(img)
end

function yuv(img)
    return image.rgb2yuv(img)
end

function transformInput(inp)
    f = tnt.transform.compose{
        [1] = resize,
	[2] = subtractMean,
	[3] = yuv
    }
    return f(inp)
end

function getTrainSample(dataset, idx)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file))
end

function getTrainLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file))
end



local train_dataset = {}
train_dataset.nbr_elements = 0
function train_dataset:size() return train_dataset.nbr_elements end
local trainData = torch.load(DATA_PATH..'train.t7')
local testData = torch.load(DATA_PATH..'test.t7')

function generateDataWithoutJitter() 
    for i =1,trainData:size(1) do
        image_data =  getTrainSample(trainData, i)
        image_label = getTrainLabel(trainData, i)
        train_dataset.nbr_elements = train_dataset.nbr_elements + 1
        train_dataset[train_dataset.nbr_elements] = {image_data, image_label}
    end
end
function jitter()

    local n = train_dataset:size()
    for idx = 1,n do
        local original_image = train_dataset[idx][1]

        local target = train_dataset[idx][2]
        for i=1,2 do
            local rand_angle = (torch.randn(1)*15*3.14/180)[1]
            local rand_position_x = (torch.randn(1)*2)[1]
            local rand_position_y = (torch.randn(1)*2)[1]
            image_data = original_image:clone()
            image_data = image.rotate(image_data, rand_angle)
            image_data = image.translate(image_data, rand_position_x, rand_position_y)
           

            train_dataset.nbr_elements = train_dataset.nbr_elements + 1
            train_dataset[train_dataset.nbr_elements] = {image_data, target}
        end
    end          
end

generateDataWithoutJitter()
jitter()

function getIterator(dataset)
    --[[
    -- Hint:  Use ParallelIterator for using multiple CPU cores
    --]]
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
        }
    }
end
trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',
    --[[
    --  Hint:  Use a resampling strategy that keeps the 
    --  class distribution even during initial training epochs 
    --  and then slowly converges to the actual distribution 
    --  in later stages of training.
    --]]
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, train_dataset:size(1)):long(),
            load = function(idx)
                return {
                    input =  train_dataset[idx][1],
                    target = train_dataset[idx][2]
                }
            end

        }
    }
}

testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            sampleId = torch.LongTensor{testData[idx][1]}
        }
    end
}


--[[
-- Hint:  Use :cuda to convert your model to use GPUs
--]]
local model = require("models/".. opt.model)
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1
model = model:cuda()
criterion = criterion:cuda()
-- print(model)

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end

--[[
-- Hint:  Use onSample function to convert to
--        cuda tensor for using GPU
--]]
local input  = torch.CudaTensor()
local target = torch.CudaTensor()
engine.hooks.onSample = function(state)
  input:resize(
      state.sample.input:size()
  ):copy(state.sample.input)
  state.sample.input  = input
  if state.sample.target then
      target:resize( state.sample.target:size()):copy(state.sample.target)
      state.sample.target = target
  end
end


engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
end

local epoch = 1
local lr = 0.01
print(model)
while epoch <= opt.nEpochs do
    trainDataset:select('train')
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = lr,
            momentum = opt.momentum,
	    weightDecay = 1e-4,
	learningRateDecay = .0001
        }
    }

    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }
    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

local submission = assert(io.open(opt.logDir .. "/submission_jitter_wd50.csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
    local fileNames  = state.sample.sampleId
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

print("The End!")
