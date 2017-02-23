require "torch"
require "nn"
require "nngraph"
require "util"
require "optim"
require "lfs"

require "util.plot"
require "util.data"
require "util.misc"
require "external.hungarian"

-- model and data paths
seq_name 	= "train/KITTI-13"
model_bin 	= "bin/rnnTracker_r300_l1_n1_m1_d4.t7"

-- f***ing global variables used in other files
opt = {
	max_n=1,
	max_m=1,
	rnn_size=300, -- whatever this is
	num_layers=1,
	model="rnn",
	length=50, -- this is max number of frames!
	det_fail=.1,
	det_false=.2,
	state_dim=4,
	mini_batch_size=1,
	gpuid=-1,
	opencl=0,
	verbose=2,
	seed=12,
	real_data=1
}
opt.temp_win = opt.length
imSizes = {
	Synth={imH=1000,imW=1000},
	[seq_name]={imH=375,imW=1242}
}
miniBatchSize = 1
stateDim = opt.state_dim
maxTargets = 1
maxDets = 1
nClasses = maxDets + 1

-- torch configuration
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)

-- load the model
checkpoint = torch.load(model_bin)
protos = checkpoint.protos
protos.rnn:evaluate()

-- get initial state of cell/hidden states
init_state = {}
for L=1,opt.num_layers do
	local h_init = torch.zeros(1, opt.rnn_size)
	table.insert(init_state, h_init:clone())
end

-- read data
realTracksTab, realDetsTab, realLabTab, realExTab, realDetExTab, realSeqNames = prepareData('real', {seq_name}, {seq_name},  true)


