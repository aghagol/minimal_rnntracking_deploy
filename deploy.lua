require "torch"
require "nn"
require "nngraph"
require "util"
require "optim"
require "lfs"
-- local util
require "util.plot"
require "util.data"
require "util.misc"
require "external.hungarian"
require "auxBFPAR"
------------------------------------------------------------
-- define some helper functions:
function getLocFromFullState(state)
  if opt.vel~=0 then
    ind = torch.linspace(1,fullStateDim-1,fullStateDim/2):long()
    state = state:index(3,ind)
  end
  return state
end
------------------------------------------------------------
-- model and data paths:
seq_name    = "train/KITTI-13"
model_bin   = "bin/rnnTracker_r300_l1_n1_m1_d4.t7"
------------------------------------------------------------
-- global variables:
opt = {
  model_split=0,
  dalink=1,
  max_n=1,
  freq_reshuffle=0,
  init_from="",
  plot_every=10000,
  statePredIndex2=3,
  det_false=0.2,
  rnn_size=300,
  lrng_rate=3e-05,
  dropout=0.03,
  random_epoch=0,
  seed=12,
  kappa=1,
  miss_thr=0.05,
  synth_valid=1,
  verbose=2,
  exPredIndex=4,
  det_thr=0.03,
  order_dets=0,
  lambda=1,
  einp=1,
  use_gt_input=0,
  xi=1,
  norm_mean=1,
  exSmoothPredIndex=5,
  real_data=1,
  fixed_n=1,
  nu=0.5,
  max_epochs=100000,
  batch_size=1,
  reshuffle_dets=0,
  pert_training=0,
  det_noise=0.01,
  synth_training=1,
  forget_bias=1,
  eval_conf="config/XPOS.ini",
  modelParams={
    [1]="model_index",
    [2]="rnn_size",
    [3]="num_layers",
    [4]="max_n",
    [5]="max_m",
    [6]="state_dim" },
  lrng_rate_decay_after=10000,
  linp=0,
  trim_tracks=1,
  use_da_input=4,
  training_file="",
  ex_thr=0.5,
  loss_type=1,
  full_set=1,
  opencl=0,
  statePredIndex=2,
  detlink=1,
  one_hot=1,
  max_m=1,
  gpuid=-1,
  max_nf=0,
  decay_rate=0.97,
  suppress_x=0,
  rand_par_rng=0.08,
  config="config/0314Ea-3.txt",
  mini_batch_size=1,
  dummy_weight=1,
  grad_clip=0.1,
  save_plots=0,
  dummy_det_val=0,
  print_every=1000,
  norm_std=-1,
  num_layers=1,
  eval_mot15=1,
  model_index=3,
  mu=0,
  det_fail=0.1,
  vel=0,
  profiler=0,
  real_dets=1,
  daPredIndex=4,
  temp_win=50,
  model="rnn",
  eval_val_every=10000,
  state_dim=4,
  lrng_rate_decay=0.95
}
imSizes = {
  Synth={imH=1000,imW=1000},
  [seq_name]={imH=375,imW=1242}
}
updLoss = opt.kappa ~= 0
predLoss = opt.lambda ~= 0
daLoss = opt.mu ~= 0
exVar = opt.nu ~= 0
miniBatchSize = 1
stateDim = opt.state_dim
fullStateDim = stateDim
maxTargets = 1
maxDets = 20
maxAllDets = maxDets
nClasses = 2 -- what is this?
maxAllTargets = 5 -- this para rules them all
xSize = stateDim*maxTargets
dSize = stateDim*maxDets
T = opt.temp_win - opt.batch_size
------------------------------------------------------------
-- torch configuration:
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
------------------------------------------------------------
-- load the model:
checkpoint = torch.load(model_bin)
protos = checkpoint.protos
protos.rnn:evaluate() -- TESTING mode
------------------------------------------------------------
-- read data:
AllTracksTab, AllDetsTab, AllLabTab, AllExTab, AllDetExTab  = prepareData('real', {seq_name}, {seq_name},  true)
------------------------------------------------------------
-- some more fucking global variables:
realTracksTab = AllTracksTab
realDetsTab = AllDetsTab
realLabTab = AllLabTab
realExTab = AllExTab
realDetExTab = AllDetExTab
realSeqNames = {{seq_name}}

tracks = realTracksTab[1]
detections = realDetsTab[1]:clone()
labels = realLabTab[1]:clone() -- only for debugging
exlabels = realExTab[1]:clone()
detexlabels = realDetExTab[1]:clone()
alldetections = AllDetsTab[1]:clone()
alldetexlabels = AllDetExTab[1]:clone()
------------------------------------------------------------
-- get initial state of cell/hidden states
init_state = {}
for L=1,opt.num_layers do
  local h_init = torch.zeros(1, opt.rnn_size)
  table.insert(init_state, h_init:clone())
end
rnn_state = {[0] = init_state}

Allrnn_states = {}
for tar=1,maxAllTargets do
  Allrnn_states[tar] = {}
  for t=1,T do Allrnn_states[tar][t] = {} end
  Allrnn_states[tar][0] = init_state
end

protosClones = {}
for tar=1,maxAllTargets do
  table.insert(protosClones, protos.rnn:clone())
end
------------------------------------------------------------
-- predict:
predictions = {}
Allpredictions = {}

for tar=1,maxAllTargets do
  Allpredictions[tar] = {}
  for t=1,T do
    Allpredictions[tar][t] = {}
    for k=1, 6 do
      Allpredictions[tar][t][k] = {}
    end
  end
end

for t=1,T do
  Allrnninps, Allrnn_states = getRNNInput(t, Allrnn_states, Allpredictions)

  allLst={}
  for tar=1,maxAllTargets do
    rnninp = {}
    for i=1,#Allrnninps[tar] do table.insert(rnninp, Allrnninps[tar][i]:clone()) end
    lst =protosClones[tar]:forward(rnninp)  -- do one forward tick
    table.insert(allLst, lst)
  end

  tmpRNNstates = {}
  for tar=1,maxAllTargets do
    for k,v in pairs(allLst[tar]) do Allpredictions[tar][t][k] = v:clone() end -- deep copy
    rnn_state = {}
    for i=1,#init_state do
        table.insert(rnn_state, allLst[tar][i])
    end -- extract the state, without output
    table.insert(tmpRNNstates, rnn_state)
  end

  for tar=1,maxAllTargets do
    Allrnn_states[tar][t] = deepcopy(tmpRNNstates[tar])
  end

  predictions = moveState(Allpredictions, t)
end
------------------------------------------------------------
-- compute tracks:
AllpredTracks, AllpredTracks2, allpredDA, allpredEx = decode(Allpredictions)
predTracks = torch.zeros(1,T,stateDim)
-- predDA = torch.zeros(1,T,nClasses)
predEx = torch.zeros(1,T,1)
for tar=1,maxAllTargets do
  predTracks = predTracks:cat(AllpredTracks[tar], 1)
  -- predDA = predDA:cat(allpredDA[tar], 1)
  predEx = predEx:cat(allpredEx[tar], 1)
end
predTracks = predTracks:sub(2,-1)
-- predDA = predDA:sub(2,-1)
predEx = predEx:sub(2,-1)
-- predDA = allPredDA:clone() -- why???
predTracks = getLocFromFullState(predTracks)
predExBin = getPredEx(predEx)
-- predLab = getLabelsFromLL(predDA, false)
finalTracks = predTracks:clone()


AllunnormDetsTab={}
for k,v in pairs(AllDetsTab) do AllunnormDetsTab[k] = AllDetsTab[k]:clone() end

N,F,D = getDataSize(finalTracks)
for t=1,F do -- what is this for???
  for i=1,N do
    -- if predExBin[i][t] == 0 then finalTracks[i][t] = 0  end
  end
end

finalTracksTab={}
table.insert(finalTracksTab, finalTracks)

if realData then
  finalTracksTab = normalizeData(finalTracksTab, AllunnormDetsTab, true, maxAllTargets, maxAllDets, realSeqNames)
end

writeResTensor = finalTracksTab[1]
-- smash existence probability as dim = 5
writeResTensor = writeResTensor:cat(predEx, 3)
-- remove false tracks
writeResTensor = writeResTensor:sub(1,maxAllTargets)
writeTXT(writeResTensor, 'out/out.txt')
------------------------------------------------------------
-- fix the tracks using the existence probabilities:
fixedTracks = torch.zeros(1,F,D)
fixedEx = torch.zeros(1,F):int()
for tar=1,N do
  started, finished=0,0
  for t=1,F do
    if (t==1 and predExBin[tar][t]==1) or (t>1 and predExBin[tar][t]==1 and predExBin[tar][t-1]==0) then
      started=t
    end
    if (t==F and predExBin[tar][t]==1) or (t<F and predExBin[tar][t]==1 and predExBin[tar][t+1]==0) then
      finished=t
    end
    if started>0 and finished>0 then
      tmpTrack = torch.zeros(1,F,D)
      tmpTrack[{{1},{started,finished},{}}] = finalTracks[{{tar},{started,finished},{}}]
      fixedTracks=fixedTracks:cat(tmpTrack,1)

      started=0
      finished=0
    end
  end
end
if fixedTracks:size(1)>1 then
  fixedTracks=fixedTracks:sub(2,-1)
else
  fixedTracks=finalTracks:clone()
end
------------------------------------------------------------
-- plot tracks over detections:
trueDets = alldetexlabels:reshape(maxAllDets, opt.temp_win, 1):expand(maxAllDets, opt.temp_win, stateDim)
realDets = alldetections:clone():cmul(trueDets:float())

plotTab = {}
plotTab = getDetectionsPlotTab(realDets, plotTab, nil, nil)
plotTab = getTrackPlotTab(finalTracks, plotTab, 2,nil,nil,1)
-- plotTab = getTrackPlotTab(fixedTracks, plotTab, 2,nil,nil,1)

plot(plotTab, nil, 'out/out.png', nil, true)
------------------------------------------------------------


























