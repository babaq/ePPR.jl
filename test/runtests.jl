using Test, ePPR, FileIO, BenchmarkTools

## Simulated Data
# x: Natural Images
# y: Simulated Neuron with MFR=0.56, MIF=4.26
simdata=load(joinpath(@__DIR__,"simdata.jld2"));x=simdata["x"];y=simdata["y"];imagesize=simdata["imagesize"]
logdir = joinpath(@__DIR__,"log")

# ePPR with Linear Time Interaction
hp = ePPRHyperParams(imagesize...)
log = ePPRLog(debug=true,plot=true)
model,models = epprcv(x,y,hp,log)
plotmodel(model,hp)

# ePPR with Non-Linear Time Interaction
hp = ePPRHyperParams(imagesize...,ndelay=3,lambda=15,nft=[6])
log = ePPRLog(debug=true,plot=true)
model,models = epprcv(x,y,hp,log)
plotmodel(model,hp)

## ePPR with Hyper Parameter Search
hp = ePPRHyperParams(imagesize...,lambda=64,lambdascale=0.5)
log = ePPRLog(debug=true,plot=true,dir=logdir)
model,models = epprhypercv(x,y,hp,log)
plotmodel(model,hp)

## Data Recorded from an Anesthetized Cat
# x: Natural Images each present 40ms
# y1: Simple Cell Spike Sum
# y2: Complex Cell Spike Sum
data=load(joinpath(@__DIR__,"data.jld2"));x=data["x"];y1=data["y1"];y2=data["y2"];imagesize=data["imagesize"]

# ePPR with Linear Time Interaction
hp = ePPRHyperParams(imagesize...,lambda=15,blankcolor=mean(x))
log = ePPRLog(debug=true,plot=true)
model,models = epprcv(x,y1,hp,log)
plotmodel(model,hp)

# ePPR with Non-Linear Time Interaction
hp = ePPRHyperParams(imagesize...,ndelay=3,lambda=15,nft=[6],blankcolor=mean(x))
log = ePPRLog(debug=true,plot=true)
model,models = epprcv(x,y2,hp,log)
plotmodel(model,hp)

## Benchmark
hp = ePPRHyperParams(imagesize...)
@btime model,models = epprcv($x,$y1,$hp) samples=1 # ~19s
