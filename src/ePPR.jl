__precompile__(false)
module ePPR

import Base.length,Base.push!,Base.deleteat!
export ePPRDebugOptions,DebugNone,DebugBasic,DebugFull,DebugVisual,
delaywindowpool,delaywindowpooloperator,delaywindowpoolblankimage,cvpartitionindex!,getinitialalpha,refitmodelbetas!,laplacian2dmatrix,
ePPRModel,getterm,setterm!,clean!,ePPRHyperParams,ePPRCrossValidation,
eppr,epprcv,epprhypercv,cvmodel,forwardstepwise,refitmodel!,backwardstepwise,dropleastimportantterm!,dropterm!,lossfun,fitnewterm,newtontrustregion

using LinearAlgebra,Statistics,GLM,Roots,HypothesisTests,RCall,Dierckx,Plots
R"library('MASS')"
pyplot();clibrary(:colorcet)

const DebugNone=0
const DebugBasic=1
const DebugFull=2
const DebugVisual=3
Base.@kwdef mutable struct ePPRDebugOptions
    level::Int = DebugNone
    logdir = nothing
    logio = nothing
end

function (debug::ePPRDebugOptions)(msg;level::Int=DebugBasic,log="ePPR.log",once=false)
    if debug.level >= level
        if debug.logio==nothing
            if debug.logdir==nothing
                io=stdout
            else
                !isdir(debug.logdir) && mkpath(debug.logdir)
                io = open(joinpath(debug.logdir,log),"a")
                debug.logio = io
            end
        else
            io=debug.logio
        end
        println(io,msg)
        flush(io)
    end
    once && debug.logio!=nothing && close(debug.logio)
end

function (debug::ePPRDebugOptions)(msg::Plots.Plot;level::Int=DebugBasic,log="ePPRModel")
    if debug.level >= level
        if debug.logdir==nothing
            display(msg)
        else
            !isdir(debug.logdir) && mkpath(debug.logdir)
            png(msg,joinpath(debug.logdir,log))
        end
    end
end

"""
``\\hat{y}_i=\\bar{y}+\\sum_{d=0}^D\\sum_{m=1}^{M_d}\\beta_{m,d}\\phi_{m,d}(\\alpha_{m,d}^Tx_{i-d})``
with ``\\frac{1}{n}\\sum_{i=1}^n\\phi_{m,d}(\\alpha_{m,d}^Tx_{i-d})=0``, ``\\frac{1}{n}\\sum_{i=1}^n\\phi_{m,d}^2(\\alpha_{m,d}^Tx_{i-d})=1``
"""
Base.@kwdef mutable struct ePPRModel
    "𝑦̄"
    ymean::Float64 = 0.0
    "vector of β for each term"
    beta::Vector{Float64} = []
    "vector of Φ for each term"
    phi::Vector = []
    "vector of α for each term"
    alpha::Vector{Vector{Float64}} = []
    "vector of (temporal, spatial) index for each term"
    index::Vector = []
    "vector of ``\\phi_{m,d}(\\alpha_{m,d}^TX_{-d})`` for each term"
    phivalues::Vector{Vector{Float64}} = []
    "vector of trustregionsize for each term"
    trustregionsize::Vector{Float64} = []
    "γ"
    residuals::Vector{Float64} = []
end
length(m::ePPRModel)=length(m.beta)
(m::ePPRModel)() = m.ymean.+dropdims(sum(cat((m.beta.*m.phivalues)...,dims=2),dims=2),dims=2)
function (m::ePPRModel)(x::Matrix,xpast::Union{Matrix,Nothing})
    p = fill(m.ymean,size(x,1))
    for t in 1:length(m)
        j = m.index[t][1]
        tx = j>0 ? [xpast[end-(j-1):end,:];x[1:end-j,:]] : x
        p .+= m.beta[t].*m.phi[t](tx*m.alpha[t])
    end
    p
end
function deleteat!(model::ePPRModel,i::Integer)
    deleteat!(model.beta,i)
    deleteat!(model.phi,i)
    deleteat!(model.alpha,i)
    deleteat!(model.index,i)
    deleteat!(model.phivalues,i)
    deleteat!(model.trustregionsize,i)
end
function push!(model::ePPRModel,β::Float64,Φ,α::Vector{Float64},index,Φvs::Vector{Float64},trustregionsize::Float64=1.0)
    push!(model.beta,β)
    push!(model.phi,Φ)
    push!(model.alpha,α)
    push!(model.index,index)
    push!(model.phivalues,Φvs)
    push!(model.trustregionsize,trustregionsize)
end
function getterm(model::ePPRModel,i::Integer)
    return model.beta[i],model.phi[i],model.alpha[i],model.index[i],model.phivalues[i],model.trustregionsize[i]
end
function setterm!(model::ePPRModel,i::Integer,β::Float64,Φ,α::Vector{Float64},index,Φvs::Vector{Float64},trustregionsize::Float64=1.0)
    model.beta[i]=β
    model.phi[i]=Φ
    model.alpha[i]=α
    model.index[i]=index
    model.phivalues[i]=Φvs
    model.trustregionsize[i]=trustregionsize
end
clean!(model)=model
function clean!(model::ePPRModel)
    model.phivalues=[]
    model.trustregionsize=[]
    model.residuals=[]
    return model
end

Base.@kwdef mutable struct ePPRCrossValidation
    trainpercent::Float64 = 0.88
    trainfold::Int = 5
    testfold::Int = 8
    traintestfold::Int = 8
    trainsets = []
    tests = []
    trainsetindex::Int = 1
    h0level::Float64 = 0.05
    h1level::Float64 = 0.05
    traintestcors = []
    modeltraintestcor = []
    modeltestcor = []
end

"""
Hyper Parameters for ePPR
"""
Base.@kwdef mutable struct ePPRHyperParams
    """memory size to pool for nonlinear time interaction, ndelay=1 for linear time interaction.
    only first delay terms in `nft` is used for nonlinear time interaction."""
    ndelay::Int = 1
    "number of forward terms for each delay. [3, 2, 1] means 3 spatial terms for delay 0, 2 for delay 1, 1 for delay 2"
    nft::Vector{Int} = [3,3,3]
    "penalization parameter λ"
    lambda::Float64 = 30
    "Φ Spline degree of freedom"
    phidf::Int = 5
    "minimum number of backward terms"
    mnbt::Int = 1
    "whether to fit all spatial terms before moving to next temporal delay"
    spatialtermfirst::Bool = true
    "α priori for penalization"
    alphapenaltyoperator = []
    "`(lossₒ-lossₙ)/lossₒ`, forward converge rate threshold to decide a saturated iteration"
    forwardconvergerate::Float64 = 0.01
    "`(lossₒ-lossₙ)/lossₒ`, refit converge rate threshold to decide a saturated iteration"
    refitconvergerate::Float64 = 0.001
    "number of consecutive saturated iterations to decide a solution"
    nsaturatediteration::Int = 2
    "maximum number of iterations to fit a new term"
    newtermmaxiteration::Int = 100
    "initial size of trust region"
    trustregioninitsize::Float64 = 1
    "maximum size of trust region"
    trustregionmaxsize::Float64 = 1000
    "η of trust region"
    trustregioneta::Float64 = 0.2
    "maximum iterations of trust region"
    trustregionmaxiteration::Int = 1000
    "row vector of blank image"
    blankimage = []
    "dimension of image"
    imagesize = []
    "drop term index between backward models"
    droptermindex = []
    "ePPR Cross Validation"
    cv::ePPRCrossValidation = ePPRCrossValidation()
    "Valid Image Region"
    xindex::Vector{Int} = Int[]
    "maximum iterations of hyperparameter search"
    hypermaxiteration = 50
end
function ePPRHyperParams(nrow::Int,ncol::Int;xindex::Vector{Int}=Int[],ndelay::Int=1,blankcolor=0.5)
    hp=ePPRHyperParams()
    hp.imagesize = (nrow,ncol)
    hp.xindex=xindex
    hp.ndelay=ndelay
    hp.blankimage = delaywindowpoolblankimage(nrow,ncol,xindex,ndelay,blankcolor)
    hp.alphapenaltyoperator = delaywindowpooloperator(laplacian2dmatrix(nrow,ncol),xindex,ndelay)
    return hp
end

function delaywindowpool(x::Matrix,xindex::Vector{Int}=Int[],ndelay::Int=1,blankcolor=0.5,debug::ePPRDebugOptions=ePPRDebugOptions())
    vx = isempty(xindex) ? x : x[:,xindex]
    ndelay<=1 && return vx

    debug("Nonlinear Time Interaction, pool x[i-$(ndelay-1):i] together ...")
    nc=size(vx,2);dwvx=vx
    for j in 1:ndelay-1
        dwvx = [dwvx [fill(blankcolor,j,nc);vx[1:end-j,:]]]
    end
    return dwvx
end
delaywindowpool(x::Matrix,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())=delaywindowpool(x,hp.xindex,hp.ndelay,hp.blankimage[1],debug)

function delaywindowpooloperator(spatialoperator::Matrix,xindex::Vector{Int}=Int[],ndelay::Int=1)
    vso = isempty(xindex) ? spatialoperator : spatialoperator[xindex,xindex]
    ndelay<=1 && return vso

    nr,nc=size(vso)
    dwvo = zeros(ndelay*nr,ndelay*nc)
    for j in 0:ndelay-1
        dwvo[(1:nr).+j*nr, (1:nc).+j*nc] = vso
    end
    return dwvo
end

function delaywindowpoolblankimage(nrow::Int,ncol::Int,xindex::Vector{Int}=Int[],ndelay::Int=1,blankcolor=0.5)
    pn = isempty(xindex) ? nrow*ncol : length(xindex)
    fill(blankcolor,1,pn*ndelay)
end

function getxpast(maxmemory,xi,x,blankimage)
    if maxmemory == 0
        return nothing
    end
    pi = (xi[1]-maxmemory):(xi[1]-1)
    cat(map(i->i<=0 ? blankimage : x[i:i,:],pi)...,dims=1)
end

"""
Data partition for cross validation

cv: cross validation
n: sample number
"""
function cvpartitionindex!(cv::ePPRCrossValidation,n::Int,debug::ePPRDebugOptions=ePPRDebugOptions())
    ntrain = cv.trainpercent*n
    ntrainfold = ntrain/cv.trainfold
    ntraintestfold = Int(floor(ntrainfold/cv.traintestfold))
    ntrainfold = ntraintestfold*cv.traintestfold
    ntrain = ntrainfold*cv.trainfold
    trainsets=[]
    for tf in 0:cv.trainfold-1
        traintest = Any[tf*ntrainfold .+ (1:ntraintestfold) .+ ttf*ntraintestfold for ttf in 0:cv.traintestfold-1]
        train = setdiff(1:ntrain,tf*ntrainfold .+ (1:ntrainfold))
        push!(trainsets,(train=train,traintest=traintest))
    end
    ntestfold = Int(floor((n-ntrain)/cv.testfold))
    tests = Any[ntrain .+ (1:ntestfold) .+ tf*ntestfold for tf in 0:cv.testfold-1]
    debug("Cross Validation Data Partition: n = $n, ntrain = $ntrain in $(cv.trainfold)-fold, ntrainfold = $ntrainfold in $(cv.traintestfold)-fold, ntest = $(ntestfold*cv.testfold) in $(cv.testfold)-fold")
    cv.trainsets=trainsets;cv.tests=tests
    return cv
end

function cvmodel(models::Vector{ePPRModel},x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug("ePPR Models Cross Validation ...")
    train = hp.cv.trainsets[hp.cv.trainsetindex].train;traintest = hp.cv.trainsets[hp.cv.trainsetindex].traintest;test=hp.cv.tests
    # response and model predication
    maxmemory = length(hp.nft)-1
    traintestpredications = map(m->map(i->m(x[i,:],getxpast(maxmemory,i,x,hp.blankimage)),traintest),models)
    traintestys = map(i->y[i],traintest)
    # correlation between response and predication
    traintestcors = map(mps->cor.(traintestys,mps),traintestpredications)
    hp.cv.traintestcors = traintestcors;hp.cv.modeltraintestcor=[];hp.cv.modeltestcor=[]
    debug.level >= DebugVisual && debug(plotcor(models,traintestcors),log="Models_Goodness")
    # find the model no worse than models with more terms, and better than models with less terms
    mi=0;nmodel=length(models)
    for rm in 1:nmodel
        moretermp = [pvalue(SignedRankTest(traintestcors[rm],traintestcors[m]),tail=:left) for m in rm+1:nmodel]
        if rm==1 && (1==nmodel || all(moretermp .> hp.cv.h0level))
            mi=rm;break
        end
        if rm==nmodel || all(moretermp .> hp.cv.h0level)
            lesstermp = [pvalue(SignedRankTest(traintestcors[m],traintestcors[rm]),tail=:left) for m in 1:rm-1]
            if all(lesstermp .< hp.cv.h1level)
                mi=rm;break
            end
        end
    end
    if mi==0
        debug("No model not worse than models with more terms, and better than models with less terms.")
        return nothing
    end
    model = deepcopy(models[mi])
    debug("$(mi)th model with $(length(model)) terms is chosen.")

    # find drop terms that do not improve model predication
    droptermp = [pvalue(SignedRankTest(traintestcors[m-1],traintestcors[m]),tail=:left) for m in 2:nmodel]
    notimprove = findall(droptermp .> hp.cv.h0level)
    # find drop term models with change level(zero correlation) predication
    modelp = [pvalue(SignedRankTest(traintestcors[m]),tail=:both) for m in 2:nmodel]
    notpredictive = findall(modelp .> hp.cv.h0level)

    poorterm = hp.droptermindex[union(notimprove,notpredictive)]
    # spurious terms in the selected model
    spuriousterm = findall(in(poorterm),model.index)
    if !isempty(spuriousterm)
        debug("Model drop spurious term: $(model.index[spuriousterm]).")
        foreach(i->deleteat!(model,i),sort(spuriousterm,rev=true))
    end
    length(model)==0 && return nothing
    model = eppr(model,x[train,:],y[train],hp,debug)
    hp.cv.modeltraintestcor = map(i->cor(y[i],model(x[i,:],getxpast(maxmemory,i,x,hp.blankimage))),traintest)
    hp.cv.modeltestcor = map(i->cor(y[i],model(x[i,:],getxpast(maxmemory,i,x,hp.blankimage))),test)
    return model
end

function epprcv(x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    n = length(y);n !=size(x,1) && error("Length of x and y does not match!")
    cvpartitionindex!(hp.cv,n,debug)
    train = hp.cv.trainsets[hp.cv.trainsetindex].train
    px = delaywindowpool(x,hp,debug)
    models = eppr(px[train,:],y[train],hp,debug)
    model = cvmodel(models,px,y,hp,debug)
    debug("Cross Validated ePPR Done.",once=true)
    return model,models
end

function epprhypercv(x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    n = length(y);n !=size(x,1) && error("Length of x and y does not match!")
    cvpartitionindex!(hp.cv,n,debug)
    train = hp.cv.trainsets[hp.cv.trainsetindex].train
    px = delaywindowpool(x,hp,debug)

    hi=0;hypermodel=[];hypermodels=[];λs=[];modelcors=[];nsaturatediteration=0;maxsaturation=2
    hp.lambda = 2
    for i in 1:hp.hypermaxiteration
        debug("HyperParameter Search: λ = $(hp.lambda) ...")
        models = eppr(px[train,:],y[train],hp,debug)
        model = cvmodel(models,px,y,hp,debug)
        debug("Cross Validated ePPR Done.")
        if !isnothing(model)
            push!(hypermodel,model);push!(hypermodels,models);push!(λs,hp.lambda);push!(modelcors,[hp.cv.modeltraintestcor;hp.cv.modeltestcor])
            if length(modelcors)==1
                hp.lambda *=2
            else
                improvep = pvalue(SignedRankTest(modelcors[end-1],modelcors[end]),tail=:left)
                if improvep < hp.cv.h1level
                    nsaturatediteration=0
                    hp.lambda *= 2.5
                else
                    impirep = pvalue(SignedRankTest(modelcors[end-1],modelcors[end]),tail=:right)
                    if impirep < hp.cv.h1level
                        hi=length(hypermodel)-1;break
                    else
                        nsaturatediteration+=1
                        if nsaturatediteration>=maxsaturation
                            hi=length(hypermodel);break
                        else
                            hp.lambda *= 1.5
                        end
                    end
                end
            end
        else
            hp.lambda *=2
        end
    end
    debug.level >= DebugVisual && !isempty(modelcors) && debug(plotcor(λs,modelcors,xlabel="λ"),log="λ_Models_Goodness")
    if hi==0
        if length(hypermodel)>0
            _,hi=findmax(mean.(modelcors))
        else
            return nothing,[]
        end
    end
    debug("HyperParameter Search Done.",once=true)
    hp.lambda = λs[hi]
    return hypermodel[hi],hypermodels[hi]
end

"""
extended Projection Pursuit Regression
by minimizing ``f=\\sum_{i=1}^N(y_i-\\hat{y}(x_i))^2+\\lambda\\sum_{d=0}^D\\sum_{m=1}^{M_d}\\Vert{L\\alpha_{m,d}}\\Vert^2``

x: matrix with one image per row
y: vector of responses
hp: hyper parameters
debug: debug options
"""
function eppr(x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    model = forwardstepwise(x,y,hp,debug)
    model = refitmodel!(model,x,y,hp,debug)
    models = backwardstepwise(model,x,y,hp,debug)
end
function eppr(model::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    refitmodelbetas!(forwardstepwise(model,x,y,hp,debug),y,debug)
end

function forwardstepwise(m::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    ti = map(i->i.t,m.index);si = map(i->i.s,m.index)
    ut = sort(unique(ti));uts=Dict(t=>si[ti.==t] for t in ut);utsn=Dict(t=>length(uts[t]) for t in ut)
    debug("ePPR Model Forward Stepwise ...")
    ym = mean(y);model = ePPRModel(ymean=ym);model.residuals=y.-ym
    if hp.spatialtermfirst
        for t in ut
            tx = t>0 ? [repeat(hp.blankimage,outer=(t,1));x[1:end-t,:]] : x
            for s in 1:utsn[t]
                debug("Fit Model (Temporal-$t, Spatial-$s) New Term ...")
                α = normalize(m.alpha[findfirst(==((t=t,s=uts[t][s])),m.index)], 2)
                β,Φ,α,Φvs = fitnewterm(tx,model.residuals,α,hp.phidf,debug)
                model.residuals .-= β*Φvs
                push!(model,β,Φ,α,(t=t,s=s),Φvs)
            end
        end
    else
        for s in 1:maximum(values(utsn)), t in ut
            s>utsn[t] && continue
            debug("Fit Model (Temporal-$t, Spatial-$s) New Term ...")
            tx = t>0 ? [repeat(hp.blankimage,outer=(t,1));x[1:end-t,:]] : x
            α = normalize(m.alpha[findfirst(==((t=t,s=uts[t][s])),m.index)], 2)
            β,Φ,α,Φvs = fitnewterm(tx,model.residuals,α,hp.phidf,debug)
            model.residuals .-= β*Φvs
            push!(model,β,Φ,α,(t=t,s=s),Φvs)
        end
    end
    return model
end

function forwardstepwise(x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug("ePPR Forward Stepwise ...")
    if hp.ndelay>1
        hp.nft=hp.nft[1:1]
    end
    ym = mean(y);model = ePPRModel(ymean=ym);model.residuals=y.-ym
    if hp.spatialtermfirst
        for t in 0:length(hp.nft)-1
            tx = t>0 ? [repeat(hp.blankimage,outer=(t,1));x[1:end-t,:]] : x
            for s in 1:hp.nft[t+1]
                debug("Fit (Temporal-$t, Spatial-$s) New Term ...")
                α = getinitialalpha(tx,model.residuals,debug)
                β,Φ,α,Φvs,trustregionsize = fitnewterm(tx,model.residuals,α,hp,debug)
                model.residuals .-= β*Φvs
                push!(model,β,Φ,α,(t=t,s=s),Φvs,trustregionsize)
            end
        end
    else
        for s in 1:maximum(hp.nft), t in 0:length(hp.nft)-1
            s>hp.nft[t+1] && continue
            debug("Fit (Temporal-$t, Spatial-$s) New Term ...")
            tx = t>0 ? [repeat(hp.blankimage,outer=(t,1));x[1:end-t,:]] : x
            α = getinitialalpha(tx,model.residuals,debug)
            β,Φ,α,Φvs,trustregionsize = fitnewterm(tx,model.residuals,α,hp,debug)
            model.residuals .-= β*Φvs
            push!(model,β,Φ,α,(t=t,s=s),Φvs,trustregionsize)
        end
    end
    return model
end

function refitmodel!(model::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug("ePPR Model Refit ...")
    model = refitmodelbetas!(model,y,debug)
    for i in 1:length(model)
        oldloss = lossfun(model,y,hp)
        oldβ,oldΦ,oldα,index,oldΦvs,oldtrustregionsize = getterm(model,i)
        model.residuals .+= oldβ*oldΦvs

        t = index.t;s=index.s
        tx = t>0 ? [repeat(hp.blankimage,outer=(t,1));x[1:end-t,:]] : x
        debug("Refit (Temporal-$t, Spatial-$s) New Term ...")
        β,Φ,α,Φvs,trustregionsize = fitnewterm(tx,model.residuals,oldα,hp,debug,convergerate=hp.refitconvergerate,trustregionsize=oldtrustregionsize)
        setterm!(model,i,β,Φ,α,index,Φvs,trustregionsize)
        newloss = lossfun(model,y,hp)
        if newloss > oldloss
            debug("Model Loss increased from $oldloss to $newloss. Discard the new term, keep the old one.")
            setterm!(model,i,oldβ,oldΦ,oldα,index,oldΦvs,oldtrustregionsize)
            model.residuals .-= oldβ*oldΦvs
        else
            model.residuals .-= β*Φvs
        end
    end
    return model
end

function backwardstepwise(model::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug("ePPR Backward Stepwise ...")
    debug.level >= DebugVisual && debug(plotmodel(model,hp),log="Model_$(length(model))")
    models=[deepcopy(model)];hp.droptermindex=[]
    for i in length(model)-1:-1:hp.mnbt
        model,dropindex = dropleastimportantterm!(model,debug)
        pushfirst!(hp.droptermindex,dropindex)
        model = refitmodel!(model,x,y,hp,debug)
        debug.level >= DebugVisual && debug(plotmodel(model,hp),log="Model_$(length(model))")
        pushfirst!(models,deepcopy(model))
    end
    return models
end

dropleastimportantterm!(model::ePPRModel,debug::ePPRDebugOptions=ePPRDebugOptions())=dropterm!(model,argmin(abs.(model.beta)),debug)

function dropterm!(model::ePPRModel,i::Integer,debug::ePPRDebugOptions=ePPRDebugOptions())
    dropindex = model.index[i]
    β=model.beta[i]
    debug("Drop Term: (temporal-$(dropindex.t), spatial-$(dropindex.s)) with β: $(β).")
    deleteat!(model,i)
    return model,dropindex
end

lossfun(g::Vector) = 0.5*norm(g,2)^2
"""
Loss function for term
f(α) = sum((r-Φ(x*α)).^2) + λ*norm(hp.alphapenaltyoperator*α,2)^2
"""
lossfun(r::Vector,x::Matrix,α::Vector,Φ,hp::ePPRHyperParams) = lossfun([r-Φ(x*α);sqrt(hp.lambda)*hp.alphapenaltyoperator*α])
"Loss function for model"
function lossfun(model::ePPRModel,y::Vector,hp::ePPRHyperParams)
    modelloss = lossfun(y-model())
    penaltyloss = 0.5*hp.lambda*sum(norm.([hp.alphapenaltyoperator].*model.alpha,2).^2)
    return modelloss + penaltyloss
end

(phi::RObject)(x) = rcopy(R"predict($phi, x=$x)$y")
function fitnewterm(x::Matrix,r::Vector,α::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions();convergerate::Float64=hp.forwardconvergerate,trustregionsize::Float64=hp.trustregioninitsize)
    saturatediteration = 0;phi=nothing;Φvs=nothing;xa=nothing
    for i in 1:hp.newtermmaxiteration
        xa = x*α
        phi = R"smooth.spline(y=$r, x=$xa, df=$(hp.phidf), spar=NULL, cv=FALSE)"
        Φvs = phi(xa)
        f(a) = lossfun(r,x,a,phi,hp)
        gt = r-Φvs;gp = sqrt(hp.lambda)*hp.alphapenaltyoperator*α;g = [gt;gp]
        debug("New Term $(i)th iteration. TermLoss: $(lossfun(gt)), PenaltyLoss: $(lossfun(gp)).")
        # Loss f(α) before trust region
        lossₒ = lossfun(g)
        Φ′vs = rcopy(R"predict($phi, x=$xa, deriv=1)$y")
        gg = [-Φ′vs.*x;sqrt(hp.lambda)*hp.alphapenaltyoperator]'
        f′ = gg*g
        f″ = gg*gg'
        # α and Loss f(α) after trust region
        success,α,lossₙ,trustregionsize = newtontrustregion(f,α,lossₒ,f′,f″,trustregionsize,hp.trustregionmaxsize,hp.trustregioneta,hp.trustregionmaxiteration,debug)
        if !success
            debug("NewtonTrustRegion failed, New Term use initial α.")
            break
        end
        lossₙ > lossₒ && debug("New Term $(i)th iteration. Loss increased from $(lossₒ) to $(lossₙ).")
        cr = (lossₒ-lossₙ)/lossₒ
        if lossₙ < lossₒ && cr < convergerate
            saturatediteration+=1
            if saturatediteration >= hp.nsaturatediteration
                debug("New Term converged in $i iterations with (lossₒ-lossₙ)/lossₒ = $(cr).")
                break
            end
        else
            saturatediteration=0
        end
        i==hp.newtermmaxiteration && debug("New Term does not converge in $i iterations.")
    end
    β = std(Φvs)
    Φvs /=β

    si = sortperm(xa)
    phi = Spline1D(xa[si], phi(xa[si]), k=3, bc="extrapolate", s=0.5)
    return β,phi,α,Φvs,trustregionsize
end

function fitnewterm(x::Matrix,r::Vector,α::Vector,phidf::Int,debug::ePPRDebugOptions=ePPRDebugOptions())
    xa = x*α
    Φ = R"smooth.spline(y=$r, x=$xa, df=$phidf, spar=NULL, cv=FALSE)"
    Φvs = Φ(xa)
    β = std(Φvs)
    Φvs /=β

    si = sortperm(xa)
    Φ = Spline1D(xa[si], Φ(xa[si]), k=3, bc="extrapolate", s=0.5)
    return β,Φ,α,Φvs
end

"""
Update α only once
"Numerical Optimization, Nocedal and Wright, 2006"

subproblem: Min mᵢ(p) = fᵢ + gᵢᵀp + 0.5pᵀBᵢp , ∥p∥ ⩽ rᵢ

Theorem 4.1
a: (Bᵢ + λI)pˢ = -gᵢ , λ ⩾ 0
b: λ(rᵢ - ∥pˢ∥) = 0
c: Bᵢ + λI positive definite

λ = 0 => ∥pˢ∥ ⩽ rᵢ, Bᵢpˢ = -gᵢ, Bᵢ positive definite
∥pˢ∥ = rᵢ => λ ⩾ 0, p(λ) = -(Bᵢ + λI)⁻¹gᵢ, Bᵢ + λI positive definite
"""
function newtontrustregion(f::Function,x₀::Vector,f₀::Float64,g₀::Vector,H₀::Matrix,r::Float64,rmax::Float64,η::Float64,maxiteration::Int,debug::ePPRDebugOptions)
    eh = eigen(Symmetric(H₀))
    posdef = isposdef(eh)
    qᵀg = eh.vectors'*g₀
    if posdef
        pˢ = -eh.vectors*(qᵀg./eh.values)
        pˢₙ = norm(pˢ,2)
    end
    ehminvalue = minimum(eh.values)
    λe = eh.values.-ehminvalue
    ehminidx = λe .== 0
    C1 = sum((qᵀg./λe)[.!ehminidx].^2)
    C2 = sum(qᵀg[ehminidx].^2)
    C3 = sum(qᵀg.^2)

    for i in 1:maxiteration
        debug("NewtonTrustRegion $(i)th iteration, r = $(r)",level=DebugFull)
        # try for solution when λ = 0
        if posdef && pˢₙ <= r
            pᵢ = pˢ
            islambdazero = true
        else
            islambdazero = false
            # easy or hard-easy cases
            if C2 > 0 || C1 >= r^2
                iseasy = true
                ishard = C2==0

                λdn = sqrt(C2)/r
                λup = sqrt(C3)/r
                function ϕ(λ)
                    if λ==0
                        if C2 > 0
                            return -1/r
                        else
                            return sqrt(1/C1) - 1/r
                        end
                    end
                    return 1/norm(qᵀg./(λe.+λ),2) - 1/r
                end
                if ϕ(λup) <= 0
                    λ = λup
                elseif ϕ(λdn) >= 0
                    λ = λdn
                else
                    λ = fzero(ϕ,λdn,λup)
                end
                pᵢ = -eh.vectors*(qᵀg./(λe.+λ))
            else
                # hard-hard case
                iseasy = false
                ishard = true
                w = qᵀg./λe
                w[ehminidx]=0
                τ = sqrt(r^2-C1)
                𝑧 = eh.vectors[:,1]
                pᵢ = -eh.vectors*w + τ*𝑧
            end
        end
        # ρ: ratio of actual change versus predicted change
        xᵢ = x₀ + pᵢ
        fᵢ = f(xᵢ)
        ρ = (fᵢ - f₀) / (pᵢ'*(g₀ + H₀*pᵢ/2))
        # update trust region size
        if ρ < 0.25
            r /= 4
        elseif ρ > 0.75 && !islambdazero
            r = min(2*r,rmax)
        end
        if debug.level >= DebugFull
            debug("                                 ρ = $ρ",level=DebugFull)
            if islambdazero
                steptype="λ = 0"
            else
                if ishard
                    steptype=iseasy ? "hard-easy" : "hard-hard"
                else
                    steptype="easy"
                end
            end
            debug("                                 step is $steptype",level=DebugFull)
        end
        # accept solution only once
        if ρ > η
            return true,xᵢ,fᵢ,r
        end
    end
    debug("NewtonTrustRegion does not converge in $maxiteration iterations.",level=DebugFull)
    return false,x₀,f₀,r
end

function getinitialalpha(x::Matrix,r::Vector,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug("Get Initial α ...")
    # RCall lm.ridge with kLW lambda
    α = rcopy(R"""
    lmr = lm.ridge($r ~ 0 + $x)
    lmr = lm.ridge($r ~ 0 + $x, lambda=lmr$kLW)
    coefficients(lmr)
    """)
    α.-=mean(α);normalize!(α,2);α
end

function refitmodelbetas!(model::ePPRModel,y::Vector,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug("Refit Model βs ...")
    ml = length(model);n=length(y)
    x = Matrix{Float64}(undef,n,ml)
    for i in 1:ml
        x[:,i] = model.phivalues[i]
    end
    lmresult = lm(x, y .- model.ymean)
    β = coef(lmresult)
    debug("Old βs: $(model.beta)")
    debug("New βs: $β")
    model.beta = β
    model.residuals = residuals(lmresult)
    return model
end

"""
2D Laplacian Filtering in Matrix Form
"""
function laplacian2dmatrix(nrow::Int,ncol::Int)
    center = [-1 -1 -1;
              -1  8 -1;
              -1 -1 -1]
    firstrow=[-1  5 -1;
              -1 -1 -1;
               0  0  0]
    lastrow= [ 0  0  0;
              -1 -1 -1;
              -1  5 -1]
    firstcol=[-1 -1  0;
               5 -1  0;
              -1 -1  0]
    lastcol= [ 0 -1 -1;
               0 -1  5;
               0 -1 -1]
    topleft= [ 3 -1  0;
              -1 -1  0;
               0  0  0]
    topright=[ 0 -1  3;
               0 -1 -1;
               0  0  0]
    downleft=[ 0  0  0;
              -1 -1  0;
               3 -1  0]
    downright=[0  0  0;
               0 -1 -1;
               0 -1  3]
    lli = LinearIndices((nrow,ncol))
    lm = zeros(nrow*ncol,nrow*ncol)
    # fill center
    for r in 2:nrow-1, c in 2:ncol-1
        f=zeros(nrow,ncol)
        f[r-1:r+1,c-1:c+1]=center
        lm[lli[r,c],:]=vec(f)
    end
    for c in 2:ncol-1
        # fill first row
        r=1;f=zeros(nrow,ncol)
        f[r:r+2,c-1:c+1]=firstrow
        lm[lli[r,c],:]=vec(f)
        # fill last row
        r=nrow;f=zeros(nrow,ncol)
        f[r-2:r,c-1:c+1]=lastrow
        lm[lli[r,c],:]=vec(f)
    end
    for r in 2:nrow-1
        # fill first col
        c=1;f=zeros(nrow,ncol)
        f[r-1:r+1,c:c+2]=firstcol
        lm[lli[r,c],:]=vec(f)
        # fill last col
        c=ncol;f=zeros(nrow,ncol)
        f[r-1:r+1,c-2:c]=lastcol
        lm[lli[r,c],:]=vec(f)
    end
    # fill top left
    r=1;c=1;f=zeros(nrow,ncol)
    f[r:r+2,c:c+2]=topleft
    lm[lli[r,c],:]=vec(f)
    # fill top right
    r=1;c=ncol;f=zeros(nrow,ncol)
    f[r:r+2,c-2:c]=topright
    lm[lli[r,c],:]=vec(f)
    # fill down left
    r=nrow;c=1;f=zeros(nrow,ncol)
    f[r-2:r,c:c+2]=downleft
    lm[lli[r,c],:]=vec(f)
    # fill down right
    r=nrow;c=ncol;f=zeros(nrow,ncol)
    f[r-2:r,c-2:c]=downright
    lm[lli[r,c],:]=vec(f)
    return lm
end

include("Visualization.jl")
end # module
