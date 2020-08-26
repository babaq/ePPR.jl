using LinearAlgebra,Statistics,GLM,Roots,HypothesisTests,RCall,Dierckx,Plots
import Base: length, push!, deleteat!



"ePPR Log Options"
Base.@kwdef mutable struct ePPRLog
    debug = false
    plot = false
    io = nothing
    dir = nothing
end
function (log::ePPRLog)(msg;file="ePPRLog.txt",close=false)
    if isnothing(log.io)
        if isnothing(log.dir)
            io=stdout
        else
            isdir(log.dir) || mkpath(log.dir)
            io = open(joinpath(log.dir,file),"a")
            log.io = io
        end
    else
        io=log.io
    end
    println(io,msg)
    flush(io)
    close && !isnothing(log.io) && close(log.io)
end
function (log::ePPRLog)(msg::Plots.Plot;file="Model.png")
    if isnothing(log.dir)
        display(msg)
    else
        isdir(log.dir) || mkpath(log.dir)
        savefig(msg,joinpath(log.dir,file))
    end
end

"""
extended Projection Pursuit Regression Model

```math
\\hat{y}_i=\\bar{y}+\\sum_{d=0}^D\\sum_{m=1}^{M_d}\\beta_{m,d}\\phi_{m,d}(\\alpha_{m,d}^Tx_{i-d}), where

\\frac{1}{n}\\sum_{i=1}^n\\phi_{m,d}(\\alpha_{m,d}^Tx_{i-d})=0, \\frac{1}{n}\\sum_{i=1}^n\\phi_{m,d}^2(\\alpha_{m,d}^Tx_{i-d})=1
```
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
"Number of terms in the model"
length(m::ePPRModel) = length(m.beta)
function deleteat!(model::ePPRModel,i::Int)
    deleteat!(model.beta,i)
    deleteat!(model.phi,i)
    deleteat!(model.alpha,i)
    deleteat!(model.index,i)
    deleteat!(model.phivalues,i)
    deleteat!(model.trustregionsize,i)
    return model
end
function push!(model::ePPRModel,β::Float64,Φ,α::Vector{Float64},index,Φvs::Vector{Float64},trustregionsize::Float64=1.0)
    push!(model.beta,β)
    push!(model.phi,Φ)
    push!(model.alpha,α)
    push!(model.index,index)
    push!(model.phivalues,Φvs)
    push!(model.trustregionsize,trustregionsize)
    return model
end
function getterm(model::ePPRModel,i::Int)
    return model.beta[i],model.phi[i],model.alpha[i],model.index[i],model.phivalues[i],model.trustregionsize[i]
end
function setterm!(model::ePPRModel,i::Int,β::Float64,Φ,α::Vector{Float64},index,Φvs::Vector{Float64},trustregionsize::Float64=1.0)
    model.beta[i]=β
    model.phi[i]=Φ
    model.alpha[i]=α
    model.index[i]=index
    model.phivalues[i]=Φvs
    model.trustregionsize[i]=trustregionsize
    return model
end
function clean!(model::ePPRModel)
    model.phivalues=[]
    model.trustregionsize=[]
    model.residuals=[]
    return model
end

"Cross Validation Parameters for ePPR"
Base.@kwdef mutable struct ePPRCrossValidation
    "Percent of samples for training"
    trainpercent::Float64 = 0.88
    "N-fold of training samples"
    trainfold::Int = 5
    "N-fold of one `trainfold` samples for testing training"
    traintestfold::Int = 8
    "each combination of train and test fold"
    trainsets = []
    "N-fold of testing samples for testing model"
    testfold::Int = 8
    "test fold of testing samples"
    tests = []
    "which trainset for training"
    trainsetindex::Int = 1
    "significent level to accept null hypothesis that current model is no better than reference model"
    h0level::Float64 = 0.05
    "significent level to accept alternative hypothesis that current model is better than reference model"
    h1level::Float64 = 0.05
    "Correlation between response and model prediction on testsets of current trainset"
    modeltraintestcor = []
    "Correlation between response and model prediction on testsets"
    modeltestcor = []
    "Correlation between response and model prediction"
    modelcors = []
end

"Hyper Parameters for ePPR"
Base.@kwdef mutable struct ePPRHyperParams
    """memory size to pool for nonlinear time interaction, ndelay=1 for linear time interaction.
    only first delay terms in `nft` is used for nonlinear time interaction."""
    ndelay::Int = 1
    "number of forward terms for each delay. [3, 2, 1] means 3 spatial terms for delay 0, 2 for delay 1, 1 for delay 2"
    nft::Vector{Int} = [3,3,3]
    "penalization parameter λ"
    lambda::Float64 = 15
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
    "number of consecutive saturated iterations to decide a new term"
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
    "dimension of image"
    imagesize = ()
    "row vector of blank image"
    blankimage = []
    "drop term index between backward models"
    droptermindex = []
    "ePPR Cross Validation"
    cv::ePPRCrossValidation = ePPRCrossValidation()
    "Valid Image Region"
    xindex::Vector{Int} = Int[]
    "Index iⱼ where image sequence breaks between x[iⱼ-1,:] and x[iⱼ,:]"
    xbreak::Vector{Int} = Int[]
    "maximum iterations of hyperparameter search"
    hypermaxiteration = 25
    "number of consecutive saturated iterations to decide a hyperparameter"
    nhypersaturatediteration::Int = 2
    "scale factor for λ search"
    lambdascale::Float64 = 1.5
end
function ePPRHyperParams(nrow::Int,ncol::Int;xindex::Vector{Int}=Int[],ndelay::Int=1,nft::Vector{Int}=[3,3,3],lambda=30,blankcolor=127)
    hp = ePPRHyperParams(imagesize=(nrow,ncol),xindex=xindex,ndelay=ndelay,nft=nft,lambda=lambda)
    hp.blankimage = fill(blankcolor,1,prod(hp.imagesize))
    hp.alphapenaltyoperator = laplacian2dmatrix(nrow,ncol)
    return hp
end

"Image sequences at delay `d`"
function delayx(x,d,hp,xi=[])
    if d<=0
        tx = x
    else
        tx = [repeat(hp.blankimage,outer=(d,1));x[1:end-d,:]]
        if !isempty(hp.xbreak)
            for bi in hp.xbreak
                tx[bi-d:bi-1,:]=repeat(hp.blankimage,outer=(d,1))
            end
        end
    end
    isempty(xi) ? tx : tx[xi,:]
end

"Pool images in delay window"
function delaywindowpool!(x::Matrix,hp::ePPRHyperParams,log::ePPRLog)
    if isempty(hp.xindex)
        vx = x
    else
        vx = x[:,hp.xindex]
        hp.blankimage = hp.blankimage[:,hp.xindex]
        hp.alphapenaltyoperator=hp.alphapenaltyoperator[hp.xindex,hp.xindex]
    end
    hp.ndelay<=1 && return vx

    log.debug && log("Nonlinear Time Interaction, pool x[i-$(hp.ndelay-1):i, :] together ...")
    dwpx=vx
    for d in 1:hp.ndelay-1
        dwpx = [dwpx delayx(vx,d,hp)]
    end
    hp.alphapenaltyoperator = delaywindowpooloperator(hp.alphapenaltyoperator,hp.ndelay)
    return dwpx
end

function delaywindowpooloperator(operator::Matrix,ndelay::Int=1)
    ndelay<=1 && return operator

    nr,nc=size(operator)
    dwpo = zeros(ndelay*nr,ndelay*nc)
    for d in 0:ndelay-1
        dwpo[(1:nr).+d*nr, (1:nc).+d*nc] = operator
    end
    return dwpo
end

"Model prediction on training data"
(m::ePPRModel)() = m.ymean .+ mapreduce((b,ps)->b*ps, (i,j)->i.+j, m.beta, m.phivalues)
"""
Model prediction on data `x`

1. x: matrix with one image per row
2. hp: hyper parameters
3. xi: image indices on which predictions are made
"""
function (m::ePPRModel)(x::Matrix,hp::ePPRHyperParams,xi=[])
    ti=map(i->i.t,m.index);ut=unique(ti);uti=[findall(ti.==t) for t in ut]
    ŷ = m.ymean
    for i in eachindex(ut)
        tx = delayx(x,ut[i],hp,xi)
        for j in uti[i]
            ŷ = ŷ .+ m.beta[j] * m.phi[j](tx*m.alpha[j])
        end
    end
    return ŷ
end

"""
Data Partition for cross validation

1. cv: cross validation
2. n: number of samples
3. log: log options
"""
function cvpartition!(cv::ePPRCrossValidation,n::Int,log::ePPRLog)
    ntrain = cv.trainpercent*n
    ntrainfold = ntrain/cv.trainfold
    ntraintestfold = Int(floor(ntrainfold/cv.traintestfold))
    ntrainfold = ntraintestfold*cv.traintestfold
    ntrain = ntrainfold*cv.trainfold
    trainsets=[]
    for tf in 0:cv.trainfold-1
        traintest = [tf*ntrainfold .+ (1:ntraintestfold) .+ ttf*ntraintestfold for ttf in 0:cv.traintestfold-1]
        train = setdiff(1:ntrain,tf*ntrainfold .+ (1:ntrainfold))
        push!(trainsets,(train=train,traintest=traintest))
    end
    cv.trainsetindex=cv.trainfold
    ntestfold = Int(floor((n-ntrain)/cv.testfold))
    tests = [ntrain .+ (1:ntestfold) .+ tf*ntestfold for tf in 0:cv.testfold-1]
    log.debug && log("Cross Validation Data Partition: n = $n, ntrain = $ntrain in $(cv.trainfold)-fold, ntrainfold = $ntrainfold in $(cv.traintestfold)-fold, ntest = $(ntestfold*cv.testfold) in $(cv.testfold)-fold")
    cv.trainsets=trainsets;cv.tests=tests
    return cv
end

function cvmodel(models::Vector{ePPRModel},x::Matrix,y::Vector,hp::ePPRHyperParams,log::ePPRLog)
    log.debug && log("ePPR Models Cross Validation ...")
    train = hp.cv.trainsets[hp.cv.trainsetindex].train;traintest = hp.cv.trainsets[hp.cv.trainsetindex].traintest;test=hp.cv.tests
    # response and model predication
    traintestpredications = map(m->map(i->m(x,hp,i),traintest),models)
    traintestys = map(i->y[i],traintest)
    # correlation between response and predication
    traintestcors = map(mps->cor.(traintestys,mps),traintestpredications)
    hp.cv.modeltraintestcor=[];hp.cv.modeltestcor=[] # clear model test correlation
    log.plot && log(plotcor(models,traintestcors),file="Models_Goodness (λ=$(hp.lambda)).png")
    # find the model no worse than models with more terms and better than models with less terms
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
        log.debug && log("No model not worse than models with more terms and better than models with less terms.")
        return nothing
    end
    model = deepcopy(models[mi])
    log.debug && log("$(mi)th model with $(length(model)) terms is chosen.")

    # find drop terms that do not improve model predication
    droptermp = [pvalue(SignedRankTest(traintestcors[m-1],traintestcors[m]),tail=:left) for m in 2:nmodel]
    notimprove = findall(droptermp .> hp.cv.h0level)
    # find drop term models with chance level(zero correlation) predication
    modelp = [pvalue(SignedRankTest(traintestcors[m]),tail=:both) for m in 2:nmodel]
    notpredictive = findall(modelp .> hp.cv.h0level)

    poorterm = hp.droptermindex[union(notimprove,notpredictive)]
    # spurious terms in the selected model
    spuriousterm = findall(in(poorterm),model.index)
    if !isempty(spuriousterm)
        log.debug && log("Model drop spurious term: $(model.index[spuriousterm]).")
        foreach(i->deleteat!(model,i),sort(spuriousterm,rev=true))
    end
    length(model)==0 && return nothing
    model = eppr(model,x[train,:],y[train],hp,log)
    hp.cv.modeltraintestcor = map(i->cor(y[i],model(x,hp,i)),traintest)
    hp.cv.modeltestcor = map(i->cor(y[i],model(x,hp,i)),test)
    return model
end

"extended Projection Pursuit Regression with cross validation"
function epprcv(x::Matrix,y::Vector,hp::ePPRHyperParams,log::ePPRLog=ePPRLog())
    n = length(y);n !=size(x,1) && error("Length of x and y does not match!")
    cvpartition!(hp.cv,n,log)
    px = delaywindowpool!(x,hp,log)
    log.debug && log("Choose $(hp.cv.trainsetindex)th trainset.")
    train = hp.cv.trainsets[hp.cv.trainsetindex].train
    models = eppr(px[train,:],y[train],hp,log)
    model = cvmodel(models,px,y,hp,log)
    log.debug && log("Cross Validated ePPR Done.",close=true)
    return model,models
end

"extended Projection Pursuit Regression with cross validation and hyper parameters search"
function epprhypercv(x::Matrix,y::Vector,hp::ePPRHyperParams,log::ePPRLog=ePPRLog())
    n = length(y);n !=size(x,1) && error("Length of x and y does not match!")
    cvpartition!(hp.cv,n,log)
    px = delaywindowpool!(x,hp,log)
    log.debug && log("Choose $(hp.cv.trainsetindex)th trainset.")
    train = hp.cv.trainsets[hp.cv.trainsetindex].train

    hi=0;hypermodel=[];hypermodels=[];λs=[];modelcors=[];saturatediteration=0;chanceiteration=0;nomodeliteration=0
    for i in 1:hp.hypermaxiteration
        log.debug && log("HyperParameter Search: λ = $(hp.lambda) ...")
        models = eppr(px[train,:],y[train],hp,log)
        model = cvmodel(models,px,y,hp,log)
        log.debug && log("Cross Validated ePPR Done.")
        if !isnothing(model)
            nomodeliteration=0
            push!(hypermodel,model);push!(hypermodels,models);push!(λs,hp.lambda);push!(modelcors,[hp.cv.modeltraintestcor;hp.cv.modeltestcor])
            chancep = pvalue(SignedRankTest(modelcors[end]),tail=:both)
            if any(x->isnan(x),modelcors[end]) || (chancep > hp.cv.h0level)
                chanceiteration+=1
                if chanceiteration>=hp.nhypersaturatediteration
                    hi=-1;break
                else
                    hp.lambda *= hp.lambdascale
                end
                continue
            else
                chanceiteration=0
            end
            if length(modelcors)==1
                hp.lambda *=hp.lambdascale
            else
                improvep = pvalue(SignedRankTest(modelcors[end-1],modelcors[end]),tail=:left)
                if improvep < hp.cv.h1level
                    saturatediteration=0
                    hp.lambda *= hp.lambdascale
                else
                    impairp = pvalue(SignedRankTest(modelcors[end-1],modelcors[end]),tail=:right)
                    if impairp < hp.cv.h1level
                        hi=length(modelcors)-1;break
                    else
                        saturatediteration+=1
                        if saturatediteration>=hp.nhypersaturatediteration
                            hi=length(modelcors);break
                        else
                            hp.lambda *= hp.lambdascale
                        end
                    end
                end
            end
        else
            nomodeliteration+=1
            if nomodeliteration>=hp.nhypersaturatediteration
                hi=-1;break
            else
                hp.lambda *= hp.lambdascale
            end
        end
    end
    log.plot && !isempty(modelcors) && log(plotcor(λs,modelcors,xlabel="λ"),file="λ_Models_Goodness.png")
    if hi<0
        log.debug && log("No predictive λ and model.",close=true)
        return nothing,[]
    elseif hi==0
        if length(modelcors)>0
            _,hi=findmax(mean.(modelcors))
        else
            log.debug && log("No valid λ and model.",close=true)
            return nothing,[]
        end
    end
    hp.lambda = λs[hi];hp.cv.modelcors = modelcors[hi]
    log.debug && log("HyperParameter search done with best λ = $(hp.lambda).",close=true)
    return hypermodel[hi],hypermodels[hi]
end

"""
extended Projection Pursuit Regression

Minimizing ``f=\\sum_{i=1}^N(y_i-\\hat{y}(x_i))^2+\\lambda\\sum_{d=0}^D\\sum_{m=1}^{M_d}\\Vert{L\\alpha_{m,d}}\\Vert^2``

1. x: matrix with one image per row
2. y: vector of response for each image
3. hp: hyper parameters
4. log: log options
"""
function eppr(x::Matrix,y::Vector,hp::ePPRHyperParams,log::ePPRLog=ePPRLog())
    model = forwardstepwise(x,y,hp,log)
    model = refitmodel!(model,x,y,hp,log)
    models = backwardstepwise(model,x,y,hp,log)
end
function eppr(model::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,log::ePPRLog)
    refitmodelbeta!(forwardstepwise(model,x,y,hp,log),y,log)
end

"""
Gradually add term into the model based on terms of an existing model
"""
function forwardstepwise(m::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,log::ePPRLog)
    ti = map(i->i.t,m.index);si = map(i->i.s,m.index)
    ut = sort(unique(ti));uts = Dict(t=>si[ti.==t] for t in ut);utsn=Dict(t=>length(uts[t]) for t in ut)
    log.debug && log("ePPR Model Forward Stepwise ...")
    ym = mean(y);model = ePPRModel(ymean=ym,residuals=y.-ym)
    if hp.spatialtermfirst
        for t in ut
            tx = delayx(x,t,hp)
            for s in 1:utsn[t]
                log.debug && log("Fit Model (Temporal-$t, Spatial-$s) New Term ...")
                α = normalize(m.alpha[findfirst(==((t=t,s=uts[t][s])),m.index)], 2)
                β,Φ,α,Φvs = fitnewterm(tx,model.residuals,α,hp.phidf)
                model.residuals .-= β*Φvs
                push!(model,β,Φ,α,(t=t,s=s),Φvs)
            end
        end
    else
        for s in 1:maximum(values(utsn)), t in ut
            s>utsn[t] && continue
            log.debug && log("Fit Model (Temporal-$t, Spatial-$s) New Term ...")
            tx = delayx(x,t,hp)
            α = normalize(m.alpha[findfirst(==((t=t,s=uts[t][s])),m.index)], 2)
            β,Φ,α,Φvs = fitnewterm(tx,model.residuals,α,hp.phidf)
            model.residuals .-= β*Φvs
            push!(model,β,Φ,α,(t=t,s=s),Φvs)
        end
    end
    return model
end

"""
Gradually add term into the model
"""
function forwardstepwise(x::Matrix,y::Vector,hp::ePPRHyperParams,log::ePPRLog)
    log.debug && log("ePPR Forward Stepwise ...")
    if hp.ndelay>1
        hp.nft=hp.nft[1:1]
    end
    ym = mean(y);model = ePPRModel(ymean=ym,residuals=y.-ym)
    if hp.spatialtermfirst
        for t in 0:length(hp.nft)-1
            tx = delayx(x,t,hp)
            for s in 1:hp.nft[t+1]
                log.debug && log("Fit (Temporal-$t, Spatial-$s) New Term ...")
                α = getinitialalpha(tx,model.residuals,log)
                β,Φ,α,Φvs,trustregionsize = fitnewterm(tx,model.residuals,α,hp,log)
                model.residuals .-= β*Φvs
                push!(model,β,Φ,α,(t=t,s=s),Φvs,trustregionsize)
            end
        end
    else
        for s in 1:maximum(hp.nft), t in 0:length(hp.nft)-1
            s>hp.nft[t+1] && continue
            log.debug && log("Fit (Temporal-$t, Spatial-$s) New Term ...")
            tx = delayx(x,t,hp)
            α = getinitialalpha(tx,model.residuals,log)
            β,Φ,α,Φvs,trustregionsize = fitnewterm(tx,model.residuals,α,hp,log)
            model.residuals .-= β*Φvs
            push!(model,β,Φ,α,(t=t,s=s),Φvs,trustregionsize)
        end
    end
    return model
end

function getinitialalpha(x::Matrix,r::Vector,log::ePPRLog)
    log.debug && log("Get Initial α ...")
    # RCall lm.ridge with kLW lambda
    α = rcopy(R"""
    lmr = lm.ridge($r ~ 0 + $x)
    lmr = lm.ridge($r ~ 0 + $x, lambda=lmr$kLW)
    coefficients(lmr)
    """)
    α.-=mean(α);normalize!(α,2);α
end

function refitmodelbeta!(model::ePPRModel,y::Vector,log::ePPRLog)
    log.debug && log("Refit Model βs ...")
    ml = length(model);n=length(y)
    x = Matrix{Float64}(undef,n,ml)
    for i in 1:ml
        x[:,i] = model.phivalues[i]
    end
    lmresult = lm(x, y .- model.ymean)
    β = coef(lmresult)
    if log.debug
        log("Old βs: $(model.beta)")
        log("New βs: $β")
    end
    model.beta = β
    model.residuals = residuals(lmresult)
    return model
end

function refitmodel!(model::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,log::ePPRLog)
    log.debug && log("ePPR Model Refit ...")
    model = refitmodelbeta!(model,y,log)
    for i in 1:length(model)
        oldloss = lossfun(model,y,hp)
        oldβ,oldΦ,oldα,index,oldΦvs,oldtrustregionsize = getterm(model,i)
        model.residuals .+= oldβ*oldΦvs

        t = index.t;s=index.s
        tx = delayx(x,t,hp)
        log.debug && log("Refit (Temporal-$t, Spatial-$s) New Term ...")
        β,Φ,α,Φvs,trustregionsize = fitnewterm(tx,model.residuals,oldα,hp,log,convergerate=hp.refitconvergerate,trustregionsize=oldtrustregionsize)
        setterm!(model,i,β,Φ,α,index,Φvs,trustregionsize)
        newloss = lossfun(model,y,hp)
        if newloss > oldloss
            log.debug && log("Model Loss increased from $oldloss to $newloss. Discard the new term, keep the old one.")
            setterm!(model,i,oldβ,oldΦ,oldα,index,oldΦvs,oldtrustregionsize)
            model.residuals .-= oldβ*oldΦvs
        else
            model.residuals .-= β*Φvs
        end
    end
    return model
end

function backwardstepwise(model::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,log::ePPRLog)
    log.debug && log("ePPR Backward Stepwise ...")
    log.plot && log(plotmodel(model,hp),file="Model_$(length(model)) (λ=$(hp.lambda)).png")
    models=[deepcopy(model)];hp.droptermindex=[]
    for i in length(model)-1:-1:hp.mnbt
        model,dropindex = dropleastimportantterm!(model,log)
        pushfirst!(hp.droptermindex,dropindex)
        model = refitmodel!(model,x,y,hp,log)
        log.plot && log(plotmodel(model,hp),file="Model_$(length(model)) (λ=$(hp.lambda)).png")
        pushfirst!(models,deepcopy(model))
    end
    return models
end

dropleastimportantterm!(model::ePPRModel,log::ePPRLog)=dropterm!(model,argmin(abs.(model.beta)),log)
function dropterm!(model::ePPRModel,i::Int,log::ePPRLog)
    dropindex = model.index[i]
    β=model.beta[i]
    log.debug && log("Drop Term: (temporal-$(dropindex.t), spatial-$(dropindex.s)) with β: $(β).")
    deleteat!(model,i)
    return model,dropindex
end

lossfun(g::Vector) = 0.5*norm(g,2)^2
"""
Loss function for a model term

``f(α) = sum((r.-Φ(x*α)).^2) + λ*norm(hp.alphapenaltyoperator*α,2)^2``
"""
lossfun(r::Vector,x::Matrix,α::Vector,Φ,hp::ePPRHyperParams) = lossfun([r.-Φ(x*α);sqrt(hp.lambda)*hp.alphapenaltyoperator*α])
"Loss function for model terms"
function lossfun(model::ePPRModel,y::Vector,hp::ePPRHyperParams)
    modelloss = lossfun(y.-model())
    penaltyloss = 0.5*hp.lambda*sum(norm.([hp.alphapenaltyoperator].*model.alpha,2).^2)
    return modelloss + penaltyloss
end

(phi::RObject)(x) = rcopy(R"predict($phi, x=$x)$y")
"Alternately fit `Φ` and `α` using smooth spline and newton trust region"
function fitnewterm(x::Matrix,r::Vector,α::Vector,hp::ePPRHyperParams,log::ePPRLog;convergerate::Float64=hp.forwardconvergerate,trustregionsize::Float64=hp.trustregioninitsize)
    saturatediteration = 0;xa=phi=Φvs=nothing
    for i in 1:hp.newtermmaxiteration
        xa = x*α
        phi = R"smooth.spline(y=$r, x=$xa, df=$(hp.phidf), spar=NULL, cv=FALSE)"
        Φvs = phi(xa)
        gt = r.-Φvs;gp = sqrt(hp.lambda)*hp.alphapenaltyoperator*α;g = [gt;gp]
        log.debug && log("New Term $(i)th iteration. TermLoss: $(lossfun(gt)), PenaltyLoss: $(lossfun(gp)).")
        f(a) = lossfun(r,x,a,phi,hp)
        # Loss before trust region
        lossₒ = lossfun(g)
        Φ′vs = rcopy(R"predict($phi, x=$xa, deriv=1)$y")
        gg = [-Φ′vs.*x;sqrt(hp.lambda)*hp.alphapenaltyoperator]'
        f′ = gg*g
        f″ = gg*gg'
        # α and Loss after trust region
        success,α,lossₙ,trustregionsize = newtontrustregion(f,α,lossₒ,f′,f″,trustregionsize,hp.trustregionmaxsize,hp.trustregioneta,hp.trustregionmaxiteration,log)
        if !success
            log.debug && log("NewtonTrustRegion failed, New Term use old α.")
            break
        end
        log.debug && lossₙ > lossₒ && log("New Term $(i)th iteration. Loss increased from $(lossₒ) to $(lossₙ).")
        cr = (lossₒ-lossₙ)/lossₒ
        if lossₙ < lossₒ && cr < convergerate
            saturatediteration+=1
            if saturatediteration >= hp.nsaturatediteration
                log.debug && log("New Term converged in $i iterations with (lossₒ-lossₙ)/lossₒ = $(cr).")
                break
            end
        else
            saturatediteration=0
        end
        log.debug && i==hp.newtermmaxiteration && log("New Term does not converge in $i iterations.")
    end
    β = std(Φvs)

    si = sortperm(xa)
    Φ = Spline1D(xa[si], Φvs[si], bc="nearest", s=0.5)
    # xknots = range(extrema(xa)...,length=21)[2:end-1]
    # Φ = Spline1D(xa[si], Φvs[si], xknots, bc="nearest")

    return β,Φ,α,Φvs/β,trustregionsize
end

function fitnewterm(x::Matrix,r::Vector,α::Vector,phidf::Int)
    xa = x*α
    phi = R"smooth.spline(y=$r, x=$xa, df=$phidf, spar=NULL, cv=FALSE)"
    Φvs = phi(xa)
    β = std(Φvs)

    si = sortperm(xa)
    Φ = Spline1D(xa[si], Φvs[si], bc="nearest", s=0.5)
    # xknots = range(extrema(xa)...,length=21)[2:end-1]
    # Φ = Spline1D(xa[si], Φvs[si], xknots, bc="nearest")

    return β,Φ,α,Φvs/β
end

"""
Update `α` only once [^1]

subproblem: Min mᵢ(p) = fᵢ + gᵢᵀp + 0.5pᵀBᵢp , ∥p∥ ⩽ rᵢ

Theorem 4.1
a: (Bᵢ + λI)pˢ = -gᵢ , λ ⩾ 0
b: λ(rᵢ - ∥pˢ∥) = 0
c: Bᵢ + λI positive definite

λ = 0 => ∥pˢ∥ ⩽ rᵢ, Bᵢpˢ = -gᵢ, Bᵢ positive definite
∥pˢ∥ = rᵢ => λ ⩾ 0, p(λ) = -(Bᵢ + λI)⁻¹gᵢ, Bᵢ + λI positive definite

[^1]

"Numerical Optimization, Nocedal and Wright, 2006"
"""
function newtontrustregion(f::Function,x₀::Vector,f₀::Float64,g₀::Vector,H₀::Matrix,r::Float64,rmax::Float64,η::Float64,maxiteration::Int,log::ePPRLog)
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
        log.debug && log("NewtonTrustRegion $(i)th iteration, r = $(r)")
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
        if log.debug
            log("                                 ρ = $ρ")
            if islambdazero
                steptype="λ = 0"
            else
                if ishard
                    steptype=iseasy ? "hard-easy" : "hard-hard"
                else
                    steptype="easy"
                end
            end
            log("                                 step is $steptype")
        end
        # accept solution only once
        if ρ > η
            return true,xᵢ,fᵢ,r
        end
    end
    log.debug && log("NewtonTrustRegion does not converge in $maxiteration iterations.")
    return false,x₀,f₀,r
end

"""
2D Laplacian Filter in Matrix Form
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



## Visualization
"Plot ePPR Model"
function plotmodel(model::ePPRModel,hp::ePPRHyperParams;color=:coolwarm,linkclim=true,xlim=200,size=(650,850))
    plot(plotalpha(model,hp;color,linkclim),plotphi(model,hp;xlim),layout=(2,1),size=size,link=:none)
end

"Plot ePPR α for each term"
function plotalpha(model::ePPRModel,hp::ePPRHyperParams;color=:coolwarm,linkclim=true)
    ti = map(i->i.t+1,model.index);si = map(i->i.s,model.index);tmin,tmax=extrema(ti);smax=maximum(si)
    utsmax=Dict(t=>maximum(si[ti.==t]) for t in unique(ti));inpx=prod(hp.imagesize);xnpx=length(hp.xindex);αlim=0
    p = plot(layout=(tmax,smax),yflip=true,framestyle=:none)
    for i in 1:length(model)
        t=ti[i];s=si[i];iα=model.alpha[i]
        α = mapfoldl(d->begin
                            if xnpx==0
                                da = reshape(iα[(1:inpx).+d*inpx],hp.imagesize)
                            else
                                da = zeros(hp.imagesize)
                                da[hp.xindex]=iα[(1:xnpx).+d*xnpx]
                            end
                            da
                        end,(a1,a2)->[a1;a2],0:hp.ndelay-1)
        if linkclim
            αlim=max(αlim,maximum(abs.(α)))
        end
        colorbar = linkclim ? ( (t==tmin && s==utsmax[t]) ? :right : :none ) : :right
        heatmap!(p[t,s],α,color=color,ratio=:equal,colorbar=colorbar)
    end
    for t in 1:tmax
        ylabel!(p[t,1],"Delay $(hp.ndelay>1 ? "$(hp.ndelay-1) - 0" : t-1)")
    end
    linkclim && plot!(p,clims=(-αlim,αlim))
    p
end

"Plot ePPR Φ function for each term"
function plotphi(model::ePPRModel,hp::ePPRHyperParams;xlim=200)
    ti = map(i->i.t+1,model.index);si = map(i->i.s,model.index);tmax=maximum(ti);smax=maximum(si)
    p = plot(layout=(tmax,smax),leg=false,grid=false,framestyle=:none,xtick=[-xlim,0,xlim])
    for i in 1:length(model)
        t=ti[i];s=si[i]
        vline!(p[t,s],[0],linewidth=0.5,color=:grey80);hline!(p[t,s],[0],linewidth=0.5,color=:grey80)
        plot!(p[t,s],x->model.phi[i](x),-xlim,xlim,color=:deepskyblue,linewidth=2,link=:all,framestyle=:axes,title="β=$(round(model.beta[i],digits=3))")
        if t < tmax
            xaxis!(p[t,s],xformatter=_->"")
        end
        if s > 1
            yaxis!(p[t,s],yformatter=_->"")
        end
    end
    for t in 1:tmax
        ylabel!(p[t,1],"Delay $(hp.ndelay>1 ? "$(hp.ndelay-1) - 0" : t-1)")
    end
    p
end

"Plot Correlation between response and ePPR prediction"
plotcor(models::Vector{ePPRModel},mcors)=plotcor(length.(models),mcors,xlabel="Number of Terms")
function plotcor(ms,mcors;xlabel="Models")
    scatter(ms,hcat(mcors...)',ylabel="Pearson Correlation",xlabel=xlabel,leg=false,xtick=ms)
end
