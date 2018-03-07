__precompile__(false)
module ePPR

import Base.length,Base.push!,Base.deleteat!,StatsBase.predict
export ePPRDebugOptions,DebugNone,DebugBasic,DebugFull,DebugVisual,
delaywindowpool,delaywindowpooloperator,delaywindowpoolblankimage,cvpartitionindex,getinitialalpha,refitmodelbetas,laplacian2dmatrix,
ePPRModel,getterm,setterm,ePPRHyperParams,ePPRCrossValidation,
eppr,epprcv,cvmodel,forwardstepwise,refitmodel,backwardstepwise,dropterm,lossfun,fitnewterm,newtontrustregion

using GLM,Roots,HypothesisTests,RCall,Dierckx
R"library('MASS')"

const DebugNone=0
const DebugBasic=1
const DebugFull=2
const DebugVisual=3
mutable struct ePPRDebugOptions
    level::Int
end
ePPRDebugOptions()=ePPRDebugOptions(DebugNone)

"""
``\hat{y}_i=\bar{y}+\sum_{d=0}^D\sum_{m=1}^{M_d}\beta_{m,d}\phi_{m,d}(\alpha_{m,d}^Tx_{i-d})``
with ``\frac{1}{n}\sum_{i=1}^n\phi_{m,d}(\alpha_{m,d}^Tx_{i-d})=0``, ``\frac{1}{n}\sum_{i=1}^n\phi_{m,d}^2(\alpha_{m,d}^Tx_{i-d})=1``
"""
mutable struct ePPRModel
    "𝑦̄"
    ymean::Float64
    "vector of β for each term"
    beta::Vector{Float64}
    "vector of Φ for each term"
    phi::Vector
    "vector of α for each term"
    alpha::Vector{Vector{Float64}}
    "vector of [temporal, spatial] index for each term"
    index::Vector{Vector{Int}}
    "vector of ``\phi_{m,d}(\alpha_{m,d}^TX_{-d})`` for each term"
    phivalues::Vector{Vector{Float64}}
    "γ"
    residuals::Vector{Float64}
end
ePPRModel() = ePPRModel(0)
ePPRModel(ymean) = ePPRModel(ymean,[],[],[],[],[],[])
length(m::ePPRModel)=length(m.beta)
predict(m::ePPRModel)=m.ymean+squeeze(sum(cat(2,(m.beta.*m.phivalues)...),2),2)
predict(m::ePPRModel,x::Matrix)=m.ymean+squeeze(sum(cat(2,(m.beta.*map((Φ,xα)->Φ(xα),m.phi,[x].*m.alpha))...),2),2)
(m::ePPRModel)() = predict(m)
(m::ePPRModel)(x::Matrix) = predict(m,x)
function deleteat!(model::ePPRModel,i::Integer)
    deleteat!(model.beta,i)
    deleteat!(model.phi,i)
    deleteat!(model.alpha,i)
    deleteat!(model.index,i)
    deleteat!(model.phivalues,i)
end
function push!(model::ePPRModel,β::Float64,Φ,α::Vector{Float64},index::Vector{Int},Φvs::Vector{Float64})
    push!(model.beta,β)
    push!(model.phi,Φ)
    push!(model.alpha,α)
    push!(model.index,index)
    push!(model.phivalues,Φvs)
end
function getterm(model::ePPRModel,i::Integer)
    return model.beta[i],model.phi[i],model.alpha[i],model.index[i],model.phivalues[i]
end
function setterm(model::ePPRModel,i::Integer,β::Float64,Φ,α::Vector{Float64},index::Vector{Int},Φvs::Vector{Float64})
    model.beta[i]=β
    model.phi[i]=Φ
    model.alpha[i]=α
    model.index[i]=index
    model.phivalues[i]=Φvs
end

"""
Hyper Parameters for ePPR
"""
mutable struct ePPRHyperParams
    """memory size to pool for nonlinear time interaction, ndelay=1 for linear time interaction.
    only first delay terms in `nft` is used for nonlinear time interaction."""
    ndelay::Int
    "number of forward terms for each delay. [3, 2, 1] means 3 spatial terms for delay 0, 2 for delay 1, 1 for delay 2"
    nft::Vector{Int}
    "penalization parameter λ"
    lambda::Float64
    "Φ Spline degree of freedom"
    phidf::Int
    "minimum number of backward terms"
    mnbt::Int
    "whether to fit all spatial terms before moving to next temporal"
    spatialtermfirst::Bool
    "α priori for penalization"
    alphapenaltyoperator
    "`(lossₒ-lossₙ)/lossₒ`, forward converge rate threshold to decide a saturated iteration"
    forwardconvergerate::Float64
    "`(lossₒ-lossₙ)/lossₒ`, refit converge rate threshold to decide a saturated iteration"
    refitconvergerate::Float64
    "number of consecutive saturated iterations to decide a solution"
    nsaturatediteration::Int
    "maximum number of iterations to fit a new term"
    newtermmaxiteration::Int
    "initial size of trust region"
    trustregioninitsize::Float64
    "maximum size of trust region"
    trustregionmaxsize::Float64
    "η of trust region"
    trustregioneta::Float64
    "maximum iterations of trust region"
    trustregionmaxiteration::Int
    "row vector of blank image"
    blankimage
    "dimension of image"
    imagesize
    "drop term index between backward models"
    droptermindex
    "ePPR Cross Validation"
    cv
end
ePPRHyperParams()=ePPRHyperParams(1,[2,2],15,5,1,true,[],0.01,0.001,2,100,1,1000,0.2,1000,[],[],[],ePPRCrossValidation())
function ePPRHyperParams(nrow::Int,ncol::Int;ndelay::Int=1,blankcolor=0.5)
    hp=ePPRHyperParams()
    hp.imagesize = (nrow,ncol)
    hp.ndelay=ndelay
    hp.blankimage = delaywindowpoolblankimage(nrow,ncol,ndelay,blankcolor)
    hp.alphapenaltyoperator = delaywindowpooloperator(laplacian2dmatrix(nrow,ncol),ndelay)
    return hp
end
ePPRHyperParams(nrowncol::Int;ndelay::Int=1,blankcolor=0.5)=ePPRHyperParams(nrowncol,nrowncol,ndelay=ndelay,blankcolor=blankcolor)

mutable struct ePPRCrossValidation
    trainpercent::Float64
    trainfold::Int
    testfold::Int
    traintestfold::Int
    trainindex::Int
    modelselectpvalue::Float64
    trains
    tests
end
ePPRCrossValidation() = ePPRCrossValidation(0.9,5,8,8,1,0.08,[],[])

function delaywindowpool(x::Matrix,ndelay::Int,blankcolor=0.5)
    if ndelay>1
        xcol=size(x,2);dwx=x
        for j in 1:ndelay-1
            dwx = [dwx [fill(blankcolor,j,xcol);x[1:end-j,:]]]
        end
        return dwx
    end
    return x
end
delaywindowpool(x::Matrix,hp::ePPRHyperParams)=delaywindowpool(x,hp.ndelay,hp.blankimage[1])

function delaywindowpooloperator(spatialoperator::Matrix,ndelay::Int)
    if ndelay>1
        nr,nc=size(spatialoperator)
        dwo = zeros(ndelay*nr,ndelay*nc)
        for j in 0:ndelay-1
            dwo[(1:nr)+j*nr, (1:nc)+j*nc] = spatialoperator
        end
        return dwo
    end
    return spatialoperator
end

delaywindowpoolblankimage(nrow::Int,ncol::Int,ndelay::Int=1,blankcolor=0.5)=fill(blankcolor,1,nrow*ncol*ndelay)

"""
Data partition for cross validation

n: data size
cv: cross validation
"""
function cvpartitionindex(n::Int,cv::ePPRCrossValidation,debug::ePPRDebugOptions=ePPRDebugOptions())
    ntrain = cv.trainpercent*n
    ntrainfold = ntrain/cv.trainfold
    ntraintestfold = Int(floor(ntrainfold/cv.traintestfold))
    ntrainfold = Int(ntraintestfold*cv.traintestfold)
    ntrain = Int(ntrainfold*cv.trainfold)
    trains=[]
    for tf in 0:cv.trainfold-1
        traintest = Any[tf*ntrainfold + (1:ntraintestfold)+ttf*ntraintestfold for ttf in 0:cv.traintestfold-1]
        train = setdiff(1:ntrain,tf*ntrainfold + (1:ntrainfold))
        push!(trains,Any[train,traintest])
    end
    ntestfold = Int(floor((n-ntrain)/cv.testfold))
    tests = Any[ntrain + (1:ntestfold)+tf*ntestfold for tf in 0:cv.testfold-1]
    debug.level>DebugNone && println("Cross Validation Data Partition, n = $n, ntrain = $ntrain in $(cv.trainfold)-fold, ntrainfold = $ntrainfold in $(cv.traintestfold)-fold, ntest = $(ntestfold*cv.testfold) in $(cv.testfold)-fold")
    cv.trains=trains;cv.tests=tests
    return cv
end

function cvmodel(models::Vector{ePPRModel},x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug.level>DebugNone && println("ePPR Cross Validation ...")
    train = hp.cv.trains[hp.cv.trainindex][1];traintest = hp.cv.trains[hp.cv.trainindex][2]
    # response and model predication
    traintestpredications = map(m->map(i->m(x[i,:]),traintest),models)
    traintestys = map(i->y[i],traintest)
    # correlation between response and predication
    traintestcors = map(mps->cor.(traintestys,mps),traintestpredications)
    debug.level>DebugFull && display(plotcor(models,traintestcors))
    # find the model no worse than models with more terms, and better than models with less terms
    mi=0;nmodel=length(models)
    for rm in 1:nmodel
        moretermp = [pvalue(SignedRankTest(traintestcors[rm],traintestcors[m]),tail=:left) for m in rm+1:nmodel]
        if rm==1 && all(moretermp .> hp.cv.modelselectpvalue)
            mi=rm
            break
        end
        if all(moretermp .> hp.cv.modelselectpvalue)
            lesstermp = [pvalue(SignedRankTest(traintestcors[m],traintestcors[rm]),tail=:left) for m in 1:rm-1]
            if all(lesstermp .< hp.cv.modelselectpvalue)
                mi=rm
                break
            end
        end
    end
    if mi==0
        warn("No model not worse than models with more terms, and better than models with less terms")
        return nothing
    end
    model = deepcopy(models[mi])

    # find drop terms that do not improve model predication
    droptermp = [pvalue(SignedRankTest(traintestcors[m-1],traintestcors[m]),tail=:left) for m in 2:nmodel]
    notimprove = find(droptermp .> hp.cv.modelselectpvalue)
    # find models with change level predication
    modelp = [pvalue(SignedRankTest(traintestcors[m]),tail=:both) for m in 1:nmodel]
    notpredictive = find(modelp .> hp.cv.modelselectpvalue)
    # spurious terms in the selected model
    spuriousterm = findin(model.index,hp.droptermindex[union(notimprove,notpredictive)])
    !isempty(spuriousterm) && deleteat!.(model,spuriousterm)

    return eppr(model,x[train,:],y[train],hp,debug)
end

function epprcv(x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    n = length(y);n !=size(x,1) && error("Length of x and y does not match!")
    cvpartitionindex(n,hp.cv,debug)
    train = hp.cv.trains[hp.cv.trainindex][1]
    px = delaywindowpool(x,hp)
    models = eppr(px[train,:],y[train],hp,debug)
    return cvmodel(models,px,y,hp,debug),models
end

"""
extended Projection Pursuit Regression
by minimizing ``f=\sum_{i=1}^N(y_i-\hat{y}(x_i))^2+\lambda\sum_{d=0}^D\sum_{m=1}^{M_d}\Vert{L\alpha_{m,d}}\Vert^2``

x: matrix with one image per row
y: vector of response
hp: hyper parameters
debug: debug options
"""
function eppr(x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    model = forwardstepwise(x,y,hp,debug)
    model = refitmodel(model,x,y,hp,debug)
    models = backwardstepwise(model,x,y,hp,debug)
end
function eppr(model::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    model = forwardstepwise(model,x,y,hp,debug)
    model,model.residuals = refitmodelbetas(model,y,debug)
    return model
end

function forwardstepwise(m::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    js = map(t->t[1],m.index);is = map(t->t[2],m.index)
    ujs = unique(js);jis=Dict(j=>is[js.==j] for j in ujs);njis=Dict(j=>length(jis[j]) for j in ujs)
    debug.level>DebugNone && println("ePPR Forward Stepwise ...")
    ym = mean(y);model = ePPRModel(ym);r=y-ym
    if hp.spatialtermfirst
        for j in ujs
            tx = j>0?[repmat(hp.blankimage,j);x[1:end-j,:]]:x
            for i in 1:njis[j]
                debug.level>DebugNone && println("Fit [Temporal-$j, Spatial-$i] New Term ...")
                α = normalize(m.alpha[ m.index .== [[j,jis[j][i]]] ][1],2)
                β,Φ,α,Φvs = fitnewterm(tx,r,α,hp.phidf,debug)
                r -= β*Φvs
                push!(model,β,Φ,α,[j,i],Φvs)
            end
        end
    else
        for i in 1:maximum(values(njis)),j in ujs
            i>njis[j] && continue
            debug.level>DebugNone && println("Fit [Temporal-$j, Spatial-$i] New Term ...")
            tx = j>0?[repmat(hp.blankimage,j);x[1:end-j,:]]:x
            α = normalize(m.alpha[ m.index .== [[j,jis[j][i]]] ][1],2)
            β,Φ,α,Φvs = fitnewterm(tx,r,α,hp.phidf,debug)
            r -= β*Φvs
            push!(model,β,Φ,α,[j,i],Φvs)
        end
    end
    model.residuals=r
    return model
end

function forwardstepwise(x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug.level>DebugNone && println("ePPR Forward Stepwise ...")
    if hp.ndelay>1
        hp.nft=hp.nft[1:1]
    end
    ym = mean(y);model = ePPRModel(ym);r=y-ym
    if hp.spatialtermfirst
        for j in 0:length(hp.nft)-1
            tx = j>0?[repmat(hp.blankimage,j);x[1:end-j,:]]:x
            for i in 1:hp.nft[j+1]
                debug.level>DebugNone && println("Fit [Temporal-$j, Spatial-$i] New Term ...")
                α = getinitialalpha(tx,r,debug)
                β,Φ,α,Φvs = fitnewterm(tx,r,α,hp,debug)
                r -= β*Φvs
                push!(model,β,Φ,α,[j,i],Φvs)
            end
        end
    else
        for i in 1:maximum(hp.nft),j in 0:length(hp.nft)-1
            i>hp.nft[j+1] && continue
            debug.level>DebugNone && println("Fit [Temporal-$j, Spatial-$i] New Term ...")
            tx = j>0?[repmat(hp.blankimage,j);x[1:end-j,:]]:x
            α = getinitialalpha(tx,r,debug)
            β,Φ,α,Φvs = fitnewterm(tx,r,α,hp,debug)
            r -= β*Φvs
            push!(model,β,Φ,α,[j,i],Φvs)
        end
    end
    model.residuals=r
    return model
end

function refitmodel(model::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug.level>DebugNone && println("ePPR Model Refit ...")
    model,r = refitmodelbetas(model,y,debug)
    for t in 1:length(model)
        oldloss = lossfun(model,y,hp)
        oldβ,oldΦ,oldα,index,oldΦvs = getterm(model,t)
        r += oldβ*oldΦvs

        j = index[1];i=index[2]
        tx = j>0?[repmat(hp.blankimage,j);x[1:end-j,:]]:x
        debug.level>DebugNone && println("Refit [Temporal-$j, Spatial-$i] New Term ...")
        β,Φ,α,Φvs = fitnewterm(tx,r,oldα,hp,debug,forward=false)
        setterm(model,t,β,Φ,α,index,Φvs)
        newloss = lossfun(model,y,hp)
        if newloss > oldloss
            debug.level>DebugNone && println("Model Loss increased from $oldloss to $newloss. Discard the new term, keep the old one.")
            setterm(model,t,oldβ,oldΦ,oldα,index,oldΦvs)
            r -= oldβ*oldΦvs
        else
            r -= β*Φvs
        end
    end
    model.residuals=r
    return model
end

function backwardstepwise(model::ePPRModel,x::Matrix,y::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug.level>DebugNone && println("ePPR Backward Stepwise ...")
    if debug.level>DebugFull
        display(plotalpha(model,hp))
        display(plotphi(model))
    end
    models=[deepcopy(model)];droptermindex=[]
    for i in length(model):-1:hp.mnbt+1
        model,index = dropleastimportantterm(model,debug)
        unshift!(droptermindex,index)
        model = refitmodel(model,x,y,hp,debug)
        if debug.level>DebugFull
            display(plotalpha(model,hp))
            display(plotphi(model))
        end
        unshift!(models,deepcopy(model))
    end
    hp.droptermindex = droptermindex
    return models
end

dropleastimportantterm(model::ePPRModel,debug::ePPRDebugOptions=ePPRDebugOptions())=dropterm(model,indmin(abs.(model.beta)),debug)

function dropterm(model::ePPRModel,i::Integer,debug::ePPRDebugOptions=ePPRDebugOptions())
    index = model.index[i]
    β=model.beta[i]
    debug.level>DebugNone && println("Drop Term: [temporal-$(index[1]), spatial-$(index[2])] with β: $(β).")
    deleteat!(model,i)
    return model,index
end

lossfun(g::Vector) = 0.5*norm(g,2)^2
"""
Loss function for term
f(α) = sum((r-Φ(x*α)).^2) + λ*norm(hp.alphapenaltyoperator*α,2)^2
"""
lossfun(r::Vector,x::Matrix,α::Vector,Φ,hp::ePPRHyperParams) = lossfun([r-Φ(x*α);sqrt(hp.lambda)*hp.alphapenaltyoperator*α])
"Loss function for model"
function lossfun(model::ePPRModel,y::Vector,hp::ePPRHyperParams)
    modelloss = lossfun(y-predict(model))
    penaltyloss = 0.5*hp.lambda*sum(norm.([hp.alphapenaltyoperator].*model.alpha,2).^2)
    return modelloss + penaltyloss
end

(Φro::RObject)(xα) = rcopy(R"predict($Φro, x=$xα)$y")
function fitnewterm(x::Matrix,r::Vector,α::Vector,hp::ePPRHyperParams,debug::ePPRDebugOptions=ePPRDebugOptions();forward::Bool=true)
    saturatediteration = 0;Φ=nothing;Φvs=nothing;xα=nothing;crt = forward?hp.forwardconvergerate:hp.refitconvergerate
    for i in 1:hp.newtermmaxiteration
        xα = x*α
        Φ = R"smooth.spline(y=$r, x=$xα, df=$(hp.phidf), spar=NULL, cv=FALSE)"
        Φvs = Φ(xα)
        f(a) = lossfun(r,x,a,Φ,hp)
        gt = r-Φvs;gp = sqrt(hp.lambda)*hp.alphapenaltyoperator*α;g = [gt;gp]
        debug.level>DebugNone && println("New Term $(i)th iteration. TermLoss: $(lossfun(gt)), PenaltyLoss: $(lossfun(gp)).")
        # Loss f(α) before trust region
        lossₒ = lossfun(g)
        Φ′vs = rcopy(R"predict($Φ, x=$xα, deriv=1)$y")
        gg = [-Φ′vs.*x;sqrt(hp.lambda)*hp.alphapenaltyoperator]'
        f′ = gg*g
        f″ = gg*gg'
        # α and Loss f(α) after trust region
        success,α,lossₙ = newtontrustregion(f,α,lossₒ,f′,f″,hp.trustregioninitsize,hp.trustregionmaxsize,hp.trustregioneta,hp.trustregionmaxiteration,debug)
        if !success
            warn("NewtonTrustRegion failed, New Term use initial α.")
            break
        end
        lossₙ > lossₒ && debug.level>DebugNone && println("New Term $(i)th iteration. Loss increased from $(lossₒ) to $(lossₙ).")
        cr = (lossₒ-lossₙ)/lossₒ
        if lossₙ < lossₒ && cr < crt
            saturatediteration+=1
            if saturatediteration >= hp.nsaturatediteration
                debug.level>DebugNone && println("New Term converged in $i iterations with (lossₒ-lossₙ)/lossₒ = $(cr).")
                break
            end
        else
            saturatediteration=0
        end
        i==hp.newtermmaxiteration && warn("New Term does not converge in $i iterations.")
    end
    β = std(Φvs)
    Φvs /=β
    si = sortperm(xα)
    Φ = Spline1D(xα[si], Φ(xα[si]), k=3, bc="extrapolate", s=50)
    return β,Φ,α,Φvs
end

function fitnewterm(x::Matrix,r::Vector,α::Vector,phidf::Int,debug::ePPRDebugOptions=ePPRDebugOptions())
    xα = x*α
    Φ = R"smooth.spline(y=$r, x=$xα, df=$phidf, spar=NULL, cv=FALSE)"
    Φvs = Φ(xα)
    β = std(Φvs)
    Φvs /=β
    si = sortperm(xα)
    Φ = Spline1D(xα[si], Φ(xα[si]), k=3, bc="extrapolate", s=50)
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
    eh = eigfact(Symmetric(H₀))
    posdef = isposdef(eh)
    qᵀg = eh[:vectors]'*g₀
    if posdef
        pˢ = -eh[:vectors]*(qᵀg./eh[:values])
        pˢₙ = norm(pˢ,2)
    end
    λe = eh[:values]-eh[:values][1]
    C1 = sum((qᵀg./λe)[2:end].^2)
    C2 = qᵀg[1]^2
    C3 = sum(qᵀg.^2)

    for i in 1:maxiteration
        debug.level>DebugBasic && println("NewtonTrustRegion $(i)th iteration, r = $(r)")
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
                    return 1/norm(qᵀg./(λe+λ),2) - 1/r
                end
                if ϕ(λup) <= 0
                    λ = λup
                elseif ϕ(λdn) >= 0
                    λ = λdn
                else
                    λ = fzero(ϕ,λdn,λup)
                end
                pᵢ = -eh[:vectors]*(qᵀg./(λe+λ))
            else
                iseasy = false
                ishard = true
                # hard-hard case
                w = qᵀg./λe
                w[1]=0
                τ = sqrt(r^2-C1)
                𝑧 = eh[:vectors][:,1]
                pᵢ = -eh[:vectors]*w + τ*𝑧
            end
        end
        # ρ: ratio of actual change versus predicted change
        xᵢ = x₀ + pᵢ
        fᵢ = f(xᵢ)
        ρ = (fᵢ - f₀) / (pᵢ'*g₀ + pᵢ'*H₀*pᵢ/2)
        # update trust region size
        if ρ < 0.25
            r /= 4
        elseif ρ > 0.75 && !islambdazero
            r = min(2r,rmax)
        end
        if debug.level>DebugBasic
            println("                                 ρ = $ρ")
            if islambdazero
                steptype="λ = 0"
            else
                if ishard
                    steptype=iseasy?"hard-easy":"hard-hard"
                else
                    steptype="easy"
                end
            end
            println("                                 step is $steptype")
        end
        # accept solution only once
        if ρ > η
            return true,xᵢ,fᵢ
        end
    end
    debug.level>DebugBasic && warn("NewtonTrustRegion does not converge in $maxiteration iterations.")
    return false,x₀,f₀
end

function getinitialalpha(x::Matrix,r::Vector,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug.level>DebugNone && println("Get Initial α ...")
    # RCall lm.ridge with kLW lambda
    α = rcopy(R"""
    lmr = lm.ridge($r ~ 0 + $x)
    lmr = lm.ridge($r ~ 0 + $x, lambda=lmr$kLW)
    coefficients(lmr)
    """)
    α-=mean(α);normalize!(α,2);α
end

function refitmodelbetas(model::ePPRModel,y::Vector,debug::ePPRDebugOptions=ePPRDebugOptions())
    debug.level>DebugNone && println("Refit βs ...")
    x = cat(2,model.phivalues...)
    res = lm(x,y-model.ymean)
    β = coef(res)
    if debug.level>DebugNone
        println("Old βs: $(model.beta)")
        println("New βs: $β")
    end
    model.beta = β
    return model,residuals(res)
end

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
    ldim=(nrow,ncol)
    lm = zeros(nrow*ncol,nrow*ncol)
    # fill center
    for r in 2:nrow-1, c in 2:ncol-1
        f=zeros(nrow,ncol)
        f[r-1:r+1,c-1:c+1]=center
        lm[sub2ind(ldim,r,c),:]=vec(f)
    end
    for c in 2:ncol-1
        # fill first row
        r=1;f=zeros(nrow,ncol)
        f[r:r+2,c-1:c+1]=firstrow
        lm[sub2ind(ldim,r,c),:]=vec(f)
        # fill last row
        r=nrow;f=zeros(nrow,ncol)
        f[r-2:r,c-1:c+1]=lastrow
        lm[sub2ind(ldim,r,c),:]=vec(f)
    end
    for r in 2:nrow-1
        # fill first col
        c=1;f=zeros(nrow,ncol)
        f[r-1:r+1,c:c+2]=firstcol
        lm[sub2ind(ldim,r,c),:]=vec(f)
        # fill last col
        c=ncol;f=zeros(nrow,ncol)
        f[r-1:r+1,c-2:c]=lastcol
        lm[sub2ind(ldim,r,c),:]=vec(f)
    end
    # fill top left
    r=1;c=1;f=zeros(nrow,ncol)
    f[r:r+2,c:c+2]=topleft
    lm[sub2ind(ldim,r,c),:]=vec(f)
    # fill top right
    r=1;c=ncol;f=zeros(nrow,ncol)
    f[r:r+2,c-2:c]=topright
    lm[sub2ind(ldim,r,c),:]=vec(f)
    # fill down left
    r=nrow;c=1;f=zeros(nrow,ncol)
    f[r-2:r,c:c+2]=downleft
    lm[sub2ind(ldim,r,c),:]=vec(f)
    # fill down right
    r=nrow;c=ncol;f=zeros(nrow,ncol)
    f[r-2:r,c-2:c]=downright
    lm[sub2ind(ldim,r,c),:]=vec(f)
    return lm
end

include("Visualization.jl")
end # module
