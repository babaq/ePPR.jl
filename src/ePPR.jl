__precompile__(false)
module ePPR

import Base:length
export ePPRModel,ePPRHyperParams,predict,ePPRFit,ForwardStepwise,RefitModel,BackwardStepwise,Laplacian2DMatrix,
ePPRDebugOptions,DebugNone,DebugBasic,
DelayWindowed,DelayWindowedOperator

using GLM,GLMNet,Dierckx,Optim
"""
``\hat{y}_i=\bar{y}+\sum_{d=0}^D\sum_{m=1}^{M_d}\beta_{m,d}\phi_{m,d}(\alpha_{m,d}^Tx_{i-d})``
"""
mutable struct ePPRModel
    "𝑦̄"
    ymean
    "vector of β for each term"
    beta
    "vector of Φ for each term"
    phi
    "vector of α for each term"
    alpha
    "vector of (temporal,spatial) index for each term"
    index
    "vector of ``\phi_{m,d}(\alpha_{m,d}^TX_{-d})`` for each term"
    phivalues
end
ePPRModel(ymean) = ePPRModel(ymean,[],[],[],[],[])
length(m::ePPRModel)=length(m.beta)
predict(m::ePPRModel)=m.ymean+squeeze(sum(cat(2,(m.beta.*m.phivalues)...),2),2)
#predict(m::ePPRModel,x::Matrix)=m.ymean+sum(m.beta.* x*m.alpha)

"""
Hyper Parameters for ePPR
"""
mutable struct ePPRHyperParams
    "number of forward terms for each delay. [3, 2, 1] means 3 spatial terms for delay0, 2 for delay1, 1 for delay2"
    nft::Vector{Int}
    "penalization parameter λ"
    lambda::Float64
    "Φ Spline smoothness"
    smooth::Float64
    "minimum number of backward terms"
    mnbt::Int
    "whether to fit all spatial terms before moving to next temporal"
    isspatialtermfirst::Bool
    "α priori for penalization"
    alphapenaltyoperator
    "`(loss₁-loss₂)/loss₁`, degree of convergence threshold to decide a saturated iteration"
    convergentpercent::Float64
    "number of consecutive saturated iterations to decide a solution"
    convergentiteration::Int
    "max number of iterations to fit new term"
    newtermmaxiteration::Int
    "initial size of trust region"
    trustregioninitsize::Float64
    "max size of trust region"
    trustregionmaxsize::Float64
    "η of trust region"
    trustregioneta::Float64
    "options for Optim"
    optimoptions
    "row vector of blank image"
    blankimage
end
ePPRHyperParams()=ePPRHyperParams([1],15,1,1,true,[],0.005,2,100,1,5,1/5,Optim.Options(iterations=100),[])
function ePPRHyperParams(nrow::Int,ncol::Int,blankcolor=0)
    hp=ePPRHyperParams()
    hp.blankimage = fill(blankcolor,1,nrow*ncol)
    hp.alphapenaltyoperator = Laplacian2DMatrix(nrow,ncol)
    return hp
end
ePPRHyperParams(nrowncol::Int,blankcolor=0)=ePPRHyperParams(nrowncol,nrowncol,blankcolor)

function DelayWindowed(x,ndelay,blankcolor=0)
    if ndelay>1
        xcol=size(x,2);dwx=x
        for j in 1:ndelay-1
            dwx = [dwx [fill(blankcolor,j,xcol);x[1:end-j,:]]]
        end
        return dwx
    end
    return x
end
function DelayWindowedOperator(spatialoperator,ndelay)
    if ndelay>1
        nr,nc=size(spatialoperator)
        dwo = zeros(ndelay*nr,ndelay*nc)
        for j in 1:ndelay
            dwo[1+(j-1)*nr:nr*j,1+(j-1)*nc:nc*j] = spatialoperator
        end
        return dwo
    end
    return spatialoperator
end

const DebugNone=0
const DebugBasic=1
mutable struct ePPRDebugOptions
    level::Int
end
ePPRDebugOptions()=ePPRDebugOptions(DebugNone)

"""
x: matrix with one image per row
y: vector of response
hp: hyperparameters
debug: debugoptions
"""
function ePPRFit(x,y,hp=ePPRHyperParams(sqrt(size(x,2))),debug=ePPRDebugOptions())
    model,r = ForwardStepwise(x,y,hp,debug)
    model,r = RefitModel(model,x,y,hp,debug)
    models = BackwardStepwise(model,x,y,hp,debug)
end

function ForwardStepwise(x,y,hp,debug=ePPRDebugOptions())
    if debug.level>DebugNone
        println("Start Forward Stepwise Fitting ...")
    end
    ym = mean(y);model = ePPRModel(ym);r=y-ym
    if hp.isspatialtermfirst
        for j in 0:length(hp.nft)-1
            tx = j>0?[repmat(hp.blankimage,j);x[1:end-j,:]]:x
            for i in 1:hp.nft[j+1]
                α = GetInitialAlpha(tx,r,debug)
                if debug.level>DebugNone
                    println("Fitting (Temporal-$j, Spatial-$i) New Term ...")
                end
                β,Φ,α,Φvs = FitNewTerm(tx,r,α,hp,debug)
                r -= β*Φvs
                push!(model.beta,β)
                push!(model.phi,Φ)
                push!(model.alpha,α)
                push!(model.phivalues,Φvs)
                push!(model.index,(j,i))
            end
        end
    else
        for i in 1:maximum(hp.nft),j in 0:length(hp.nft)-1
            i>hp.nft[j+1] && continue
            tx = j>0?[repmat(hp.blankimage,j);x[1:end-j,:]]:x
            α = GetInitialAlpha(tx,r,debug)
            if debug.level>DebugNone
                println("Fitting (Temporal-$j, Spatial-$i) New Term ...")
            end
            β,Φ,α,Φvs = FitNewTerm(tx,r,α,hp,debug)
            r -= β*Φvs
            push!(model.beta,β)
            push!(model.phi,Φ)
            push!(model.alpha,α)
            push!(model.phivalues,Φvs)
            push!(model.index,(j,i))
        end
    end
    return model,r
end

function BackwardStepwise(model,x,y,hp,debug=ePPRDebugOptions())
    if debug.level>DebugNone
        println("Start Backward Stepwise Fitting ...")
    end
    models=[deepcopy(model)]
    for i in length(model):-1:hp.mnbt+1
        β,Φvs,model = DropLeastImportantTerm(model,debug)
        model,r = RefitModel(model,x,y,hp,debug)
        push!(models,deepcopy(model))
    end
    return models
end

function DropLeastImportantTerm(model,debug=ePPRDebugOptions())
    i= indmin(abs.(model.beta))
    index = model.index[i]
    β=model.beta[i]
    Φvs=model.phivalues[i]
    if debug.level>DebugNone
        println("Droping Least Important Term: (temporal-$(index[1]), spatial-$(index[2])) with β: $(β).")
    end
    deleteat!(model.beta,i)
    deleteat!(model.phi,i)
    deleteat!(model.alpha,i)
    deleteat!(model.phivalues,i)
    deleteat!(model.index,i)
    return β,Φvs,model
end

function LossValue(model,y,hp)
    dataloss = 0.5*norm(y-predict(model),2)^2
    penaltyloss = 0.5*hp.lambda*sum(norm.([hp.alphapenaltyoperator].*model.alpha,2).^2)
    return dataloss+penaltyloss
end

function RefitModel(model,x,y,hp,debug=ePPRDebugOptions())
    if debug.level>DebugNone
        println("Start Model ReFitting ...")
    end
    model,r = RefitModelBetas(model,y,debug)
    for t in 1:length(model)
        oldloss = LossValue(model,y,hp)
        oldβ=model.beta[t]
        oldΦ=model.phi[t]
        oldΦvs=model.phivalues[t]
        oldα = model.alpha[t]
        index = model.index[t]
        r += oldβ*oldΦvs

        j = index[1];i=index[2]
        if j>0
            tx=[repmat(hp.blankimage,j);x[1:end-j,:]]
        else
            tx=x
        end
        if debug.level>DebugNone
            println("ReFitting (Temporal-$j, Spatial-$i) New Term ...")
        end
        β,Φ,α,Φvs = FitNewTerm(tx,r,oldα,hp,debug)

        model.beta[t]=β
        model.phi[t]=Φ
        model.alpha[t]=α
        model.phivalues[t]=Φvs
        newloss = LossValue(model,y,hp)
        if newloss > oldloss
            if debug.level>DebugNone
                warn("New Term Model Loss increased from $oldloss to $newloss. Discard the new term, keep the old one.")
            end
            model.beta[t]=oldβ
            model.phi[t]=oldΦ
            model.alpha[t]=oldα
            model.phivalues[t]=oldΦvs
            r -= oldβ*oldΦvs
        else
            r -= β*Φvs
        end
    end
    return model,r
end

function FitNewTerm(x,r,α,hp,debug=ePPRDebugOptions())
    saturateiteration = 0;Φ=nothing;Φvs=nothing
    for i in 1:hp.newtermmaxiteration
        xα = x*α;si = sortperm(xα)
        Φ = Spline1D(xα[si],r[si],k=3,s=hp.smooth,bc="extrapolate")
        Φvs = Φ(xα)
        # f(a) = sum((r-Φ(x*a))^2) + λ*norm(hp.alphapenaltyoperator*a,2)^2
        f(a) = 0.5*norm([r-Φ(x*a);sqrt(hp.lambda)*hp.alphapenaltyoperator*a],2)^2
        fg(a) = 0.5*norm(a,2)^2
        gd = r-Φvs;gp = sqrt(hp.lambda)*hp.alphapenaltyoperator*α;g = [gd;gp]
        if debug.level>DebugNone
            println("New Term $(i)th iteration DataLoss: $(fg(gd)), PenaltyLoss: $(fg(gp)).")
        end
        loss1 = fg(g) # = f(α)
        Φ′ = derivative(Φ,xα,nu=1)
        gg = [-Φ′.*x;sqrt(hp.lambda)*hp.alphapenaltyoperator]'
        f′ = gg*g
        f″ = gg*gg'
        g!(storage,a)=  storage = f′
        h!(storage,a)=  storage = f″
        res = optimize(f, g!, h!, α,
        NewtonTrustRegion(initial_delta=hp.trustregioninitsize,delta_hat=hp.trustregionmaxsize,eta=hp.trustregioneta),hp.optimoptions)
        α = Optim.minimizer(res)
        loss2 = Optim.minimum(res)
        if loss2 < loss1
            cp = (loss1-loss2)/loss1
            if cp < hp.convergentpercent
                saturateiteration+=1
                if saturateiteration >= hp.convergentiteration
                    if debug.level>DebugNone
                        println("New Term converged in $i iterations with (loss1-loss2)/loss1 = $(cp).")
                    end
                    break
                end
            else
                saturateiteration=0
            end
        else
            if debug.level>DebugNone
                warn("New Term $(i)th iteration Loss increased from $loss1 to $loss2")
            end
        end
        if i==hp.newtermmaxiteration && debug.level>DebugNone
            warn("New Term does not converge after $i iterations.")
        end
    end
    β = std(Φvs)
    Φvs /=β
    return β,Φ,α,Φvs
end

function GetInitialAlpha(x,r,debug=ePPRDebugOptions())
    # Ridge Regression
    # a = ridge(x,r,1,trans=false,bias=true)
    # a=a[1:end-1]

    if debug.level>DebugNone
        println("Getting Initial α ...")
    end
    # ElasticNet Regularization, alpha=1 for Lasso, alpha=0 for Ridge
    cv = glmnetcv(x,r,alpha=0)
    α = cv.path.betas[:,indmin(cv.meanloss)]
    α-=mean(α);normalize!(α,2);α
end

function RefitModelBetas(model,y,debug=ePPRDebugOptions())
    if debug.level>DebugNone
        println("Refitting Model βs ...")
    end
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

function Laplacian2DMatrix(nrow::Int,ncol::Int)
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



end # module
