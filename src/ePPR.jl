__precompile__(false)
module ePPR

import Base:length
export ePPRModel,ePPRHyperParams,ePPRFit

using GLM,GLMNet,MultivariateStats,Dierckx,Optim
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
length(x::ePPRModel)=length(x.beta)
predict(x::ePPRModel)=x.ymean+sum(x.beta.*x.phivalue)

"""
Hyper Parameters for ePPR
"""
mutable struct ePPRHyperParams
  "true to fit all spatial terms then move onto next temporal"
  isspatialtermfirst::Bool
  "alpha priori for penalization"
  alphapenaltyoperator
  "(loss₁-loss₂)/loss₁, degree of convergence threshold to decide a saturated iteration"
  convergentpercent
  "number of consecutive saturated iterations to decide a solution"
  convergentiteration
  "max number of iterations to fit new term"
  newtermiterationmax
  "inital size of trust region"
  trustregioninitsize
  "max size of trust region"
  trustregionmaxsize
  "η of trust region"
  trustregioneta
  "options for Optim"
  optimoptions
  "row vector of blank image"
  blankimage
end
ePPRHyperParams()=ePPRHyperParams(true,[],0.005,2,100,1,5,1/5,Optim.Options(iterations=100),[])

"""
x: matrix with one image per row
y: vector of responses
fts: number of forward terms for each delay
λ:
s:
mbt: minimum number of backward terms
hp: hyperparameters
"""
function ePPRFit(x,y,fts,λ,s,mbt,hp)
  model,r = ForwardStepwise(x,y,fts,λ,s,hp)
  #models = BackwardStepwise(model,x,y,r,λ,s,mbt,hp)
  #model = SelectModel(models)
end

function ForwardStepwise(x,y,fts,λ,s,hp)
  ym = mean(y);model = ePPRModel(ym);r=y-ym
  if hp.isspatialtermfirst
      for j in 0:length(fts)-1
          if j>0
            tx=[repmat(hp.blankimage,j);x[1:end-j,:]]
          else
            tx=x
          end
        for i in 1:fts[j+1]
          α = GetInitialAlpha(tx,r)
          β,Φ,α,Φvs = FitNewTerm(tx,r,α,λ,s,hp)
          r -= β*Φvs
          push!(model.beta,β)
          push!(model.phi,Φ)
          push!(model.alpha,α)
          push!(model.phivalues,Φvs)
          push!(model.index,(j,i))
        end
      end
  else
  end
  return model,r
end

function BackwardStepwise(model,x,y,r,λ,s,mbt,hp)
  models=[deepcopy(model)]
  for i in length(model):-1:mbt+1
    β,Φvs,model = DropInsignificantTerm(model)
    r += β*Φvs
    model,r = RefitModel(model,x,y,r,λ,s,hp)
    push!(models,deepcopy(model))
  end
  return models
end

function DropInsignificantTerm(model)
  i= indmin(abs(model.beta))
  β=model.beta[i]
  Φvs=model.phivalues[i]
  deleteat!(model.beta,i)
  deleteat!(model.phi,i)
  deleteat!(model.alpha,i)
  deleteat!(model.phivalues,i)
  deleteat!(model.index,i)
  return β,Φvs,model
end

function RefitModel(model,x,y,r,λ,s,hp)
  model,r = RefitModelBetas(model,y)
  for i in 1:length(model)
      β=model.beta[1]
      Φvs=model.phivalues[1]
      index = model.index[1]
      r += β*Φvs
      deleteat!(model.beta,1)
      deleteat!(model.phi,1)
      deleteat!(model.alpha,1)
      deleteat!(model.phivalues,1)
      deleteat!(model.index,1)

    j = index[1]
    if j>0
      tx=[repmat(hp.blankimage,j);x[1:end-j,:]]
    else
      tx=x
    end
    β,Φ,α,Φvs = FitNewTerm(tx,r,α,λ,s,hp)
    r -= β*Φvs
    push!(model.beta,β)
    push!(model.phi,Φ)
    push!(model.alpha,α)
    push!(model.phivalues,Φvs)
    push!(model.index,index)
  end
  return model,r
end

function FitNewTerm(x,r,α,λ,s,hp)
  saturateiteration = 0;Φ=nothing;Φvs=nothing
  for i in 1:hp.newtermiterationmax
    xα = x*α;si = sortperm(xα)
    Φ = Spline1D(xα[si],r[si],k=3,s=s,bc="extrapolate")
    Φvs = Φ(xα)
    # f = a->sum((r-Φ(x*a))^2) + λ*norm(hp.alphasmoothoperator*a)^2
    f(a) = 0.5*norm([r-Φ(x*a);sqrt(λ)*hp.alphapenaltyoperator*a],2)^2
    g = [r-Φvs;sqrt(λ)*hp.alphapenaltyoperator*α]
    loss1 = 0.5*norm(g,2)^2 # f(α)
    Φ′ = derivative(Φ,xα,nu=1)
    gg = [-Φ′*x;sqrt(λ)*hp.alphapenaltyoperator']
    f′ = gg*g
    f″ = gg*gg'
    g!(storage,a)=  storage = f′
    h!(storage,a)=  storage = f″
    res = optimize(f, g!, h!, α,
     NewtonTrustRegion(inital_delta=hp.trustregioninitsize,delta_hat=hp.trustregionmaxsize,eta=hp.trustregioneta),
     hp.optimoption)
    α = Optim.minimizer(res)
    loss2 = Optim.minimum(res)
    if loss2 < loss1
      if (loss1-loss2)/loss1 < hp.convergentpercent
        saturateiteration+=1
        saturateiteration == hp.convergentiteration && break
      else
        saturateiteration=0
      end
    end
  end
  β = std(Φvs)
  Φvs /=β
  return β,Φ,α,Φvs
end

function GetInitialAlpha(x,r)
  # Ridge Regression
  # a = ridge(x,r,1,trans=false,bias=true)
  # a=a[1:end-1]

  # ElasticNet Regularization, alpha=1 for Lasso, alpha=0 for Ridge
  cv = glmnetcv(x,r,alpha=0)
  α = cv.path.betas[:,indmin(cv.meanloss)]
  α-=mean(α);normalize!(α,2);α
end

function RefitModelBetas(model,y)
  x = cat(2,model.phivalues...)
  res = lm(x,y-model.ymean)
  model.beta = coef(res)
  return model,residuals(res)
end


end # module
