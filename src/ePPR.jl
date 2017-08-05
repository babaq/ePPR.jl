module ePPR

export ePPR,ePPRModel,ePPRHyperParams


mutable struct ePPRModel
  ymean
  beta
  phi
  alpha
end

mutable struct ePPRHyperParams
  isspatialtermfirst = true
end

"""
y: vector of responses
x: matrix with one image per row
ntd: vector of number of terms for each delay
"""
function ePPR(y,x,fts,λ,s,mbt,hp)
  model,r = ForwardStepwise(y,x,fts,λ,s,hp)
  models = BackwardStepwise(model,r,x,λ,s,mbt,hp)
  model = SelectModel(models)
end

function ForwardStepwise(y,x,fts,λ,s,hp)
  ym = mean(y);r=y-ym
  for j in 1:length(fts)
    for i in 1:fts[j]
      a = GetInitialAlpha(r,x)
      β[j][i],Φ[j][i],α[j][i] = FitNewTerm(r,x,a,λ,s)
      r -= β[j][i]*Φ[j][i](α[j][i]*x)
    end
    # get shifted x
    sx = x
    x = last(x)
  end
  return ePPRModel(ym,β,Φ,α),r
end

function BackwardStepwise(model,r,x,λ,s,mbt,hp)
  models=[deepcopy(model)]
  for i in length(model):mbt+1
  #for m = reverse(collect(2:length(model)))
    β,Φ,α,model = DropInsignificantTerm(model)
    r += β*Φ(α*x)
    model,r = RefitModel(model,r,x,λ,s,hp)
    models=[models;deepcopy(model)]
  end
  return models
end

function DropInsignificantTerm(model)
end

function RefitModel(model,r,x,λ,s,hp)
  model,r = RefitModelBetas(model,r)
  for i in 1:length(model)
    β,Φ,α,delay,model = RemoveTerm(i,model)
    xi = x[i-delay]
    r += β*Φ(α*xi)
    β,Φ,α = FitNewTerm(r,xi,α,λ,s,hp)
    model = AddTerm(β,Φ,α,model) # need index
    r -= β*Φ(α*xi)
  end
  return model,r
end

function FitNewTerm(r,x,α,λ,s,hp)
while true

end
end

function GetInitialAlpha(r,x)
  a = ridge(x,r,1,trans=false,bias=false)
  a-=mean(a);normalize!(a,2);a
end

function RefitModelBetas(model,r)
  # lm
end


end # module
