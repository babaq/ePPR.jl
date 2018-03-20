export plotalpha,plotphi,plotcor

using Plots
plotlyjs()
clibrary(:colorcet)

function plotalpha(model::ePPRModel,hp::ePPRHyperParams)
    js = map(t->t[1]+1,model.index);is = map(t->t[2],model.index);maxj=maximum(js);maxi=maximum(is);np=prod(hp.imagesize)
    p = plot(layout=grid(maxj,maxi),yflip=true,leg=false,framestyle=:none)
    for t in 1:length(model)
        j=js[t];i=is[t]
        α = mapfoldl(d->reshape(normalize(model.alpha[t],2)[(1:np)+d*np],hp.imagesize),(a0,a1)->[a0;a1],0:hp.ndelay-1)
        plot!(p[j,i],α,seriestype=:heatmap,color=:fire,ratio=:equal)
    end
    for j in 1:maxj
        ylabel!(p[j,1],"Delay$(j-1)")
    end
    p
end
function plotphi(model::ePPRModel,xrange=-200:200)
    minx,maxx=extrema(xrange);xtick=[minx,0,maxx]
    js = map(t->t[1]+1,model.index);is = map(t->t[2],model.index);maxj=maximum(js);maxi=maximum(is)
    p = plot(layout=grid(maxj,maxi),leg=false,framestyle=:none)#,grid=false)
    for t in 1:length(model)
        j=js[t];i=is[t]
        plot!(p[j,i],x->model.phi[t](x),minx,maxx,seriestype=:line,linewidth=2,link=:all,framestyle=:axes,title="β=$(round(model.beta[t],3))")
        xaxis!(p[j,i],xtick=xtick)
        if j < maxj
            xaxis!(p[j,i],xformatter=_->"")
        end
    end
    for j in 1:maxj
        ylabel!(p[j,1],"Delay$(j-1)")
    end
    p
end
function plotcor(models::Vector{ePPRModel},cors)
    x = length.(models)
    scatter(x,cat(2,cors...)',ylabel="Correlation Coefficients",xlabel="Number of Terms",leg=false,xtick=x)
end
