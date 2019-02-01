export plotmodel,plotalpha,plotphi,plotcor

function plotmodel(model::ePPRModel,hp::ePPRHyperParams;color=:coolwarm,xrange=-200:200,linkcolorbar=true)
    plot(plotalpha(model,hp,color=color,linkcolorbar=linkcolorbar),plotphi(model,hp,xrange),layout=grid(2,1),size=(650,850),link=:none)
end

function plotalpha(model::ePPRModel,hp::ePPRHyperParams;color=:coolwarm,linkcolorbar=true)
    ti = map(i->i.t+1,model.index);si = map(i->i.s,model.index);maxt=maximum(ti);mint=minimum(ti);maxs=maximum(si)
    utsmax=Dict(t=>maximum(si[ti.==t]) for t in unique(ti));ipn=prod(hp.imagesize);xpn=length(hp.xindex);αlim=0
    p = plot(layout=grid(maxt,maxs),yflip=true,framestyle=:none)
    for i in 1:length(model)
        t=ti[i];s=si[i];iα=model.alpha[i]
        α = mapfoldl(d->begin
                            if isempty(hp.xindex)
                                da = reshape(iα[(1:ipn).+d*ipn],hp.imagesize)
                            else
                                da = zeros(hp.imagesize)
                                da[hp.xindex]=iα[(1:xpn).+d*xpn]
                            end
                            da
                        end,(a1,a2)->[a1;a2],0:hp.ndelay-1)
        if linkcolorbar
            αlim=max(αlim,maximum(abs.(α)))
        end
        colorbar = linkcolorbar ? ( (t==mint && s==utsmax[t]) ? :right : :none ) : :right
        plot!(p[t,s],α,seriestype=:heatmap,color=color,ratio=:equal,colorbar=colorbar)
    end
    for t in 1:maxt
        ylabel!(p[t,1],"Delay $(hp.ndelay>1 ? "$(hp.ndelay-1) - 0" : t-1)")
    end
    linkcolorbar && plot!(p,clims=(-αlim,αlim))
    p
end
function plotphi(model::ePPRModel,hp::ePPRHyperParams,xrange=-200:200)
    minx,maxx=extrema(xrange);xtick=[minx,0,maxx]
    ti = map(i->i.t+1,model.index);si = map(i->i.s,model.index);tmax=maximum(ti);smax=maximum(si)
    p = plot(layout=grid(tmax,smax),leg=false,grid=false,framestyle=:none)
    for i in 1:length(model)
        t=ti[i];s=si[i]
        plot!(p[t,s],x->model.phi[i](x),minx,maxx,seriestype=:line,linewidth=2,link=:all,framestyle=:axes,title="β=$(round(model.beta[i],digits=3))")
        vline!(p[t,s],[0],linewidth=0.3,linecolor=:grey75);hline!(p[t,s],[0],linewidth=0.3,linecolor=:grey75)
        xaxis!(p[t,s],xtick=xtick)
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
plotcor(models::Vector{ePPRModel},cors)=plotcor(length.(models),cors,xlabel="Number of Terms")
function plotcor(x,cors;xlabel="Models")
    scatter(x,cat(cors...,dims=2)',ylabel="Pearson Correlation",xlabel=xlabel,leg=false,xtick=x)
end
