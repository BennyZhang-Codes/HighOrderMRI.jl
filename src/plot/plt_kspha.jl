"""
    fig = plt_kspha(kspha::AbstractArray{<:Real, 2}, dt::Real; kwargs...)

# Description
    Plots k coefficients of spherical harmonic terms using matplotlib 

# Arguments
- `kspha::AbstractArray{<:Real, 2}`: `[nSample, nTerm]` coefficients in units of rad, rad/m and rad/m^2.
- `dt::Real`: sampling time in seconds.

# Keywords
- `width`: (`::Real`, `=15`) figure's width
- `height`: (`::Real`, `=7`) figure's height
- `fontsize_label`: (`::Real`, `=7`) font size of labels
- `fontsize_legend`: (`::Real`, `=7`) font size of legend
- `fontsize_ticklabel`: (`::Real`, `=6`) font size of tick labels
- `color_facecolor`: (`::String`, `="#1F1F1F"`) background color of the figure  
- `color_label`: (`::String`, `="#CCCCCC"`) color of labels and tick labels
- `linewidth`: (`::Real`, `=0.5`) line width
- `ticklength`: (`::Real`, `=1.5`) length of ticks
- `pad_label`: (`::Real`, `=2`) padding between labels and tick labels
- `pad_labeltick`: (`::Real`, `=2`) padding between tick labels and axis

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> fig = plt_kspha(kspha, dt)
```
"""
function plt_kspha(
    kspha  :: AbstractArray{<:Real, 2} ,
    dt     :: Real                     ;
    width              = 15            ,
    height             = 7             ,
    fontsize_label     = 7             ,
    fontsize_legend    = 7             ,
    fontsize_ticklabel = 6             ,
    color_facecolor    = "#1F1F1F"     ,
    color_label        = "#CCCCCC"     ,
    linewidth          = 0.5           ,
    ticklength         = 1.5           ,
    pad_label          = 2             ,
    pad_labeltick      = 2             ,
    )
    matplotlib.rc("mathtext", default="regular")

    nSample, nTerm = size(kspha)
    @assert nTerm in [9, 16] "nTerm must be 9 or 16 for up to 2nd or 3rd order spherical harmonics"
    t = (1:nSample)*dt*1e3;
    nRow = nTerm == 9 ? 3 : 4;

    fig, axs = plt.subplots(nrows=nRow, ncols=1, figsize=(width/2.53999863, height/2.53999863), sharex=true, facecolor=color_facecolor, squeeze=false)

    for ax in axs
        ax.set_xlim(t[1]-1, t[end]+1);
        ax.set_facecolor(color_facecolor)
        ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
            color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
        for spine in ax.spines  # "left", "right", "bottom", "top"
            ax.spines[spine].set_visible(false)
        end
    end
    
    ax0, ax1, ax2 = axs[1,1], axs[2,1], axs[3,1];
    ax0.plot(t, kspha[:, 1], linewidth=linewidth, color=spha_color[1], label=spha_basis_latex[1])
    
    ax1.plot(t, kspha[:, 2], linewidth=linewidth, color=spha_color[2], label=spha_basis_latex[2])
    ax1.plot(t, kspha[:, 3], linewidth=linewidth, color=spha_color[3], label=spha_basis_latex[3])
    ax1.plot(t, kspha[:, 4], linewidth=linewidth, color=spha_color[4], label=spha_basis_latex[4])
    
    ax2.plot(t, kspha[:, 5], linewidth=linewidth, color=spha_color[5], label=spha_basis_latex[5])
    ax2.plot(t, kspha[:, 6], linewidth=linewidth, color=spha_color[6], label=spha_basis_latex[6])
    ax2.plot(t, kspha[:, 7], linewidth=linewidth, color=spha_color[7], label=spha_basis_latex[7])
    ax2.plot(t, kspha[:, 8], linewidth=linewidth, color=spha_color[8], label=spha_basis_latex[8])
    ax2.plot(t, kspha[:, 9], linewidth=linewidth, color=spha_color[9], label=spha_basis_latex[9])
    ax0.set_ylabel(L"0^{th} \ order \ [rad]"    , fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax1.set_ylabel(L"1^{st} \ order \ [rad/m]"  , fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax2.set_ylabel(L"2^{nd} \ order \ [rad/m^2]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    for ax in [ax0, ax1, ax2]
        ax.legend(bbox_to_anchor=(0.01, 1.1), fontsize=fontsize_legend, labelcolor=color_label, 
                scatteryoffsets=[0.5],
                ncols=5, loc="upper left", frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)
    end

    if nTerm == 16
        ax3 = axs[4, 1]
        ax3.plot(t, kspha[:, 10], linewidth=linewidth, color=spha_color[10], label=spha_basis_latex[10])
        ax3.plot(t, kspha[:, 11], linewidth=linewidth, color=spha_color[11], label=spha_basis_latex[11])
        ax3.plot(t, kspha[:, 12], linewidth=linewidth, color=spha_color[12], label=spha_basis_latex[12])
        ax3.plot(t, kspha[:, 13], linewidth=linewidth, color=spha_color[13], label=spha_basis_latex[13])
        ax3.plot(t, kspha[:, 14], linewidth=linewidth, color=spha_color[14], label=spha_basis_latex[14])
        ax3.plot(t, kspha[:, 15], linewidth=linewidth, color=spha_color[15], label=spha_basis_latex[15])
        ax3.plot(t, kspha[:, 16], linewidth=linewidth, color=spha_color[16], label=spha_basis_latex[16])
        ax3.set_ylabel(L"3^{rd} \ order \ [rad/m^3]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
        ax3.legend(bbox_to_anchor=(0.01, 1.1), fontsize=fontsize_legend, labelcolor=color_label, 
                scatteryoffsets=[0.5],
                ncols=4, loc="upper left", frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)
    end

    if nTerm == 9
        ax2.set_xlabel(L"Time \ [ms]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    elseif nTerm == 16
        ax3.set_xlabel(L"Time \ [ms]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    end
    
    fig.align_ylabels()
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    return fig
end


"""
    fig = plt_ksphas(ksphas::AbstractVector{T};, kspha2::AbstractArray{<:Real, 2}, dt::Float64; kwargs...)

# Description
    Plots k coefficients of spherical harmonic terms using matplotlib 

# Arguments
- `ksphas::AbstractVector{T};`: `[nSpha]` coefficients in units of rad, rad/m, rad/m^2 and rad/m^3.
- `dt::Real`: sampling time in seconds.

# Keywords
- `width`: (`::Real`, `=15`) figure's width
- `height`: (`::Real`, `=7`) figure's height
- `fontsize_label`: (`::Real`, `=7`) font size of labels
- `fontsize_legend`: (`::Real`, `=7`) font size of legend
- `fontsize_ticklabel`: (`::Real`, `=6`) font size of tick labels
- `color_facecolor`: (`::String`, `="#1F1F1F"`) background color of the figure  
- `color_label`: (`::String`, `="#CCCCCC"`) color of labels and tick labels
- `linewidth`: (`::Real`, `=0.5`) line width
- `ticklength`: (`::Real`, `=1.5`) length of ticks
- `pad_label`: (`::Real`, `=2`) padding between labels and tick labels
- `pad_labeltick`: (`::Real`, `=2`) padding between tick labels and axis

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> fig = plt_kspha_com(kspha1, kspha2, dt)
```
"""
function plt_ksphas(
    ksphas  :: AbstractVector{T}       ,
    dt      :: Real                    ;
    labels             = nothing       ,
    width              = 15            ,
    height             = 8             ,
    fontsize_label     = 7             ,
    fontsize_legend    = 7             ,
    fontsize_ticklabel = 6             ,
    color_facecolor    = "#1F1F1F"     ,
    color_label        = "#CCCCCC"     ,
    linewidth          = 0.5           ,
    ticklength         = 1.5           ,
    pad_label          = 2             ,
    pad_labeltick      = 2             ,
    ) where T<:AbstractArray{<:Real, 2}
    matplotlib.rc("mathtext", default="regular")
    if !isnothing(labels)
        @assert length(ksphas) == length(labels) "labels should have the same length as ys"
    else
        labels = ["kspha $(idx)" for idx = 1:length(ksphas)] 
    end

    for kspha in ksphas
        @assert size(kspha, 2) in [9, 16] "kspha should be a 2D array with 9 or 16 rows for 0th-2nd or 0th-3rd order terms"
    end

    nTerms   = [size(kspha, 2) for kspha in ksphas]
    nSamples = [size(kspha, 1) for kspha in ksphas]
    maxOrder = maximum(nTerms) == 9 ? 2 : 3;
    nRow = nCol = maximum(nTerms) == 9 ? 3 : 4;
    t_max = (1:maximum(nSamples))*dt*1e3;
    
    fig, axs = plt.subplots(nrows=nRow, ncols=nCol, figsize=(width/2.53999863, height/2.53999863),facecolor=color_facecolor, squeeze=false)
    for (kspha, label, nTerm) in zip(ksphas, labels, nTerms)
        for row = 1:nRow, col = 1:nCol
            ax = axs[row, col]
            term = (row-1)*nCol + col

            nSample = size(kspha, 1)
            t = (1:nSample)*dt*1e3;
            ax.plot(t, kspha[:, term], linewidth=linewidth, label=label)
            if term == nTerm
                break
            end
        end
    end

    for row = 1:nRow, col = 1:nCol
        ax = axs[row, col]
        term = (row-1)*nCol + col
    
        ax.set_xlim(t_max[1]-1, t_max[end]+1);
        ax.set_facecolor(color_facecolor)
        ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
            color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
        for spine in ax.spines  # "left", "right", "bottom", "top"
            # ax.spines[spine].set_linewidth(linewidth)
            ax.spines[spine].set_visible(false)
        end
    
        ax.set_xlabel(L"Time \ [ms]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)

        if term == 1
            ylabel = spha_basis_latex[term] * L"\ [rad]"
        elseif term <=4
            ylabel = spha_basis_latex[term] * L"\ [rad/m]"
        elseif term <= 9
            ylabel = spha_basis_latex[term] * L"\ [rad/m^2]"
        elseif term <= 16
            ylabel = spha_basis_latex[term] * L"\ [rad/m^3]"
        end
        ax.set_ylabel(ylabel, fontsize=fontsize_label, color=spha_color[term], labelpad=pad_label)
        
        ax.legend(bbox_to_anchor=(0.01, 1.05), fontsize=fontsize_legend, labelcolor=color_label, 
                scatteryoffsets=[0.5],
                ncols=3, loc="upper left", frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)
    end
    
    fig.align_ylabels()
    fig.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    return fig
end