"""
    fig = plt_bfield(bfield::AbstractArray{<:Real, 2}, dt::Float64; kwargs...)

# Description
    Plots k coefficients of spherical harmonic terms using matplotlib 

# Arguments
- `bfield::AbstractArray{<:Real, 2}`: `[nSample, nTerms]` coefficients in units of rad, rad/m and rad/m^2.
- `dt::Float64`: sampling time in seconds.

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
julia> fig = plt_bfield(bfield, dt)
```
"""
function plt_bfield(
    bfield::AbstractArray{<:Real, 2},
    dt::Float64;
    width              = 15       ,
    height             = 7        ,
    fontsize_label     = 7        ,
    fontsize_legend    = 7        ,
    fontsize_ticklabel = 6        ,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    linewidth          = 0.5      ,
    ticklength         = 1.5      ,
    pad_label          = 2        ,
    pad_labeltick      = 2        ,
    )
    @assert size(bfield, 2) >= 9 "bfield should be a 2D array with 9 rows for 0th-2nd order terms"
    nSample = size(bfield, 1)
    t = (1:nSample)*dt*1e3;
    
    color_dfc  = ["#5f4690", "#1d6996", "#38a6a5", "#0f8554", "#73af48", "#edad08", "#e17c05", "#cc503e", "#94346e"];
    legend_dfc = [L"1", L"x", L"y", L"z", L"xy", L"zy", L"2z^2-(x^2+y^2)", L"xz", L"x^2-y^2"];
    
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(width/2.53999863, height/2.53999863), sharex=true, facecolor=color_facecolor, squeeze=false)
    
    ax0, ax1, ax2 = axs;
    
    for ax in axs
        ax.set_xlim(t[1]-1, t[end]+1);
        ax.set_facecolor(color_facecolor)
        ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
            color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
        for spine in ax.spines  # "left", "right", "bottom", "top"
            # ax.spines[spine].set_linewidth(linewidth)
            ax.spines[spine].set_visible(false)
        end
    end
    
    ax0.plot(t, bfield[:, 1], linewidth=linewidth, color=color_dfc[1], label=legend_dfc[1])
    
    ax1.plot(t, bfield[:, 2], linewidth=linewidth, color=color_dfc[2], label=legend_dfc[2])
    ax1.plot(t, bfield[:, 3], linewidth=linewidth, color=color_dfc[3], label=legend_dfc[3])
    ax1.plot(t, bfield[:, 4], linewidth=linewidth, color=color_dfc[4], label=legend_dfc[4])
    
    ax2.plot(t, bfield[:, 5], linewidth=linewidth, color=color_dfc[5], label=legend_dfc[5])
    ax2.plot(t, bfield[:, 6], linewidth=linewidth, color=color_dfc[6], label=legend_dfc[6])
    ax2.plot(t, bfield[:, 7], linewidth=linewidth, color=color_dfc[7], label=legend_dfc[7])
    ax2.plot(t, bfield[:, 8], linewidth=linewidth, color=color_dfc[8], label=legend_dfc[8])
    ax2.plot(t, bfield[:, 9], linewidth=linewidth, color=color_dfc[9], label=legend_dfc[9])
    for ax in axs
        ax.legend(bbox_to_anchor=(0.01, 1.1), fontsize=fontsize_legend, labelcolor=color_label, 
                scatteryoffsets=[0.5],
                ncols=5, loc="upper left", frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)
    end
    ax0.set_ylabel(L"0^{th} \ order \ [mT]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax1.set_ylabel(L"1^{st} \ order \ [mT/m]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax2.set_ylabel(L"2^{nd} \ order \ [mT/m^2]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    
    ax2.set_xlabel(L"Time \ [ms]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    
    fig.align_ylabels()
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    return fig
end


"""
    fig = plt_bfield_com(bfield1::AbstractArray{<:Real, 2}, bfield2::AbstractArray{<:Real, 2}, dt::Float64; kwargs...)

# Description
    Plots k coefficients of spherical harmonic terms using matplotlib 

# Arguments
- `bfield1::AbstractArray{<:Real, 2}`: `[nSample, nTerms]` coefficients in units of rad, rad/m and rad/m^2.
- `dt::Float64`: sampling time in seconds.

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
julia> fig = plt_bfield_com(bfield1, bfield2, dt)
```
"""
function plt_bfield_com(
    bfield1::AbstractArray{<:Real, 2},
    bfield2::AbstractArray{<:Real, 2},
    dt::Float64;
    width              = 15       ,
    height             = 8        ,
    fontsize_label     = 7        ,
    fontsize_legend    = 7        ,
    fontsize_ticklabel = 6        ,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    linewidth          = 0.5      ,
    ticklength         = 1.5      ,
    pad_label          = 2        ,
    pad_labeltick      = 2        ,
    )

    @assert size(bfield1, 2) >= 9 "bfield should be a 2D array with 9 rows for 0th-2nd order terms"
    nSample = size(bfield1, 1)
    t = (1:nSample)*dt*1e3;
    
    color_dfc  = ["#5f4690", "#1d6996", "#38a6a5", "#0f8554", "#73af48", "#edad08", "#e17c05", "#cc503e", "#94346e"];
    legend_dfc = [L"1", L"x", L"y", L"z", L"xy", L"zy", L"2z^2-(x^2+y^2)", L"xz", L"x^2-y^2"];
    
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(width/2.53999863, height/2.53999863),facecolor=color_facecolor, squeeze=false)
    
    for row = 1:3, col = 1:3
        ax = axs[row, col]
        term = (row-1)*3 + col
    
        ax.set_xlim(t[1]-1, t[end]+1);
        ax.set_facecolor(color_facecolor)
        ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
            color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
        for spine in ax.spines  # "left", "right", "bottom", "top"
            # ax.spines[spine].set_linewidth(linewidth)
            ax.spines[spine].set_visible(false)
        end
    
        ax.plot(t, bfield1[:, term], linewidth=linewidth, label="bfield 1")
        ax.plot(t, bfield2[:, term], linewidth=linewidth, label="bfield 2")
        ax.set_xlabel(L"Time \ [ms]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
        
        if term == 1
            ylabel = legend_dfc[term] * L"\ [mT]"
        elseif term <=4
            ylabel = legend_dfc[term] * L"\ [mT/m]"
        elseif term <= 9
            ylabel = legend_dfc[term] * L"\ [mT/m^2]"
        end
        ax.set_ylabel(ylabel, fontsize=fontsize_label, color=color_dfc[term], labelpad=pad_label)
        ax.legend(bbox_to_anchor=(0.01, 1.05), fontsize=fontsize_legend, labelcolor=color_label, 
                scatteryoffsets=[0.5],
                ncols=5, loc="upper left", frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)
    end
    
    fig.align_ylabels()
    fig.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    return fig
end