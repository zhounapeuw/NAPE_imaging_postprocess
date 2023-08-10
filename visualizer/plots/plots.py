import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from visualizer import misc
import os

class EventTicksPlot:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.fig = go.Figure()

    def create_traces(self):
        for idx_roi in np.arange(self.data_processor.fparams['num_rois']):
            self.fig.add_trace(
                go.Scatter(
                    visible=False,
                    name='Activity',
                    line=dict(color="green", width=1.5),
                    x=self.data_processor.tvec,
                    y=self.data_processor.signal_to_plot[idx_roi, :]))

        if self.data_processor.events_content:
            for idx_cond, cond in enumerate(self.data_processor.conditions):
                for idx_ev, event in enumerate(self.data_processor.event_times[cond]):
                    if idx_ev == 0:
                        self.fig.add_trace(go.Scatter(x=[event, event], y=[self.data_processor.min_max_all[0], self.data_processor.min_max_all[1]], visible=True,
                                                      mode='lines',
                                                      line=dict(color=self.data_processor.cond_colors[idx_cond], width=1.5, dash='dash'),
                                                      showlegend=True, legendgroup=cond,
                                                      name='{}'.format(cond)))
                    else:
                        self.fig.add_trace(go.Scatter(x=[event, event], y=[self.data_processor.min_max_all[0], self.data_processor.min_max_all[1]], visible=True,
                                                      mode='lines',
                                                      showlegend=False, legendgroup=cond,
                                                      line=dict(color=self.data_processor.cond_colors[idx_cond], width=1.5, dash='dash'),
                                                      name='{} {}'.format(cond, str(idx_ev))))

    def create_sliders(self):
        steps = []
        for iroi in np.arange(self.data_processor.fparams['num_rois']):
            step = dict(
                method="restyle",
                args=[{"visible": ([False] * self.data_processor.fparams['num_rois']) + [True] * (len(self.fig.data) - self.data_processor.fparams['num_rois'])},
                      {"title": "Viewing ROI " + str(iroi)},
                ])
            step["args"][0]["visible"][iroi] = True
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Viewing "},
            pad={"t": 50},
            steps=steps
        )]

        self.fig.update_layout(sliders=sliders)

        for idx in np.arange(self.data_processor.fparams['num_rois']):
            self.fig['layout']['sliders'][0]['steps'][idx]['label'] = 'ROI ' + str(idx)

    def setup_layout(self):
        self.fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Fluorescence ({})".format(self.data_processor.fparams['flag_normalization']),
            legend_title="Legend Title",
            font=dict(size=12),
            showlegend=True,
            legend_title_text='Legend',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        self.fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black')
        self.fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black')
    
    def generate_session_plot(self):
        self.create_traces()
        self.create_sliders()
        self.setup_layout()

        self.fig.data[0].visible = True
        return self.fig
    
class S2PActivityPlot:
    def __init__(self, data_processor):
        self.data_processor = data_processor

    def generate_contour_plot(self, package="plotly"):
        if package == "matplotlib":
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(self.data_processor.s2p_data_dict['ops']['meanImg'], cmap='gray',
                    vmin=np.min(self.data_processor.s2p_data_dict['ops']['meanImg']) / 3,
                    vmax=np.max(self.data_processor.s2p_data_dict['ops']['meanImg']) / 3)
            ax.axis('off')

            idx_color_rois = 0
            for idx, roi_id in enumerate(self.data_processor.plot_vars['cell_ids']):
                if roi_id in self.data_processor.plot_vars['rois_to_tseries'] or self.data_processor.plot_vars[
                    'color_all_rois']:
                    this_roi_color = self.data_processor.plot_vars['colors_roi_name'][idx_color_rois]
                    idx_color_rois += 1
                else:
                    this_roi_color = 'grey'
                ax.contour(self.data_processor.plot_vars['s2p_masks'][idx, :, :], colors=[this_roi_color], linewidths=0.5)
                if self.data_processor.show_labels and roi_id in self.data_processor.plot_vars['rois_to_tseries']:
                    ax.text(self.data_processor.plot_vars['roi_centroids'][idx][1] - 1,
                            self.data_processor.plot_vars['roi_centroids'][idx][0] - 1, str(roi_id),
                            fontsize=18, weight='bold', color=this_roi_color)

            return plt
        else:
            to_plot = self.data_processor.s2p_data_dict['ops']['meanImg']
            fig = go.Figure()

            # Add the image as a heatmap trace
            heatmap_trace = go.Heatmap(z=to_plot,
                                    colorscale='gray',
                                    zmin=np.min(to_plot),
                                    zmax=np.max(to_plot),
                                    showscale=False)  # Set showscale to False to hide the colorbar
            fig.add_trace(heatmap_trace)

            # Create a scatter trace for each contour
            for idx, roi_id in enumerate(self.data_processor.plot_vars['cell_ids']):
                if roi_id in self.data_processor.plot_vars['rois_to_tseries'] or self.data_processor.plot_vars['color_all_rois']:
                    this_roi_color = self.data_processor.plot_vars['colors_roi'][idx]
                else:
                    this_roi_color = 'grey'

                # Find the contour points for this ROI
                contours = np.where(self.data_processor.plot_vars['s2p_masks'][idx] > 0)

                # Create scatter trace with mode 'lines' to draw contour lines
                contour_trace = go.Scatter(x=contours[1],
                                        y=contours[0],
                                        mode='lines',
                                        line=dict(color=this_roi_color),
                                        name=f"ROI {roi_id}")
                fig.add_trace(contour_trace)

                # Add the ROI label
                if self.data_processor.show_labels and roi_id in self.data_processor.plot_vars['rois_to_tseries']:
                    fig.add_annotation(text=str(roi_id),
                                    x=self.data_processor.plot_vars['roi_centroids'][idx][1] - 1,
                                    y=self.data_processor.plot_vars['roi_centroids'][idx][0] - 1,
                                    font=dict(family="Arial", size=18, color=this_roi_color),
                                    showarrow=False)

            # Remove axis ticks and labels
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)

            # Adjust margins and size
            fig.update_layout(margin=dict(l=10, b=10, t=30), font=dict(family="Arial", size=15))

            return fig

    def generate_time_series_plot(self, package="plotly"):
        if package == "matplotlib":
            if self.data_processor.plot_vars['num_rois_to_tseries'] > 1:
                fig, ax = plt.subplots(self.data_processor.plot_vars['num_rois_to_tseries'], 1,
                                    figsize=(9, 2 * self.data_processor.plot_vars['num_rois_to_tseries']))
                for idx in range(self.data_processor.plot_vars['num_rois_to_tseries']):
                    to_plot = self.data_processor.s2p_data_dict['F_npil_corr_dff'][self.data_processor.plot_vars['rois_to_tseries'][idx]]
                    tvec = np.linspace(0, to_plot.shape[0] / self.data_processor.s2p_data_dict['ops']['fs'], to_plot.shape[0])
                    ax[idx].plot(tvec, np.transpose(to_plot), color=self.data_processor.plot_vars['colors_roi_name'][idx])

                    ax[idx].set_title(f"ROI {self.data_processor.plot_vars['rois_to_tseries'][idx]}")
                    ax[idx].tick_params(axis='both', which='major', labelsize=13)
                    ax[idx].tick_params(axis='both', which='minor', labelsize=13)

                    if self.data_processor.path_dict['tseries_start_end']:
                        ax.set_xlim(self.data_processor.path_dict['tseries_start_end'][0], self.data_processor.path_dict['tseries_start_end'][1])

                plt.subplots_adjust(hspace=0.5)
                plt.setp(ax, xlim=None, ylim=[np.min(self.data_processor.s2p_data_dict['F_npil_corr_dff']) * 1.1,
                                                np.max(self.data_processor.s2p_data_dict['F_npil_corr_dff']) * 1.1])

                ax[idx].set_xlabel('Time (s)', fontsize=16)
                ax[idx].yaxis.set_label_coords(-0.06, 1) 
                fig.text(0.05, 0.5, 'Fluorescence Level', va='center', rotation='vertical', fontsize=16)

                return plt
            else:
                fig, ax = plt.subplots(figsize=(9, 2))  # You can adjust the figsize as needed
    
                to_plot = self.data_processor.s2p_data_dict['F_npil_corr_dff'][self.data_processor.plot_vars['rois_to_tseries'][0]]
                tvec = np.linspace(0, to_plot.shape[0] / self.data_processor.s2p_data_dict['ops']['fs'], to_plot.shape[0])
                ax.plot(tvec, np.transpose(to_plot), color=self.data_processor.plot_vars['colors_roi_name'][0])

                ax.set_title(f"ROI {self.data_processor.plot_vars['rois_to_tseries'][0]}", fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=13)
                ax.tick_params(axis='both', which='minor', labelsize=13)

                ax.set_xlabel('Time (s)', fontsize=16)
                ax.set_ylabel('Fluorescence Level', fontsize=16, fontweight='500')

                if self.data_processor.path_dict['tseries_start_end']:
                    ax.set_xlim(self.data_processor.path_dict['tseries_start_end'][0], self.data_processor.path_dict['tseries_start_end'][1])

                plt.setp(ax, xlim=None, ylim=[np.min(self.data_processor.s2p_data_dict['F_npil_corr_dff']) * 1.1, np.max(self.data_processor.s2p_data_dict['F_npil_corr_dff']) * 1.1])

                return plt
        else:
            tvec, trace_data_selected = self.data_processor.generate_tsv_and_trace()

            # Create a DataFrame for the trace data
            df_trace_data = pd.DataFrame(trace_data_selected.T, columns=[f"ROI {roi}" for roi in self.data_processor.plot_vars['rois_to_tseries']])
            df_trace_data['Time (s)'] = tvec

            # Melt the DataFrame to have a 'variable' column for ROI names and 'value' column for trace values
            df_trace_data_melted = pd.melt(df_trace_data, id_vars=['Time (s)'], value_vars=[f"ROI {roi}" for roi in self.data_processor.plot_vars['rois_to_tseries']])

            colors_dict = {}

            for roi, i in enumerate(self.data_processor.plot_vars['rois_to_tseries']):
                colors_dict[f"ROI {roi}"] = self.data_processor.plot_vars['colors_roi'][i]

            # Create the figure using plotly.express
            fig = px.line(df_trace_data_melted, x='Time (s)', y='value', color='variable', color_discrete_map=colors_dict)

            # Update layout
            fig.update_layout(
                margin=dict(l=20, b=10, t=30),
                xaxis_title="Time (s)",
                yaxis_title="Fluorescence Level",
                showlegend=True,
                font=dict(family="Arial", size=15)
            )

            return fig

    def generate_heatmap_plot(self, package="plotly"):
        tvec, trace_data_selected = self.data_processor.generate_tsv_and_trace()

        if package == "matplotlib":
            # Create the heatmap plot
            fig, ax = plt.subplots(figsize=(10, 6))
            heatmap = ax.imshow(trace_data_selected, extent=[tvec[0], tvec[-1], self.data_processor.plot_vars['cell_ids'][0], self.data_processor.plot_vars['cell_ids'][-1]], aspect='auto', cmap='viridis')

            # Add colorbar
            cbar = plt.colorbar(heatmap)
            cbar.set_label('Fluorescence (dF/F)')

            # Set labels and title
            ax.set_xlabel("Time (s)", fontsize=16)
            ax.set_ylabel("ROI", fontsize=16)

            ax.set_yticks(self.data_processor.rois_to_plot)
            ax.set_yticklabels(self.data_processor.rois_to_plot)

            ax.set_title("Heatmap of Fluorescence Activity", fontsize=16)

            return plt
        else:
            trace = go.Heatmap(z=trace_data_selected, x=tvec, y=self.data_processor.plot_vars['cell_ids'])
            fig = go.Figure(data=trace)

            fig.update_layout(
                margin=dict(l=60, b=50, t=40),
                xaxis_title="Time (s)",
                yaxis_title="ROI",
                font=dict(family="Arial", size=15)
            )

            return fig

class EventAnalysisPlot:
    def __init__(self, data_processor):
        self.data_processor = data_processor
    
    def get_cmap(self, n, name='plasma'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    # calculate all the color limits for heatmaps; useful for locking color limits across different heatmap subplots   
    def generate_clims(self, data_in, norm_type):
        # get min and max for all data across conditions 
        clims_out = [np.nanmin(data_in), np.nanmax(data_in)]
        if 'zscore' in norm_type: # if data are zscored, make limits symmetrical and centered at 0
            clims_max = np.max(np.abs(clims_out)) # then we take the higher of the two magnitudes
            clims_out = [-clims_max*0.5, clims_max*0.5] # and set it as the negative and positive limit for plotting
        return clims_out

    def generate_individual_line_graph(self, iROI):
        line_shades = []

        for idx, cond in enumerate(self.data_processor.conditions):
            # prep data to plot
            num_trials = self.data_processor.data_dict[cond]['num_trials']
            to_plot = np.nanmean(self.data_processor.data_dict[cond][self.data_processor.fparams["data_trial_resolved_key"]][:,iROI,:], axis=0)
            to_plot_err = np.nanstd(self.data_processor.data_dict[cond][self.data_processor.fparams["data_trial_resolved_key"]][:,iROI,:], axis=0)/np.sqrt(num_trials)
            
            # plot trace
            line = self.ax.plot(self.data_processor.tvec, to_plot)
            
            if self.data_processor.fparams['flag_trial_avg_errbar']:
                shade = self.ax.fill_between(self.data_processor.tvec, to_plot - to_plot_err, to_plot + to_plot_err, alpha=0.5)
                line_shades.append((line[0], shade))
            else:
                line_shades.append(line[0])
            
        # plot x, y labels, and legend
        self.ax.set_ylabel(self.data_processor.fparams["ylabel"])
        self.ax.set_xlabel('Time [s]')
        self.ax.set_title('ROI # {}; Trial-avg'.format(str(iROI)))
        self.ax.legend(self.data_processor.conditions, fontsize=12, loc='upper left')
        self.ax.legend(line_shades, self.data_processor.conditions, fontsize=12, loc='upper left')
        self.ax.autoscale(enable=True, axis='both', tight=True)
        self.ax.axvline(0, color='0.5', alpha=0.65) # plot vertical line for time zero
        self.ax.annotate('', xy=(self.data_processor.event_bound_ratio[0], -0.01), xycoords='axes fraction', 
                                    xytext=(self.data_processor.event_bound_ratio[1], -0.01), 
                                    arrowprops=dict(arrowstyle="-", color='g'))
        self.ax.tick_params(axis = 'both', which = 'major')

    def generate_heatmap(self, data_in, tvec, event_bound_ratio, clims, title, subplot_index, cond, cmap_='inferno'):
        # set imshow extent to replace x and y axis ticks/labels
        plot_extent = [tvec[0], tvec[-1], data_in.shape[0], 0] # [x min, x max, y min, ymax]

        # prep labels; plot x and y labels for first subplot
        if subplot_index == (0, 0) or subplot_index == 0 :
            self.ax[subplot_index].set_ylabel('Trial')
            self.ax[subplot_index].set_xlabel('Time [s]');
        self.ax[subplot_index].tick_params(axis = 'both', which = 'major')

        # prep the data
        to_plot = np.squeeze(data_in)
        if len(self.data_processor.event_frames[cond]) == 1: # accomodates single trial data
            to_plot = to_plot[np.newaxis, :]

        # plot the data
        im = misc.subplot_heatmap(self.ax[subplot_index], title, to_plot, cmap=cmap_, clims=clims, extent_=plot_extent)

        # add meta data lines
        self.ax[subplot_index].axvline(0, color='0.5', alpha=1) # plot vertical line for time zero
        # plots green horizontal line indicating event duration
        self.ax[subplot_index].annotate('', xy=(event_bound_ratio[0], -0.01), xycoords='axes fraction', 
                                xytext=(event_bound_ratio[1], -0.01), 
                                arrowprops=dict(arrowstyle="-", color='g'))

        cbar = self.fig.colorbar(im, ax = self.ax[subplot_index], shrink = 0.5)
        cbar.ax.set_ylabel(self.data_processor.fparams["ylabel"])
    
    def sort_heatmap_peaks(self, data, tvec, sort_epoch_start_time, sort_epoch_end_time, sort_method = 'peak_time'):
        
        # find start/end samples for epoch
        sort_epoch_start_samp = self.tvec2samp(tvec, sort_epoch_start_time)
        sort_epoch_end_samp = self.tvec2samp(tvec, sort_epoch_end_time)
        
        if sort_method == 'peak_time':
            epoch_peak_samp = np.argmax(data[:,sort_epoch_start_samp:sort_epoch_end_samp], axis=1)
            final_sorting = np.argsort(epoch_peak_samp)
        elif sort_method == 'max_value':
    
            time_max = np.nanmax(data[:,sort_epoch_start_samp:sort_epoch_end_samp], axis=1)
            final_sorting = np.argsort(time_max)[::-1]

        return final_sorting

    def plot_trial_avg_heatmap(self, data_in, conditions, tvec, event_bound_ratio, cmap, clims, sorted_roi_order = None, rois_oi = None):
        num_subplots = len(conditions)
        n_columns = np.min([num_subplots, 3.0])
        n_rows = int(np.ceil(num_subplots/n_columns))

        # set imshow extent to replace x and y axis ticks/labels (replace samples with time)
        plot_extent = [tvec[0], tvec[-1], self.data_processor.num_rois, 0 ]

        fig, ax = plt.subplots(nrows=n_rows, ncols=int(n_columns), figsize = (n_columns*5, n_rows*4))
        if not isinstance(ax,np.ndarray): # this is here to make the code below compatible with indexing a single subplot object
            ax = [ax]

        for idx, cond in enumerate(conditions):

            # determine subplot location index
            if n_rows == 1:
                subplot_index = idx
            else:
                subplot_index = np.unravel_index(idx, (n_rows, int(n_columns))) # turn int index to a tuple of array coordinates

            # prep labels; plot x and y labels for first subplot
            if subplot_index == (0, 0) or subplot_index == 0 :
                ax[subplot_index].set_ylabel('ROI #')
                ax[subplot_index].set_xlabel('Time [s]');
            ax[subplot_index].tick_params(axis = 'both', which = 'major')
            
            # plot the data
            if sorted_roi_order is not None:
                roi_order = sorted_roi_order
            else:
                roi_order = slice(0, self.data_processor.num_rois)
            to_plot = data_in[cond][self.data_processor.fparams["data_trial_avg_key"]][roi_order,:] # 

            im = misc.subplot_heatmap(ax[subplot_index], cond, to_plot, cmap=self.data_processor.fparams["cmap_"], clims=clims, extent_=plot_extent)
            ax[subplot_index].axvline(0, color='k', alpha=0.3) # plot vertical line for time zero
            ax[subplot_index].annotate('', xy=(event_bound_ratio[0], -0.01), xycoords='axes fraction', 
                                        xytext=(event_bound_ratio[1], -0.01), 
                                        arrowprops=dict(arrowstyle="-", color='g'))
            if rois_oi is not None:
                for ROI_OI in rois_oi:
                    ax[subplot_index].annotate('', xy=(1.005, 1-(ROI_OI/self.data_processor.num_rois)-0.015), xycoords='axes fraction', 
                                            xytext=(1.06, 1-(ROI_OI/self.data_processor.num_rois)-0.015), 
                                            arrowprops=dict(arrowstyle="->", color='k'))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        cbar = fig.colorbar(im, ax = ax, shrink = 0.7)
        cbar.ax.set_ylabel(self.data_processor.fparams["ylabel"], fontsize=13)
        
        # hide empty subplot
        for a in ax.flat[num_subplots:]:
            a.axis('off')
        
        return fig
    
    def generate_bar_graph(self):
        ### Quantification of roi-, trial-, time-averaged data
        fig, ax = plt.subplots(1,1, figsize = (10,6))
        analysis_window = self.data_processor.fparams['event_sort_analysis_win']
        analysis_win_samps = [ misc.find_nearest_idx(self.data_processor.tvec, time)[0] for time in analysis_window ]

        to_plot = []
        to_plot_err = []

        for idx, cond in enumerate(self.data_processor.conditions):
            line_color = self.cmap_lines(idx)
            # first trial avg the data
            trial_avg = np.nanmean(self.data_processor.data_dict[cond]['zdata'], axis=0)
            
            # z-score trial-avg data for each respective ROI
            # apply zscore function to each row of data
            apply_axis = 1 
            zscore_trial_avg = np.apply_along_axis(misc.zscore_, apply_axis, trial_avg, self.data_processor.baseline_svec)
            
            # take avg across time
            zscore_trial_time_avg = np.nanmean(zscore_trial_avg[:,analysis_win_samps[0]:analysis_win_samps[1],:], axis=1)
            
            # take avg/std across ROIs
            zscore_roi_trial_time_avg = np.nanmean(zscore_trial_time_avg, axis=0)
            zscore_roi_trial_time_std = np.nanstd(zscore_trial_time_avg, axis=0)
            
            to_plot.append(zscore_roi_trial_time_avg[0])
            to_plot_err.append(zscore_roi_trial_time_std[0]/np.sqrt(len(zscore_trial_time_avg)))
            
        barlist = ax.bar(self.data_processor.conditions, to_plot, yerr=to_plot_err, align='center', alpha=0.5, ecolor='black', capsize=10 )
        for idx in range(len(self.data_processor.conditions)):
            barlist[idx].set_color(self.cmap_lines(idx))
        ax.set_ylabel('Normalized Fluorescence', fontsize=13)
        ax.set_title('ROI Time-averaged Quant', fontsize=15)
        ax.yaxis.grid(True)
        ax.tick_params(axis = 'both', which = 'major')
        ax.tick_params(axis = 'x', which = 'major', rotation = 45)

        return fig
    
    def generate_line_graph(self):
        line_shades = []
        fig, axs = plt.subplots(1,1, figsize = (10,6))
        for idx, cond in enumerate(self.data_processor.conditions):
            line_color = self.cmap_lines(idx)
            # first trial avg the data
            trial_avg = np.nanmean(self.data_processor.data_dict[cond]['zdata'], axis=0)
            
            # z-score trial-avg data for each respective ROI
            # apply zscore function to each row of data
            app_axis = 1 
            zscore_trial_avg = np.apply_along_axis(misc.zscore_, app_axis, trial_avg, self.data_processor.baseline_svec)
            
            # take avg/std across ROIs
            zscore_roi_trial_avg = np.nanmean(zscore_trial_avg, axis=0)
            zscore_roi_trial_std = np.nanstd(zscore_trial_avg, axis=0)
            
            to_plot = np.squeeze(zscore_roi_trial_avg)
            to_plot_err = np.squeeze(zscore_roi_trial_std)/np.sqrt(self.data_processor.num_rois)
            
            axs.plot(self.data_processor.tvec, to_plot, color=line_color)
            if self.data_processor.fparams['opto_blank_frame']:
                line = axs.plot(self.data_processor.tvec[self.data_processor.t0_sample:self.data_processor.event_end_sample], to_plot[self.data_processor.t0_sample:self.data_processor.event_end_sample], marker='', color=line_color)
            else:
                line = axs.plot(self.data_processor.tvec[self.data_processor.t0_sample:self.data_processor.event_end_sample], to_plot[self.data_processor.t0_sample:self.data_processor.event_end_sample], marker='', color=line_color)
            
            if self.data_processor.fparams['flag_roi_trial_avg_errbar']:
                shade = axs.fill_between(self.data_processor.tvec, to_plot - to_plot_err, to_plot + to_plot_err, color = line_color,
                            alpha=0.2) # this plots the shaded error bar
                line_shades.append((line[0],shade))
                    
        axs.set_ylabel(self.data_processor.fparams["ylabel"])
        axs.set_xlabel('Time [s]')
        axs.set_title('ROI Trial-avg')
        axs.legend(self.data_processor.conditions)
        axs.legend(line_shades, self.data_processor.conditions, fontsize=15)
        axs.axvline(0, color='0.5', alpha=0.65) # plot vertical line for time zero
        axs.annotate('', xy=(self.data_processor.event_bound_ratio[0], -0.01), xycoords='axes fraction', 
                                    xytext=(self.data_processor.event_bound_ratio[1], -0.01), 
                                    arrowprops=dict(arrowstyle="-", color='g'))
        axs.tick_params(axis = 'both', which = 'major')
        axs.autoscale(enable=True, axis='both', tight=True)

        axs.set_ylim([-1.5, 10])

        return fig
    
    def generate_roi_plots(self):
        self.cmap_lines = self.get_cmap(len(self.data_processor.conditions))

        num_subplots = len(self.data_processor.conditions) # plus one for trial-avg traces
        n_columns = np.min([num_subplots, 4.0])
        n_rows = int(np.ceil(num_subplots/n_columns))

        all_plots = []

        for iROI in range(self.data_processor.num_rois):
            plots = {}
            
            # calculate color limits. This is outside of heatmap function b/c want lims across conditions
            # loop through each condition's data and flatten before concatenating values
            roi_clims = self.generate_clims(np.concatenate([self.data_processor.data_dict[cond][self.data_processor.fparams["data_trial_resolved_key"]][:, iROI, :].flatten() for cond in self.data_processor.conditions]), self.data_processor.fparams['flag_normalization'])
            
            self.fig, self.ax = plt.subplots(nrows=n_rows, ncols=int(n_columns), 
                                figsize=(n_columns*4, n_rows*3),
                                constrained_layout=True)
            
            ### Plot heatmaps for each condition
            for idx_cond, cond in enumerate(self.data_processor.conditions):
                
                subplot_index = self.data_processor.subplot_loc(idx_cond, n_rows, n_columns) # determine subplot location index
                data_to_plot = self.data_processor.data_dict[cond][self.data_processor.fparams["data_trial_resolved_key"]][:, iROI, :]
                title = 'ROI {}; {}'.format(str(iROI), cond)
                
                self.generate_heatmap(data_to_plot, self.data_processor.tvec, self.data_processor.event_bound_ratio, roi_clims, title, subplot_index, cond, self.data_processor.fparams["cmap_"])
            
            plots["heatmap"] = self.fig
            
            self.fig, self.ax = plt.subplots(1,1, figsize = (5, 4))
            self.generate_individual_line_graph(iROI)
            plots["linegraph"] = self.fig

            all_plots.append(plots)
        
        self.tvec2samp = lambda tvec, time: np.argmin(np.abs(tvec - time))

        # if flag is true, sort ROIs (usually by average fluorescence within analysis window)
        if self.data_processor.fparams['flag_sort_rois']:
            if not self.data_processor.fparams['roi_sort_cond']: # if no condition to sort by specified, use first condition
                self.data_processor.fparams['roi_sort_cond'] = self.data_processor.data_dict.keys()[0]
            if not self.data_processor.fparams['roi_sort_cond'] in self.data_processor.data_dict.keys():
                sorted_roi_order = range(self.data_processor.num_rois)
                interesting_rois = self.data_processor.fparams['interesting_rois']
                print('Specified condition to sort by doesn\'t exist! ROIs are in default sorting.')
            else:
                # returns new order of rois sorted using the data and method supplied in the specified window
                sorted_roi_order = self.sort_heatmap_peaks(self.data_processor.data_dict[self.data_processor.fparams['roi_sort_cond']]['ztrial_avg_data'], self.data_processor.tvec, sort_epoch_start_time=0, 
                                sort_epoch_end_time = self.data_processor.trial_start_end_sec[-1], 
                                sort_method = self.data_processor.fparams['user_sort_method'])
                # finds corresponding interesting roi (roi's to mark with an arrow) order after sorting
                interesting_rois = np.in1d(sorted_roi_order, self.data_processor.fparams['interesting_rois']).nonzero()[0] 
        else:
            sorted_roi_order = range(self.data_processor.num_rois)
            interesting_rois = self.data_processor.fparams['interesting_rois']

        if not self.data_processor.all_nan_rois[0].size == 0:
            set_diff_keep_order = lambda main_list, remove_list : [i for i in main_list if i not in remove_list]
            sorted_roi_order = set_diff_keep_order(sorted_roi_order, self.data_processor.all_nan_rois)
            interesting_rois = [i for i in self.data_processor.fparams['interesting_rois'] if i not in self.data_processor.all_nan_rois]
        
        heatmap_fig = self.plot_trial_avg_heatmap(self.data_processor.data_dict, self.data_processor.conditions, self.data_processor.tvec, self.data_processor.event_bound_ratio, self.data_processor.fparams["cmap_"], clims = self.generate_clims(np.concatenate([self.data_processor.data_dict[cond][self.data_processor.fparams["data_trial_avg_key"]].flatten() for cond in self.data_processor.conditions]), self.data_processor.fparams['flag_normalization']), sorted_roi_order = sorted_roi_order, rois_oi = interesting_rois)

        all_plots.append({
            "heatmap": heatmap_fig,
            "bargraph": self.generate_bar_graph(),
            "linegraph": self.generate_line_graph()
        })

        return all_plots