import os
import numpy as np
import glob
import pickle
import json
import seaborn as sns
import matplotlib.ticker as ticker
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
import util_funcs as utils

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

        if self.data_processor.fparams['fname_events']:
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
                    this_roi_color = plt.cm.viridis(idx_color_rois)
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
            show_labels_=True
            cmap_scale_ratio=1

            to_plot = self.data_processor.s2p_data_dict['ops']['meanImg']
            fig = go.Figure()

            # Add the image as a heatmap trace
            heatmap_trace = go.Heatmap(z=to_plot,
                                    colorscale='gray',
                                    zmin=np.min(to_plot) * (1.0 / cmap_scale_ratio),
                                    zmax=np.max(to_plot) * (1.0 / cmap_scale_ratio),
                                    showscale=False)  # Set showscale to False to hide the colorbar
            fig.add_trace(heatmap_trace)

            # Create a scatter trace for each contour
            for idx, roi_id in enumerate(self.data_processor.plot_vars['cell_ids']):
                if roi_id in self.data_processor.plot_vars['rois_to_tseries']:
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
                if show_labels_ and roi_id in self.data_processor.plot_vars['rois_to_tseries']:
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
            fig, ax = plt.subplots(self.data_processor.plot_vars['num_rois_to_tseries'], 1,
                                figsize=(9, 2 * self.data_processor.plot_vars['num_rois_to_tseries']))
            for idx in range(self.data_processor.plot_vars['num_rois_to_tseries']):
                to_plot = self.data_processor.s2p_data_dict['F_npil_corr_dff'][self.data_processor.plot_vars['rois_to_tseries'][idx]]
                tvec = np.linspace(0, to_plot.shape[0] / self.data_processor.s2p_data_dict['ops']['fs'], to_plot.shape[0])
                ax[idx].plot(tvec, np.transpose(to_plot), color=self.data_processor.plot_vars['colors_roi_name'][idx])

                ax[idx].set_title(f"ROI {self.data_processor.plot_vars['rois_to_tseries'][idx]}")
                ax[idx].tick_params(axis='both', which='major', labelsize=13)
                ax[idx].tick_params(axis='both', which='minor', labelsize=13)
                if idx == np.ceil(self.data_processor.plot_vars['num_rois_to_tseries'] / 2 - 1):
                    ax[idx].set_ylabel('Fluorescence Level', fontsize=20)

            plt.setp(ax, xlim=None, ylim=[np.min(self.data_processor.s2p_data_dict['F_npil_corr_dff']) * 1.1,
                                            np.max(self.data_processor.s2p_data_dict['F_npil_corr_dff']) * 1.1])

            ax[idx].set_xlabel('Time (s)', fontsize=20)

            return plt
        else:
            tvec, trace_data_selected = self.generate_tsv_and_trace()

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
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ROI")
            ax.set_title("Heatmap of Fluorescence Activity")

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