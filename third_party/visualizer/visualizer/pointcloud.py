from flask import Flask, render_template_string
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.cm as cm
from termcolor import cprint
import os


def visualize_pointcloud(pointcloud, color:tuple=None, img_path=None):
    vis = Visualizer()
    return vis.visualize_pointcloud(pointcloud, color=color, img_path=img_path)
    

def visualize_pointcloud_and_save(pointcloud, color:tuple=None, save_path=None):
    vis = Visualizer()
    vis.visualize_pointcloud_and_save(pointcloud, color=color, save_path=save_path)
    
class Visualizer:
    def __init__(self):
        self.app = Flask(__name__)
        self.pointclouds = []
        
    def _generate_trace(self, pointcloud, color:tuple=None, size=5, opacity=0.7):
        x_coords = pointcloud[:, 0]
        y_coords = pointcloud[:, 1]
        z_coords = pointcloud[:, 2]

        if pointcloud.shape[1] == 3:
            if color is None:
                # # design a colorful point cloud based on 3d coordinates
                # # Normalize coordinates to range [0, 1]
                # min_coords = pointcloud.min(axis=0)
                # max_coords = pointcloud.max(axis=0)
                # normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
                # try:
                #     # Use normalized coordinates as RGB values
                #     colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
                # except: # maybe meet NaN error
                #     # use simple cyan color
                #     colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]


                # Normalize coordinates to range [0, 1]
                min_coords = pointcloud.min(axis=0)
                max_coords = pointcloud.max(axis=0)
                normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)

                # Choose a dimension to base the heatmap on, e.g., the Z-coordinate
                z_values = normalized_coords[:, 2]  # assuming Z is the third column

                # Normalize z_values to [0, 1]
                z_min = z_values.min()
                z_max = z_values.max()
                normalized_z = (z_values - z_min) / (z_max - z_min)

                # Apply a colormap
                cmap_list = ['viridis', 'jet', 'plasma', 'inferno', 'magma', 'cividis', "turbo"]
                colormap = cm.get_cmap(cmap_list[1])  # You can choose other colormaps like 'plasma', 'inferno', etc.
                colors = [colormap(z) for z in normalized_z]

                # Convert the colors to the RGB format expected in the original code
                colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]


            else:    
                colors = ['rgb({},{},{})'.format(color[0], color[1], color[2]) for _ in range(len(x_coords))]
        else:
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]

        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=size,
                opacity=opacity,
                color=colors
            )
        )


    def colorize(self, pointcloud):
        if pointcloud.shape[1] == 3:

            # design a colorful point cloud based on 3d coordinates
            # Normalize coordinates to range [0, 1]
            min_coords = pointcloud.min(axis=0)
            max_coords = pointcloud.max(axis=0)
            normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
            try:
                # Use normalized coordinates as RGB values
                colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
            except: # maybe meet NaN error
                # use simple cyan color
                x_coords = pointcloud[:, 0]
                colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]

        else:
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]
        return colors
    

    def visualize_pointcloud(self, pointcloud, color:tuple=None, img_path=None):
        point_size = 3
        trace = self._generate_trace(pointcloud, color=color, size=point_size, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(
            
            scene=dict(
                # aspectmode='cube', 
                # xaxis=dict(
                #     showbackground=False,  # 隐藏背景网格
                #     showgrid=True,        # 隐藏网格
                #     showline=True,         # 显示轴线
                #     linecolor='grey',      # 设置轴线颜色为灰色
                #     zerolinecolor='grey',  # 设置0线颜色为灰色
                #     zeroline=False,        # 关闭0线
                #     gridcolor='grey',      # 设置网格颜色为灰色
                    
                # ),
                # yaxis=dict(
                #     showbackground=False,
                #     showgrid=True,
                #     showline=True,
                #     linecolor='grey',
                #     zerolinecolor='grey',
                #     zeroline=False,        # 关闭0线
                #     gridcolor='grey',      # 设置网格颜色为灰色
                #     # set range
                #     # range=[-0.1, 0.1]
                # ),
                # zaxis=dict(
                #     showbackground=False,
                #     showgrid=True,
                #     showline=True,
                #     linecolor='grey',
                #     zerolinecolor='grey',
                #     zeroline=False,        # 关闭0线
                #     gridcolor='grey',      # 设置网格颜色为灰色
                # ),
                # close all axis, all line, all grid, all ticks
                xaxis=dict(visible=False, range=[0, 2]),
                yaxis=dict(visible=False, range=[-1, 1]),
                zaxis=dict(visible=False, range=[-0.5, 1.6]),
                bgcolor='white',  # 设置背景色为白色
                camera=dict(
                    eye=dict(x=1.0, y=-2.0, z=-0.5),  # Position camera to view from top-right
                    up=dict(x=0, y=0, z=1),   # Keep y-axis as up
                    center=dict(x=0.5, y=0, z=0) # Look at the origin
                )
            )
        )
        if img_path is not None:
            fig.write_image(img_path, width=800*2, height=600*2)
            # import pdb; pdb.set_trace()
        else:
            div = pio.to_html(fig, full_html=False)

            @self.app.route('/')
            def index():
                return render_template_string('''<div>{{ div|safe }}</div>''', div=div)
            
            self.app.run(debug=True, use_reloader=False)

    def visualize_pointcloud_and_save(self, pointcloud, color:tuple=None, save_path=None):
        trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(
            
            scene=dict(
                # aspectmode='cube', 
                xaxis=dict(
                    showbackground=False,  # 隐藏背景网格
                    showgrid=True,        # 隐藏网格
                    showline=True,         # 显示轴线
                    linecolor='grey',      # 设置轴线颜色为灰色
                    zerolinecolor='grey',  # 设置0线颜色为灰色
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                    
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                bgcolor='white'  # 设置背景色为白色
            )
        )
        # save
        fig.write_image(save_path, width=800, height=600)
        

    def save_visualization_to_file(self, pointcloud, file_path, color:tuple=None):
        # visualize pointcloud and save as html
        trace = self._generate_trace(pointcloud, color=color)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig_html = pio.to_html(go.Figure(data=[trace], layout=layout), full_html=True)

        with open(file_path, 'w') as file:
            file.write(fig_html)
        print(f"Visualization saved to {file_path}")
    
