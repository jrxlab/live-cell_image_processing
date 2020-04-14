from __future__ import annotations
from type.cell import cell_t
import matplotlib.pyplot as pl_
from mpl_toolkits import mplot3d
import networkx as nx_
from typing import Optional
import os



class tracks_t(nx_.DiGraph): # create a graph
    
    def TrackContainingCell(self, cell: cell_t) -> Optional[nx_.DiGraph]:
        
        """
        TrackContainingCell method
        Return a subgraph with the weakly connected components (cells) as nodes.
        """
    
        for component in nx_.weakly_connected_components(self): # get the weakly connected components
        #for component in nx_.strongly_connected_components(self):
            if cell in component: # if the cell is in the weakly connected components
                return self.subgraph(component) # return the subgraph view ( components =cells = nodes)

        return None


    def Plot(self, show_figure: bool = True) -> None:
        
        """
        Plot method
        Plots the cell tracking 3D graphs
        """
    
        figure = pl_.figure()
        axes = figure.add_subplot(projection=mplot3d.Axes3D.name)
        axes.set_xlabel("row positions")
        axes.set_ylabel("column positions")
        axes.set_zlabel("time points")
        colors = "bgrcmyk"

        for c_idx, component in enumerate(nx_.weakly_connected_components(self)): # for each weakly connected components of the graph
            color_idx = c_idx % colors.__len__() # color index
            for from_cell, to_cell in self.subgraph(component).edges: # get the graph edeges 
                time_points = (from_cell.time_point, to_cell.time_point) # get the time points
                rows = (from_cell.position[0], to_cell.position[0]) # cell row positions
                cols = (from_cell.position[1], to_cell.position[1]) # cell col positions
                axes.plot3D(rows, cols, time_points, colors[color_idx]) #plot the graph 

        if show_figure:
            pl_.show()
        
        if not os.path.exists("outputs/tracking"):
            os.makedirs("outputs/tracking")
            
        pl_.savefig("outputs/tracking/tracks")