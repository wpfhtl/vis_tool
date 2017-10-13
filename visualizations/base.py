"""
all the visualization class should extend this base
,and implement the make_visualization method
"""
class BaseVisualization:
    def __init__(self):
        self.data=[]
        pass
    def make_visualization(self, inputs, output_dir, settings=None):
        raise NotImplementedError
