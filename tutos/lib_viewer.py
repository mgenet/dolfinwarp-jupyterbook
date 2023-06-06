#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2020                                            ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob
import ipywidgets
import itkwidgets
import vtk

################################################################################

class Viewer:
    def __init__(self,
            images=None,
            meshes=None):

        if (images is None):
            self.has_images = False
        else:
            self.has_images = True

            self.images_filenames = sorted(glob.glob(images))
            self.n_images = len(self.images_filenames)
            assert (self.n_images > 0),\
                "There must be at least one image. Aborting."

            if   (self.images_filenames[0].endswith("vtk")):
                self.images_reader = vtk.vtkImageReader()
            elif (self.images_filenames[0].endswith("vti")):
                self.images_reader = vtk.vtkXMLImageDataReader()
            else:
                assert 0, "Images filenames must be .vtk or .vti. Aborting."
            self.images_reader.SetFileName(self.images_filenames[0])
            self.images_reader.Update()
            self.image = self.images_reader.GetOutput()

        if (meshes is None):
            self.has_meshes = False
        else:
            self.has_meshes = True

            if (type(meshes) is str): meshes = [meshes]

            self.meshes_filenames = []
            self.n_meshes = []
            self.meshes_readers = []
            self.warps = []
            self.meshes = []
            for k_meshes in range(len(meshes)):
                self.meshes_filenames += [sorted(glob.glob(meshes[k_meshes]))]
                self.n_meshes += [len(self.meshes_filenames[-1])]
                assert (self.n_meshes[-1] > 0),\
                    "There must be at least one mesh. Aborting."

                if   (self.meshes_filenames[-1][0].endswith("vtk")):
                    self.meshes_readers += [vtk.vtkUnstructuredGridReader()]
                elif (self.meshes_filenames[-1][0].endswith("vtu")):
                    self.meshes_readers += [vtk.vtkXMLUnstructuredGridReader()]
                else:
                    assert 0, "Meshes filenames must be .vtk or .vtu. Aborting."
                self.meshes_readers[-1].SetFileName(self.meshes_filenames[-1][0])
                self.meshes_readers[-1].Update()
                self.meshes_readers[-1].GetOutput().GetPointData().SetActiveVectors("displacement")
                self.warps += [vtk.vtkWarpVector()]
                self.warps[-1].SetInputData(self.meshes_readers[-1].GetOutput())
                self.warps[-1].Update()
                self.meshes += [self.warps[-1].GetOutput()]
            self.n_meshes = max(self.n_meshes)

        if       (self.has_images) and not (self.has_meshes):
            self.n = self.n_images
            self.viewer = itkwidgets.view(image=self.image)
        elif not (self.has_images) and     (self.has_meshes):
            self.n = self.n_meshes
            self.viewer = itkwidgets.view(geometries=[*self.meshes])
        elif     (self.has_images) and     (self.has_meshes):
            self.n = max(self.n_images, self.n_meshes)
            self.viewer = itkwidgets.view(image=self.image, geometries=[*self.meshes])

        if (self.n > 1):
            self.has_index_slider = True
        else:
            self.has_index_slider = False

        if (self.has_images):
            self.has_warp_slider = False
        else:
            self.has_warp_slider = True


    def update_index(self, index=0):
        if (self.has_images):
            self.images_reader.SetFileName(self.images_filenames[index])
            self.images_reader.Update()
            self.viewer.image = self.image
        if (self.has_meshes):
            for k_meshes in range(len(self.meshes)):
                self.meshes_readers[k_meshes].SetFileName(self.meshes_filenames[k_meshes][index])
                self.meshes_readers[k_meshes].Update()
                self.meshes_readers[k_meshes].GetOutput().GetPointData().SetActiveVectors("displacement")
                self.warps[k_meshes].Update()
            self.viewer.geometries = [*self.meshes]


    def update_warp(self, warp=1.):
        if (self.has_meshes):
            for k_meshes in range(len(self.meshes)):
                self.warps[k_meshes].SetScaleFactor(warp)
                self.warps[k_meshes].Update()
            self.viewer.geometries = [*self.meshes]


    def update_index_and_warp(self, index=0, warp=1.):
        self.update_index(index)
        self.update_warp(warp)


    def view(self):
        if       (self.has_index_slider) and not (self.has_warp_slider):
            slider = ipywidgets.interactive(
                self.update_index,
                index=(0, self.n-1, 1),
                continuous_update=True)
            i_slider = slider.children[0]
            i_slider.description = "index"
        elif not (self.has_index_slider) and     (self.has_warp_slider):
            slider = ipywidgets.interactive(
                self.update_warp,
                warp=(0., 2., 0.1),
                continuous_update=True)
            w_slider = slider.children[0]
            w_slider.description = "warp"
        elif     (self.has_index_slider) and     (self.has_warp_slider):
            slider = ipywidgets.interactive(
                self.update_index_and_warp,
                index=(0, self.n-1, 1),
                warp=(0., 2., 0.1),
                continuous_update=True)
            i_slider, w_slider = slider.children[0:2]
            i_slider.description = "index"
            w_slider.description = "warp"

        if (self.has_index_slider):
            b_prev = ipywidgets.Button(description="<")
            def _prev(b): i_slider.value -= 1
            b_prev.on_click(_prev)

            b_next = ipywidgets.Button(description=">")
            def _next(b): i_slider.value += 1
            b_next.on_click(_next)

        if       (self.has_index_slider) and not (self.has_warp_slider):
            grid = ipywidgets.GridspecLayout(1,3)
            grid[0,0] = b_prev
            grid[0,1] = i_slider
            grid[0,2] = b_next
        elif not (self.has_index_slider) and     (self.has_warp_slider):
            grid = ipywidgets.GridspecLayout(1,1)
            grid[0,0] = w_slider
        elif     (self.has_index_slider) and     (self.has_warp_slider):
            grid = ipywidgets.GridspecLayout(2,3)
            grid[0,0] = b_prev
            grid[0,1] = i_slider
            grid[0,2] = b_next
            grid[1,1] = w_slider
        else:
            grid = ipywidgets.GridspecLayout(1,1)

        grid.layout.width   = "400px"
        grid.layout.padding =   "0px"
        if (self.has_index_slider):
            b_prev.layout.width =  "50px"
            b_next.layout.width =  "50px"

        return ipywidgets.VBox([self.viewer, grid])
