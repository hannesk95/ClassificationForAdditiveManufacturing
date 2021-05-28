import numpy as np
import copy


class Defector:
    """# TODO"""
    def __init__(self, arg):
        """# TODO"""
        self.mesh = arg

    def __call__(self, model):
        """# TODO"""
        # TODO fix class
        # get the voxel grid indices out of the occupancy grid
        model = np.load("5.npz")['model']
        voxels = np.argwhere(model == 1)

        # find appropriate axis and radius of the cylinder
        axis, radius = find_axis_and_radius(voxels)

        # find center of the voxels grid
        center = np.round(voxels.mean(axis = 0))

        # maximum and minimum grid index 
        maxx = np.max(voxels, axis=0)
        minn = np.min(voxels, axis=0)

        # set cylinder boundary points to the center
        pt1 = copy.deepcopy(center)
        pt2 = copy.deepcopy(center)

        # change 1 component based on the selected axis
        idx = axis
        pt1[idx] =  minn[idx]
        pt2[idx] =  maxx[idx]

        # identify voxels to be removed
        idx = self.points_in_cylinder(pt1, pt2, radius, voxels)
        to_be_removed = voxels[idx]

        # remove selected voxels from the occupancy grid
        for v in to_be_removed:
            model[v[0], v[1], v[2]] = 0


    def points_in_cylinder(pt1, pt2, radius, points):
        # reference: https://stackoverflow.com/questions/47932955/how-to-check-if-a-3d-point-is-inside-a-cylinder
        """
        find the points in a cylinder
        :pt1: 1st bounding point of the cylinder
        :pt2: 2nd bounding point of the cylinder
        :radius: radius of the cylinder
        :points: NumPy array of 3D points
        :return: NumPy boolean array of points inside of cylinder
        """
        vec = pt2 - pt1
        const = radius * np.linalg.norm(vec)
        cond1 = np.dot(points - pt1, vec) >= 0
        cond2 = np.dot(points - pt2, vec) <= 0
        cond3 = np.linalg.norm(np.cross(points - pt1, vec), axis=1) <= const

        cond4 = np.logical_and(cond1, cond2)
        cond5 = np.logical_and(cond3, cond4)
        return cond5


    def find_radius(self, voxels, cylinder_axis):
        """
        find an appropriate radius for the cylinder
        :voxels: voxel grid indices
        :cylinder_axis: axis of the cylinder {0, 1, 2}
        :return: selected radius
        """
        # list of possible diameters [1, 10]
        diameters = list(range(10, 0, -1))
        
        # 2d plane perpendicular to the cylinder axis
        plane = list(range(3))
        plane.remove(cylinder_axis)    

        # maximum and minimum grid index 
        maxx = np.max(voxels, axis=0)
        minn = np.min(voxels, axis=0)
        
        # perpendicular plane dimensions 
        fir_dim = maxx[plane[0]] - minn[plane[0]]
        sec_dim = maxx[plane[1]] - minn[plane[1]]
        
        # perpendicular plane smaller dimension
        min_dim = np.min([fir_dim, sec_dim])

        # heuristic: there should be more than 10 voxels after making the hole
        for d in diameters:
            diff = min_dim - d
            if diff > 10:
                return int(d/2)
            
        return None


    def find_axis_and_radius_greedy(self, voxels):
        """
        find an appropriate radius for the cylinder
        :voxels: voxel grid indices
        :return: pair: selected axis and radius
        """
        # try out the 3 possible axes
        axis = range(3)
        for a in axis:
            # return 1st axis with largest diameter
            r = self.find_radius(voxels, a)
            print(a, r)
            if r != None:
                return a, r

    
    def find_axis_and_radius_exhaustive(self, voxels):
        """
        find an appropriate radius for the cylinder
        :voxels: voxel grid indices
        :return: pair: selected axis and radius
        """
        # try out the 3 possible axes
        axis = range(3)
        result = []
        for a in axis:
            r = self.find_radius(voxels, a)
            print(a, r)
            result.append(r)
        # return axis, radius 
        return np.argmax(result), np.max(result)

    def find_axis_and_radius(self, voxels, version = 1):
        """
        find an appropriate radius for the cylinder
        :voxels: voxel grid indices
        :version: 1 is greedy, otherwise exhaustive
        :return: pair: selected axis and radius
        """
        if version == 1:
            return self.find_axis_and_radius_greedy(voxels)
        else:
            return self.find_axis_and_radius_exhaustive(voxels)
    