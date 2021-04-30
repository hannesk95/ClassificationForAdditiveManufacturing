import pymeshlab


class Normalizer:
    def __init__(self, model):
        self.model = model

    def __call__(self, stl_path, target_path=None):
        """
        Align 3D models using pymeshlab
        :param stl_path: Source path to the stl model
        :param target_path: path of the resulting stl file.
        :return: stl file containing the model aligned
        """

        ms = pymeshlab.MeshSet()
        # load mesh
        ms.load_new_mesh(stl_path)
        # apply filter to align the model to principal component
        ms.transform_align_to_principal_axis()
        ms.save_current_mesh(target_path)
