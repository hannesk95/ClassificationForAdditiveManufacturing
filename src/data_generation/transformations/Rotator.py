
def rotate_model(model_data: np.ndarray, x_rotation: int, y_rotation: int, z_rotation: int) -> np.ndarray:
    """
    Rotates the voxelized model
    :param model_data: Voxelized model data
    :param x_rotation: Degrees of rotation around the x axis
    :param y_rotation: Degrees of rotation around the y axis
    :param z_rotation:Degrees of rotation around the z axis
    :return: Rotated voxelized model
    """

    model_data = np.around(rotate(model_data, x_rotation,(1,2),reshape = False))
    model_data = np.around(rotate(model_data, y_rotation, (0, 2), reshape = False))
    model_data = np.around(rotate(model_data, z_rotation, (0, 1), reshape = False))
    

    return model_data



class Rotator:
    def __init__(self, x_rotation,y_rotation,z_rotation):
        self.x_rotation = x_rotation
        self.y_rotation = y_rotation
        self.z_rotation = z_rotation
        random.seed(42)

    def __call__(self, model):
        
        
        model_data = model.model
        model_data_tmp = deepcopy(model_data)
        voxels = np.argwhere(model_data_original == 1)
        if voxels.size == 0:
            logging.warning(f"Model empty")
            return None
        
        else: 
            
            #pad the model
            padding = model_data.shape[0]/2
            model_data_tmp = np.pad(model_data_tmp, ((padding,padding), (padding,padding), (padding, padding)), 'constant')

            
            #Rotate the model and preserve the shape
            model_data_tmp = rotate_model(model_data_tmp, x_rotation, y_rotation, z_rotation)
                                    
            #Voxel_model rotated                        
            model_rotated = VoxelModel(model_data_tmp, np.array([1]), model.model_name + f'_rotated{self.x_rotation}_{self.y_rotation}_{self.z_rotation}')
            return [model, model_rotated]
                
                
               
