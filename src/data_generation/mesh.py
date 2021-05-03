class Mesh:
    def __init__(self, path):
        self.path = path
        self.vertices, self.normals, self.faces = self.load_model(path)

    def load_model(self):
        # TODO load model with good libary
        pass

    def get_model_data(self):
        """Returns vertices, normals and faces"""
        return self.vertices, self.normals, self.faces

    def set_model_data(self, vertices, normales, faces):
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
        