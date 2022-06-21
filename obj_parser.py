#!/usr/bin/env python3

import sys
import numpy as np

"""
obj model for python.
"""


class Vertex:
    """
    The class that holds the x, y, and z coordinates of a vertex.
    """
    def __init__(self, number, x, y, z):
        """
        Initializes a vertex from values.
        """
        self.number = number
        self.x = x
        self.y = y
        self.z = z

    def from_array(array):
        """
        Creates a vertex from an array of string representing floats.
        """
        return Vertex( 0, 0, 0, 0).set(array)

    def set(self, array):
        """
        Sets a vertex from an array of string representing floats.
        """
        self.number = float(array[0])
        self.x = float(array[1])
        self.y = float(array[2])
        self.z = float(array[3])
        return self

    def copy(self):
        new_vertex = Vertex(self.number, self.x, self.y, self.z)
        return new_vertex
    
    def set_number(self, new_number):
        self.number = new_number

    def isequal(self, vertex):
        return (self.x == vertex.x) & (self.y == vertex.y) & (self.z == vertex.z)

    def __str__(self):
        return "v {} {} {}".format(self.x, self.y, self.z)

    def __add__(self, other):
        return Vertex(self.number, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vertex(self.number, self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Vertex(self.number, self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        vert = Vertex(self.number, self.x, self.y, self.z)
        if other != 0:
            vert = Vertex(self.number, self.x / other, self.y / other, self.z / other)
        return vert

    def __repr__(self):
        return str(self)

    def cross(self, other):
        return Vertex(-1, self.y*other.z - self.z*other.y, self.z*other.x - self.x*other.z, self.x*other.y -
                      self.y*other.x)
    
    def norm(self):
        return np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def distance(self, other):
        return np.sqrt(np.power(self.x-other.x, 2)+np.power(self.y-other.y, 2)+np.power(self.z-other.z, 2))


class Face:
    """
    The class that holds a, b, and c, the indices of the vertices of the face.
    """
    def __init__(self, array):
        """
        Initializes a face from an array of strings representing vertex indices (starting at 1)
        """
        self.c = None
        self.b = None
        self.a = None
        self.set(array)
        self.visible = True

    def set(self, array):
        """
        Sets a face from an array of strings representing vertex indices (starting at 1)
        """
        self.a = int(array[0].split('/')[0]) - 1
        self.b = int(array[1].split('/')[0]) - 1
        self.c = int(array[2].split('/')[0]) - 1
        return self

    def test(self, vertices, line="Unknown"):
        """
        Tests if a face references only vertices that exist when the face is declared.
        """
        if self.a >= len(vertices):
            raise VertexError(self.a + 1, line)
        if self.b >= len(vertices):
            raise VertexError(self.b + 1, line)
        if self.c >= len(vertices):
            raise VertexError(self.c + 1, line)

    def downgrade(self, vertex_number):
        if self.a + 1 > vertex_number:
            self.a -= 1
        if self.b + 1 > vertex_number:
            self.b -= 1
        if self.c + 1 > vertex_number:
            self.c -= 1

    def get_vertices(self):
        return [self.a, self.b, self.c]

    def get_normal(self, model):
        v1 = model.vertices[self.b] - model.vertices[self.a]
        v2 = model.vertices[self.c] - model.vertices[self.a]
        normal = v1.cross(v2)
        return normal

    def copy(self):
        new_face = Face([f"{self.a}", f"{self.b}", f"{self.c}"])
        return new_face

    def __str__(self):
        return "f {} {} {}".format(self.a + 1, self.b + 1, self.c + 1)

    def __repr__(self):
        return str(self)
            

class VertexError(Exception):
    """
    An operation references a vertex that does not exist.
    """
    def __init__(self, index, line):
        """
        Creates the error from index of the referenced vertex and the line where the error occurred.
        """
        self.line = line
        self.index = index
        super().__init__()

    def __str__(self):
        """
        Pretty prints the error.
        """
        return f'There is no vertex {self.index} (line {self.line})'


class FaceError(Exception):
    """
    An operation references a face that does not exist.
    """
    def __init__(self, index, line):
        """
        Creates the error from index of the referenced face and the line where the error occurred.
        """
        self.line = line
        self.index = index
        super().__init__()

    def __str__(self):
        """
        Pretty prints the error.
        """
        return f'There is no face {self.index} (line {self.line})'


class FaceVertexError(Exception):
    """
    An operation references a face vertex that does not exist.
    """
    def __init__(self, index, line):
        """
        Creates the error from index of the referenced face vertex and the line where the error occurred.
        """
        self.line = line
        self.index = index
        super().__init__()

    def __str__(self):
        """
        Pretty prints the error.
        """
        return f'Face has no vertex {self.index} (line {self.line})'


class UnknownInstruction(Exception):
    """
    An instruction is unknown.
    """
    def __init__(self, instruction, line):
        """
        Creates the error from instruction and the line where the error occurred.
        """
        self.line = line
        self.instruction = instruction
        super().__init__()

    def __str__(self):
        """
        Pretty prints the error.
        """
        return f'Instruction {self.instruction} unknown (line {self.line})'


class Model:
    """
    The OBJ model.
    """
    def __init__(self):
        """
        Initializes an empty model.
        """
        self.vertices = []
        self.faces = []
        self.face_normals = []
        self.vertex_normals = {}
        self.vertex_normals_list = []
        self.line = 0
        self.name = ''

    def get_vertex_from_string(self, string):
        """
        Gets a vertex from a string representing the index of the vertex, starting at 1.
        To get the vertex from its index, simply use model.vertices[i].
        """
        index = int(string) - 1
        if index >= len(self.vertices):
            raise FaceError(index + 1, self.line)
        return self.vertices[index]

    def get_face_from_string(self, string):
        """
        Gets a face from a string representing the index of the face, starting at 1.
        To get the face from its index, simply use model.faces[i].
        """
        index = int(string) - 1
        if index >= len(self.faces):
            raise FaceError(index + 1, self.line)
        return self.faces[index]

    def parse_file(self, path):
        """
        Parses an OBJA file.
        """
        self.name = path.split('/')[-1].split('.')[0][4:]
        with open(path, "r") as file:
            for line in file.readlines():
                self.parse_line(line)
        self.vertex_normals, self.vertex_normals_list = self.compute_vertex_normals()

    def parse_line(self, line):
        """
        Parses a line of OBJA file.
        """
        self.line += 1

        split = line.split()

        if len(split) == 0:
            return

        if split[0] == "v":
            self.vertices.append(Vertex.from_array(np.array([(len(self.vertices) + 1)] + split[1:])))

        elif split[0] == "f":
            for i in range(1, len(split) - 2):
                face = Face([elt.split('/')[0] for elt in split[i:i+3]])
                face.test(self.vertices, self.line)
                self.faces.append(face)
                self.face_normals.append(face.get_normal(self))

        elif split[0] == "#":
            return

        else:
            return
            # raise UnknownInstruction(split[0], self.line)

    def get_lists(self):
        vert_list = []
        faces_list = []
        for vert in self.vertices:
            vert_list.append(tuple([vert.x, vert.y, vert.z]))
        for face in self.faces:
            faces_list.append(tuple([face.a, face.b, face.c]))
        return vert_list, faces_list

    def save(self, output_path):
        f = open(output_path, 'w+')
        for vert in self.vertices:
            f.writelines(f"{vert}\n")
        for face in self.faces:
            f.writelines(f"{face}\n")
    
    def save1(self, output_path, distances):
        colormap = create_colormap(distances)
        f = open(output_path, 'w+')
        for vert, color in zip(self.vertices, colormap):
            f.writelines(f"{vert} {color[0]} {color[1]} {color[2]}\n")
        for face in self.faces:
            f.writelines(f"{face}\n")

    def replace(self, vertex1, vertex2):
        for (ind, vert) in enumerate(self.vertices):
            if vert.number == vertex1.number:
                self.vertices.pop(ind)
            elif vert.number > vertex1.number:
                self.vertices[ind].set_number(vert.number - 1)
        for (ind, face) in enumerate(self.faces):
            if face.a + 1 == vertex1.number:
                self.faces[ind].a = int(vertex2.number - 1)
            if face.b + 1 == vertex1.number:
                self.faces[ind].b = int(vertex2.number - 1)
            if face.c + 1 == vertex1.number:
                self.faces[ind].c = int(vertex2.number - 1)
            if face.a + 1 > vertex1.number:
                self.faces[ind].a -= 1
            if face.b + 1 > vertex1.number:
                self.faces[ind].b -= 1
            if face.c + 1 > vertex1.number:
                self.faces[ind].c -= 1

    def remove_vert_n_ppv(self, vertex_number, n):
        neighbors = set()
        neighbors.add(vertex_number)
        for i in range(n):
            neigh = neighbors.copy()
            for vertex_numb in neigh:
                for face in self.faces:
                    face_vert = face.get_vertices()
                    if vertex_numb in face_vert:
                        neighbors.update(face_vert)
        for ind in reversed(sorted(neighbors)):
            self.remove_vertex(ind)
    
    def remove_vert_dist(self, vert, dist):
        for neighbor in reversed(sorted(self.getNeighbors(vert, dist))):
            self.remove_vertex(neighbor)

    def remove_vertex(self, vertex_number):
        faces_to_remove = []
        self.vertices.pop(vertex_number)
        for index, face in enumerate(self.faces):
            if vertex_number in face.get_vertices():
                faces_to_remove.append(index)
        for ind in reversed(faces_to_remove):
            self.remove_face(ind)
        for face in self.faces:
            face.downgrade(vertex_number)

    def remove_face(self, face_index):
        self.faces.pop(face_index)

    def compute_vertex_normals(self):
        vert_normals = {}
        vert_normals_list = []
        for face in self.faces:
            for vert in face.get_vertices():
                if vert in vert_normals.keys():
                    vert_normals[vert] += face.get_normal(self)
                else :
                    vert_normals[vert] = face.get_normal(self)
        for vert in vert_normals:
            vert_normalised = vert_normals[vert]/vert_normals[vert].norm()
            vert_normals[vert] = vert_normalised
            vert_normals_list.append(vert_normalised)
        return vert_normals, vert_normals_list

    def copy(self):
        new_model = Model()
        new_model.vertices = [vert.copy() for vert in self.vertices]
        new_model.faces = [face.copy() for face in self.faces]
        new_model.face_normals = []
        new_model.vertex_normals = {}
        new_model.line = self.line
        new_model.name = self.name
        return new_model

    def getDiagonal(self):
        (min_x, max_x, min_y, max_y, min_z, max_z) = self.getBbox()
        d = np.sqrt(np.power(max_x-min_x, 2) + np.power(max_y-min_y, 2) + np.power(max_z-min_z, 2))
        return d

    def getBbox(self):
        min_x = min_y = min_z = np.inf
        max_x = max_y = max_z = -np.inf
        for vertex in self.vertices:
            if vertex.x < min_x:
                min_x = vertex.x
            elif vertex.x > max_x:
                max_x = vertex.x
            if vertex.y < min_y:
                min_y = vertex.y
            elif vertex.y > max_y:
                max_y = vertex.y
            if vertex.z < min_z:
                min_z = vertex.z
            elif vertex.z > max_z:
                max_z = vertex.z
        return min_x, max_x, min_y, max_y, min_z, max_z

    def getNeighbors(self, germe, dist):
        voisins = set()
        voisins_restants = set()
        voisins.add(germe)
        voisins_restants.add(germe)
        cont = True
        while cont:
            to_add = []
            for vertex_numb in voisins_restants:
                for face in self.faces:
                    face_vert = face.get_vertices()
                    if vertex_numb in face_vert:
                        for vert in face_vert:
                            if self.vertices[germe].distance(self.vertices[vert]) <= dist and vert not in voisins :
                                to_add.append(vert)
                                voisins.add(vert)
            voisins_restants.clear()
            voisins_restants.update(to_add)
            cont = (len(to_add) != 0)
        return voisins


def create_colormap(distances):
    colormap = []
    n = len(distances)
    min_dist = min(distances)
    max_dist = max(distances)
    if (min_dist != max_dist) :
        for dist in distances:
            coeff = (dist-min_dist)/(max_dist-min_dist)
            R = int(255*coeff)
            V = 255-R
            color = [R,V,0]
            colormap.append(color)
    else:
        colormap = n*[[0,255,0]]
    return colormap


def parse_file(path):
    """
    Parses a file and returns the model.
    """
    model = Model()
    model.parse_file(path)
    return model


def main():
    if len(sys.argv) == 1:
        print("obj needs a path to an obj file")
        return

    model = parse_file(sys.argv[1])


if __name__ == "__main__":
    main()
