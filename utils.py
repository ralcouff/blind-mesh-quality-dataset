# utils.py
import pymeshlab
import os


def clean_mesh(filename):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.Percentage(0.1), removeunref=True)
    ms.save_current_mesh(file_name=filename, save_vertex_color=False, save_vertex_normal=False)


def explore_folder(path, extension):
    files_to_compute = []
    reference_file = []
    for root, dirs, files in os.walk(path):
        for f in files:
            filename = f.split('.')
            if filename[1] == extension:
                full_path = os.path.join(path, f)
                if filename[0].split('_')[0] == "ref":
                    reference_file.append(full_path)
                    files_to_compute.append(full_path)
                else:
                    files_to_compute.append(full_path)
    if not reference_file:
        reference_file = []
    return reference_file, files_to_compute
