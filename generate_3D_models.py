from typing import List

from obj_parser import parse_file, Model
from config import Config
from utils import clean_mesh, explore_folder

import sys
import numpy as np
import os
import pymeshlab
import random


def add_noise_normal(model: Model, scale: float):
    """Apply a global noise on each vertex along its normal.

    Parameters
    ----------
    model : Model
        The 3D model onto which we will apply the global noise deformation along its normals
    scale : float
        The maximum/minimum intensity of the noise applied

    Returns
    -------
    Model
        The model with the noise deformation
    """
    noised_model = Model()
    noised_model.faces = model.faces
    uniform_noise = 2*np.random.uniform(0, scale, (len(model.vertices), 1)) - scale
    noise = [vert * float(sca) for vert, sca in zip(model.vertex_normals_list, uniform_noise)]
    noised_model.vertices = [x + n for x, n in zip(model.vertices, noise)]
    return noised_model


def multi_noise_normals(ref_file: str, output_file: str):
    """Generate all the global noised models according to the list in config file

    Parameters
    ----------
    ref_file : str
        The reference file containing the model
    output_file : str
        The name to give to the output model

    Returns
    -------
    None
    """
    noises = Config.noises
    for noise in noises:
        save_filename = f"{output_file}_{int(noise*1000)}.obj"
        model = parse_file(ref_file)
        scale = noise*model.getDiagonal()
        print(f"Noise normals : {noise} \nCorresponding scale : {scale}")
        noised_model = add_noise_normal(model, scale)
        noised_model.save(save_filename)
        clean_mesh(save_filename)


def multi_decimation(ref_file: str, output_file: str):
    """Generate the decimated models according to the list in the config file

    Parameters
    ----------
    ref_file : str
        The 3D file to decimate
    output_file : str
        The name of the output files

    Returns
    -------
    None
    """
    for decimation in Config.decimation:
        print(f"Decimation : {decimation}")
        decimate_model(ref_file, output_file, decimation)


def decimate_model(ref_file: str, output_file: str, decimation: float):
    """Decimate the reference model by quadric edge collapse, according to the decimation percentage
    and save the decimated models

    Parameters
    ----------
    ref_file : str
        The 3D file to decimate
    output_file : str
        The name of the output file
    decimation : float
        The percentage of vertex we want to conserve in the decimation

    Returns
    -------
    None
    """
    output_file = f"{output_file}_{int(100*decimation)}"
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ref_file)
    ms.meshing_decimation_quadric_edge_collapse(targetperc=decimation)
    ms.save_current_mesh(file_name=f"{output_file}.obj", save_vertex_color=False, save_vertex_normal=False)
    clean_mesh(f"{output_file}.obj")


def make_hole_dist(model: Model, n_hole: list, dist: float):
    hole_model = model.copy()
    hole_model.faces = model.faces
    hole_model.vertices = model.vertices
    dist2 = dist*model.getDiagonal()
    for i in range(max(n_hole)):
        vertex_2_remove = random.randint(0, len(hole_model.vertices)-1)
        hole_model.remove_vert_dist(vertex_2_remove, dist2)
        if i+1 in n_hole:
            print(f"Holes #{i}, {dist2}")
            save_filename = os.path.join(in_path, f"{model.name}_hole_{i+1}_{int(100*dist)}.obj")
            hole_model.save(save_filename)
            clean_mesh(save_filename)


def multi_hole_dist(ref_file: str):
    pctge_holes = Config.pctge_holes
    for dist in Config.DIST:
        model = parse_file(ref_file)
        n_hole = [int(pctge*len(model.vertices)) for pctge in pctge_holes]
        make_hole_dist(model, n_hole, dist)


def multi_noise_dist(ref_file: str):
    pctge_patches = Config.pctge_patches
    for dist in Config.noises_dist:
        model = parse_file(ref_file)
        n_noises = [int(pctge*len(model.vertices)) for pctge in pctge_patches]
        make_noise_dist(model, n_noises, dist)


def make_noise_dist(model: Model, n_noises: list(), dist: float):
    noise_model = model.copy()
    noise_model.faces = model.faces
    noise_model.vertices = model.vertices
    dist2 = dist*model.getDiagonal()
    for i in range(max(n_noises)):
        germe = random.randint(0, len(noise_model.vertices)-1)
        voisins = model.getNeighbors(germe, dist2)
        voisins_dist = [1 - model.vertices[vert].distance(model.vertices[germe])/dist2 for vert in voisins]
        noise = float(2*np.random.uniform(0,dist2,(1,1)) - dist2)
        for vois,vois_dist in zip(voisins,voisins_dist):
            noise_1 = noise*vois_dist
            noise_2 = (model.vertex_normals_list[germe]/model.vertex_normals_list[germe].norm())*float(noise_1)
            noise_model.vertices[vois] += noise_2
        if i+1 in n_noises:
            print(f"Noises #{i}, {dist2}")
            save_filename = os.path.join(in_path, f"{model.name}_noise_patch_{i+1}_{int(100*dist)}.obj")
            noise_model.save(save_filename)
            clean_mesh(save_filename)


def subdivide(ref_file: str, output_file: str, n_iter: int):
    """Subdivide the model according to a certain number of iterations
    The subdivision is a loop subdivision from pymeshlab

    Parameters
    ----------
    ref_file : str
        The reference 3D file to subdivide
    output_file : str
        The name of the output file
    n_iter : int
        The number of iterations used for the subdivision

    Returns
    -------
    None
    """
    output_file = f"{output_file}_{n_iter}"
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ref_file)
    ms.meshing_surface_subdivision_loop(iterations=n_iter, threshold=pymeshlab.Percentage(0.0))
    ms.save_current_mesh(file_name=f"{output_file}.obj", save_vertex_color=False, save_vertex_normal=False)
    clean_mesh(f"{output_file}.obj")


def multi_subdivide(ref_file: str, output_file: str):
    for n_iter in Config.subdivision_iterations:
        print(f"Subdivision #{n_iter}")
        subdivide(ref_file, output_file, n_iter)


def generate_models(ref_file):
    clean_mesh(ref_file)
    ref_model = parse_file(ref_file)
    try:
        multi_decimation(ref_file, os.path.join(in_path, f"{ref_model.name}_decimated"))
        multi_subdivide(ref_file, os.path.join(in_path, f"{ref_model.name}_subdivide"))
        multi_noise_normals(ref_file, os.path.join(in_path, f"{ref_model.name}_noise_normals"))
        multi_hole_dist(ref_file)
        multi_noise_dist(ref_file)
    except Exception as e:
        print(f"Error concerning '{ref_file}' : {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dirs = [sys.argv[1]]
    else:
        dirs = []
        for _, folders, _ in os.walk(Config.FILES_FOLDER):
            for d in folders:
                dirs.append(d)

    print(f"Models to generate :\n{dirs}")
    for d in dirs:
        if not d.startswith('_'):
            print(f"=====================  {d}  =====================")
            in_path = os.path.join(Config.FILES_FOLDER, d)
            reference_file, files_to_compute = explore_folder(in_path, "obj")
            generate_models(reference_file[0])
