import random
import numpy as np
import odl

from PIL import Image, ImageDraw
from helper_functions import spectrum
from odl_fbp import fbp_op
from xraylib_np import CS_Energy
from projections import prj_factory, poly_projection

Nx, Ny = 256, 256
Np, Nd = 1000, 500
Ne = 100
materials = [35, 40, 45, 50]
Es, Is = spectrum(100, 1e7)
material_profile = CS_Energy(np.array(materials), np.array(Es))
rect_profile = CS_Energy(np.array([14]), np.array(Es))

fwd, bwd = prj_factory(Nx, Ny, Np, Nd, ret_A=False)
reco_space = odl.uniform_discr([0, 0], [1, 1], [Nx, Ny], dtype='float32')

angle_partition = odl.uniform_partition(0, 2 * np.pi, Np)
detector_partition = odl.uniform_partition(-0.1, 1.1, Nd)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl="astra_cuda")
fbp = fbp_op(ray_trafo)


def rand_circs():
    N = np.random.randint(2, 11)
    rs = np.random.uniform(3, 19, size=N)
    ps = np.random.uniform(58.4, 197.6, size=(2, N))
    ms = np.array([random.choice(materials) for i in range(N)])
    return rs, ps, ms


def make_circ(p, r, d, m):
    x, y = p
    bbox = [(x - r, y - r), (x + r, y + r)]
    d.ellipse(bbox, fill=(2, m, 0))


def make_data():
    base = Image.new('RGBA', (Nx, Ny), (0, 0, 0))
    d = ImageDraw.Draw(base)

    rectx1, recty1, rectx2, recty2 = 0.15, 0.15, 0.85, 0.85
    rectx1 = rectx1 * Nx
    recty1 = (1 - recty1) * Ny
    rectx2 = rectx2 * Nx
    recty2 = (1 - recty2) * Ny

    rectbbox = [(rectx1, recty1), (rectx2, recty2)]
    d.rectangle(rectbbox, fill=(1, 0, 0))

    rs, ps, ms = rand_circs()
    [make_circ(p, r, d, m) for p, r, m in zip([x for x in ps.T], rs, ms)]

    base_array = np.array(base.getdata())
    material1_array = base_array.reshape(Nx, Ny, 4)[:, :, 0]
    material2_array = base_array.reshape(Nx, Ny, 4)[:, :, 1]

    outidx = np.where(material1_array == 0)
    rectidx = np.where(material1_array == 1)
    midx = [np.where((material1_array == 2) & (material2_array == m))
            for m in materials]

    img = np.zeros((Ne, Nx, Ny))

    for ind, m in enumerate(midx):
        img[:, m[0], m[1]
            ] += np.tile(material_profile[ind, :], (len(m[0]), 1)).T

    img[:, rectidx[0], rectidx[1]] = np.tile(
        rect_profile, (len(rectidx[0]), 1)).T

    sino = poly_projection(fwd, img, Is)

    recon = fbp(sino)
    min_diff = np.argmin(
        np.array([np.linalg.norm(img[i, :] - recon.asarray()) for i in range(100)]))
    img_min_diff = img[min_diff, :]
    mid_energy = img[50, :]

    return recon.asarray(), img_min_diff, mid_energy


def make_data_multi():

    Es, Is = spectrum(5, 1e7)
    material_profile = CS_Energy(np.array(materials), np.array(Es))
    rect_profile = CS_Energy(np.array([14]), np.array(Es))
    base = Image.new('RGBA', (Nx, Ny), (0, 0, 0))
    d = ImageDraw.Draw(base)

    rectx1, recty1, rectx2, recty2 = 0.15, 0.15, 0.85, 0.85
    rectx1 = rectx1 * Nx
    recty1 = (1 - recty1) * Ny
    rectx2 = rectx2 * Nx
    recty2 = (1 - recty2) * Ny

    rectbbox = [(rectx1, recty1), (rectx2, recty2)]
    d.rectangle(rectbbox, fill=(1, 0, 0))

    rs, ps, ms = rand_circs()
    [make_circ(p, r, d, m) for p, r, m in zip([x for x in ps.T], rs, ms)]

    base_array = np.array(base.getdata())
    material1_array = base_array.reshape(Nx, Ny, 4)[:, :, 0]
    material2_array = base_array.reshape(Nx, Ny, 4)[:, :, 1]

    outidx = np.where(material1_array == 0)
    rectidx = np.where(material1_array == 1)
    midx = [np.where((material1_array == 2) & (material2_array == m))
            for m in materials]

    img = np.zeros((5, Nx, Ny))

    for ind, m in enumerate(midx):
        img[:, m[0], m[1]
            ] += np.tile(material_profile[ind, :], (len(m[0]), 1)).T

    img[:, rectidx[0], rectidx[1]] = np.tile(
        rect_profile, (len(rectidx[0]), 1)).T

    sino = poly_projection(fwd, img, Is)

    recon = fbp(sino)

    return recon.asarray(), img
