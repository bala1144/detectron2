# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from scipy.io import loadmat
import scipy.spatial.distance
import os
import pickle
from densepose.vis.densepose import DensePoseResult
import random
import time
import cv2
import math

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class MyTimer():

    def __init__(self, call_name = 'function'):
        self.start = time.time()
        self.call_name = call_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start
        msg = 'The {function} took {time} seconds to complete'
        print(msg.format(function=self.call_name, time=runtime))

def VisualizeObj(vertices, second_vertices=None):
    import pyrender
    import trimesh
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])

    sm = trimesh.creation.uv_sphere(radius=0.01)
    sm.visual.vertex_colors = [0.0, 1.0, 0.0]
    tfs = np.tile(np.eye(4), (len(vertices), 1, 1))
    tfs[:, :3, 3] = vertices
    Joints_Render = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(Joints_Render, pose=np.eye(4))


    if second_vertices is not None:
        # visulize the GT point cloud
        sm = trimesh.creation.uv_sphere(radius=0.01)
        sm.visual.vertex_colors = [0.0, 0.0, 1.0]
        tfs = np.tile(np.eye(4), (len(second_vertices), 1, 1))
        tfs[:, :3, 3] = second_vertices
        second_vertices_Render = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(second_vertices_Render, pose=np.eye(4))

    pyrender.Viewer(scene, use_raymond_lighting=True)

def SMPLVisualizer(vertices, faces, joints=None, GT_joints=None):
    import pyrender
    import trimesh
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)

    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

    if joints is not None:
        sm = trimesh.creation.uv_sphere(radius=0.02)
        sm.visual.vertex_colors = [1.0, 0.0, 0.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        Joints_Render = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(Joints_Render, pose=np.eye(4))

    if GT_joints is not None:
        sm = trimesh.creation.uv_sphere(radius=0.02)
        sm.visual.vertex_colors = [0.0, 1.0, 0.0]
        tfs = np.tile(np.eye(4), (len(GT_joints), 1, 1))
        tfs[:, :3, 3] = GT_joints
        Joints_Render = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(Joints_Render, pose=np.eye(4))

    scene.add(mesh, pose=np.eye(4))
    pyrender.Viewer(scene, use_raymond_lighting=True)

    scene.add(light, pose=np.eye(4))
    # scene.add(camera, pose=np.eye(4))
    # c = 2**-0.5
    # scene.add(camera, pose=[[ 1,  0,  0,  0],
    #                         [ 0,  c, -c, -2],
    #                         [ 0,  c,  c,  2],
    #                         [ 0,  0,  0,  1]])

    # camera pose for getting to identity
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 2],
                            [0, 0, 0, 1]])

    # render scene
    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene)

    return color

class DensePoseMethods:
    def __init__(self):
        #
        ALP_UV = loadmat(os.path.join(os.path.dirname(__file__), '../UV_data/UV_Processed.mat'))
        self.FaceIndices = np.array(ALP_UV['All_FaceIndices']).squeeze()
        self.FacesDensePose = ALP_UV['All_Faces'] - 1
        self.U_norm = ALP_UV['All_U_norm'].squeeze()
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.All_vertices = ALP_UV['All_vertices'][0]
        ## Info to compute symmetries.
        self.SemanticMaskSymmetries = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
        self.Index_Symmetry_List = [1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24,23]
        UV_symmetry_filename = os.path.join(os.path.dirname(__file__),
                                            '../UV_data/UV_symmetry_transforms.mat')
        self.UV_symmetry_transformations = loadmat(UV_symmetry_filename)

    def get_symmetric_densepose(self, I, U, V, x, y, Mask):
        ### This is a function to get the mirror symmetric UV labels.
        Labels_sym = np.zeros(I.shape)
        U_sym = np.zeros(U.shape)
        V_sym = np.zeros(V.shape)
        ###
        for i in (range(24)):
            if i + 1 in I:
                Labels_sym[I == (i + 1)] = self.Index_Symmetry_List[i]
                jj = np.where(I == (i + 1))
                ###
                U_loc = (U[jj] * 255).astype(np.int64)
                V_loc = (V[jj] * 255).astype(np.int64)
                ###
                V_sym[jj] = self.UV_symmetry_transformations['V_transforms'][0, i][V_loc, U_loc]
                U_sym[jj] = self.UV_symmetry_transformations['U_transforms'][0, i][V_loc, U_loc]
        ##
        Mask_flip = np.fliplr(Mask)
        Mask_flipped = np.zeros(Mask.shape)
        #
        for i in (range(14)):
            Mask_flipped[Mask_flip == (i + 1)] = self.SemanticMaskSymmetries[i + 1]
        #
        [y_max, x_max] = Mask_flip.shape
        y_sym = y
        x_sym = x_max - x
        #
        return Labels_sym, U_sym, V_sym, x_sym, y_sym, Mask_flipped

    def barycentric_coordinates_exists(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        if (np.dot(vCrossW, vCrossU) < 0):
            return False;
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        if (np.dot(uCrossW, uCrossV) < 0):
            return False;
        #
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        #
        return ((r <= 1) & (t <= 1) & (r + t <= 1))

    def barycentric_coordinates(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        #
        return (1 - (r + t), r, t)

    def IUV2FBC(self, I_point, U_point, V_point):
        #with MyTimer('PrepareFace'):
        P = [U_point, V_point, 0]
        FaceIndicesNow = np.where(self.FaceIndices == I_point)
        FacesNow = self.FacesDensePose[FaceIndicesNow]

        #with MyTimer('Finding P'):
        P_0 = np.vstack((self.U_norm[FacesNow][:, 0], self.V_norm[FacesNow][:, 0],np.zeros(self.U_norm[FacesNow][:, 0].shape))).transpose()
        P_1 = np.vstack((self.U_norm[FacesNow][:, 1], self.V_norm[FacesNow][:, 1],np.zeros(self.U_norm[FacesNow][:, 1].shape))).transpose()
        P_2 = np.vstack((self.U_norm[FacesNow][:, 2], self.V_norm[FacesNow][:, 2],np.zeros(self.U_norm[FacesNow][:, 2].shape))).transpose()
        #

        #with MyTimer('Find the inside face'): # this is the one taking more time
        for i, [P0, P1, P2] in enumerate(zip(P_0, P_1, P_2)):
            # print(i)
            if (self.barycentric_coordinates_exists(P0, P1, P2, P)):
                [bc1, bc2, bc3] = self.barycentric_coordinates(P0, P1, P2, P)
                return (FaceIndicesNow[0][i], bc1, bc2, bc3)
        #
        # If the found UV is not inside any faces, select the vertex that is closest!
        #
        #with MyTimer('Calculating D'):
        D1 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_0[:, 0:2]).squeeze()
        D2 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_1[:, 0:2]).squeeze()
        D3 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_2[:, 0:2]).squeeze()
        #
        #with MyTimer('Finding min D'):
        minD1 = D1.min()
        minD2 = D2.min()
        minD3 = D3.min()
        #
        #with MyTimer('Find the nearest vertex'):
        if ((minD1 < minD2) & (minD1 < minD3)):
            return (FaceIndicesNow[0][np.argmin(D1)], 1., 0., 0.)
        elif ((minD2 < minD1) & (minD2 < minD3)):
            return (FaceIndicesNow[0][np.argmin(D2)], 0., 1., 0.)
        else:
            return (FaceIndicesNow[0][np.argmin(D3)], 0., 0., 1.)

    def FBC2PointOnSurface(self, FaceIndex, bc1, bc2, bc3, Vertices):
        ##
        Vert_indices = self.All_vertices[self.FacesDensePose[FaceIndex]] - 1
        ##
        p = Vertices[Vert_indices[0], :] * bc1 + Vertices[Vert_indices[1], :] * bc2 + Vertices[Vert_indices[2], :] * bc3
        ##
        return (p)

    def Face2Vertices(self,FaceIndex):
        Vert_indices = self.All_vertices[self.FacesDensePose[FaceIndex]] - 1
        return Vert_indices

def smpl_view_set_axis_full_body(ax, azimuth=0):
    ## Manually set axis
    ax.view_init(0, azimuth)
    max_range = 0.55
    ax.set_xlim(- max_range, max_range)
    ax.set_ylim(- max_range, max_range)
    ax.set_zlim(-0.2 - max_range, -0.2 + max_range)
    ax.axis('off')

def smpl_view_set_axis_face(ax, azimuth=0):
    ## Manually set axis
    ax.view_init(0, azimuth)
    max_range = 0.1
    ax.set_xlim(- max_range, max_range)
    ax.set_ylim(- max_range, max_range)
    ax.set_zlim(0.45 - max_range, 0.45 + max_range)
    ax.axis('off')

def visulizeDenseposePredictionsOnModel(iuv_arr):
    #load the smpl model
    # Now read the smpl model.
    with MyTimer('load smpl'):
        model_path = '/media/bala/OSDisk/Users/bala/Documents/myprojects/GuidedResearch/models/smpl/SMPL_MALE.pkl'
        with open(model_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1' )
            Vertices = data['v_template']  ##  Loaded vertices of size (6890, 3)
            X, Y, Z = [Vertices[:, 0], Vertices[:, 1], Vertices[:, 2]]
            faces = data['f']

    with MyTimer('Prepocess'):
        # create dense pose methods
        DP = DensePoseMethods()
        # extarct the no zero indexes of the iuv array for display
        iuv_arr = iuv_arr.reshape(3, -1)
        print(iuv_arr.shape)
        non_zero_id = np.nonzero(iuv_arr[0])
        print('Total Non zero id', len(non_zero_id[0]))
        num_pts_to_vis = 10000
        # num_pts_to_vis = len(non_zero_id[0])

        # sample 1000 in equal intervals
        skip = int(len(non_zero_id[0]) / num_pts_to_vis)
        selcted_idx = np.arange(len(non_zero_id[0]), step=skip)
        selcted_idx = non_zero_id[0][selcted_idx[1:]]


        collected_x = np.zeros(num_pts_to_vis)
        collected_y = np.zeros(num_pts_to_vis)
        collected_z = np.zeros(num_pts_to_vis)
        Interpolated_FBC = []
        FBC = []


    with MyTimer('For'):
        for i,id in enumerate(selcted_idx):
            print()
            print(i, id)
            (ii, uu, vv) = (iuv_arr[0,id], iuv_arr[1,id] / 255.0, iuv_arr[2,id] / 255.0)
            with MyTimer('IUV2FBC'):
                FaceIndex, bc1, bc2, bc3 = DP.IUV2FBC(ii, uu, vv)
                v1,v2,v3 = DP.Face2Vertices(FaceIndex)
                FBC.append((FaceIndex, v1, v2, v3, bc1, bc2, bc3))



    #         p = DP.FBC2PointOnSurface(FaceIndex, bc1, bc2, bc3, Vertices)
    #         collected_x[i] = p[0]
    #         collected_y[i] = p[1]
    #         collected_z[i] = p[2]
    #         Interpolated_FBC.append(p)
    #
    # # for visulization
    # Interpolated_FBC = np.asarray(Interpolated_FBC)
    # SMPLVisualizer(Vertices,faces,Interpolated_FBC)

    # save after every 100 images
    # FBC = np.asarray(FBC)
    # print('FBC Shape', len(FBC))
    print('FBC Shape', FBC.shape)
    pickling_on = open("../../FBC_frame0.pkl", "wb")
    pickle.dump(FBC, pickling_on)
    pickling_on.close()

def IUVtoFBC(iuv_arr ,num_pts_to_vis = 10):


    # create dense pose methods
    DP = DensePoseMethods()
    # extarct the no zero indexes of the iuv array for display
    iuv_arr_cp = np.copy(iuv_arr)
    iuv_arr = iuv_arr.reshape(3, -1)
    print(iuv_arr.shape)
    # non_zero_id = np.nonzero(iuv_arr[0])
    non_zero_id = np.where(iuv_arr[0] == 2)
    print('Total Non zero id', len(non_zero_id[0]))

    U, V = np.where(iuv_arr_cp[0] == 2)
    print('Total Non zero V', len(V))
    print('Total Non zero U', len(U))

    # sample 1000 in equal intervals
    # uniformly samples n points form the list
    selcted_idx_org = np.random.randint(len(non_zero_id[0]), size=num_pts_to_vis)
    print('selcted_idx', selcted_idx_org)
    selcted_idx = non_zero_id[0][selcted_idx_org[:]]
    print('selcted_idx', selcted_idx)


    FBC = []
    for i,id in enumerate(selcted_idx):
        print()
        # print(i, id)
        (ii, uu, vv) = (iuv_arr[0,id], iuv_arr[1,id], iuv_arr[2,id])
        # with MyTimer('IUV2FBC'):
        FaceIndex, bc1, bc2, bc3 = DP.IUV2FBC(ii, uu / 255.0, vv / 255.0)
        v1,v2,v3 = DP.Face2Vertices(FaceIndex)
        curr_FBC = (FaceIndex, v1, v2, v3, bc1, bc2, bc3)
        if curr_FBC not in FBC:
            print(ii, uu, vv, uu/255.0, vv/255.0, curr_FBC)
            FBC.append(curr_FBC)

    # remove the duplicates and return it
    return list(set(FBC))


class FBCGenerator:

    def __init__(self, dataRootPath):
        self.dataRootPath = dataRootPath

    def IUVtoFBC_withDepth(self, fileName, iuv_arr, origin,num_pts_to_vis = 1000):

        def checkDuplicate(new, list):
            # if not list:
            #     return True
            if new in list:
                return False
            else:
                return True

        # create dense pose methods
        DP = DensePoseMethods()
        # load the point cloud
        pc = PointCloudProcessing()
        pc_file = os.path.join(self.dataRootPath, 'pointcloud', fileName + '.obj')
        print('pcfile', pc_file)
        pc.loadPointCloud(pc_file)

        # extarct the no zero indexes of the iuv array for display
        # iuv_arr = iuv_arr.reshape(3, -1)
        print(iuv_arr.shape)
        V, U = np.where(iuv_arr[0] != 0)
        print('Total Non zero id', len(U))

        # sample 1000 in equal intervals
        # uniformly samples n points form the list
        selcted_idx = np.random.randint(len(U), size=int(num_pts_to_vis + 0.5 * num_pts_to_vis ) )
        print('selcted_idx.shape',selcted_idx.shape)
        selected_U = U[selcted_idx[:]] # cartesien coordinates
        selected_V = V[selcted_idx[:]]

        # select U for UV map, which ranges from 1 to 255 max

        FBC = []
        Extracted3DPoint = []
        filtered_U = []
        filtered_V = []
        for id in range(len(selected_U)):
            # print(id)
            (ii, uu, vv) = (iuv_arr[0, selected_V[id], selected_U[id]], iuv_arr[1, selected_V[id], selected_U[id]], iuv_arr[2, selected_V[id], selected_U[id]])
            # print(id, ii, uu, vv, uu / 255.0, vv /255.0)

            # extract the Depth Points
            DepthUV = pc.colourtoDepth(np.array([selected_U[id] + origin[0],  selected_V[id] + origin[1], 1]))  # adding the BBox x and y to get UV in the image
            extracted_point = pc.extractPointsInDepthUV(DepthUV)
            # print(extracted_point, sum(extracted_point))

            if sum(extracted_point) != 0.0 :
                FaceIndex, bc1, bc2, bc3 = DP.IUV2FBC(ii, uu / 255.0, vv / 255.0)
                v1, v2, v3 = DP.Face2Vertices(FaceIndex)
                curr_FBC = (FaceIndex, v1, v2, v3, bc1, bc2, bc3)
                # print('curr_FBC', curr_FBC)
                # print('checkDuplicate(curr_FBC,FBC)', checkDuplicate(curr_FBC,FBC))

                if checkDuplicate(curr_FBC, FBC):
                    # print(id)
                    # print('adding valid points', extracted_point)
                    Extracted3DPoint.append(extracted_point)
                    filtered_U.append(selected_U[id])
                    filtered_V.append(selected_V[id])
                    FBC.append(curr_FBC)
                    # print(id, len(Extracted3DPoint))
                    if len(Extracted3DPoint) >= num_pts_to_vis:
                        break

            # check condition for valid point
            # extracted point from the Depth map should not be 0.0
            # Extracted FBC should not be already present in the list
            # if the both of conditions are valid, we can them to the depth FBC
        # # sanity checking for unqiue
        # sanity_FBC = []
        # for id in range(len(selected_U)):
        #     (ii, uu, vv) = (iuv_arr[0, selected_V[id], selected_U[id]], selected_U[id], selected_V[id])
        #     print(ii, uu, vv)
        #     FaceIndex, bc1, bc2, bc3 = DP.IUV2FBC(ii, uu / 255.0, vv / 255.0)
        #     v1, v2, v3 = DP.Face2Vertices(FaceIndex)
        #     sanity_FBC.append((FaceIndex, v1, v2, v3, bc1, bc2, bc3))    # print('len of all sanity FBC', len(sanity_FBC))
        #     # print('len of set sanity FBC', len(list(set(sanity_FBC))))


        # # sanity check for dupliate
        # print('len of Valid of 3D point from depth map', len(Extracted3DPoint))
        print('len of all FBC', len(FBC))
        print('len of set FBC', len(list(set(FBC))))


        # visualize the points
        # image_path = '../../outputres.0001.png'
        image_path = os.path.join(self.dataRootPath,  'denseposeVisulization', fileName + '.png')
        from skimage import io
        image_final = io.imread(image_path)
        selected_U = selected_U + origin[0]
        selected_V = selected_V + origin[1]
        # for idx in range(len(selected_U)):
        #     cv2.circle(image_final, (selected_U[idx], selected_V[idx]), 1, (0, 0, 255), 4)


        filtered_U = filtered_U + origin[0]
        filtered_V = filtered_V + origin[1]
        for idx in range(len(filtered_U)):
            cv2.circle(image_final, (filtered_U[idx], filtered_V[idx]), 1, (0, 255, 0), 4)

        image_path = os.path.join(self.dataRootPath, 'DepthFBCVisualization_pts'+ str(num_pts_to_vis), fileName + '.png')
        out_dir = os.path.dirname(image_path)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        cv2.imwrite(image_path, image_final)
        # cv2.imshow('image', image_final)
        # cv2.waitKey(0)
        #
        # # # selcted_idx = np.random.randint(len(pc.pointcloud_vertices), size=10000)
        # selcted_idx = np.random.randint(len(pc.pointcloud_vertices), size=10000)
        # sample_pc = np.array(pc.pointcloud_vertices)
        # sample_pc = sample_pc[selcted_idx[:]]
        # VisualizeObj(np.array(Extracted3DPoint),sample_pc)
        # # remove the duplicates and return it


        return FBC, Extracted3DPoint

def ExtarctFBC(I,U,V):
    DP = DensePoseMethods()
    FBC = []
    for id in range(len(U)):
        (ii, uu, vv) = (I[id], U[id] / 255.0, V[id] / 255.0)
        # print()
        # print(id, ii, U[id], V[id])
        # with MyTimer('IUV2FBC'):
        FaceIndex, bc1, bc2, bc3 = DP.IUV2FBC(ii, uu, vv)
        v1, v2, v3 = DP.Face2Vertices(FaceIndex)
        FBC.append((FaceIndex, v1, v2, v3, bc1, bc2, bc3))
        # print(FaceIndex, v1, v2, v3, bc1, bc2, bc3)

    return FBC

def sample_perpart_IUV(iuv_arr, xyxy, num_pts_to_vis =1000 ):
    print('sample_perpart_IUV')

    # load the image
    # image_path = '../../YawarSeq/frame000000.jpg'
    image_path = '../../outputres.0001.png'
    from skimage import io
    image = io.imread(image_path)

    # load the point cloud
    pc = PointCloudProcessing()
    pc.loadPointCloud()



    # sample the point and draw red dots on top of it
    print('IUV arr shape', iuv_arr[0].shape)
    selected_V, selected_U = np.nonzero(iuv_arr[0])
    print('Total Non zero id', len(selected_V))

    selcted_idx = np.random.randint(len(selected_V), size=num_pts_to_vis)
    # print('selcted_idx',selcted_idx)

    selected_U = selected_U[selcted_idx[:]]
    selected_V = selected_V[selcted_idx[:]]

    # with the whole image
    selected_U = selected_U + xyxy[0]
    selected_V = selected_V + xyxy[1]
    # print('selected_I',selected_I )

    filtered_U = []
    filtered_V = []
    extracted_3D_points = []
    for point_id in range(len(selected_U)):
        # print()
        # print('selected point', [selected_U[point_id], selected_V[point_id], 1])
        DepthUV = pc.colourtoDepth(np.array([selected_U[point_id], selected_V[point_id], 1]))
        extracted_point = pc.extractPointsInDepthUV(DepthUV)
        if sum(extracted_point) == 0.0:
            continue
        else:
            extracted_3D_points.append(extracted_point)
            filtered_U.append(selected_U[point_id])
            filtered_V.append(selected_V[point_id])


    print('number of filtered points', len(extracted_3D_points))

    # again subsample 1000 points for generation of FBC
    selcted_idx = np.random.randint(len(filtered_U), size=num_pts_to_vis)
    extracted_3D_points = np.asarray(extracted_3D_points)[selcted_idx[:]]

    # subtractin the bounding box x y to extract the FBC, because the FBC are extracted in the
    filtered_U = np.asarray(filtered_U)[selcted_idx[:]] - xyxy[0]
    filtered_V = np.asarray(filtered_V)[selcted_idx[:]] - xyxy[1]
    filtered_I = iuv_arr[0, filtered_V[:], filtered_U[:]]

    print('number of Depth Valid points', len(extracted_3D_points))

    FBC = ExtarctFBC(filtered_I,filtered_U,filtered_V )
    print('FBC_len', len(FBC))
    print('FBC_len', len(list(set(FBC))))

    # adding the he bounding box x y to extract the FBC, for displaying
    filtered_U = filtered_U + xyxy[0]
    filtered_V = filtered_V + xyxy[1]

    for id in range(len(filtered_U)):
        cv2.circle(image, (filtered_U[id], filtered_V[id]), 1, (0, 0, 255), 4)

    # get the partwise mean display them
    partwise_mean = find_partwise_mean(iuv_arr, origin=(xyxy[0], xyxy[1]))
    for key in partwise_mean:
        cv2.circle(image, partwise_mean[key], 3, (0, 255, 0), 5)

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    #
    # # visualize the 3D points
    # selcted_idx = np.random.randint(len(pc.pointcloud_vertices), size=10000)
    # sample_pc = np.array(pc.pointcloud_vertices)
    # sample_pc = sample_pc[selcted_idx[:]]
    # VisualizeObj(np.array(extracted_3D_points))

def find_partwise_mean(iuv_arr, origin = (0,0)):
    partwise_mean = {}

    # find all the unique body part ids in the array
    part_idxs = np.unique(iuv_arr[0])

    for id in part_idxs:
        if id != 0:
            V, U = np.where(iuv_arr[0] == id)
            U_mean = np.mean(U) + origin[0]
            V_mean = np.mean(V) + origin[1]
            partwise_mean[id] = (int(U_mean), int(V_mean))
    return partwise_mean

class PointCloudProcessing:
    def __init__(self):
        self.pointcloud_vertices = None
        colorIntrinsics = np.array([[1161.04, 0, 648.21],
                                    [0, 1161.72, 485.785],
                                    [0, 0, 1]])

        self.depthIntrinsics = np.array([[573.353, 0,	319.85],
                                    [0,	576.057, 240.632],
                                    [0, 0, 1]])

        self.colorIntrinsics_inv = np.linalg.inv(colorIntrinsics)
        # self.colorIntrinsics_inv = np.linalg.inv(self.depthIntrinsics)
        self.m_depthImageWidth = 640

    def loadPointCloud(self, pc_path=None):
        print('Load Point Cloud')
        vertices = []
        if pc_path is None:
            pc_path = '../../YawarSeq/frame000000.obj'

        for line in open(pc_path, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                vertices.append(v)

        print('vertices.shape', len(vertices))
        self.pointcloud_vertices = vertices
        return vertices

    def colourtoDepth(self,colourUV = np.array([601,650,1])):
        # take a UV point and convert it into the Depth Plane

        DepthUV = self.depthIntrinsics.dot(self.colorIntrinsics_inv.dot(colourUV))
        # print('DepthUV', DepthUV)

        return DepthUV

    def extractPointsInDepthUV(self,DepthUV):
        # print('extractPointsInUV')
        extracted_point = None
        interpolate_point = True


        # select the top and bottom pixels
        q11x = math.floor(DepthUV[0])
        q11y = math.floor(DepthUV[1])

        q22x = q11x + 1
        q22y = q11y + 1

        # print(q11x, q11y, q22x, q22y)

        # check all the pixels has valid points from the point cloud
        # check the sum
        q11 = self.pointcloud_vertices[q11x + q11y * self.m_depthImageWidth]
        q12 = self.pointcloud_vertices[q11x + q22y * self.m_depthImageWidth]
        q21 = self.pointcloud_vertices[q22x + q11y * self.m_depthImageWidth]
        q22 = self.pointcloud_vertices[q22x + q22y * self.m_depthImageWidth]
        Points3D = [q11, q12, q21, q22]

        for Point3D in Points3D:
            # print(Point3D, sum(Point3D))
            if sum(Point3D) == 0.0:
                interpolate_point = False
                break

        # extract the bilinear weight
        A = np.array([[1.0, q11x, q11y, q11x * q11y],
                     [1.0, q11x, q22y, q11x * q22y],
                     [1.0, q22x, q11y, q22x * q11y],
                     [1.0, q22x, q22y, q22x * q22y]])

        A_inv_Trans = np.linalg.inv(A).T
        X = np.array([1.0, DepthUV[0], DepthUV[1], DepthUV[0] * DepthUV[1]])
        B = A_inv_Trans.dot(X)

        if interpolate_point:
            # print('Implement Bilinear Interpolation')
            Points3D = np.asarray(Points3D)
            extracted_point = B[0] * Points3D[0] + B[1] * Points3D[1] + B[2] * Points3D[2] + B[3] * Points3D[3]
        else:
            # print('Find the closest point')
            B = np.asarray(B)
            extracted_point = Points3D[np.argmax(B)]

        return extracted_point

def SamplebasedonMean(iuv_arr, origin = (0,0)):

    image_path = '../../outputres.0001.png'
    from skimage import io
    image_final = io.imread(image_path)


    # load the point cloud
    pc = PointCloudProcessing()
    pc.loadPointCloud()

    partwise_mean = find_partwise_mean(iuv_arr, origin=origin)

    # find all the unique body part ids in the array
    part_idxs = np.unique(iuv_arr[0])
    num_pts_to_vis = 1000

    # for part_id in [19]:
    # extract the 3D points
    filtered_U = []
    filtered_V = []
    extracted_3D_points = []
    for part_id in part_idxs:
        selected_U = []
        selected_V = []
        if part_id != 0:
            V, U = np.where(iuv_arr[0] == part_id) # select all the point belonging to that id
            # U_min = np.amin(U)
            # U_max = np.amax(U)
            # V_min = np.amin(V)
            # V_max = np.amax(V)
            # print(U_min,U_max,V_min, V_max)
            # print(U_max-U_min,  V_max-V_min)
            print(part_id, len(V))
            # count = 0
            # for sample_idx in range(len(V)):
            #     # sample 100 points
            #     if U[sample_idx] - partwise_mean[id][0] < 100 and V[sample_idx] - partwise_mean[id][1] < 200 :
            #         print(sample_idx, U[sample_idx],V[sample_idx], partwise_mean[id][0],partwise_mean[id][1], U[sample_idx] - partwise_mean[id][0], V[sample_idx] - partwise_mean[id][1]  )
            #     #     count = count + 1
            #     #
            #     # if count >= num_sampl_points-1:
            #     #     break
            #
            # select 10 points from each part
            # uniformly sample points from the set
            # count = 0
            # for sample_idx in range(len(V)):
            #     # sample 100 points
            #     if U[sample_idx] - partwise_mean[id][0] < 100 and V[sample_idx] - partwise_mean[id][1] < 200 :
            #         print(sample_idx, U[sample_idx],V[sample_idx], partwise_mean[id][0],partwise_mean[id][1], U[sample_idx] - partwise_mean[id][0], V[sample_idx] - partwise_mean[id][1]  )
            #     #     count = count + 1
            #     #
            #     # if count >= num_sampl_points-1:
            #     #     break
            #
            # select 10 points from each part
            # uniformly sample points from the set
            # selcted_idx = np.random.randint(len(U), size=num_pts_to_vis)
            # selcted_idx = np.random.randint(len(U), size=500)
            selcted_idx = np.random.randint(len(U), size=len(V))
            selected_U.append(U[selcted_idx[:]])
            selected_V.append(V[selcted_idx[:]])

        selected_U = np.asarray(selected_U).flatten()
        selected_V = np.asarray(selected_V).flatten()

        selected_U = selected_U + origin[0]
        selected_V = selected_V + origin[1]

        for point_id in range(len(selected_U)):
            # print()
            # print('selected point', [selected_U[point_id], selected_V[point_id], 1])
            DepthUV = pc.colourtoDepth(np.array([selected_U[point_id], selected_V[point_id], 1]))
            extracted_point = pc.extractPointsInDepthUV(DepthUV)
            if sum(extracted_point) == 0.0:
                continue
            else:
                extracted_3D_points.append(extracted_point)
                filtered_U.append(selected_U[point_id])
                filtered_V.append(selected_V[point_id])

        # printed number of valid points
        # print('number of valid 3D points', len(extracted_3D_points))
        # for idx in range(len(selected_U)):
        #     cv2.circle(image_final, (selected_U[idx], selected_V[idx]), 1, (0, 0, 255),1)

    # sample 1000 points from the filtered points


    for idx in range(len(filtered_U)):
        cv2.circle(image_final, (filtered_U[idx], filtered_V[idx]), 1, (0, 255, 0), 1)

    for key in partwise_mean:
        cv2.circle(image_final, partwise_mean[key], 3, (0, 255, 0), 5)

    cv2.imshow('image', image_final)
    cv2.waitKey(0)

    # # visualize the 3D points
    # selcted_idx = np.random.randint(len(pc.pointcloud_vertices), size=10000)
    # sample_pc = np.array(pc.pointcloud_vertices)
    # sample_pc = sample_pc[selcted_idx[:]]
    # VisualizeObj(np.array(extracted_3D_points), sample_pc)


    return (selected_U, selected_V)


if __name__ == "__main__":

    print('main')
    datasetDirectory = '/home/bala/Documents/Link to GuidedResearch/Datasets/RenderedDataset/autumn_man'
    pickleFilename = 'densePosePrediction.pkl'
    with MyTimer('Loading and parsing pickel'):
        # pickle_in = open("../../FBC_frame0-1000.pkl", "rb")
        # pickle_in = open("../../YawarSeq/Yawar00.pkl", "rb")
        pickle_in = open(os.path.join(datasetDirectory, pickleFilename), "rb")
        example_dict = pickle.load(pickle_in)

    num_of_file = 36
    num_pts_to_vis = 1000
    generatorFBC = FBCGenerator(datasetDirectory)

    with MyTimer('Total Time Taken for Generation'):
        Agg_FBC_Framewise = []
        for i, row in enumerate(example_dict):
            frame = row['file_name']
            xyxy = row['pred_boxes_XYXY'].numpy().flatten().astype('int64')
            frame = frame[frame.find('frame') : frame.find('.png')]
            # frame = frame[frame.find(".png") - 13: frame.find(".png")]
            print()
            print()
            print(frame)
            print(xyxy)
            for j, result_encoded_w_shape in enumerate(row['pred_densepose'].results):
                print(row)
                iuv_arr = DensePoseResult.decode_png_data(*result_encoded_w_shape)
                print(iuv_arr.shape)

                # sample_perpart_IUV(iuv_arr, (xyxy[0], xyxy[1]),num_pts_to_vis)
                # # selected_U, selected_V = SamplebasedonMean(iuv_arr, (xyxy[0], xyxy[1]))

                # # find_partwise_mean(iuv_arr)
                # FBC = IUVtoFBC(iuv_arr, num_pts_to_vis)
                FBC, Extracted3DPoint = generatorFBC.IUVtoFBC_withDepth(frame, iuv_arr, (xyxy[0], xyxy[1]), num_pts_to_vis)
                print('FBC len', len(FBC))

            Agg_FBC_Framewise.append((frame, FBC, Extracted3DPoint))

            if i >= num_of_file-1:
                break

        pickling_on = open( datasetDirectory + "/DepthFBC_frame_f"+ str(num_of_file) + "_pts"+ str(num_pts_to_vis)+".pkl", "wb")
        pickle.dump(Agg_FBC_Framewise, pickling_on)
        pickling_on.close()
