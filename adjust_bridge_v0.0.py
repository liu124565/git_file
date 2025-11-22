import trimesh
import os
from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn
from adjust_crown_position4 import *
# from adjust_crown_top5 import adjust_crown_compactness
from adjust_bridge_mesh3 import mesh_cut
import numpy as np
from scipy.spatial.transform import Rotation as R



def adjust_mesh_grooves(bridge_mesh, is_upper=True, scale=0.6):  # 压平冠面
    bridge_mm = mn.meshFromFacesVerts(verts=bridge_mesh.vertices, faces=bridge_mesh.faces)
    # bridge_mm_copy1 = mm.copyMesh(bridge_mm)
    box = bridge_mm.computeBoundingBox()
    # Construct deformer on mesh vertices
    ffDeformer = mm.FreeFormDeformer(bridge_mm.points, bridge_mm.topology.getValidVerts())
    ffDeformer.init(mm.Vector3i.diagonal(3), box)
    # Move some control points of grid to the center

    move_z = abs(ffDeformer.getRefGridPointPosition(mm.Vector3i(0, 0, 1))[2]
                   - ffDeformer.getRefGridPointPosition(mm.Vector3i(0, 0, 0))[2])
    # move_z_copy = move_z
    move_z = move_z * scale

    if is_upper:
        move_z = -move_z
    control_move_lists = [
         [0, 1, 1],[1, 1, 1], [2, 1, 1]
    ]
    for control_move_list in control_move_lists:
        # print(control_move_list)
        orig_pos = ffDeformer.getRefGridPointPosition(mm.Vector3i(control_move_list[0], control_move_list[1], control_move_list[2]))
        new_pos = orig_pos + mm.Vector3f(0, 0, move_z)
        ffDeformer.setRefGridPointPosition(mm.Vector3i(control_move_list[0], control_move_list[1], control_move_list[2]), new_pos)

    ffDeformer.apply()
    # Invalidate mesh because of external vertices changes
    bridge_mm.invalidateCaches()

    # settings = mm.RemeshSettings()
    # settings.targetEdgeLen = 0.3
    # settings.useCurvature = True
    # settings.projectOnOriginalMesh = True
    # # settings.region = new_faces
    # mm.remesh(bridge_mm, settings)  # 执行 remesh
    # bridge_mm.pack()

    trimesh_vertices = mn.getNumpyVerts(bridge_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(bridge_mm.topology)
    adjust_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(), process=False)

    return adjust_mesh


def angle_between(v1, v2, degrees=True):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    # 归一化
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    # 点积求 cosθ
    cos_theta = np.dot(v1, v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止浮点误差
    angle = np.arccos(cos_theta)

    return np.degrees(angle) if degrees else angle


def get_ref_idx(mesh_mm, is_upper, is_right):
    box = mesh_mm.computeBoundingBox()
    ffDeformer = mm.FreeFormDeformer(mesh_mm.points, mesh_mm.topology.getValidVerts())
    ffDeformer.init(mm.Vector3i.diagonal(2), box)

    # if is_upper:
    #     if is_right:
    #         move_vec = mm.Vector3f(0.05, 0, 0.0)
    #         control_move_lists = [
    #             [1, 0, 1], [1, 1, 1]
    #         ]
    #         top_move_lists = [
    #             [1, 0, 0], [1, 1, 0]
    #         ]
    #     else:
    #         move_vec = mm.Vector3f(-0.05, 0, 0.0)
    #         control_move_lists = [
    #             [0, 0, 1], [0, 1, 1]
    #         ]
    #         top_move_lists = [
    #             [0, 0, 0], [0, 1, 0]
    #         ]
    # else:
    if is_right:
        move_vec = mm.Vector3f(0, -0.05, 0.0)
        control_move_lists = [
            [0, 0, 0], [1, 0, 0]
        ]
        top_move_lists = [
            [0, 0, 1], [1, 0, 1]
        ]
    else:
        move_vec = mm.Vector3f(0, 0.05, 0.0)
        control_move_lists = [
            [0, 1, 0], [1, 1, 0]
        ]
        top_move_lists = [
            [0, 1, 1], [1, 1, 1]
        ]

    orig_pos_list = []
    for control_move_list in control_move_lists:
        orig_pos = ffDeformer.getRefGridPointPosition(
            mm.Vector3i(control_move_list[0], control_move_list[1], control_move_list[2]))
        orig_pos_list.append(orig_pos)
    orig_pos_array = np.array(orig_pos_list)
    ref_point1_ = np.mean(orig_pos_array, axis=0)
    ref_point1_ = np.array(list(ref_point1_))
    # print(orig_pos_array, ref_point)

    orig_pos_list_ = []
    for control_move_list in top_move_lists:
        orig_pos = ffDeformer.getRefGridPointPosition(
            mm.Vector3i(control_move_list[0], control_move_list[1], control_move_list[2]))
        orig_pos_list_.append(orig_pos)
    orig_pos_array = np.array(orig_pos_list_)
    ref_point2_ = np.mean(orig_pos_array, axis=0)
    ref_point2_ = np.array(list(ref_point2_))

    ref_point3_ = np.array([ref_point2_[0], ref_point2_[1], 0.5 * (ref_point2_[2] + ref_point1_[2])])

    tree = cKDTree(mn.getNumpyVerts(mesh_mm))
    _, ref_idx1_0 = tree.query(np.array(list(orig_pos_list[0])))
    _, ref_idx1_2 = tree.query(np.array(list(orig_pos_list[1])))
    _, ref_idx1_1 = tree.query(ref_point1_)
    _, ref_idx2 = tree.query(ref_point2_)
    _, ref_idx3 = tree.query(ref_point3_)
    return [ref_idx1_0, ref_idx1_1, ref_idx1_2, ref_idx3], ref_idx2, move_vec, control_move_lists


def adjust_opening_angle_by_ffd(adjust_mesh, bilateral_teeth_list, is_upper, is_right):
    mesh_mm = mn.meshFromFacesVerts(verts=adjust_mesh.vertices, faces=adjust_mesh.faces)

    dir_vec_list = []
    for b_tooth in bilateral_teeth_list:
        b_tooth_mm = mn.meshFromFacesVerts(verts=b_tooth.vertices, faces=b_tooth.faces)
        ref_idx1, ref_idx2, _, _ = get_ref_idx(b_tooth_mm, is_upper, is_right)

        ref_point1 = mn.getNumpyVerts(b_tooth_mm)[ref_idx1[-1]]
        ref_point2 = mn.getNumpyVerts(b_tooth_mm)[ref_idx2]

        dir_vec = ref_point2 - ref_point1
        norm = np.linalg.norm(dir_vec)
        if norm != 0:
            dir_vec = dir_vec / norm
        dir_vec_list.append(dir_vec)

    b_tooth_dir_vec = np.array(dir_vec_list)
    b_dir_vec = np.mean(b_tooth_dir_vec, axis=0)

    b_dir_vec_copy = b_dir_vec.copy()
    b_dir_vec_copy[0] = 0


    ref_idx1, ref_idx2, move_vec, control_move_lists = get_ref_idx(mesh_mm, is_upper, is_right)
    # ref_idx1, ref_idx2 = idx_list[0], idx_list[1]
    box = mesh_mm.computeBoundingBox()
    ffDeformer = mm.FreeFormDeformer(mesh_mm.points, mesh_mm.topology.getValidVerts())
    ffDeformer.init(mm.Vector3i.diagonal(2), box)


    angele_delta = float('inf')
    count_ = 0
    while True:

        count_ += 1

        ref_point1 = mn.getNumpyVerts(mesh_mm)[ref_idx1[1]]
        ref_point2 = mn.getNumpyVerts(mesh_mm)[ref_idx2]
        # mm.saveMesh(mesh_mm, 'test_.stl')
        # pc = trimesh.PointCloud(ref_point1)
        # pc.export('1.ply')
        # pc = trimesh.PointCloud(ref_point2)
        # pc.export('2.ply')
        # pc = trimesh.PointCloud(mn.getNumpyVerts(mesh_mm))
        # pc.export('3.ply')

        # x_min, x_max = mn.getNumpyVerts(mesh_mm)[:, 0].min(),  mn.getNumpyVerts(mesh_mm)[:, 0].max()
        # middle_z = (x_min + x_max) / 2
        # range_x = [min((ref_point[0], middle_z)), max((ref_point[0], middle_z))]
        # verts = mn.getNumpyVerts(mesh_mm)
        # mask1 = (
        #         (verts[:, 0] >= range_x[0]) &
        #         (verts[:, 0] <= range_x[1]) &
        #         (np.abs(verts[:, 1] - ref_point[1]) <= 0.3)
        # )
        # filter_points = mn.getNumpyVerts(mesh_mm)[mask1]
        # if is_upper:
        #     top_idx = np.argmin(filter_points[:, 2])
        # else:
        #     top_idx = np.argmax(filter_points[:, 2])
        # top_point = filter_points[top_idx]
        dir_vec = ref_point2 - ref_point1
        norm = np.linalg.norm(dir_vec)
        if norm != 0:
            dir_vec = dir_vec / norm

        # best_normal_copy = best_normal.copy()
        # best_normal_copy[1] = 0
        dir_vec_copy = dir_vec.copy()
        dir_vec_copy[0] = 0

        count_angle = angle_between(b_dir_vec_copy, dir_vec_copy)


        # arrow2 = create_arrow(np.array([0, 0, 0]), dir_vec, total_length=5, shaft_ratio=1, shaft_radius=0.5)
        # arrow2.export("arrow_fixed2.stl")

        # ref_axis = np.array([0, 0, 1]) * np.sign(dir_vec[2])
        # angle_ = angle_with_axis(dir_vec, ref_axis)

        signed_angle = np.sign(dir_vec_copy[1] - b_dir_vec_copy[1]) * -1
        # signed_angle = 1
        if count_angle < angele_delta:
            mesh_mm_copy = mm.copyMesh(mesh_mm)
            # mm.saveMesh(mesh_mm, 'test.stl')
            angele_delta = count_angle
        else:
            signed_angle *= -1

        # if abs(opening_angle - angle_)

        print(count_angle, signed_angle, move_vec * signed_angle)
        # angele_delta = abs(opening_angle - angle_)

        if abs(count_angle) < 0.5:
            break

        if count_ > 500:
            mesh_mm = mesh_mm_copy
            break

        for control_move_list in control_move_lists:
            orig_ = ffDeformer.getRefGridPointPosition(mm.Vector3i(control_move_list[0], control_move_list[1], control_move_list[2]))
            new_pos = orig_ + move_vec * signed_angle
            ffDeformer.setRefGridPointPosition(mm.Vector3i(control_move_list[0], control_move_list[1], control_move_list[2]), new_pos)

        # Apply deformation to mesh vertices
        ffDeformer.apply()
        # Invalidate mesh because of external vertices changes
        mesh_mm.invalidateCaches()
        # mm.saveMesh(mesh_mm, 'test.stl')

    trimesh_vertices = mn.getNumpyVerts(mesh_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(mesh_mm.topology)
    adjust_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(), process=False)
    return adjust_mesh


def adjust_side_angle_by_ffd(adjust_mesh):

    ref_vec = np.array([0, 0, 1])

    min_z, max_z = adjust_mesh.vertices[:, 2].min(), adjust_mesh.vertices[:, 2].max()
    min_y, max_y = adjust_mesh.vertices[:, 1].min(), adjust_mesh.vertices[:, 1].max()
    min_x, max_x = adjust_mesh.vertices[:, 0].min(), adjust_mesh.vertices[:, 0].max()

    ref_point1 = np.array([min_x, (max_y + min_y) / 2., min_z])
    ref_point2 = np.array([min_x, (max_y + min_y) / 2., max_z])

    ref_point1_ = np.array([max_x, (max_y + min_y) / 2., min_z])
    ref_point2_ = np.array([max_x, (max_y + min_y) / 2., max_z])

    tree = cKDTree(adjust_mesh.vertices)
    _, ref_idx1 = tree.query(ref_point1)
    _, ref_idx2 = tree.query(ref_point2)
    _, ref_idx1_ = tree.query(ref_point1_)
    _, ref_idx2_ = tree.query(ref_point2_)

    mesh_mm = mn.meshFromFacesVerts(verts=adjust_mesh.vertices, faces=adjust_mesh.faces)

    control_move_list1 = [
        [0, 0, 0], [0, 1, 0]
    ]
    control_move_list2 = [
        [1, 0, 0], [1, 1, 0]
    ]

    box = mesh_mm.computeBoundingBox()
    ffDeformer = mm.FreeFormDeformer(mesh_mm.points, mesh_mm.topology.getValidVerts())
    ffDeformer.init(mm.Vector3i.diagonal(2), box)

    for ref_idx, control_move_list, move_vec in zip(
    [[ref_idx1, ref_idx2], [ref_idx1_, ref_idx2_]],
    [control_move_list1, control_move_list2],
    [mm.Vector3f(0.05, 0, 0.0), mm.Vector3f(-0.05, 0, 0.0)]
    ):

        angele_delta = float('inf')
        count_ = 0
        while True:

            count_ += 1

            ref_point1 = mn.getNumpyVerts(mesh_mm)[ref_idx[0]]
            ref_point2 = mn.getNumpyVerts(mesh_mm)[ref_idx[1]]
            # mm.saveMesh(mesh_mm, 'test_.stl')
            # pc = trimesh.PointCloud(ref_point1)
            # pc.export('1.ply')
            # pc = trimesh.PointCloud(ref_point2)
            # pc.export('2.ply')
            # pc = trimesh.PointCloud(mn.getNumpyVerts(mesh_mm))
            # pc.export('3.ply')

            # x_min, x_max = mn.getNumpyVerts(mesh_mm)[:, 0].min(),  mn.getNumpyVerts(mesh_mm)[:, 0].max()
            # middle_z = (x_min + x_max) / 2
            # range_x = [min((ref_point[0], middle_z)), max((ref_point[0], middle_z))]
            # verts = mn.getNumpyVerts(mesh_mm)
            # mask1 = (
            #         (verts[:, 0] >= range_x[0]) &
            #         (verts[:, 0] <= range_x[1]) &
            #         (np.abs(verts[:, 1] - ref_point[1]) <= 0.3)
            # )
            # filter_points = mn.getNumpyVerts(mesh_mm)[mask1]
            # if is_upper:
            #     top_idx = np.argmin(filter_points[:, 2])
            # else:
            #     top_idx = np.argmax(filter_points[:, 2])
            # top_point = filter_points[top_idx]
            dir_vec = ref_point2 - ref_point1
            norm = np.linalg.norm(dir_vec)
            if norm != 0:
                dir_vec = dir_vec / norm

            # best_normal_copy = best_normal.copy()
            # best_normal_copy[1] = 0
            dir_vec_copy = dir_vec.copy()
            dir_vec_copy[1] = 0

            # print(dir_vec_copy, ref_vec)
            dir_vec_copy = np.sign(dir_vec_copy[2] - 0) * dir_vec_copy
            # print(dir_vec_copy, ref_vec)
            count_angle = angle_between(ref_vec, dir_vec_copy)


            # arrow2 = create_arrow(np.array([0, 0, 0]), dir_vec, total_length=5, shaft_ratio=1, shaft_radius=0.5)
            # arrow2.export("arrow_fixed2.stl")

            # ref_axis = np.array([0, 0, 1]) * np.sign(dir_vec[2])
            # angle_ = angle_with_axis(dir_vec, ref_axis)

            signed_angle = np.sign(dir_vec_copy[0] - ref_vec[0]) * -1
            # signed_angle = 1
            if count_angle < angele_delta:
                mesh_mm_copy = mm.copyMesh(mesh_mm)
                # mm.saveMesh(mesh_mm, 'test.stl')
                angele_delta = count_angle
            else:
                signed_angle *= -1

            # if abs(opening_angle - angle_)

            print(count_angle, signed_angle, move_vec * signed_angle)
            # angele_delta = abs(opening_angle - angle_)

            if abs(count_angle) < 0.3:
                break

            if count_ > 500:
                mesh_mm = mesh_mm_copy
                break

            # mesh_mm = Laplace_stretched_1(mesh_mm, ref_idx1, ref_normal=move_vec * signed_angle, expand=50,
            #                               fix_points_idx=list(np.append(fix_points_idx, ref_idx2)))
            for control_move in control_move_list:
                orig_ = ffDeformer.getRefGridPointPosition(mm.Vector3i(control_move[0], control_move[1], control_move[2]))
                new_pos = orig_ + move_vec * signed_angle
                ffDeformer.setRefGridPointPosition(mm.Vector3i(control_move[0], control_move[1], control_move[2]), new_pos)

            # Apply deformation to mesh vertices
            ffDeformer.apply()
            # Invalidate mesh because of external vertices changes
            mesh_mm.invalidateCaches()
            # mm.saveMesh(mesh_mm, 'test.stl')

    trimesh_vertices = mn.getNumpyVerts(mesh_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(mesh_mm.topology)
    adjust_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(), process=False)
    return adjust_mesh



def adjust_best_normal_by_ffd(adjust_mesh, v_perp, is_upper, is_right):
    mesh_mm = mn.meshFromFacesVerts(verts=adjust_mesh.vertices, faces=adjust_mesh.faces)

    box = mesh_mm.computeBoundingBox()
    ffDeformer = mm.FreeFormDeformer(mesh_mm.points, mesh_mm.topology.getValidVerts())
    ffDeformer.init(mm.Vector3i.diagonal(2), box)

    ref_z1 = mn.getNumpyVerts(mesh_mm)[:, 0].max()
    ref_z2 = mn.getNumpyVerts(mesh_mm)[:, 0].min()
    mask1 = (np.abs(mn.getNumpyVerts(mesh_mm)[:, 0] - ref_z1) <= 1)
    mask2 = (np.abs(mn.getNumpyVerts(mesh_mm)[:, 0] - ref_z2) <= 1)


    if is_upper:
        move_vec = mm.Vector3f(0.0, 0, -0.05)
        filter_points = mn.getNumpyVerts(mesh_mm)[mask1].reshape(-1, 3)
        ref_idx1 = np.argmin(filter_points[:, 2])
        filter_points = mn.getNumpyVerts(mesh_mm)[mask2].reshape(-1, 3)
        ref_idx2 = np.argmin(filter_points[:, 2])
        if not is_right:
            top_move_lists = [
                [0, 1, 0], [1, 1, 0]
            ]
            temp_z = mn.getNumpyVerts(mesh_mm)[mask1][ref_idx1][2]
        else:
            top_move_lists = [
                [0, 0, 0], [1, 0, 0]
            ]
            temp_z = mn.getNumpyVerts(mesh_mm)[mask2][ref_idx2][2]
    else:
        move_vec = mm.Vector3f(0.0, 0, 0.05)

        filter_points = mn.getNumpyVerts(mesh_mm)[mask1]
        ref_idx1 = np.argmax(filter_points[:, 2])
        filter_points = mn.getNumpyVerts(mesh_mm)[mask2]
        ref_idx2 = np.argmax(filter_points[:, 2])

        if not is_right:
            top_move_lists = [
                [0, 1, 0], [1, 1, 0]
            ]
            temp_z = mn.getNumpyVerts(mesh_mm)[mask1][ref_idx1][2]
        else:
            top_move_lists = [
                [0, 0, 0], [1, 0, 0]
            ]
            temp_z = mn.getNumpyVerts(mesh_mm)[mask2][ref_idx2][2]

    # ref_idx1, ref_idx2, move_vec, control_move_lists = get_ref_idx(mesh_mm, is_upper, is_right)
    # ref_idx1, ref_idx2 = idx_list[0], idx_list[1]
    # box = mesh_mm.computeBoundingBox()
    # ffDeformer = mm.FreeFormDeformer(mesh_mm.points, mesh_mm.topology.getValidVerts())
    # ffDeformer.init(mm.Vector3i.diagonal(2), box)

    angele_delta = float('inf')
    count_ = 0
    while True:

        count_ += 1

        ref_point1 = mn.getNumpyVerts(mesh_mm)[mask1][ref_idx1]
        ref_point2 = mn.getNumpyVerts(mesh_mm)[mask2][ref_idx2]
        # mm.saveMesh(mesh_mm, 'test_.stl')
        # pc = trimesh.PointCloud(ref_point1)
        # pc.export('1.ply')
        # pc = trimesh.PointCloud(ref_point2)
        # pc.export('2.ply')
        # pc = trimesh.PointCloud(mn.getNumpyVerts(mesh_mm))
        # pc.export('3.ply')

        # x_min, x_max = mn.getNumpyVerts(mesh_mm)[:, 0].min(),  mn.getNumpyVerts(mesh_mm)[:, 0].max()
        # middle_z = (x_min + x_max) / 2
        # range_x = [min((ref_point[0], middle_z)), max((ref_point[0], middle_z))]
        # verts = mn.getNumpyVerts(mesh_mm)
        # mask1 = (
        #         (verts[:, 0] >= range_x[0]) &
        #         (verts[:, 0] <= range_x[1]) &
        #         (np.abs(verts[:, 1] - ref_point[1]) <= 0.3)
        # )
        # filter_points = mn.getNumpyVerts(mesh_mm)[mask1]
        # if is_upper:
        #     top_idx = np.argmin(filter_points[:, 2])
        # else:
        #     top_idx = np.argmax(filter_points[:, 2])
        # top_point = filter_points[top_idx]
        dir_vec = ref_point2 - ref_point1
        norm = np.linalg.norm(dir_vec)
        if norm != 0:
            dir_vec = dir_vec / norm

        if not is_upper:
            dir_vec *= -1

        # best_normal_copy = best_normal.copy()
        # best_normal_copy[1] = 0
        dir_vec_copy = dir_vec.copy()
        dir_vec_copy[1] = 0

        count_angle = angle_between(v_perp, dir_vec_copy)


        # arrow2 = create_arrow(np.array([0, 0, 0]), dir_vec, total_length=5, shaft_ratio=1.5, shaft_radius=0.5)
        # arrow2.export("arrow_fixed2.stl")

        signed_angle = np.sign(dir_vec_copy[2] - v_perp[2])
        # signed_angle = 1
        if count_angle < angele_delta:
            mesh_mm_copy = mm.copyMesh(mesh_mm)
            # mm.saveMesh(mesh_mm, 'test.stl')
            angele_delta = count_angle
        # else:
        #     signed_angle *= -1

        # print(count_angle, signed_angle, move_vec * signed_angle)

        if abs(count_angle) < 0.5:
            break

        if count_ > 500:
            mesh_mm = mesh_mm_copy
            break

        if not is_right:
            if abs(mn.getNumpyVerts(mesh_mm)[mask1][ref_idx1][2] - temp_z) > 1:
                break
        else:
            if abs(mn.getNumpyVerts(mesh_mm)[mask2][ref_idx2][2] - temp_z) > 1:
                break

        # mesh_mm = Laplace_stretched_1(mesh_mm, ref_idx1, ref_normal=move_vec * signed_angle, expand=50,
        #                               fix_points_idx=list(np.append(fix_points_idx, ref_idx2)))
        for control_move_list in top_move_lists:
            orig_ = ffDeformer.getRefGridPointPosition(mm.Vector3i(control_move_list[0], control_move_list[1], control_move_list[2]))
            new_pos = orig_ + move_vec * signed_angle
            ffDeformer.setRefGridPointPosition(mm.Vector3i(control_move_list[0], control_move_list[1], control_move_list[2]), new_pos)

        # Apply deformation to mesh vertices
        ffDeformer.apply()
        # Invalidate mesh because of external vertices changes
        mesh_mm.invalidateCaches()
        # mm.saveMesh(mesh_mm, 'test.stl')

    trimesh_vertices = mn.getNumpyVerts(mesh_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(mesh_mm.topology)
    adjust_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(), process=False)
    return adjust_mesh

def contour_extrude_mesh(pc, scale_xy=1.2, close_loop=True):
    pc_copy = pc.copy()
    points = pc_copy.vertices
    # 放大 XY
    center = pc.centroid
    pc.apply_translation(-center)
    pc.apply_scale([scale_xy, scale_xy, scale_xy])
    pc.apply_translation(center)

    points_scaled = pc.vertices

    N = len(pc.vertices)
    # 创建侧面三角形索引
    faces = []
    for i in range(N):
        i_next = (i + 1) % N if close_loop else i + 1
        if i_next >= N:
            break
        # 原始层：i, i_next
        # 放大层：i+N, i_next+N
        faces.append([i, i_next, i_next + N])
        faces.append([i, i_next + N, i + N])

    # 合并顶点
    vertices = np.vstack([points, points_scaled])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)

    return mesh, pc_copy, pc

# 根据基台轮廓线获取对于曲面
def groove_boundary_mesh(base_mesh:trimesh.Trimesh,
              is_upper: bool =True):

    _, base_boundary_idx = find_boundary_edges(base_mesh)
    base_boundary_points = base_mesh.vertices[base_boundary_idx]
    pc = trimesh.PointCloud(base_boundary_points)

    mesh, pc1, pc2 = contour_extrude_mesh(pc, scale_xy=3)

    mesh_mm = mn.meshFromFacesVerts(verts=mesh.vertices, faces=mesh.faces)
    mesh_mm_Verts = mm.VertBitSet()
    mesh_mm_Verts.resize(mesh_mm.topology.getValidVerts().size())
    for ancV in range(mesh_mm.topology.getValidVerts().size()):
        mesh_mm_Verts.set(mm.VertId(ancV), True)

    facebit = mm.getIncidentFaces(mesh_mm.topology, mesh_mm_Verts)
    settings = mm.SubdivideSettings()
    settings.smoothMode = True
    settings.maxEdgeLen = 0.3
    settings.maxEdgeSplits = 2000
    settings.maxDeviationAfterFlip = 0.2
    settings.region = facebit

    # 根据细分后的新顶点进行可选曲率定位.
    newVertsBitSet = mm.VertBitSet()
    settings.newVerts = newVertsBitSet
    mm.subdivideMesh(mesh_mm, settings)

    fix_verts_tree = cKDTree(mn.getNumpyVerts(mesh_mm))
    _, idx1 = fix_verts_tree.query(pc1.vertices)
    _, idx2 = fix_verts_tree.query(pc2.vertices)
    fix_idx = list(set(idx1 + idx2))

    mesh_mm_Verts = mm.VertBitSet()
    mesh_mm_Verts.resize(mesh_mm.topology.getValidVerts().size())
    for ancV in range(mesh_mm.topology.getValidVerts().size()):
        if ancV in fix_idx:
            continue
        mesh_mm_Verts.set(mm.VertId(ancV), True)

    mm.positionVertsSmoothly(mesh_mm, mesh_mm_Verts,
                             mm.LaplacianEdgeWeightsParam.Cotan,
                             mm.VertexMass.NeiArea
                             )

    holes = mesh_mm.topology.findHoleRepresentiveEdges()
    hole_info = []
    for e in holes:
        dir_area = mm.holeDirArea(mesh_mm.topology, mesh_mm.points, e)
        area = dir_area.length()
        hole_info.append((e, area, dir_area))
    # print(hole_info)
    # # 按面积降序排列
    hole_info.sort(key=lambda x: x[1], reverse=False)

    newFacesBitSet = mm.FaceBitSet()
    params = mm.FillHoleParams()
    # params.metric = mrmeshpy.getUniversalMetric(mesh)
    params.outNewFaces = newFacesBitSet
    mm.fillHole(mesh_mm, hole_info[0][0], params)

    settings = mm.SubdivideSettings()
    settings.smoothMode = True
    settings.maxEdgeLen = 0.3
    settings.maxEdgeSplits = 2000
    settings.maxDeviationAfterFlip = 0.2
    settings.region = newFacesBitSet

    # 根据细分后的新顶点进行可选曲率定位.
    newVertsBitSet = mm.VertBitSet()
    settings.newVerts = newVertsBitSet
    mm.subdivideMesh(mesh_mm, settings)

    mm.positionVertsSmoothly(mesh_mm, newVertsBitSet,
                             mm.LaplacianEdgeWeightsParam.Cotan,
                             mm.VertexMass.NeiArea
                             )

    if is_upper:
        if mesh_mm.normal(mm.VertId(0))[2] > 0:
            mesh_mm.topology.flipOrientation()
    else:
        if mesh_mm.normal(mm.VertId(0))[2] < 0:
            mesh_mm.topology.flipOrientation()
    return mesh_mm


def ffd_mesh_1(bridge_mesh, groove_mesh, bilateral_teeth_list,
                bad_tooth_num=None):  # 收束下截面 调整开口 尖点 再桥接

    is_upper = False
    # if (bad_teeth_num // 10) % 10 in [2, 4]:
    #     is_right = True
    # else:
    #     is_right = False
    is_right = bad_tooth_num > 30

    bridge_mm = mn.meshFromFacesVerts(verts=bridge_mesh.vertices, faces=bridge_mesh.faces)
    # ref_mesh_mm = mn.meshFromFacesVerts(verts=ref_mesh.vertices, faces=ref_mesh.faces)
    # groove_mesh_mm =  mn.meshFromFacesVerts(verts=groove_mesh.vertices, faces=groove_mesh.faces)
    # groove_mesh_mm.topology.flipOrientation()

    bilateral_teeth_boundary_points = []
    for bilateral_tooth in bilateral_teeth_list:
        _, boundary_indices = find_boundary_edges(bilateral_tooth)
        bilateral_teeth_boundary_points.append(bilateral_tooth.vertices[boundary_indices])

    _, boundary_indices = find_boundary_edges(groove_mesh)
    groove_boundary_points = groove_mesh.vertices[boundary_indices]

    components = groove_mesh.split(only_watertight=False)  # 关闭 only_watertight 以保留非封闭的面片
    # 按面片数量排序，取最大的一个
    largest_component = max(components, key=lambda m: len(m.faces))
    # largest_component.export('rotated_base_mesh.stl')

    box = bridge_mm.computeBoundingBox()
    # Construct deformer on mesh vertices
    ffDeformer = mm.FreeFormDeformer(bridge_mm.points, bridge_mm.topology.getValidVerts())
    ffDeformer.init(mm.Vector3i.diagonal(3), box)
    # Move some control points of grid to the center
    if is_upper:
        control_move_lists = [
            [0, 0, 2], [0, 1, 2], [0, 2, 2],
            [1, 0, 2], [1, 1, 2], [1, 2, 2],
            [2, 0, 2], [2, 1, 2], [2, 2, 2],
        ]
    else:
        control_move_lists = [
            [0, 0, 0], [0, 1, 0], [0, 2, 0],
            [1, 0, 0], [1, 1, 0], [1, 2, 0],
            [2, 0, 0], [2, 1, 0], [2, 2, 0],
        ]

    z_list = []
    z_dist_list = []
    z_ref_tree = cKDTree(groove_boundary_points)
    for b_, bilateral_boundary_points in enumerate(bilateral_teeth_boundary_points):
        z_ref_dist, z_ref_idx = z_ref_tree.query(bilateral_boundary_points)
        z_list.append(bilateral_boundary_points[np.argmin(z_ref_dist)][2])
        z_dist_list.append(np.min(z_ref_dist))

    if is_upper:
        z_mask = groove_boundary_points[:, 2].max()
        bridge_move_z = ffDeformer.getRefGridPointPosition(mm.Vector3i(0, 0, 2))[2]
        move_z = z_mask - bridge_move_z
        slice_z = min(z_list)

        move_z_ = -1

    else:
        # z_mask = min(z_list)
        z_mask = groove_boundary_points[:, 2].min()
        bridge_move_z = ffDeformer.getRefGridPointPosition(mm.Vector3i(0, 0, 0))[2]
        move_z = z_mask - bridge_move_z
        slice_z = max(z_list)

        move_z_ = 1

    for control_move_list in control_move_lists:
        orig_pos = ffDeformer.getRefGridPointPosition(mm.Vector3i(control_move_list[0], control_move_list[1], control_move_list[2]))
        new_pos = orig_pos + mm.Vector3f(0, 0, move_z)
        ffDeformer.setRefGridPointPosition(mm.Vector3i(control_move_list[0], control_move_list[1], control_move_list[2]), new_pos)

    # Apply deformation to mesh vertices
    ffDeformer.apply()
    # Invalidate mesh because of external vertices changes
    bridge_mm.invalidateCaches()

    # mm.saveMesh(bridge_mm, 'ffd_bridge.stl')

    trimesh_vertices = mn.getNumpyVerts(bridge_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(bridge_mm.topology)
    adjust_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(), process=False)

    # 有邻牙根据邻牙 切 无邻牙在基台上面切
    if len(bilateral_teeth_list) > 0:
        z_threshold_list = []
        for b_tooth in bilateral_teeth_list:
            z_threshold_list.append(b_tooth.centroid[2].copy())

        if is_upper:
            z_threshold1 = min(z_threshold_list)
            z_threshold = max(z_threshold1, slice_z)
        else:
            z_threshold1 = max(z_threshold_list)
            z_threshold = min(z_threshold1, slice_z)
    else:
        if is_upper:
            z_threshold = groove_boundary_points[:, 2].min() - 1

        else:
            z_threshold = groove_boundary_points[:, 2].max() + 1

    # 根据z轴高度切
    adjust_mesh = slice_mesh_z(adjust_mesh, z_threshold=z_threshold, is_upper=is_upper)
    adjust_mesh.process()
    adjust_mesh.export('boolean_mesh.stl')

    #--------------------------------------------------  轮廓线布尔切
    # from get_groove_boundary_mesh import groove_boundary_mesh
    groove_boundary_mm = groove_boundary_mesh(largest_component, is_upper)
    groove_points = mn.getNumpyVerts(groove_boundary_mm)
    groove_points[:, 2] += move_z_
    groove_boundary_mm.points.vec = mn.fromNumpyArray(
        groove_points)
    # mm.saveMesh(groove_boundary_mm, 'groove_boundary_.stl')

    bridge_mm1 = mn.meshFromFacesVerts(verts=adjust_mesh.vertices, faces=adjust_mesh.faces)
    boolean_mesh = mm.boolean(groove_boundary_mm, bridge_mm1, mm.BooleanOperation.OutsideB).mesh
    bridge_mm1 = boolean_mesh
    #---------------------------------------------------
    mm.saveMesh(boolean_mesh, 'boolean_mesh1.stl')
    # bridge_mm1 = mn.meshFromFacesVerts(verts=adjust_mesh.vertices, faces=adjust_mesh.faces)

    # settings = mm.RemeshSettings()
    # settings.targetEdgeLen = 0.2
    # settings.useCurvature = True
    # settings.projectOnOriginalMesh = True
    # mm.remesh(bridge_mm1, settings)  # 执行 remesh
    # bridge_mm1.pack()
    # # mm.saveMesh(bridge_mm1, 'boolean_mesh1.stl')

    trimesh_vertices = mn.getNumpyVerts(bridge_mm1)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(bridge_mm1.topology)
    adjust_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(), process=False)
    adjust_mesh.merge_vertices(
        merge_tex=True,
        merge_norm=True,
        digits_vertex=1  # 0.1 的量级
    )
    adjust_mesh.export('boolean_mesh2.stl')

    # from 最佳投影方向 import find_best_visible_direction
    # best_normal = find_best_visible_direction(largest_component, is_upper=bad_teeth_num<30)
    # best_normal = np.array(best_normal).reshape(-1, 3).squeeze()
    #
    # best_normal_copy = best_normal.copy()
    # best_normal_copy[0] = 0
    #
    #
    # v_xz = best_normal.copy()
    # v_xz[1] = 0  # 投影到XZ平面
    # n = np.array([0, 1, 0])  # XZ平面法线（Y轴）
    # v_perp = np.cross(n, v_xz)
    # v_perp = v_perp / np.linalg.norm(v_perp)

    # 根据最佳投影调整尖点位置（）
    # adjust_mesh = adjust_best_normal_by_ffd(adjust_mesh, v_perp, is_upper, is_right)
    # adjust_mesh.export('boolean_mesh_.stl')

    # 调整开口
    # idx1, idx2, _, _ = get_ref_idx(bridge_mm1, is_upper, is_right)
    # adjust_mesh_copy = adjust_mesh.copy()
    # adjust_mesh = adjust_opening_angle_by_ffd(adjust_mesh, bilateral_teeth_list, is_upper, is_right)
    # # if bad_tooth_num % 10 >= 6:
    # #     adjust_mesh = adjust_opening_angle_by_ffd(adjust_mesh, bilateral_teeth_list, is_upper, not is_right)
    # adjust_mesh.export('boolean_mesh2.stl')

    # adjust_mesh = adjust_side_angle_by_ffd(adjust_mesh)
    # adjust_mesh.export('boolean_mesh3.stl')

    from adjust_bridge_mesh3 import adjust_mesh_compare_base
    adjust_mesh = adjust_mesh_compare_base(adjust_mesh, base_mesh, base_expend=0.6)
    adjust_mesh.export('boolean_mesh4.stl')

    # 再次调整位置
    # move_center_point = adjust_mesh_copy.centroid.copy()
    # x_adjust1 = adjust_mesh_copy.vertices[:, 0].max() - adjust_mesh_copy.vertices[:, 0].min()
    # x_adjust2 = adjust_mesh.vertices[:, 0].max() - adjust_mesh.vertices[:, 0].min()
    #
    # x_scale = x_adjust1 / x_adjust2
    # adjust_mesh.apply_scale([x_scale, 1, 1])  # 缩放mesh
    #
    # adjust_mesh.apply_translation(move_center_point - adjust_mesh.centroid)  # 移动到指定位置

    # adjust_mesh.export('boolean_mesh___.stl')
    # cube_mesh.export('cube_mesh.obj')
    # groove_mesh.export('groove_mesh.obj')


    adjust_mesh.process()
    largest_component.process()
    _, bridge_boundary_indices = find_boundary_edges(adjust_mesh)

    largest_component.faces = largest_component.faces[:, ::-1] # 基台的面法向量取反
    combined_mesh = trimesh.util.concatenate([largest_component, adjust_mesh])
    combined_mesh.export('combined_mesh.stl')

    bridge_mm_copy1 = mn.meshFromFacesVerts(verts=combined_mesh.vertices, faces=combined_mesh.faces)
    # bridge_mm_copy1.pack()
    holes = bridge_mm_copy1.topology.findHoleRepresentiveEdges()
    # print(holes)
    hole_info = []
    for e in holes:
        dir_area = mm.holeDirArea(bridge_mm_copy1.topology, bridge_mm_copy1.points, e)
        area = dir_area.length()
        hole_info.append((e, area, dir_area))
    # 按面积降序排列
    hole_info.sort(key=lambda x: x[1], reverse=True)

    newFacesBitSet = mm.FaceBitSet()
    params = mm.StitchHolesParams()
    params.outNewFaces = newFacesBitSet
    params.metric = mm.getEdgeLengthStitchMetric(bridge_mm_copy1)
    mm.buildCylinderBetweenTwoHoles(bridge_mm_copy1, hole_info[0][0], hole_info[1][0], params)

    # 仅细分新网格
    settings = mm.SubdivideSettings()
    settings.smoothMode = False
    settings.maxEdgeLen = 0.3
    settings.maxEdgeSplits = 2000
    settings.maxDeviationAfterFlip = 0.2
    settings.region = newFacesBitSet

    # 根据细分后的新顶点进行可选曲率定位.
    newVertsBitSet = mm.VertBitSet()
    settings.newVerts = newVertsBitSet
    mm.subdivideMesh(bridge_mm_copy1, settings)

    settings = mm.InflateSettings(
        pressure=8,
        iterations=3,
        preSmooth=True,
        gradualPressureGrowth=True
    )
    mm.inflate(bridge_mm_copy1, newVertsBitSet, settings)

    # mm.positionVertsSmoothly(bridge_mm_copy1, newVertsBitSet, mm.LaplacianEdgeWeightsParam.Cotan,
    #                              mm.VertexMass.NeiArea)



    mm.saveMesh(bridge_mm_copy1, 'combined_bridge_mesh.stl')

    base_y_min, base_y_max = largest_component.vertices[:, 1].min(), largest_component.vertices[:, 1].max()
    y_mask = (
            (mn.getNumpyVerts(bridge_mm_copy1)[:, 0] >= base_y_min) &
            (mn.getNumpyVerts(bridge_mm_copy1)[:, 0] <= base_y_max)
    )
    filter_y_points = mn.getNumpyVerts(bridge_mm_copy1)[y_mask]

    # 可选定位
    tree = cKDTree(mn.getNumpyVerts(bridge_mm_copy1))
    _, groove_idx = tree.query(largest_component.vertices)
    _, filter_idx = tree.query(filter_y_points)
    #
    ref_z_range_list = []
    ref_x_idx = np.argmax(adjust_mesh.vertices[:, 0])
    ref_x = adjust_mesh.vertices[ref_x_idx]
    mask_ = (np.abs(adjust_mesh.vertices[:, 0] - ref_x[0]) <= 0.3)
    f_points = adjust_mesh.vertices[mask_]
    ref_z_range_list.append(abs(f_points[:, 2].max() - f_points[:, 2].min()))

    ref_x_idx = np.argmin(adjust_mesh.vertices[:, 0])
    ref_x = adjust_mesh.vertices[ref_x_idx]
    mask_ = (np.abs(adjust_mesh.vertices[:, 0] - ref_x[0]) <= 0.3)
    f_points = adjust_mesh.vertices[mask_]
    ref_z_range_list.append(abs(f_points[:, 2].max() - f_points[:, 2].min()))

    ref_z_range = min(ref_z_range_list)

    if ref_z_range < 1.5:
        r = ref_z_range * 0.5
    else:
        r = 1

    smooth_idx = tree.query_ball_point(adjust_mesh.vertices[bridge_boundary_indices], r=r)
    groove_boundary_idx = tree.query_ball_point(groove_boundary_points, r=0.3)
    groove_boundary_idx = set([x for sub in groove_boundary_idx for x in sub])
    filter_idx = list(set(filter_idx) - groove_boundary_idx)

    smooth_idx = list(set([x for sub in smooth_idx for x in sub]) - set(groove_idx) - groove_boundary_idx)

    smooth_Verts = mm.VertBitSet()
    smooth_Verts.resize(bridge_mm_copy1.topology.getValidVerts().size())
    for ancV in smooth_idx:
        smooth_Verts.set(mm.VertId(ancV), True)

    filter_Verts = mm.VertBitSet()
    filter_Verts.resize(bridge_mm_copy1.topology.getValidVerts().size())
    for ancV in filter_idx:
        filter_Verts.set(mm.VertId(ancV), True)

    # for _ in range(3):
    mm.positionVertsSmoothly(bridge_mm_copy1, (smooth_Verts | newVertsBitSet) & filter_Verts, mm.LaplacianEdgeWeightsParam.Cotan,
                                 mm.VertexMass.NeiArea)
    mm.saveMesh(bridge_mm_copy1, 'smooth1.stl')
    left_mesh = mm.Mesh()
    # faceidx = mm.getInnerFaces(bridge_mm_copy1.topology, (smooth_Verts | newVertsBitSet) & filter_Verts)
    faceidx = mm.getInnerFaces(bridge_mm_copy1.topology, smooth_Verts)

    left_mesh.addPartByMask(bridge_mm_copy1, faceidx)
    mm.saveMesh(left_mesh, "left_part_.stl")


    for _ in range(3):
        mm.positionVertsSmoothly(bridge_mm_copy1, smooth_Verts, mm.LaplacianEdgeWeightsParam.Cotan,
                                     mm.VertexMass.NeiArea
                                 )

    mm.saveMesh(bridge_mm_copy1, 'smooth2.stl')
    pc = trimesh.PointCloud(mn.getNumpyVerts(bridge_mm_copy1))
    pc.export('smooth.ply')

    region_face_bit = mm.getIncidentFaces(bridge_mm_copy1.topology, smooth_Verts)
    settings = mm.RemeshSettings()
    settings.finalRelaxIters = 3
    settings.targetEdgeLen = 0.2
    settings.useCurvature = True
    settings.projectOnOriginalMesh = False
    settings.region = region_face_bit
    mm.remesh(bridge_mm_copy1, settings)  # 执行 remesh
    mm.saveMesh(bridge_mm_copy1, 'smooth3.stl')

    # 调整开口
    # idx1, idx2, _, _ = get_ref_idx(bridge_mm1, is_upper, is_right)
    # adjust_mesh_copy = adjust_mesh.copy()
    # adjust_mesh = adjust_opening_angle_by_ffd(adjust_mesh, bilateral_teeth_list, is_upper, is_right)

    trimesh_vertices = mn.getNumpyVerts(bridge_mm_copy1)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(bridge_mm_copy1.topology)
    adjust_bridge_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(), process=False)

    return adjust_bridge_mesh



def slice_mesh_z(mesh: trimesh.Trimesh, z_threshold: float =None, is_upper: bool =True) -> trimesh.Trimesh:
    """
    沿 z=阈值 平面平整切割 mesh，不封口。
    """
    verts = mesh.vertices

    if z_threshold is None:
        if is_upper:
            z_threshold = np.max(verts[:, :2]) - 2
        else:
            z_threshold = np.min(verts[:, :2]) + 2

    # 定义切割平面
    plane_origin = [0, 0, z_threshold]
    if is_upper:
        plane_normal = [0, 0, -1]
    else:
        plane_normal = [0, 0, 1]

    # 用 slice_plane 切出平面交线
    # 这个函数会直接修改 mesh 顶点并生成新的边界
    sliced = mesh.slice_plane(plane_origin=plane_origin, plane_normal=plane_normal)

    return sliced


def angle_with_axis(dir_vec, ref_axis):
    # x_axis = np.array([1, 0, 0])
    dir_vec = np.array(dir_vec)
    cos_theta = np.dot(dir_vec, ref_axis) / (np.linalg.norm(dir_vec) * np.linalg.norm(ref_axis))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差
    theta = np.arccos(cos_theta)  # 弧度
    return np.degrees(theta)  # 返回角度

def get_Rz_by_normal(dir_vec, ref_axis, direction_axis, clockwise=False, rotated_centroid=[0, 0, 0]):
    dir_vec = np.array(dir_vec)
    angle_ = angle_with_axis(dir_vec, ref_axis)
    # print(angle_)
    if not clockwise:
        angle_ = -angle_

    Rz = trimesh.transformations.rotation_matrix(
        angle=np.radians(angle_), # 负号表示顺时针
        direction=direction_axis,  # 旋转轴
        point=rotated_centroid  # 旋转中心
    )
    # 应用旋转
    # mesh.apply_transform(Rz)
    return Rz


def ray_point_num(meshA, meshB, vector_base_to_crown=np.array([0, 0, 1])):
    ray_origins = meshA.vertices
    # ray_directions = move_normals / np.linalg.norm(move_normals, axis=1, keepdims=True)
    ray_directions = np.tile(-vector_base_to_crown, (len(ray_origins), 1))  # shape (N,3)

    # 计算每条射线的最近交点
    # 返回：location.shape = (N, 3)，其中命中失败的会是 NaN
    locations, index_ray, index_tri = meshB.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False  # 单次交点
    )

    # 对于所有射线按顺序获取最近交点（包括 None）
    # 创建 N 行 3 列，填 NaN
    hit_points = np.full((len(ray_origins), 3), np.nan)
    hit_points[index_ray] = locations

    # 计算每条射线的距离（失败为 NaN）
    distances = np.linalg.norm(hit_points - ray_origins, axis=1)

    # -----------------------
    # 结果统计
    # -----------------------
    # 有效命中数量
    valid_mask = ~np.isnan(distances)
    hit_count = valid_mask.sum()

    # 平均距离（忽略 NaN）
    mean_distance = distances[valid_mask].mean() if hit_count > 0 else 0.0
    return hit_count, mean_distance


def adjust_tooth_bulge(crown_top_mesh, upper_tooth, vector_base_to_crown=np.array([0, 0, 1])):
    """" 将假牙与对面牙齿穿模部分进行调整，使穿模部分吸附在对面牙齿上"""
    adsorb_final_mesh = crown_top_mesh.copy()
    sign = np.sign(vector_base_to_crown[2])
    # final_boolean_mesh = perform_boolean_operation(adsorb_final_mesh, upper_tooth, shear_type="InsideA")

    hit_count, mean_distance = ray_point_num(adsorb_final_mesh, upper_tooth)
    # print(hit_count, mean_distance)

    # print(len(final_boolean_mesh.vertices), '----------------')
    # 多次循环调整冠面高度，防止牙冠穿模太多破坏冠面纹理
    for i in range(50):
        if hit_count == 0:
            return adsorb_final_mesh
        if hit_count < 900:
            break
        adsorb_final_mesh.vertices[:, 2] -= sign * 0.05
        # final_boolean_mesh = perform_boolean_operation(adsorb_final_mesh, upper_tooth, shear_type="InsideA")
        hit_count, mean_distance = ray_point_num(adsorb_final_mesh, upper_tooth)
        # print(hit_count, mean_distance)

    return adsorb_final_mesh


def adjust_mesh_top(mesh, bilateral_teeth_list, sphere_mesh):

    max_top_z = np.max(mesh.vertices[:, 2])
    # 遍历每个牙齿模型
    z_values = []
    for tooth in bilateral_teeth_list:
        # if not is_upper:
        max_z = np.max(tooth.vertices[:, 2])
        # if max_z < np.max(base_mesh.vertices[:, 2]):
        #     z_values.append(np.max(base_mesh.vertices[:, 2]) + 1)
        # else:
        z_values.append(max_z)


    if len(z_values) > 0:
        # max_tooth_z = np.mean(z_values)
        max_tooth_z = max(z_values)
    else:
        max_tooth_z = 0.0

    move_distance = max_tooth_z - max_top_z
    mesh.vertices[:, 2] += move_distance  # 根据邻牙调整高度

    # 根据上颌牙调整高度
    mesh = adjust_tooth_bulge(mesh, sphere_mesh)

    return mesh


def adjust_scale(implant_mesh, bilateral_teeth, base_mesh):

    implant_mesh_center = implant_mesh.centroid
    max_x = implant_mesh.vertices[:, 0].max()
    min_x = implant_mesh.vertices[:, 0].min()

    if len(bilateral_teeth) == 2:
        implant_mesh_size = abs(max_x - min_x)

        center_list = []
        for bilateral_mesh in bilateral_teeth:
            bilateral_teeth_mm = get_face_mesh(bilateral_mesh, implant_mesh)
            trimesh_vertices = mn.getNumpyVerts(bilateral_teeth_mm)  # 转为trimesh格式
            trimesh_faces = mn.getNumpyFaces(bilateral_teeth_mm.topology)
            bilateral_teeth_tri = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy())

            bilateral_teeth_center = np.mean(bilateral_teeth_tri.vertices, axis=0)
            b_tree = cKDTree(bilateral_teeth_tri.vertices)
            _, b_idx = b_tree.query(bilateral_teeth_center)
            bilateral_teeth_center = bilateral_teeth_tri.vertices[b_idx]
            del b_tree

            center_list.append(bilateral_teeth_center[0])

        ref_dist = abs(center_list[0] - center_list[1])
        scale_factor = ref_dist / implant_mesh_size
        return scale_factor, center_list

    elif len(bilateral_teeth) == 1:
        implant_mesh_size = abs(max_x - implant_mesh_center[0])

        bilateral_teeth_mm = get_face_mesh(bilateral_teeth[0], implant_mesh)
        trimesh_vertices = mn.getNumpyVerts(bilateral_teeth_mm)  # 转为trimesh格式
        trimesh_faces = mn.getNumpyFaces(bilateral_teeth_mm.topology)
        bilateral_teeth_tri = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy())

        bilateral_teeth_center = np.mean(bilateral_teeth_tri.vertices, axis=0)
        b_tree = cKDTree(bilateral_teeth_tri.vertices)
        _, b_idx = b_tree.query(bilateral_teeth_center)
        bilateral_teeth_center = bilateral_teeth_tri.vertices[b_idx]
        del b_tree

        ref_dist = abs(bilateral_teeth_center[0] - implant_mesh_center[0])
        scale_factor = ref_dist / implant_mesh_size
        return scale_factor, []

    elif len(bilateral_teeth) == 0:
        implant_mesh_size = abs(max_x - implant_mesh_center[0])
        base_mesh_size = abs(base_mesh.vertices[:, 0].max() - base_mesh.vertices[:, 0].min())
        scale_factor = (base_mesh_size * 1.1) /implant_mesh_size

        return scale_factor, []


def get_dir_normal(cube_mesh):

    cube_obb = cube_mesh.bounding_box_oriented
    cube_obb_points = cube_obb.vertices
    mean_z = np.mean(cube_obb_points[:, 2])
    z_cube_mask = cube_obb_points[:, 2] < mean_z
    cube_obb_points = cube_obb.vertices[z_cube_mask]

    cube_point1 = cube_obb_points[0]
    cube_point2 = cube_obb_points[1]
    cube_point3 = cube_obb_points[2]

    # x_dist, y_dist = (np.linalg.norm(cube_point2 - cube_point1),
    #                   np.linalg.norm(cube_point3 - cube_point1))
    # extents = implant_mesh.bounding_box.extents

    dir_normal0 = cube_point2 - cube_point1
    dir_normal0[2] = 0
    dir_normal0 = dir_normal0 / np.linalg.norm(dir_normal0)

    dir_normal1 = cube_point3 - cube_point1
    dir_normal1[2] = 0
    dir_normal1 = dir_normal1 / np.linalg.norm(dir_normal1)

    ref_axis_x = np.array([1, 0, 0]) * np.sign(dir_normal0[0])
    angle0 = angle_with_axis(dir_normal0, ref_axis_x)
    ref_axis_x = np.array([1, 0, 0]) * np.sign(dir_normal1[0])
    angle1 = angle_with_axis(dir_normal1, ref_axis_x)
    if angle0 < angle1:
        dir_normal = dir_normal0
    else:
        dir_normal = dir_normal1
    return dir_normal



def align_vector_to_z(vec, is_upper, threshold_angele=15):
    vec = np.array(vec, dtype=float)
    vec = vec / np.linalg.norm(vec)  # normalize
    if is_upper:
        z = np.array([0, 0, -1], dtype=float)
    else:
        z = np.array([0, 0, 1], dtype=float)
    # 如果已经平行 Z 轴
    if np.allclose(vec, z):
        return np.eye(4)  # identity
    # if np.allclose(vec, -z):
    #     # 反向，绕X或Y旋转180度
    #     return R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
    # rotation axis = cross product
    axis = np.cross(vec, z)
    axis = axis / np.linalg.norm(axis)
    # angle = arccos(dot)
    angle = np.arccos(np.dot(vec, z))
    # Rodrigues formula
    R_mat = R.from_rotvec(axis * angle).as_matrix()

    # print("旋转角度 (°):", np.degrees(angle))
    # 若最佳投影与z的夹角大于__ 返回单位矩阵
    if np.degrees(angle) > threshold_angele:
        return np.eye(4)

    # 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R_mat
    return T

def create_arrow(origin, direction, total_length=1.0, shaft_ratio=0.8, shaft_radius=0.02):
    """
    创建箭头Mesh
    origin: 起点 (3,)
    direction: 方向向量 (3,)
    total_length: 箭头总长度
    shaft_ratio: 杆体长度占比 (0~1)
    shaft_radius: 杆体半径
    """
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    head_length = total_length * (1 - shaft_ratio)
    shaft_length = total_length * shaft_ratio
    head_radius = shaft_radius * 2.0

    # 创建杆体（圆柱）
    shaft = trimesh.creation.cylinder(
        radius=shaft_radius,
        height=shaft_length,
        sections=32
    )
    shaft.apply_translation([0, 0, shaft_length / 2])

    # 创建箭头头部（圆锥）
    head = trimesh.creation.cone(
        radius=head_radius,
        height=head_length,
        sections=32
    )
    head.apply_translation([0, 0, shaft_length + head_length / 2])

    # 合并
    arrow = trimesh.util.concatenate([shaft, head])

    # 旋转到指定方向
    z_axis = np.array([0, 0, 1])
    rot_matrix = trimesh.geometry.align_vectors(z_axis, direction)
    if rot_matrix is not None:
        arrow.apply_transform(rot_matrix)

    # 平移到起点
    arrow.apply_translation(origin)

    return arrow


def get_fix_idxs(implant_mesh, bad_tooth_num=None):

    implant_mesh_vertices = implant_mesh.vertices
    x_max, x_min = implant_mesh_vertices[:, 0].max(), implant_mesh_vertices[:, 0].min()
    y_max, y_min = implant_mesh_vertices[:, 1].max(), implant_mesh_vertices[:, 1].min()
    if bad_tooth_num < 30:
        implant_z = implant_mesh.vertices[:, 2].min()
    else:
        implant_z = implant_mesh.vertices[:, 2].max()

    if bad_tooth_num % 10 >= 6:
        y_unit = (y_max - y_min) / 4

        implant_point_x1 = np.array([x_min, y_min + y_unit, implant_z])
        implant_point_x1_ = np.array([x_min, y_max - y_unit, implant_z])
        implant_point_x2 = np.array([x_max, y_min + y_unit, implant_z])
        implant_point_x2_ = np.array([x_max, y_max - y_unit, implant_z])
        implant_point = np.vstack([implant_point_x1, implant_point_x1_, implant_point_x2, implant_point_x2_])

    else:
        implant_point_x1 = np.array([x_min, (y_max + y_min) / 2, implant_z])
        implant_point_x2 = np.array([x_max, (y_max + y_min) / 2, implant_z])
        implant_point = np.vstack([implant_point_x1, implant_point_x2])

    # print(implant_point)
    implant_tree = cKDTree(implant_mesh_vertices)
    _, fix_idx = implant_tree.query(implant_point)

    # pc = trimesh.PointCloud(implant_mesh.vertices[fix_idx])
    # pc.export(r'test.ply')

    return fix_idx


def repare_mesh(mesh):
    mesh_mm = mn.meshFromFacesVerts(verts=mesh.vertices, faces=mesh.faces)

    # 2) 准备 FindParams：upDirection 与 wallAngle（单位同文档，wallAngle 可为正/负）
    #    upDirection: 可用 Vector3f(0,0,1) 表示 Z+ 为“上”
    #    wallAngle: 0 => 严格垂直墙；正数扩大下方墙；负数缩小
    find_params = mm.FixUndercuts.FindParams(mm.Vector3f(0.0, 0.0, 1.0), 0.0)

    # 3) 计算合适的 voxelSize（体素分辨率），通常按 bbox 大小的比例设定
    bbox_diag = mesh_mm.computeBoundingBox().diagonal()
    voxel_size = float(bbox_diag) * 5e-3  # 举例：bbox * 0.005 -> 精度/内存权衡

    # 4) bottomExtension：底部延伸最小值（用于在下方构建垂直墙体时延伸基底）
    bottom_ext = float(bbox_diag) * 2e-2  # 举例：bbox * 0.02

    # 5) region：如果想修复整个 mesh，传 None；要固定局部则传 FaceBitSet（见下）
    region = None  # 整个网格

    # 6) smooth：是否对 voxels 做一次高斯平滑（若存在薄壁建议 True）
    smooth = True

    # 7) 进度回调 cb(progress: float) -> bool。返回 False 可中断操作
    def progress_cb(p: float) -> bool:
        # print(f"[FixUndercuts] progress: {p * 100:.1f}%")
        pass
        return True

    # 8) 构造 FixParams（使用文档中的聚合构造器）
    params = mm.FixUndercuts.FixParams(find_params, voxel_size, bottom_ext, region, smooth, progress_cb)

    # 9) 执行修复（在原 mesh 上原地修改；操作是 voxel-based，会重建网格）
    mm.FixUndercuts.fix(mesh_mm, params)

    # 10) 保存结果
    # mm.saveMesh(mesh_mm, "input_fixed.stl")

    settings = mm.RemeshSettings()
    settings.targetEdgeLen = 0.25
    settings.useCurvature = True
    settings.projectOnOriginalMesh = False
    mm.remesh(mesh_mm, settings)  # 执行 remesh
    mesh_mm.pack()
    # mm.saveMesh(mesh_mm, 'input_fixed1.stl')
    trimesh_vertices = mn.getNumpyVerts(mesh_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(mesh_mm.topology)
    adjust_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(),
                                         process=False)
    return adjust_mesh


def adjust_implant(implant_mesh, bilateral_teeth, base_mesh, health_mesh, bad_teeth_num):

    is_upper = False

    # 对牙及补洞------------------------------------------------------------
    new_vertices, new_faces, _ = mesh_cut(health_mesh.vertices,
                                          health_mesh.faces, base_mesh.centroid, r=10, is_upper=bad_teeth_num < 30)
    sphere_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    sphere_mm = mn.meshFromFacesVerts(verts=sphere_mesh.vertices, faces=sphere_mesh.faces)
    holes = sphere_mm.topology.findHoleRepresentiveEdges()
    hole_info = []
    for e in holes:
        dir_area = mm.holeDirArea(sphere_mm.topology, sphere_mm.points, e)
        area = dir_area.length()
        hole_info.append((e, area, dir_area))
    # 按面积降序排列
    hole_info.sort(key=lambda x: x[1], reverse=True)
    Edge_mean_Len = 0.5 * (np.array(list(mm.edgeLengths(sphere_mm.topology, sphere_mm.points.vec))).max()
                           - np.array(list(mm.edgeLengths(sphere_mm.topology, sphere_mm.points.vec))).min())  # 中值

    all_bitset = mm.VertBitSet()
    all_bitset.resize(sphere_mm.topology.getValidVerts().size())
    for i in range(1, len(hole_info)):
        params = mm.FillHoleParams()
        edge_metric = mm.getEdgeLengthFillMetric(sphere_mm)
        params.metric = edge_metric
        # 创建一个 FaceBitSet 用来接收新面
        bitset = mm.FaceBitSet()
        params.outNewFaces = bitset  # 记录补洞时的新面
        e, _, _ = hole_info[i]
        mm.fillHole(sphere_mm, e, params)

        settings = mm.SubdivideSettings()
        settings.smoothMode = True
        settings.maxEdgeLen = Edge_mean_Len
        settings.maxEdgeSplits = 2000
        settings.maxDeviationAfterFlip = 0.2
        settings.region = bitset

        # 根据细分后的新顶点进行可选曲率定位.
        newVertsBitSet = mm.VertBitSet()
        settings.newVerts = newVertsBitSet
        mm.subdivideMesh(sphere_mm, settings)

        # mm.saveMesh(mesh_mm, 'repair_mesh.stl')

        mm.expand(sphere_mm.topology, newVertsBitSet, 0)  #扩展平滑区域

        for _ in range(3):
            mm.positionVertsSmoothly(sphere_mm, newVertsBitSet,
                                     mm.LaplacianEdgeWeightsParam.Cotan,
                                     mm.VertexMass.NeiArea
                                     )

    trimesh_vertices = mn.getNumpyVerts(sphere_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(sphere_mm.topology)
    sphere_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(), process=False)
    sphere_mesh.export('sphere_mesh.stl')
    # 对牙及补洞------------------------------------------------------------

    # base_mesh_copy = base_mesh.copy()
    implant_mesh = adjust_mesh_grooves(implant_mesh, is_upper, scale=1)  # 压平冠面纹理  若模板经过手动处理 注释该项  scale模板压平程度 越大越平

    # direction_axis = [0, 0, -1]
    # dir_normal = get_dir_normal(cube_mesh)
    #
    # ref_axis_x = np.array([1, 0, 0]) * np.sign(dir_normal[0])
    # # Rz = get_Rz_by_normal(dir_normal, ref_axis_x, direction_axis, clockwise=is_right)  #顺时针旋转矩阵
    # # implant_mesh.apply_transform(Rz)  #旋转
    #
    # # 牙弓矩阵
    # Rz = get_Rz_by_normal(dir_normal, ref_axis_x, direction_axis, clockwise= not is_right)  #逆时针旋转矩阵
    #
    # # 最佳投影旋转 矩阵
    # best_normal = find_best_visible_direction(base_mesh, is_upper=bad_teeth_num < 30)
    # best_normal = np.array(best_normal).reshape(-1, 3).squeeze()
    # T = align_vector_to_z(best_normal, is_upper, threshold_angele=15)  # 若最佳投影与z的夹角大于threshold_angele 返回单位矩阵
    # # base_mesh.apply_transform(T)
    #
    #
    # # 根据颈缘线修剪邻牙
    # bilateral_teeth_list = []
    # for tooth_ in bilateral_teeth:
    #     pre_tooth_mesh, _ = contour_cut_mesh(tooth_, bad_whole_mesh)  # 错误分割返回原来mesh
    #     bilateral_teeth_list.append(pre_tooth_mesh.copy())
    #
    # # 基台 邻牙 旋转到轴对称方向
    # bilateral_teeth_list_rotated = []
    # for b, b_tooth in enumerate(bilateral_teeth_list):
    #     copy_mesh = b_tooth.copy()
    #     copy_mesh.apply_transform(Rz)
    #     copy_mesh.apply_transform(T)
    #     bilateral_teeth_list_rotated.append(copy_mesh)
    #     # copy_mesh.export(rf'rotated_bilateral_tooth_{b}.stl')
    #
    # # implant_mesh.apply_transform(Rz)
    # base_mesh.apply_transform(Rz)

    #
    # base_mesh.export(rf'rotated_base_mesh0.stl')
    # base_mesh.apply_transform(T)
    # cube_mesh_copy = cube_mesh.copy()
    # cube_mesh_copy.apply_transform(Rz)
    # cube_mesh_copy.export(rf'cube_mesh.stl')
    # cube_mesh_copy1 = cube_mesh_copy.copy()
    # cube_mesh_copy1.apply_transform(T)
    # cube_mesh_copy1.export(rf'rotated_cube_mesh.stl')
    # base_mesh.export(rf'rotated_base_mesh1.stl')
    #
    # # arrow = create_arrow(base_mesh.centroid, best_normal, total_length=5, shaft_ratio=1.5, shaft_radius=0.5)
    # # arrow.export("arrow_fixed1.stl")
    # #
    # # best_normal1 = find_best_visible_direction(base_mesh, is_upper=bad_teeth_num < 30)
    # # best_normal1 = np.array(best_normal1).reshape(-1, 3).squeeze()
    # #
    # # arrow = create_arrow(base_mesh.centroid, best_normal1, total_length=5, shaft_ratio=1.5, shaft_radius=0.5)
    # # arrow.export("arrow_fixed2.stl")

    # 调整模板大小
    scale_factor, center_list = adjust_scale(implant_mesh, bilateral_teeth, base_mesh)

    move_center_point = implant_mesh.centroid.copy()
    implant_mesh.apply_scale([scale_factor, 1, 1])  # 根据盒子缩放mesh
    move_center_point[2] = implant_mesh.centroid[2]
    implant_mesh.apply_translation(move_center_point - implant_mesh.centroid)  # 缩放会改变牙齿位置
    implant_mesh.export(rf'adjust_scale.stl')

    #根据邻牙调整高度
    move_center_point = implant_mesh.centroid.copy()
    implant_mesh = adjust_mesh_top(implant_mesh, bilateral_teeth, sphere_mesh)
    implant_mesh.export(rf'adjust_implant_mesh0.stl')

    if len(center_list) > 0:

        move_center_point[0] = 0.5 * (center_list[0] + center_list[1])
        # move_center_point[2] = implant_mesh.centroid[2]
        implant_mesh.apply_translation(move_center_point - implant_mesh.centroid)  # 根据邻牙左右微调位置
        implant_mesh.export(rf'adjust_implant_mesh1.stl')

    implant_mesh = ffd_mesh_1(implant_mesh, base_mesh.copy(), bilateral_teeth,
                              bad_teeth_num)

    top_fix_idx = get_fix_idxs(implant_mesh, bad_teeth_num)

    implant_mesh.export(rf'adjust_implant_mesh2.stl')


    # groove_tree = cKDTree(implant_mesh.vertices)
    # _, g_idx = groove_tree.query(base_mesh_copy.vertices)
    # implant_mesh.vertices[g_idx] = base_mesh_copy.vertices

    return implant_mesh, bilateral_teeth, top_fix_idx



if __name__ == '__main__':


    # # # bad_teeth_num = 45
    root_path = r'Z:\10-算法组\gbz\测试数据11170900\27'
    files = os.listdir(root_path)

    num_files = [f for f in files if 'control_points_raw_' in f and '.ply' in f]
    #         if len(num_files) == 0:
    #             continue
    bad_teeth_num = int(num_files[0].split('.')[0].split('_')[-1])

    implant_mesh = trimesh.load_mesh(fr"{root_path}\r_crown_top_mesh.ply")

    implant_mesh = repare_mesh(implant_mesh)

    bilateral_files = [f for f in files if 'r_bil_' in f and '.ply' in f]

    base_mesh = trimesh.load_mesh(fr"{root_path}\r_groove_mesh.stl")

    bad_mesh = trimesh.load_mesh(fr"{root_path}\r_bad_mesh.ply")
    health_mesh = trimesh.load_mesh(fr"{root_path}\r_health_mesh.ply")


    # #__________________________________________________________________
    # from edge_utils import find_boundary_edges, perform_boolean_operation, parallel_generate_faces, \
    #     encrypt_control_points, point_adsorb_mesh, create_offset, calculate_normal, simplified_mesh
    #
    # angle=45
    # first_offset_distance = 0.2
    # second_offset_distance = 0.1
    # vector_base_to_crown = np.array([0, 0, 1]).astype(float)
    # # 4. 计算每个边界点的法向量，基于边界点进行偏移
    # boundary_edges, boundary_indices = find_boundary_edges(base_mesh)
    # boundary_vertices = base_mesh.vertices[boundary_indices]
    # boundary_normals = calculate_normal(boundary_vertices)  # 计算边缘点法线
    # # 执行第一次偏移
    # first_offset_points = create_offset(boundary_vertices, boundary_normals, first_offset_distance, angle=0)
    # # 执行第二次偏移
    # angel = -angle * np.sign(vector_base_to_crown[2])
    # second_offset_points = create_offset(first_offset_points, boundary_normals, second_offset_distance, angel)
    #
    # # 5. 基于偏移点生成面, 得到具有凹槽的膨胀mesh
    # offset_mesh = parallel_generate_faces(base_mesh, boundary_vertices, first_offset_points)
    # groove_mesh = parallel_generate_faces(offset_mesh, first_offset_points, second_offset_points)
    # base_mesh = groove_mesh
    # #________________________________________________________________

    base_mesh_copy = base_mesh.copy()

    bilateral_teeth = []

    for bilateral_file in bilateral_files:
        b_mesh = trimesh.load(os.path.join(root_path, bilateral_file))
        bilateral_teeth.append(b_mesh)

    implant_mesh, bilateral_teeth_list, top_fix_idx = adjust_implant(implant_mesh, bilateral_teeth,
                                  base_mesh.copy(), health_mesh, bad_teeth_num)

    # implant_mesh.export('1.stl')    # crown_center = base_mesh_copy.centroid
    #
    # from adjust_bridge_mesh3 import adjust_mesh_bridge, mesh_cut
    #
    # health_mesh = trimesh.load_mesh(fr"{root_path}\health_tooth.stl")
    # base_mesh = trimesh.load_mesh(fr"{root_path}\groove_mesh.stl")
    #
    # new_vertices, new_faces, _ = mesh_cut(health_mesh.vertices,
    #                                       health_mesh.faces, base_mesh.centroid, r=10, is_upper=bad_teeth_num < 30)
    # sphere_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    #
    # implant_mesh = adjust_mesh_bridge(implant_mesh, base_mesh, sphere_mesh, top_fix_idx, is_upper=bad_teeth_num < 30)
    # implant_mesh.export(rf'implant_.stl')


    # import time
    #
    # root_path = r'Z:\10-算法组\gbz\测试数据11170900'
    # for root, dirs, files in os.walk(root_path):
    #
    #     if len(dirs) == 0:
    #
    #         num_files = [f for f in files if 'control_points_raw_' in f and '.ply' in f]
    #         if len(num_files) == 0:
    #             continue
    #         bad_teeth_num = int(num_files[0].split('.')[0].split('_')[-1])
    #         # if bad_teeth_num % 10 != 6:
    #         #     continue
    #
    #         start_time = time.time()
    #         print(root, bad_teeth_num)
    #         # filtered = [s for s in files if "adj_" in s]
    #         filtered = [f for f in files if 'r_bil_' in f and '.ply' in f]
    #         bilateral_teeth = []
    #         try:
    #             for f in filtered:
    #                 tooth = trimesh.load(os.path.join(root, f))
    #                 bilateral_teeth.append(tooth)
    #
    #             implant_mesh = trimesh.load(os.path.join(root, 'r_crown_top_mesh.ply'))
    #             implant_mesh = repare_mesh(implant_mesh)
    #             # crown = trimesh.load_mesh(rf"Z:\10-算法组\gbz\lmp 测试数据\牙型3\牙齿_{str(bad_teeth_num)}.stl")
    #             base_mesh = trimesh.load(os.path.join(root, "r_groove_mesh.stl"))
    #             bad_mesh = trimesh.load(os.path.join(root, "r_bad_mesh.ply"))
    #
    #             health_mesh = trimesh.load(os.path.join(root, "r_health_mesh.ply"))
    #             # new_vertices, new_faces, _ = mesh_cut(health_mesh.vertices,
    #             #                                     health_mesh.faces, base_mesh.centroid, r=10, is_upper=bad_teeth_num < 30)
    #             # sphere_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    #             #
    #             # sphere_mm = mn.meshFromFacesVerts(verts=sphere_mesh.vertices, faces=sphere_mesh.faces)
    #             # holes = sphere_mm.topology.findHoleRepresentiveEdges()
    #             # hole_info = []
    #             # for e in holes:
    #             #     dir_area = mm.holeDirArea(sphere_mm.topology, sphere_mm.points, e)
    #             #     area = dir_area.length()
    #             #     hole_info.append((e, area, dir_area))
    #             # # 按面积降序排列
    #             # hole_info.sort(key=lambda x: x[1], reverse=True)
    #             # Edge_mean_Len = 0.5 * (np.array(list(mm.edgeLengths(sphere_mm.topology, sphere_mm.points.vec))).max()
    #             #                        - np.array(
    #             #             list(mm.edgeLengths(sphere_mm.topology, sphere_mm.points.vec))).min())  # 中值
    #             #
    #             # all_bitset = mm.VertBitSet()
    #             # all_bitset.resize(sphere_mm.topology.getValidVerts().size())
    #             # for i in range(1, len(hole_info)):
    #             #     params = mm.FillHoleParams()
    #             #     edge_metric = mm.getEdgeLengthFillMetric(sphere_mm)
    #             #     params.metric = edge_metric
    #             #     # 创建一个 FaceBitSet 用来接收新面
    #             #     bitset = mm.FaceBitSet()
    #             #     params.outNewFaces = bitset  # 记录补洞时的新面
    #             #     e, _, _ = hole_info[i]
    #             #     mm.fillHole(sphere_mm, e, params)
    #             #
    #             #     settings = mm.SubdivideSettings()
    #             #     settings.smoothMode = True
    #             #     settings.maxEdgeLen = Edge_mean_Len
    #             #     settings.maxEdgeSplits = 2000
    #             #     settings.maxDeviationAfterFlip = 0.2
    #             #     settings.region = bitset
    #             #
    #             #     # 根据细分后的新顶点进行可选曲率定位.
    #             #     newVertsBitSet = mm.VertBitSet()
    #             #     settings.newVerts = newVertsBitSet
    #             #     mm.subdivideMesh(sphere_mm, settings)
    #             #
    #             #     # mm.saveMesh(mesh_mm, 'repair_mesh.stl')
    #             #
    #             #     mm.expand(sphere_mm.topology, newVertsBitSet, 0)
    #             #
    #             #     for _ in range(3):
    #             #         mm.positionVertsSmoothly(sphere_mm, newVertsBitSet,
    #             #                                  mm.LaplacianEdgeWeightsParam.Cotan,
    #             #                                  mm.VertexMass.NeiArea
    #             #                                  )
    #             #
    #             # trimesh_vertices = mn.getNumpyVerts(sphere_mm)  # 转为trimesh格式
    #             # trimesh_faces = mn.getNumpyFaces(sphere_mm.topology)
    #             # sphere_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(),
    #             #                               process=False)
    #             # sphere_mesh.export('sphere_mesh.stl')
    #
    #             # __________________________________________________________________
    #             # from edge_utils import find_boundary_edges, perform_boolean_operation, parallel_generate_faces, \
    #             #     encrypt_control_points, point_adsorb_mesh, create_offset, calculate_normal, simplified_mesh
    #             #
    #             # angle = 45
    #             # first_offset_distance = 0.2
    #             # second_offset_distance = 0.1
    #             # vector_base_to_crown = np.array([0, 0, 1]).astype(float)
    #             # # 4. 计算每个边界点的法向量，基于边界点进行偏移
    #             # boundary_edges, boundary_indices = find_boundary_edges(base_mesh)
    #             # boundary_vertices = base_mesh.vertices[boundary_indices]
    #             # boundary_normals = calculate_normal(boundary_vertices)  # 计算边缘点法线
    #             # # 执行第一次偏移
    #             # first_offset_points = create_offset(boundary_vertices, boundary_normals, first_offset_distance, angle=0)
    #             # # 执行第二次偏移
    #             # angel = -angle * np.sign(vector_base_to_crown[2])
    #             # second_offset_points = create_offset(first_offset_points, boundary_normals, second_offset_distance,
    #             #                                      angel)
    #             #
    #             # # 5. 基于偏移点生成面, 得到具有凹槽的膨胀mesh
    #             # offset_mesh = parallel_generate_faces(base_mesh, boundary_vertices, first_offset_points)
    #             # groove_mesh = parallel_generate_faces(offset_mesh, first_offset_points, second_offset_points)
    #             # base_mesh = groove_mesh
    #             # # ________________________________________________________________
    #             # groove_mesh.export(os.path.join(root, 'r_groove_mesh.stl'))
    #             # base_mesh_copy = base_mesh.copy()
    #             #
    #             implant_mesh, bilateral_teeth_list, top_fix_idx = adjust_implant(implant_mesh, bilateral_teeth,
    #                                                                              base_mesh.copy(), health_mesh,
    #                                                                              bad_teeth_num)
    #
    #             # implant_mesh.export('1.stl')  # crown_center = base_mesh_copy.centroid
    #             implant_mesh.export(os.path.join(root, 'r_bridge_mesh.stl'))
    #
    #         except Exception as e:
    #             print(type(e).__name__, e)
    #         print("--- %s seconds ---" % (time.time() - start_time))

