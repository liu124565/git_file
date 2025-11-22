import numpy as np
import trimesh
from scipy.spatial import cKDTree, KDTree
from edge_utils import find_boundary_edges
from meshlib import mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
import pyvista as pv
import math
from trimesh import grouping
import heapq
from collections import defaultdict
import time


def get_evenly_spaced_points(points, num_points=12):
    """
    从点序列中随机选择一个起始点，等间隔提取 num_points 个点（环状取法）

    参数:
        points (np.ndarray): (N, D) 点数组（2D 或 3D）
        num_points (int): 要提取的点数（默认4）

    返回:
        np.ndarray: (num_points, D) 的等间隔点数组
    """
    N = len(points)
    if N < num_points:
        # raise ValueError("点数量不足")
        return points

    step = N // num_points  # 向下取整
    start = np.random.randint(0, N)  # 随机起点

    indices = [(start + i * step) % N for i in range(num_points)]
    return points[indices]


def get_graph(base_mesh):
    # 获取曲率数据
    curv = base_mesh['Mean_Curvature'].copy()
    curv[curv < 0] *= 0.3
    curvatures = np.abs(curv)
    epsilon = 1e-5  # 避免除零

    # 构建邻接图
    n_points = base_mesh.n_points
    graph = [[] for _ in range(n_points)]

    # 遍历所有三角形面构建邻接关系
    # faces = base_mesh.faces.reshape(-1, 4)[:, 1:]  # 重构为N×3的三角形面
    # 获取所有三角形面片（自动处理混合类型）
    triangles = base_mesh.extract_surface().triangulate()
    faces = triangles.faces.reshape(-1, 4)[:, 1:]
    for face in faces:
        for i in range(3):
            u, v = face[i], face[(i + 1) % 3]
            avg_curvature = (curvatures[u] + curvatures[v]) / 2

            # 最大曲率路径：曲率越大 → 权重越小
            weight = 1.0 / (avg_curvature + epsilon)

            graph[u].append((v, weight))
            graph[v].append((u, weight))
    return graph


# Dijkstra算法实现
def dijkstra(graph, start, end):
    dist = np.full(len(graph), np.inf)
    prev = np.full(len(graph), -1, dtype=int)
    dist[start] = 0.0

    queue = [(0.0, start)]
    while queue:
        current_dist, u = heapq.heappop(queue)
        if u == end:
            break
        if current_dist > dist[u]:
            continue
        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(queue, (new_dist, v))

    # 回溯路径
    path = []
    current = end
    while current != -1:
        path.append(current)
        current = prev[current]
    return path[::-1]

def has_discontinuity(points, tol=0.6):
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[i - 1]) > tol :  # 距离太远则认为断裂
            # print(points[i], points[i - 1])
            return True
    if np.linalg.norm(points[0] - points[ - 1]) > tol:
        return True
    return False

def find_path_points(in_points, base_mesh):
    all_path_points = np.zeros((0, 3))
    graph = get_graph(base_mesh)
    for i in range(len(in_points)):
        start_index = i
        if i + 1 == len(in_points):
            end_index = 0
        else:
            end_index = i + 1
        start_point = np.array(in_points[start_index])
        end_point = np.array(in_points[end_index])
        # 构建KDTree快速查找最近点
        points = base_mesh.points
        kdtree = KDTree(points)
        start_idx = kdtree.query(start_point)[1]
        end_idx = kdtree.query(end_point)[1]

        # 获取路径
        path_indices = dijkstra(graph, start_idx, end_idx)
        path_points = points[path_indices]
        all_path_points = np.vstack((all_path_points, path_points))
    return all_path_points


def pre_control_points(control_points, base_mesh):

    max_edges_length = base_mesh.edges_unique_length.max() * 1.5

    vertices = base_mesh.vertices  # 顶点坐标，形状为 (n_vertices, 3)
    faces = base_mesh.faces  # 面数据，形状为 (n_faces, 3)
    # 将面数据转换为 PyVista 格式：每个面开头添加顶点数量（3），然后展平
    faces_pv = np.hstack([3 * np.ones((faces.shape[0], 1)), faces]).astype(int).ravel()
    # 创建 PyVista 网格
    base_mesh = pv.PolyData(vertices, faces_pv)

    base_mesh['Mean_Curvature'] = base_mesh.curvature(curv_type='mean')

    in_points = get_evenly_spaced_points(control_points, 20)


    find_control_points = None

    all_path_points = find_path_points(in_points, base_mesh)



    in_points = get_evenly_spaced_points(all_path_points)

    for _ in range(10):

        all_path_points1 = find_path_points(in_points, base_mesh)

        n_points = 6
        is_break = has_discontinuity(all_path_points1, max_edges_length)
        if not is_break:
            find_control_points = all_path_points1
            in_points = get_evenly_spaced_points(find_control_points, n_points)
        elif find_control_points is None:
            in_points = get_evenly_spaced_points(all_path_points)
        elif find_control_points is not None:
            in_points = get_evenly_spaced_points(find_control_points)

    if find_control_points is not None:
        return find_control_points
    else:
        return control_points


# def Laplace_stretched_list(mesh, ancV_list, ref_normal, shiftAmount = 0.01, expand=10, fix_points_idx=None, adjust_base_idx=None):
#     lDeformer = mm.Laplacian(mesh)
#
#     freeVerts = mm.VertBitSet()
#     freeVerts.resize(mesh.topology.getValidVerts().size())
#     for ancV in ancV_list:
#         freeVerts.set(mm.VertId(ancV), True)
#
#     mm.expand(mesh.topology, freeVerts, expand)  # 扩展
#     # 初始化拉普拉斯变形器
#     lDeformer.init(freeVerts, mm.EdgeWeights.Cotan, mm.VertexMass.NeiArea)
#
#     for ancV in ancV_list:
#         ancV = mm.VertId(ancV)
#         direction_normal = mm.Vector3f(*[0, 0, -1])
#         normalized_shift = mesh.normal(ancV)
#         normalized = -smallest_angle_vector(ref_normal, np.array(list(normalized_shift)))
#         normalized = mm.Vector3f(*normalized)
#         # print(mesh.normal(ancV))
#         # 沿法线方向移动锚点 反向
#         lDeformer.fixVertex(ancV, mesh.points.vec[ancV.get()] + normalized * shiftAmount )
#
#     # 固定指定的点
#     if fix_points_idx is not None:
#         for v_index in fix_points_idx:
#             v_index = mm.VertId(v_index)
#             lDeformer.fixVertex(v_index, mesh.points.vec[v_index.get()])
#
#     if adjust_base_idx is not None:
#         for v_index in adjust_base_idx:
#             v_index = mm.VertId(v_index)
#             lDeformer.fixVertex(v_index, mesh.points.vec[v_index.get()])
#
#     # 执行变形计算
#     lDeformer.apply()
#     # 更新网格缓存
#     mesh.invalidateCaches()
#     return mesh
#
#
# def smallest_angle_vector(v_array: np.ndarray, v_target: np.ndarray):
#     """
#     v_array: shape (N, 3) 多个向量
#     v_target: shape (3,) 目标向量
#
#     返回与目标向量夹角最小的向量
#     """
#     # 归一化
#     v_array_norm = v_array / np.linalg.norm(v_array, axis=1, keepdims=True)
#     v_target_norm = v_target / np.linalg.norm(v_target)
#
#     # 计算点积
#     dots = np.dot(v_array_norm, v_target_norm)  # shape (N,)
#
#     # 夹角 = arccos(dot), 由于点积范围[-1,1]，安全裁剪
#     dots = np.clip(dots, -1.0, 1.0)
#     angles = np.arccos(dots)  # 角度单位为弧度
#
#     # 找最小夹角的索引
#     min_idx = np.argmin(angles)
#
#     return v_array[min_idx]
#
# def Laplace_stretched_list1(mesh, ancV_list, point_idx, bridge_center,
#                             ref_crown_normals, shiftAmount = 0.01, expand=10, fix_points_idx=None, adjust_base_idx=None):
#     cut_points_index = point_idx
#     # ref_crown_normal1, ref_crown_normal2 = ref_crown_normals[0], ref_crown_normals[1]
#
#     lDeformer = mm.Laplacian(mesh)
#     freeVerts = mm.VertBitSet()
#     freeVerts.resize(mesh.topology.getValidVerts().size())
#     for ancV in ancV_list:
#         freeVerts.set(mm.VertId(ancV), True)
#     mm.expand(mesh.topology, freeVerts, expand)  # 扩展
#     # 初始化拉普拉斯变形器
#     lDeformer.init(freeVerts, mm.EdgeWeights.Cotan, mm.VertexMass.NeiArea)
#
#     for i, ancV in enumerate(ancV_list):
#     # for i in range(0, len(ancV_list), 10):
#         ancV = mm.VertId(ancV_list[i])
#         normalized_shift = mesh.normal(ancV)
#         if ancV_list[i] in cut_points_index:
#             normalized = smallest_angle_vector(ref_crown_normals, np.array(list(normalized_shift)))
#         else:
#             mesh_point = np.array(list(mesh.points.vec[ancV.get()]))
#             normal_shift = -bridge_center + mesh_point
#             magnitude = np.linalg.norm(normal_shift)
#             normalized = normal_shift / magnitude
#             # normalized = smallest_angle_vector(ref_crown_normal1, normalized_shift)
#         normalized = mm.Vector3f(*normalized)
#         # 沿法线方向移动锚点 反向
#         lDeformer.fixVertex(ancV, mesh.points.vec[ancV.get()] + normalized * shiftAmount)
#
#     # 固定指定的点
#     if fix_points_idx is not None:
#         for v_index in fix_points_idx:
#             v_index = mm.VertId(v_index)
#             lDeformer.fixVertex(v_index, mesh.points.vec[v_index.get()])
#
#     if adjust_base_idx is not None:
#         for v_index in adjust_base_idx:
#             v_index = mm.VertId(v_index)
#             lDeformer.fixVertex(v_index, mesh.points.vec[v_index.get()])
#
#     # 执行变形计算
#     lDeformer.apply()
#     # 更新网格缓存
#     mesh.invalidateCaches()
#     return mesh


def Laplace_stretched_list(mesh, ancV_list, shiftAmount = 0.01, expand=10, is_upper=True, adjust_base_idx=None):
    lDeformer = mm.Laplacian(mesh)

    freeVerts = mm.VertBitSet()
    freeVerts.resize(mesh.topology.getValidVerts().size())
    for ancV in ancV_list:
        freeVerts.set(mm.VertId(ancV), True)

    mm.expand(mesh.topology, freeVerts, expand)  # 扩展
    # 初始化拉普拉斯变形器
    lDeformer.init(freeVerts, mm.EdgeWeights.Cotan, mm.VertexMass.NeiArea)

    if is_upper:
        normalized = mm.Vector3f(*[0, 0, 1])
    else:
        normalized = mm.Vector3f(*[0, 0, -1])

    for ancV in ancV_list:
        ancV = mm.VertId(ancV)

        # print(mesh.normal(ancV))
        lDeformer.fixVertex(ancV, mesh.points.vec[ancV.get()] + normalized * shiftAmount )

    if adjust_base_idx is not None:
        for v_index in adjust_base_idx:
            v_index = mm.VertId(v_index)
            lDeformer.fixVertex(v_index, mesh.points.vec[v_index.get()])

    # 执行变形计算
    lDeformer.apply()
    # 更新网格缓存
    mesh.invalidateCaches()
    return mesh


def Laplace_stretched_list1(mesh, ancV_list, bridge_center,
                            shiftAmount = 0.01, expend=10, adjust_base_idx=None):

    lDeformer = mm.Laplacian(mesh)
    freeVerts = mm.VertBitSet()
    freeVerts.resize(mesh.topology.getValidVerts().size())

    for ancV in ancV_list:
        freeVerts.set(mm.VertId(ancV), True)

    mm.expand(mesh.topology, freeVerts, expend)
    # 初始化拉普拉斯变形器
    lDeformer.init(freeVerts, mm.EdgeWeights.Cotan, mm.VertexMass.NeiArea)

    normals_ =  -bridge_center + mn.getNumpyVerts(mesh)[ancV_list]

    # for i, ancV in enumerate(ancV_list):
    for i in range(0, len(ancV_list), 10):
        ancV = mm.VertId(ancV_list[i])
        # normalized_shift = mesh.normal(ancV)

        # mesh_point = np.array(list(mesh.points.vec[ancV.get()]))
        # normal_shift = -bridge_center + mesh_point
        normal_shift = normals_[i]
        magnitude = np.linalg.norm(normal_shift)
        normalized = normal_shift / magnitude
            # normalized = smallest_angle_vector(ref_crown_normal1, normalized_shift)
        normalized = mm.Vector3f(*normalized)
        # 沿法线方向移动锚点 反向
        lDeformer.fixVertex(ancV, mesh.points.vec[ancV.get()] + normalized * shiftAmount)

    if adjust_base_idx is not None:
        for v_index in adjust_base_idx:
            v_index = mm.VertId(v_index)
            lDeformer.fixVertex(v_index, mesh.points.vec[v_index.get()])

    # 执行变形计算
    lDeformer.apply()
    # 更新网格缓存
    mesh.invalidateCaches()
    return mesh


def filtered_mesh(base_mesh):
    unique_vertices, inverse = grouping.unique_rows(base_mesh.vertices)
    new_vertices = base_mesh.vertices[unique_vertices]
    new_faces = inverse[base_mesh.faces].reshape((-1, 3))
    valid_faces_mask = (new_faces[:, 0] != new_faces[:, 1]) & \
                       (new_faces[:, 1] != new_faces[:, 2]) & \
                       (new_faces[:, 2] != new_faces[:, 0])
    # 应用过滤
    filtered_faces = new_faces[valid_faces_mask]
    base_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=filtered_faces,
        process=False  # 不再需要自动处理
    )
    return base_mesh

def mesh_cut(vertices, faces, center, r, keep_inside=True, is_upper=True):
    """
    切割网格：保留圆柱内部或外部的部分

    参数:
        vertices (np.ndarray): 网格顶点数组，形状为 (N, 3)
        faces (np.ndarray): 网格面数组，形状为 (M, 3)，每个面由三个顶点索引组成
        center (tuple): 圆柱中心坐标 (cx, cy)，忽略z轴
        r (float): 圆柱半径
        keep_inside (bool): True保留圆柱内部，False保留外部

    返回:
        new_vertices (np.ndarray): 切割后的顶点数组
        new_faces (np.ndarray): 切割后的面数组
        vertex_mask (np.ndarray): 原始顶点的掩码（满足圆柱条件的布尔数组）
    """
    # 计算顶点到`圆柱轴心（XY平面）的距离平方

    dist_sq = (vertices[:, 0] - center[0]) ** 2 + (vertices[:, 1] - center[1]) ** 2
    max_z = max(vertices[:, 2])
    min_z = min(vertices[:, 2])
    # if is_upper:
    #     max_z = min_z + 20
    # else:
    #     min_z = max_z - 20
    r_sq = r ** 2

    # 根据keep_inside生成顶点掩码
    if keep_inside:
        vertex_mask = dist_sq <= r_sq
    else:
        vertex_mask = dist_sq > r_sq

    z_mask = (vertices[:, 2] >= min_z) & (vertices[:, 2] <= max_z)
    # 组合掩码 (XY平面条件 AND Z轴条件)
    vertex_mask = vertex_mask & z_mask

    # 标记完全保留的面（所有顶点满足条件）
    face_mask = np.all(vertex_mask[faces], axis=1) if keep_inside else \
        np.all(~vertex_mask[faces], axis=1)
    valid_faces = faces[face_mask]

    # 获取保留面中使用的顶点索引
    used_vertex_indices = np.unique(valid_faces.flatten())

    # 创建新顶点数组
    new_vertices = vertices[used_vertex_indices]

    # 建立旧索引到新索引的映射
    index_map = np.full(vertices.shape[0], -1, dtype=int)
    index_map[used_vertex_indices] = np.arange(len(used_vertex_indices))

    # 更新面中的顶点索引
    new_faces = index_map[valid_faces]

    return new_vertices, new_faces, vertex_mask

def corp_trimesh(bilateral_teeth, is_upper=True, z_range=1.5):
    vertices = bilateral_teeth.vertices
    if is_upper:
        centroid_z = vertices[:, 2].min() + z_range
        filtered_vertex_indices = np.where(vertices[:, 2] < centroid_z)[0]
    else:
        centroid_z = vertices[:, 2].max() - z_range
        filtered_vertex_indices = np.where(vertices[:, 2] > centroid_z)[0]
    # 创建一个布尔掩码，表示哪些顶点被保留
    vertex_mask = np.zeros(len(vertices), dtype=bool)
    vertex_mask[filtered_vertex_indices] = True
    selected_vertices = vertices[vertex_mask]
    # 获取选中的顶点在原网格中的索引
    selected_vertex_indices = np.where(vertex_mask)[0]
    # 创建顶点到新索引的映射
    vertex_to_new_index = {old_index: new_index for new_index, old_index in
                           enumerate(selected_vertex_indices)}  # 筛选面，只保留由选中顶点组成的面
    new_faces = []
    for face in bilateral_teeth.faces:
        # 检查面中的所有顶点是否都被选中
        if all(vertex in selected_vertex_indices for vertex in face):
            # 更新面的顶点索引
            new_face = [vertex_to_new_index[vertex] for vertex in face]
            new_faces.append(new_face)
    new_faces = np.array(new_faces)
    # trimesh.Trimesh(vertices=selected_vertices, faces=new_faces).export('corp.stl')
    crop_mesh = mn.meshFromFacesVerts(verts=selected_vertices, faces=new_faces)
    return crop_mesh

def get_sdf_point(meshA, meshB):
    trimesh_verticesA = mn.getNumpyVerts(meshA)  # 转为trimesh格式
    trimesh_facesA = mn.getNumpyFaces(meshA.topology)
    trimesh_meshA = trimesh.Trimesh(vertices=trimesh_verticesA.copy(), faces=trimesh_facesA.copy(), process=False )

    trimesh_verticesB = mn.getNumpyVerts(meshB)  # 转为trimesh格式
    trimesh_facesB = mn.getNumpyFaces(meshB.topology)
    trimesh_meshB = trimesh.Trimesh(vertices=trimesh_verticesB.copy(), faces=trimesh_facesB.copy(), process=False )

    # normal = trimesh_meshA.centroid - trimesh_meshB.centroid
    # normal = np.array([0, 0, 1]).astype(float)
    # normal /= np.linalg.norm(normal)
    # origins = trimesh_meshA.vertices  # (N,3)
    # directions = np.tile(normal, (len(origins), 1))  # (N,3)
    #
    origins = trimesh_meshA.vertices  # (N,3)
    directions = trimesh_meshA.vertices - trimesh_meshB.centroid

    rmi = trimesh.ray.ray_pyembree.RayMeshIntersector(trimesh_meshB)
    # 批量求交
    locations, index_ray, index_tri = rmi.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False
    )

    # index_ray 表示 locations 对应的射线点索引
    sdf_points = origins[index_ray]
    sd_dist = np.linalg.norm(locations - origins[index_ray], axis=1)

    # idx_tree = cKDTree(trimesh_verticesA)
    # _, inside_indices = idx_tree.query(sdf_points)
    inside_indices = index_ray.tolist()
    # print('sd_dist_max', sd_dist.max())
    return sdf_points, list(inside_indices)
    # return sdf_points, list(inside_indices), np.array(sd_dist)


def detach_mesh(meshA, meshB, is_sdf=True):
    collidingFacePairs = mm.findCollidingTriangles(meshA,
                                                        meshB)  # find each pair of colliding faces
    colliding_points_index = []
    for fp in collidingFacePairs:
        for v_index in meshA.topology.getTriVerts(fp.aFace):
            colliding_points_index.append(v_index.get())
    if is_sdf:
        sdf_points, sdf_idx = get_sdf_point(meshA, meshB)
        colliding_points_index = list(set(colliding_points_index + sdf_idx))
        # pc = trimesh.PointCloud(sdf_points)
        # pc.export('sdf_points.ply')

    colliding_points_index = list(set(colliding_points_index))
    isColliding = not collidingFacePairs.empty()
    return isColliding, colliding_points_index

def Laplace_stretched_batch(mesh, move_vertices_idx, ref_normals, shiftAmount=0.6, expand=50, fix_points_idx=None, adjust_base_idx=None):
    """
    mesh: mrmeshpy 网格
    move_vertices_idx: 待移动顶点列表 (list of int)
    ref_normals: 每个顶点对应的移动方向 (list of np.array)
    shiftAmount: 移动距离
    expand: 拉普拉斯展开步数
    fix_points_idx: 需要固定的顶点索引
    adjust_base_idx: 需要固定的其他顶点索引
    """
    lDeformer = mm.Laplacian(mesh)

    # 1. 初始化自由顶点集合
    freeVerts = mm.VertBitSet()
    freeVerts.resize(mesh.topology.getValidVerts().size())

    # 将待移动顶点标记为自由顶点
    for v_idx in move_vertices_idx:
        freeVerts.set(mm.VertId(v_idx), True)

    # 2. 扩展影响区域
    mm.expand(mesh.topology, freeVerts, expand)

    # 3. 初始化拉普拉斯变形器
    lDeformer.init(freeVerts, mm.EdgeWeights.Cotan, mm.VertexMass.NeiArea)

    # 4. 固定待移动顶点的新位置
    for v_idx, normal in zip(move_vertices_idx, ref_normals):
        v_id = mm.VertId(v_idx)
        target_pos = mesh.points.vec[v_id.get()] + mm.Vector3f(*normal) * shiftAmount
        lDeformer.fixVertex(v_id, target_pos)

    # 5. 固定指定点
    if fix_points_idx is not None:
        for v_idx in fix_points_idx:
            v_id = mm.VertId(v_idx)
            lDeformer.fixVertex(v_id, mesh.points.vec[v_id.get()])

    if adjust_base_idx is not None:
        for v_idx in adjust_base_idx:
            v_id = mm.VertId(v_idx)
            lDeformer.fixVertex(v_id, mesh.points.vec[v_id.get()])

    # 6. 执行变形
    lDeformer.apply()
    mesh.invalidateCaches()
    return mesh



def constriction_mesh(expend_base_mm, cut_base_mm, base_expend):
    cut_base_mm_points = mn.getNumpyVerts(cut_base_mm)
    trimesh_vertices = mn.getNumpyVerts(expend_base_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(expend_base_mm.topology)
    expend_base_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy())
    expend_base_mm_points = trimesh_vertices
    # 1. 找到边界
    _, boundary_index = find_boundary_edges(expend_base_mesh)
    boundary_vertices = expend_base_mm_points[boundary_index]

    # 2. 计算收缩方向（指向质心）
    mesh_center = np.mean(boundary_vertices, axis=0)
    boundary_normals = []
    for p in boundary_vertices:
        v_pos = p
        dir_vec = mesh_center - v_pos
        dir_vec /= np.linalg.norm(dir_vec)
        boundary_normals.append(dir_vec)

    fix_tree = cKDTree(expend_base_mm_points)
    _, fix_cut_points_idx = fix_tree.query(cut_base_mm_points)
    # 3. 批量收缩边界顶点
    # mesh_mm = expend_base_mm
    expend_base_mm = Laplace_stretched_batch(
        expend_base_mm,
        move_vertices_idx=boundary_index,
        ref_normals=boundary_normals,
        shiftAmount=base_expend,
        expand=500,
        fix_points_idx=fix_cut_points_idx
    )
    return expend_base_mm


def adjust_mesh_bridge(brigde_mesh, base_mesh, sphere_mesh, top_fix_idxs, is_upper):
    # 面片滤除
    # crown_tooth = filtered_mesh(crown_tooth)
    base_mesh = filtered_mesh(base_mesh)
    # brigde_mesh = filtered_mesh(brigde_mesh)

    crown_center = np.array(base_mesh.vertices.mean(axis=0))
    # if is_upper:
    #     crown_center[2] = base_mesh.vertices[:, 2].max()
    # else:
    #     crown_center[2] = base_mesh.vertices[:, 2].min()

    # _, boundary_indices = find_boundary_edges(crown_tooth)
    # boundary_points = crown_tooth.vertices[boundary_indices]
    # control_points = pre_control_points(boundary_points, crown_tooth)  #4、5牙参数设置为6 67参数设置为 8

    # point_cloud = trimesh.points.PointCloud(control_points)
    # point_cloud.export('control.ply')

    crown_mm = mn.meshFromFacesVerts(verts=brigde_mesh.vertices, faces=brigde_mesh.faces)
    opposite_mm = mn.meshFromFacesVerts(verts=sphere_mesh.vertices, faces=sphere_mesh.faces)
    base_mm = mn.meshFromFacesVerts(verts=base_mesh.vertices, faces=base_mesh.faces)

    crown_points = np.array([list(v) for v in np.array(crown_mm.points.vec)])
    opposite_mm_points = np.array([list(v) for v in np.array(opposite_mm.points.vec)])
    base_mm_points = np.array([list(v) for v in np.array(base_mm.points.vec)])

    base_fix_tree = cKDTree(crown_points)
    _, base_fix_idx = base_fix_tree.query(base_mm_points)

    # try:
    #     points_vec = mn.fromNumpyArray(control_points)
    #     faceidx = mm.cutMeshByContour(crown_mm, points_vec)
    #
    #     cut_points_index = []
    #     for fp in faceidx:
    #         for v_index in crown_mm.topology.getTriVerts(fp):
    #             cut_points_index.append(v_index.get())
    #     cut_points_index = set(cut_points_index)
    #     cut_points_index = list(cut_points_index)
    #
    #     if len(cut_points_index) == len(crown_points):
    #         print('错误分割')
    #         cut_points_index=[]
    # except:
    #     print('自交')
    #     cut_points_index = []

    base_kdtree = cKDTree(base_mm_points)
    dists_base_opposite, _ = base_kdtree.query(opposite_mm_points)

    min_dist = min(dists_base_opposite)
    print(min_dist)
    if min_dist > 0.6:
        expend_base_mm = 0.6
    elif min_dist > 0.2:
        expend_base_mm = min_dist - 0.1
    else:
        print("基台高度过高")
        return brigde_mesh
    # expend_base_mm=0.05

    # # 基台外扩0.6与冠面做检测
    # params = mm.GeneralOffsetParameters()
    # params.voxelSize = base_mm.computeBoundingBox().diagonal() * 5e-3  # 设置体素大小（控制精度）
    # params.signDetectionMode = mm.SignDetectionMode.Unsigned
    # mp = mm.MeshPart(base_mm)
    # base_mm = mm.offsetOneDirection(mp, expend_base_mm, params)
    # # 删除小面片
    # trimesh_vertices = mn.getNumpyVerts(base_mm)  # 转为trimesh格式
    # trimesh_faces = mn.getNumpyFaces(base_mm.topology)
    # expend_base_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy())
    # components = expend_base_mesh.split(only_watertight=False)  # 关闭 only_watertight 以保留非封闭的面片
    # # 按面片数量排序，取最大的一个
    # largest_component = max(components, key=lambda m: len(m.faces))
    # cut_base_mm = corp_trimesh(largest_component, is_upper, z_range=1.5) #裁剪基台顶部用以固定
    # # mm.saveMesh(cut_base_mm, 'cut_base.stl')
    # base_mm = mn.meshFromFacesVerts(verts=largest_component.vertices, faces=largest_component.faces)
    # # mm.saveMesh(base_mm, rf"base_expend.stl")
    #
    # base_mm = constriction_mesh(base_mm, cut_base_mm)

    # mm.saveMesh(base_mm, rf"base_expend1.stl")

    # ref_crown_normal2 = np.array([[0, 0, 1], [0, 0, -1]])

    isColliding_base, colliding_points_index_base = detach_mesh(crown_mm, base_mm)
    # 对颌牙接触检测
    isColliding, colliding_points_index = detach_mesh(crown_mm, opposite_mm)
    expend_base = int(len(colliding_points_index_base) / 50) + 1
    expend = int(len(colliding_points_index) / 50) + 1
    # factor = expend / (max(len(colliding_points_index), len(colliding_points_index_base)) + 1)

    pc = trimesh.PointCloud(mn.getNumpyVerts(crown_mm)[top_fix_idxs])
    pc.export(r'test.ply')

    count = 0
    crown_mm_copy = None
    while True:
        # 对颌牙接触检测
        isColliding, colliding_points_index = detach_mesh(crown_mm, opposite_mm)
        colliding_points_index = list(set(colliding_points_index) - set(base_fix_idx))
        if not isColliding and not isColliding_base:
            break

        if 0 < len(colliding_points_index) < 50 and crown_mm_copy is None and count>1:
            verts = mn.getNumpyVerts(crown_mm)
            faces = mn.getNumpyFaces(crown_mm.topology)
            crown_mm_copy = mn.meshFromFacesVerts(verts=verts, faces=faces)

        # if int(factor) != 20:
        #     expend = int(factor * len(colliding_points_index))

        count_opp=0
        colliding_opp=len(colliding_points_index)
        while isColliding:

            diff = np.setdiff1d(top_fix_idxs, colliding_points_index)
            # print(diff)
            base_fix_idx = list(base_fix_idx) + list(diff)

            crown_mm = Laplace_stretched_list(crown_mm, colliding_points_index,
                                              expand=expend,
                                              adjust_base_idx=base_fix_idx,
                                              is_upper=is_upper
                                              )

            # mm.saveMesh(crown_mm, rf"crown_mm.stl")

            isColliding, colliding_points_index = detach_mesh(crown_mm, opposite_mm)
            colliding_points_index = list(set(colliding_points_index) - set(base_fix_idx))

            # expend = int(factor * len(colliding_points_index))
            # if len(colliding_points_index) <= 50:
            #     expend += 5 * int((len(colliding_points_index) + 1) / 50) + 2

            expend = int(len(colliding_points_index) / 50) + 1
            print('对颌牙', len(colliding_points_index), expend)

            if colliding_opp > len(colliding_points_index):
                colliding_opp = len(colliding_points_index)
                count_opp = 0  # 有改进 → 归零
            else:
                count_opp += 1  # 没改进 → 累计
                if count_opp >= 100:
                    print(f"参数在 {100} 次迭代未减小，提前停止")
                    return bridge_mesh

        # 0.6基台检测
        _, colliding_points_index_base = detach_mesh(crown_mm, base_mm)
        colliding_points_index_base = list(set(colliding_points_index_base) - set(base_fix_idx))
        if len(colliding_points_index_base) > 0:
            isColliding_base = True
        else:
            isColliding_base = False
        # if int(factor) != 20:
        #     expend_base = int(factor * len(colliding_points_index_base))
        colliding_base = len(colliding_points_index_base)
        count_base = 0
        while isColliding_base:
            crown_mm = Laplace_stretched_list1(crown_mm, colliding_points_index_base,
                                               crown_center,
                                               expend=expend_base,
                                               adjust_base_idx=base_fix_idx
                                               )

            _, colliding_points_index_base = detach_mesh(crown_mm, base_mm, is_sdf=False)
            colliding_points_index_base = list(set(colliding_points_index_base) - set(base_fix_idx))
            if len(colliding_points_index_base) > 0:
                isColliding_base = True
            else:
                isColliding_base = False

            # expend_base = int(factor * len(colliding_points_index_base))
            # if len(colliding_points_index_base) <= 50:
            #     expend_base += 5 * int((len(colliding_points_index_base) + 1) / 50) + 2

            expend_base = int(len(colliding_points_index_base) / 50) + 1
            print('基台', len(colliding_points_index_base), expend_base)

            if colliding_base > len(colliding_points_index_base):
                colliding_base = len(colliding_points_index_base)
                count_base = 0  # 有改进 → 归零
            else:
                count_base += 1  # 没改进 → 累计
                if count_base >= 100:
                    print(f"参数在 {100} 次迭代未减小，提前停止")
                    return brigde_mesh


        # isColliding = not mm.findCollidingTriangles(crown_mm, opposite_mm).empty()
        count+=1
        if count > 20:
            print('对颌牙与基台间距可能不足')
            if crown_mm_copy is not None:
                crown_mm = crown_mm_copy
            break

    # mm.saveMesh(crown_mm, rf"adjust_bridge_mesh.stl")
    # settings = mm.DenoiseViaNormalsSettings()
    # settings.fastIndicatorComputation = True  # 使用快速近似计算
    # settings.beta = 0.01  # 0.001 → 保留锐利边界 0.01 → 保留中等锐度边 0.1 → 更平滑（特征保留少
    # settings.gamma = 5  # 平滑强度：0 → 不平滑 1 → 中等平滑 >1 → 更强的平滑
    # settings.normalIters = 10  # 法线迭代次数
    # settings.pointIters = 10  # 顶点迭代次数
    # settings.guideWeight = 0.1  # 保持接近原始位置
    # settings.limitNearInitial = True  # 限制点的移动
    # settings.maxInitialDist = 0.5  # 最大移动距离
    # # 执行去噪平滑
    # mm.meshDenoiseViaNormals(crown_mm, settings)

    trimesh_vertices = mn.getNumpyVerts(crown_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(crown_mm.topology)
    crown_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy())

    return crown_mesh



def adjust_mesh_compare_base(brigde_mesh, base_mesh, base_expend=0.6, is_upper=False):
    # 面片滤除
    base_mesh = filtered_mesh(base_mesh)

    crown_center = np.array(base_mesh.vertices.mean(axis=0))
    crown_center[2] = base_mesh.vertices[:, 2].min()

    # print(crown_center, brigde_mesh.centroid, base_mesh.centroid)

    crown_mm = mn.meshFromFacesVerts(verts=brigde_mesh.vertices, faces=brigde_mesh.faces)
    base_mm = mn.meshFromFacesVerts(verts=base_mesh.vertices, faces=base_mesh.faces)

    # crown_points = np.array([list(v) for v in np.array(crown_mm.points.vec)])
    # base_mm_points = np.array([list(v) for v in np.array(base_mm.points.vec)])
    #
    # base_fix_tree = cKDTree(crown_points)
    # _, base_fix_idx = base_fix_tree.query(base_mm_points)

    # 基台外扩0.6与冠面做检测
    params = mm.GeneralOffsetParameters()
    params.voxelSize = base_mm.computeBoundingBox().diagonal() * 5e-3  # 设置体素大小（控制精度）
    params.signDetectionMode = mm.SignDetectionMode.Unsigned
    mp = mm.MeshPart(base_mm)
    base_mm = mm.offsetOneDirection(mp, base_expend, params)
    # 删除小面片
    trimesh_vertices = mn.getNumpyVerts(base_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(base_mm.topology)
    expend_base_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy())
    components = expend_base_mesh.split(only_watertight=False)  # 关闭 only_watertight 以保留非封闭的面片
    # 按面片数量排序，取最大的一个
    largest_component = max(components, key=lambda m: len(m.faces))
    cut_base_mm = corp_trimesh(largest_component, is_upper, z_range=1.5) #裁剪基台顶部用以固定
    # mm.saveMesh(cut_base_mm, 'cut_base.stl')
    base_mm = mn.meshFromFacesVerts(verts=largest_component.vertices, faces=largest_component.faces)
    # mm.saveMesh(base_mm, rf"base_expend.stl")

    base_mm = constriction_mesh(base_mm, cut_base_mm, base_expend)
    mm.saveMesh(base_mm, rf"base_expend.stl")

    isColliding_base, colliding_points_index_base = detach_mesh(crown_mm, base_mm)
    expend_base = int(len(colliding_points_index_base) / 100) + 1

    # 0.6基台检测
    # _, colliding_points_index_base = detach_mesh(crown_mm, base_mm)
    # colliding_points_index_base = list(set(colliding_points_index_base) - set(base_fix_idx))
    # if len(colliding_points_index_base) > 0:
    #     isColliding_base = True
    # else:
    #     isColliding_base = False
    # if int(factor) != 20:
    #     expend_base = int(factor * len(colliding_points_index_base))
    colliding_base = len(colliding_points_index_base)
    count_base = 0
    while isColliding_base:
        crown_mm = Laplace_stretched_list1(crown_mm, colliding_points_index_base,
                                           crown_center,
                                           expend=expend_base,
                                           shiftAmount=0.1,
                                           adjust_base_idx=None
                                           )

        isColliding_base, colliding_points_index_base = detach_mesh(crown_mm, base_mm, is_sdf=True)
        # colliding_points_index_base = list(set(colliding_points_index_base) - set(base_fix_idx))
        # if len(colliding_points_index_base) > 0:
        #     isColliding_base = True
        # else:
        #     isColliding_base = False

        # expend_base = int(factor * len(colliding_points_index_base))
        # if len(colliding_points_index_base) <= 50:
        #     expend_base += 5 * int((len(colliding_points_index_base) + 1) / 50) + 2
        # mm.saveMesh(crown_mm, 'crown_mesh.stl')
        # pc = trimesh.PointCloud(mn.getNumpyVerts(crown_mm)[colliding_points_index_base])
        # pc.export('crown_mesh.ply')

        expend_base = int(len(colliding_points_index_base) / 100) + 1
        if expend_base <= 3:
            expend_base = 3
        print('基台', len(colliding_points_index_base), expend_base)

        if colliding_base > len(colliding_points_index_base):
            colliding_base = len(colliding_points_index_base)
            count_base = 0  # 有改进 → 归零
        else:
            count_base += 1  # 没改进 → 累计
            if count_base >= 100:
                print(f"参数在 {100} 次迭代未减小，提前停止")
                return brigde_mesh

    trimesh_vertices = mn.getNumpyVerts(crown_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(crown_mm.topology)
    crown_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy())

    return crown_mesh


if __name__ == "__main__":

    # crown_path = r"Z:\10-算法组\gbz\测试数据09191335\24_3\18\adjust_top_crown_top_mesh_3.stl"
    # bridge_path = r"Z:\10-算法组\gbz\测试数据09191335\24_3\18\bridge_mesh_0.stl"
    # adjust_whole_path = r"Z:\10-算法组\gbz\测试数据09191335\24_3\18\health_tooth.stl"
    # base_path = r"Z:\10-算法组\gbz\测试数据09191335\24_3\18\groove_mesh.stl"

    crown_path = r"Z:\10-算法组\gbz\测试数据09230851\14_8\41\adjust_top_2.stl"
    bridge_path = r"Z:\10-算法组\gbz\测试数据09230851\14_8\41\bridge_mesh_0.stl"
    adjust_whole_path = r"Z:\10-算法组\gbz\测试数据09230851\14_8\41\health_tooth.stl"
    base_path = r"Z:\10-算法组\gbz\测试数据09230851\14_8\41\groove_mesh.stl"

    is_upper=True

    crown_tooth = trimesh.load(crown_path)
    bridge_mesh = trimesh.load(bridge_path)
    base_mesh = trimesh.load(base_path)
    whole_mesh = trimesh.load(adjust_whole_path)

    crown_center = crown_tooth.centroid
    new_vertices, new_faces, _ = mesh_cut(whole_mesh.vertices,
                                          whole_mesh.faces, crown_center, r=10)
    sphere_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)


    crown_mesh = adjust_mesh_bridge(crown_tooth, bridge_mesh, base_mesh, sphere_mesh, is_upper)
    # crown_mesh = adjust_bridge_mesh(crown_tooth, bridge_mesh, base_mesh, sphere_mesh, False)

    crown_mesh.export('adjust_bridge_mesh.stl')








