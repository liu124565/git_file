import trimesh
import numpy as np
from collections import defaultdict, deque
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from sympy.codegen.cnodes import union

from edge_utils import  perform_boolean_operation
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn

from scipy import interpolate


def find_boundary_loops(mesh):
    """
    找到 mesh 中所有边界环（闭合边界）
    返回:
        boundary_loops_vertices_sorted: 按长度从大到小排序的环，每个环为顶点索引序列
    """
    # === Step 1. 统计每条边的出现次数 ===
    edge_counts = defaultdict(int)
    edge_to_vertices = {}

    for face in mesh.faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge_key = tuple(sorted([v1, v2]))
            edge_counts[edge_key] += 1
            if edge_key not in edge_to_vertices:
                edge_to_vertices[edge_key] = [v1, v2]

    # === Step 2. 取出现 1 次的边（即边界边） ===
    boundary_edges = [edge_to_vertices[e] for e, c in edge_counts.items() if c == 1]

    # === Step 3. 构建邻接表（无重复）===
    boundary_edge_adj = defaultdict(list)
    for v1, v2 in boundary_edges:
        boundary_edge_adj[v1].append([v1, v2])
        boundary_edge_adj[v2].append([v2, v1])

    # === Step 4. 追踪所有闭合环 ===
    visited_edges = set()
    boundary_loops = []

    for edge in boundary_edges:
        ekey = tuple(edge)
        if ekey in visited_edges:
            continue

        loop = []
        start = edge[0]
        current = start

        while True:
            next_edge = None
            for e in boundary_edge_adj[current]:
                et = tuple(e)
                if et not in visited_edges:
                    next_edge = e
                    break

            if next_edge is None:
                break

            visited_edges.add(tuple(next_edge))
            loop.append(next_edge)

            # 下一个顶点
            current = next_edge[1]

            # 如果回到起点且环长度>1，则停止
            if current == start and len(loop) > 1:
                break

        if loop:
            boundary_loops.append(loop)

    # === Step 5. 将环转换为【顶点序列】 ===
    loops_vertices = []
    for loop in boundary_loops:
        vertex_ring = [loop[0][0]]
        for e in loop:
            vertex_ring.append(e[1])
        loops_vertices.append(vertex_ring)

    # === Step 6. 按环长度由大到小排序 ===
    loops_vertices_sorted = sorted(loops_vertices, key=lambda x: len(x), reverse=True)

    return loops_vertices_sorted



def find_boundary_edges(mesh):
    """
    找到mesh的边界边
    返回边界边的顶点索引对和对应的边界顶点集合
    """
    edge_counts = defaultdict(int)
    edge_to_vertices = {}

    # 统计每条边出现的次数，不对顶点索引排序，保持原始方向
    for face in mesh.faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            # 使用排序后的元组作为字典键，但保存原始顶点顺序
            edge_key = tuple(sorted([v1, v2]))
            edge_counts[edge_key] += 1
            # 保存边的原始方向
            if edge_key not in edge_to_vertices:
                edge_to_vertices[edge_key] = [v1, v2]

    # 收集所有边界边，使用原始顶点顺序
    boundary_edges = [edge_to_vertices[edge_key] for edge_key, count in edge_counts.items() if count == 1]

    # 构建边界边的邻接表
    boundary_edge_adj = defaultdict(list)
    for edge in boundary_edges:
        v1, v2 = edge
        boundary_edge_adj[v1].append(edge)
        boundary_edge_adj[v2].append(edge)

        # 构建边界边的邻接表（不变）
        boundary_edge_adj = defaultdict(list)
        for edge in boundary_edges:
            v1, v2 = edge
            boundary_edge_adj[v1].append(edge)
            boundary_edge_adj[v2].append(edge)

        # === 修改开始 ===
        visited_edges = set()  # 跟踪已访问的边
        boundary_loops = []  # 存储所有边界环
        # sorted_boundary_indices_list = []  # 存储所有边界顶点环
        max_boundary_indices = []
        # 遍历所有边界边
        for edge in boundary_edges:
            # 跳过已访问的边
            if tuple(edge) in visited_edges:
                continue

            current_loop = []  # 当前环的边
            start_vertex = edge[0]
            current_vertex = start_vertex

            # 遍历当前环
            while True:
                next_edge = None
                # 查找未访问的相邻边
                for e in boundary_edge_adj[current_vertex]:
                    e_tuple = tuple(e)
                    if e_tuple not in visited_edges:
                        next_edge = e
                        break

                if next_edge is None:
                    break

                # 标记为已访问
                visited_edges.add(tuple(next_edge))
                current_loop.append(next_edge)

                # 移动到下一个顶点
                v1, v2 = next_edge
                current_vertex = v2 if v1 == current_vertex else v1

            # 保存当前环
            if current_loop:
                boundary_loops.append(current_loop)

                # 提取当前环的顶点
                ring_vertices = []
                for e in current_loop:
                    v1, v2 = e
                    if v1 not in ring_vertices:
                        ring_vertices.append(v1)
                    if v2 not in ring_vertices:
                        ring_vertices.append(v2)
                # sorted_boundary_indices_list.append(ring_vertices)
                if len(ring_vertices) > len(max_boundary_indices):
                    max_boundary_indices = ring_vertices
    # return boundary_loops, sorted_boundary_indices_list
    return boundary_loops, max_boundary_indices


# def map_submesh_to_mesh(submesh: trimesh.Trimesh, mesh: trimesh.Trimesh):
#     """
#     使用KDTree将子mesh顶点映射到原mesh顶点
#     """
#     tree = cKDTree(mesh.vertices)
#     distances, indices = tree.query(submesh.vertices)
#     # indices[i] 表示 submesh 顶点 i 在原 mesh 的索引
#     return indices
#
# def fill_hole_on_mesh(mesh: trimesh.Trimesh, submesh: trimesh.Trimesh):
#     """
#     使用子mesh边界在原mesh上进行区域生长补洞
#     """
#     _, upper_boundary = find_boundary_edges(submesh)
#     # upper_boundary_points = mesh1.vertices[upper_boundary]
#
#     # 子mesh边界顶点
#     boundary_vertices_submesh = set(np.unique(upper_boundary))
#     interior_vertices_submesh = set(range(len(submesh.vertices))) - boundary_vertices_submesh
#
#     if not interior_vertices_submesh:
#         raise ValueError("子mesh内部没有非边界顶点可做seed")
#
#     # 子mesh顶点映射到原mesh
#     submesh_to_mesh_map = map_submesh_to_mesh(submesh, mesh)
#
#     # 映射到原mesh索引
#     boundary_vertices_in_mesh = set(submesh_to_mesh_map[list(boundary_vertices_submesh)])
#     interior_vertices_in_mesh = [submesh_to_mesh_map[v] for v in interior_vertices_submesh]
#
#     # 选第一个内部顶点作为seed
#     seed_vertex = interior_vertices_in_mesh[0]
#
#     # 区域生长
#     visited = set()
#     queue = deque([seed_vertex])
#     while queue:
#         v = queue.popleft()
#         if v in visited or v in boundary_vertices_in_mesh:
#             continue
#         visited.add(v)
#         neighbors = mesh.vertex_neighbors[v]
#         for n in neighbors:
#             if n not in visited:
#                 queue.append(n)
#
#     visited_with_boundary = visited.union(boundary_vertices_in_mesh)
#     return visited_with_boundary

def find_vertex_layers(mesh, boundary_vertices):
    """
    从边界顶点开始，标记每个顶点所在的层
    """
    vertex_layers = {v: -1 for v in range(len(mesh.vertices))}
    current_layer = 0
    current_vertices = set(boundary_vertices)

    # 为边界顶点标记层0
    for v in current_vertices:
        vertex_layers[v] = 0

    # 通过邻接关系逐层标记
    while current_vertices:
        next_vertices = set()
        for vertex in current_vertices:
            # 获取与当前顶点相连的所有顶点
            connected = set()
            for face in mesh.vertex_faces[vertex]:
                if face == -1:
                    continue
                connected.update(mesh.faces[face])

            # 添加未标记的相邻顶点到下一层
            for connected_vertex in connected:
                if vertex_layers[connected_vertex] == -1:
                    next_vertices.add(connected_vertex)
                    vertex_layers[connected_vertex] = current_layer + 1

        current_vertices = next_vertices
        current_layer += 1

    return vertex_layers

def get_invisible_idx(mesh, best_normal):
    vertices = mesh.vertices
    ray_directions = np.tile(best_normal, (len(vertices), 1))  # shape (N,3)
    # origin 就是顶点本身
    ray_origins = vertices
    index_tri, index_ray, locations = mesh.ray.intersects_id(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=True,
        return_locations=True
    )

    # 计算距离
    # locations 对应 index_ray
    hit_origins = ray_origins[index_ray]  # 对应击中的射线原点
    distances = np.linalg.norm(locations - hit_origins, axis=1)

    # 筛除距离接近0的
    eps = 1e-6  # 可根据需要调整
    valid_mask = distances > eps

    # 过滤后的结果
    index_ray_filtered = index_ray[valid_mask]
    return index_ray_filtered



def connected_components_on_vertices(mesh, vertex_indices):
    """
    输入：mesh + 部分点索引
    输出：根据 mesh 拓扑得到的连通点分组 list[list[int]]
    """
    vertex_indices = set(vertex_indices)

    # 建立顶点邻接（只连接在同一条边上的顶点）
    adjacency = defaultdict(list)
    for f in mesh.faces:
        for i in range(3):
            a = f[i]
            b = f[(i+1) % 3]
            if a in vertex_indices and b in vertex_indices:
                adjacency[a].append(b)
                adjacency[b].append(a)

    # BFS 分组
    visited = set()
    components = []

    for v in vertex_indices:
        if v in visited:
            continue

        q = deque([v])
        group = []
        visited.add(v)

        while q:
            cur = q.popleft()
            group.append(cur)

            for nxt in adjacency[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)

        components.append(group)

    # 按组大小由大到小排序
    components.sort(key=len, reverse=True)
    return components


def curve_to_mesh(xy_points, z_top=1.0, z_bottom=0.0, closed=False):
    """
    将二维曲线点 (x, y) 扩展成带厚度的三维 mesh

    Parameters
    ----------
    xy_points : ndarray, shape (n,2)
        二维曲线点
    z_top : float
        顶层 z 值
    z_bottom : float
        底层 z 值
    closed : bool
        是否闭合曲线，闭合时首尾相连

    Returns
    -------
    mesh : trimesh.Trimesh
        三维厚度 mesh
    """
    xy_points = np.asarray(xy_points)
    n = len(xy_points)

    # 顶层和底层顶点
    vertices_top = np.column_stack([xy_points, np.full((n, 1), z_top)])
    vertices_bottom = np.column_stack([xy_points, np.full((n, 1), z_bottom)])
    vertices = np.vstack([vertices_top, vertices_bottom])

    # 构建面片
    faces = []
    loop = n if closed else n - 1
    for i in range(loop):
        t0 = i
        t1 = (i + 1) % n
        b0 = i + n
        b1 = (i + 1) % n + n
        faces.append([t0, t1, b1])
        faces.append([t0, b1, b0])

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def clean_mesh(mesh: trimesh.Trimesh):
        """
        尽可能修复各种烂 mesh，避免 vertex_faces 相关异常。
        包含：
        - 清理越界 faces
        - 删除 inf/nan 顶点
        - 删除未引用顶点
        - 删除重复/退化面
        - 修法线、补洞
        - 重建拓扑（process）
        """

        # ---------- Step 1：faces 引用合法检查 ----------
        faces = mesh.faces
        vcount = len(mesh.vertices)

        valid = np.all((faces >= 0) & (faces < vcount), axis=1)
        if not np.all(valid):
            print(f"[repair] removing {np.sum(~valid)} faces with invalid vertex index")
            mesh = mesh.submesh([valid], append=True)[0]

        # ---------- Step 2：基础清理 ----------
        mesh.remove_infinite_values()
        # mesh.remove_duplicate_faces()
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        # mesh.remove_degenerate_faces()
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-12))

        # ---------- Step 3：修复法线 ----------
        mesh.fix_normals()

        # ---------- Step 4：修补洞 ----------
        try:
            mesh.fill_holes()
        except Exception as e:
            print("[repair] fill_holes failed:", e)

        # ---------- Step 5：全面处理 ----------
        mesh = mesh.process(validate=True)

        # ---------- Step 6：强制重建一个干净的 mesh ----------
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            process=True
        )

        return mesh


def generate_xy_vectors_at_point(n=180, include_endpoint=False):
    """
    在给定三维点处，沿 XY 平面生成 n 个等角度分布的向量。

    参数
    ----
    point : array-like, shape (3,)
        三维点 (x, y, z)；向量的起点。
    n : int
        向量个数（默认 180，平分 360°）。
    length : float
        向量长度（默认 1.0）。若为 None 或 1.0 则生成单位向量。
    include_endpoint : bool
        是否包含 360°（即包含与 0° 重合的向量）。通常设为 False，
        否则会在角度数组末尾包含 2π，使第一个与最后一个重复。

    返回
    ----
    origins : ndarray, shape (n, 3)
        每个向量的起点（均为输入点）。
    vectors : ndarray, shape (n, 3)
        每个向量的方向向量（长度为 length）。
    endpoints : ndarray, shape (n, 3)
        每个向量的终点 = origin + vector
    angles_deg : ndarray, shape (n,)
        每个向量对应的角度（度，0->360）
    """
    # point = np.asarray(point, dtype=float).reshape(3)
    if include_endpoint:
        angles = np.linspace(0, 2*np.pi, n, endpoint=True)
    else:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)

    # 在 XY 平面上单位方向向量 (cosθ, sinθ, 0)
    dirs = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=1)

    return dirs


def get_new_mesh(id, point_change_id, mesh,point_change,remesh=False):
    # 构造交错顶点序列: s0,b0,s1,b1,...,sN-1,bN-1
    vertices = np.empty((2 * len(id)), dtype=int)
    vertices[0::2] = id
    vertices[1::2] = point_change_id
    faces=[]
    for value in range(len(id)-1):
        k=value*2
        faces.append([vertices[k], vertices[k+1], vertices[k+2]])
        faces.append([vertices[k+2], vertices[k+1], vertices[k+3]])

    new_point = np.vstack((np.array(mesh.vertices), point_change))
    new_face = np.concatenate((np.array(mesh.faces), faces), axis=0)
    new_mesh = trimesh.Trimesh(new_point, new_face, process=True)

    # ==== 只检查新增面的法向量 ====
    new_indices = np.arange(len(mesh.vertices), len(new_point))  # 新增顶点索引
    # 找到所有包含新增顶点的面
    mask_new_faces = np.isin(new_mesh.faces, new_indices).any(axis=1)
    new_faces_idx = np.where(mask_new_faces)[0]

    if len(new_faces_idx) > 0:
        centroid = new_mesh.centroid
        face_centroids = new_mesh.triangles_center[new_faces_idx]
        face_normals = new_mesh.face_normals[new_faces_idx]
        directions = face_centroids - centroid
        dot = np.einsum('ij,ij->i', directions, face_normals)
        # 对新增面的平均点积为负则翻转这些面
        if np.mean(dot) < 0:
            new_mesh.faces[new_faces_idx] = new_mesh.faces[new_faces_idx][:, ::-1]

    # 刷新法线缓存
    new_mesh._cache.clear()
    _ = new_mesh.vertex_normals
    _ = new_mesh.face_normals

    if remesh==True:
        # ==== 可选 remesh ====
        mesh_lib = mn.meshFromFacesVerts(verts=new_mesh.vertices, faces=new_mesh.faces)
        remeshsettings = mm.RemeshSettings()
        remeshsettings.targetEdgeLen = 0.2
        mm.remesh(mesh_lib, remeshsettings)
        vertices = mn.getNumpyVerts(mesh_lib)
        faces = mn.getNumpyFaces(mesh_lib.topology)
        mesh_src = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        return new_mesh
    return mesh_src


def create_new_mesh(mesh, line1, line2):
    tree = cKDTree(mesh.vertices)
    dis, id = tree.query(line1)
    id = np.append(id, id[0])
    point_change_id = np.arange(len(id) - 1) + len(mesh.vertices)
    point_change_id = np.append(point_change_id, point_change_id[0])
    new_mesh = get_new_mesh(id, point_change_id, mesh,line2)
    return new_mesh


def get_big_index(vectors, base, boundary_centroid):
    tree = cKDTree(base.vertices)
    result = []
    for id in range(len(vectors) - 1):
        # 切割mesh
        left_point = vectors[id] * 10 + boundary_centroid
        right_point = vectors[id + 1] * 10 + boundary_centroid
        target = np.array([left_point, right_point, boundary_centroid])
        target_upp = target.copy()
        target_low = target.copy()
        target_upp[:, 2] = 100
        target_low[:, 2] = -100
        mesh = trimesh.Trimesh(target_low)
        mesh = create_new_mesh(mesh, target_low, target_upp)
        crash = perform_boolean_operation(mesh, base, "InsideB")
        # mesh.export('1.stl')
        # base.export('2.stl')
        # crash.export(os.path.join(os.path.join(r"result",file_name),r"crash.stl"))
        # crash.export(r"crash.stl")
        #找到中间点
        crash_tree=cKDTree(crash.vertices[:,:2])
        _,boundary_centroid_id=crash_tree.query(boundary_centroid[:2])
        # crash_center=crash.vertices[boundary_centroid_id]

        # 找到符合要求的点
        dis, id = tree.query(crash.vertices)
        # id = np.unique(id)

        teeth_root = np.min(crash.vertices[:, 2])
        teeth_top = np.max(crash.vertices[:, 2])
        # dis = abs(teeth_top - teeth_root)

        # 找到顶部点
        # 如果中心点不在这一部分 证明中心处过于凹陷了 需要把丢掉的加入进去
        mask = crash.vertices[:, 2] < teeth_top-1
        mask_indices = np.where(mask == 0)[0]  # 或 np.where(~mask)[0]
        # crash_top = trimesh.PointCloud(crash.vertices[mask_indices])
        # crash_top.export(os.path.join(os.path.join(r"result", file_name), r"crash_top0.ply"))
        # crash_top.export( r"crash_top0.ply")
        if boundary_centroid_id not in mask_indices:
            vertex_layers=find_vertex_layers_dl(crash,[boundary_centroid_id])
            # 键是点的索引 值是多少层
            mask_rand_indice=vertex_layers[mask_indices[0]]
            for key,value in vertex_layers.items():
                if value<mask_rand_indice:
                    mask_indices=np.append(mask_indices,key)
            mask_indices=np.unique(mask_indices)
            # crash_top=trimesh.PointCloud(crash.vertices[mask_indices])
            # crash_top.export(os.path.join(os.path.join(r"result", file_name), r"crash_top1.ply"))
            # crash_top.export(r"crash_top0_.ply")

        # 根据高度添加一波点
        # if dis<=height:
        result.extend(id[mask_indices])

    return result


def find_vertex_layers_dl(mesh, boundary_vertices):
    """
    从边界顶点开始标记层级，同时删除所有未连通区域（layer=1000）
    """
    vertex_layers = {v: -1 for v in range(len(mesh.vertices))}
    current_layer = 0
    current_vertices = set(boundary_vertices)

    # 初始边界层 = 0
    for v in current_vertices:
        vertex_layers[v] = 0

    # BFS 扩散层
    while current_vertices:
        next_vertices = set()
        for vertex in current_vertices:
            connected = set()
            for face in mesh.vertex_faces[vertex]:
                if face == -1:
                    continue
                connected.update(mesh.faces[face])

            for cv in connected:
                if vertex_layers[cv] == -1:
                    next_vertices.add(cv)
                    vertex_layers[cv] = current_layer + 1

        current_vertices = next_vertices
        current_layer += 1

    vertex_layers = {v: l for v, l in vertex_layers.items() if l != -1}

    return vertex_layers


def map_submesh_to_mesh(submesh: trimesh.Trimesh, mesh: trimesh.Trimesh):
    """
    使用KDTree将子mesh顶点映射到原mesh顶点
    """
    tree = cKDTree(mesh.vertices)
    distances, indices = tree.query(submesh.vertices)
    # indices[i] 表示 submesh 顶点 i 在原 mesh 的索引
    return indices

def fill_hole_on_mesh(mesh: trimesh.Trimesh, submesh: trimesh.Trimesh):
    """
    使用子mesh边界在原mesh上进行区域生长补洞
    """

    _, upper_boundary = find_boundary_edges(submesh)
    # upper_boundary_points = mesh1.vertices[upper_boundary]

    # 子mesh边界顶点
    boundary_vertices_submesh = set(np.unique(upper_boundary))
    interior_vertices_submesh = set(range(len(submesh.vertices))) - boundary_vertices_submesh

    if not interior_vertices_submesh:
        raise ValueError("子mesh内部没有非边界顶点可做seed")

    # 子mesh顶点映射到原mesh
    submesh_to_mesh_map = map_submesh_to_mesh(submesh, mesh)

    # 映射到原mesh索引
    boundary_vertices_in_mesh = set(submesh_to_mesh_map[list(boundary_vertices_submesh)])
    interior_vertices_in_mesh = [submesh_to_mesh_map[v] for v in interior_vertices_submesh]

    # 选第一个内部顶点作为seed
    seed_vertex = interior_vertices_in_mesh[0]

    # 区域生长
    visited = set()
    queue = deque([seed_vertex])
    while queue:
        v = queue.popleft()
        if v in visited or v in boundary_vertices_in_mesh:
            continue
        visited.add(v)
        neighbors = mesh.vertex_neighbors[v]
        for n in neighbors:
            if n not in visited:
                queue.append(n)

    visited_with_boundary = visited.union(boundary_vertices_in_mesh)
    return visited_with_boundary



def repair_mesh(mesh, best_normal, idx=None):
    # print('-----------------------------')
    #获取倒凹区域位置
    index_ray_filtered = get_invisible_idx(mesh, best_normal)
    # 滤除边界点附近索引
    _, boundary_idx = find_boundary_edges(mesh)
    vertex_layers = find_vertex_layers(mesh, boundary_idx)
    selected_indices = [v for v, layer in vertex_layers.items() if layer < 3]
    vertex_idx = list(set(index_ray_filtered) - set(selected_indices))

    # ---------- 滤除顶部点索引
    # boundary_centroid = np.mean(boundary_vertices, axis=0)
    raw_mesh = mesh.copy()
    boundary_centroid = mesh.centroid
    vectors = generate_xy_vectors_at_point(n=45)
    vectors = np.vstack([vectors, vectors[0]])
    # 找到需要最大扩大的点
    up_idx = get_big_index(vectors, mesh, boundary_centroid)
    # vertex_layers = find_vertex_layers(mesh, boundary_idx)
    # # 获取层级 < 5 的顶点索引
    # selected_indices = [v for v, layer in vertex_layers.items() if layer < 3]
    up_idx = list(set(up_idx) - set(selected_indices))

    faces_mask1 = np.all(np.isin(raw_mesh.faces, up_idx), axis=1)
    upper_faces = raw_mesh.faces[faces_mask1]
    up_idx = np.unique(upper_faces)
    old_to_new = {v: i for i, v in enumerate(up_idx)}
    remapped_faces = np.vectorize(old_to_new.get)(upper_faces)
    upper_mesh = trimesh.Trimesh(vertices=raw_mesh.vertices[up_idx], faces=remapped_faces, process=False)
    # upper_mesh.export('upper_mesh.ply')

    components = upper_mesh.split(only_watertight=False)  # 关闭 only_watertight 以保留非封闭的面片
    upper_mesh = max(components, key=lambda m: len(m.faces))

    up_idx = list(fill_hole_on_mesh(raw_mesh, upper_mesh))
    vertex_idx = list(set(vertex_idx) - set(up_idx))
    # ---------- 滤除顶部点索引

    if idx is not None:
        vertex_idx = list(set(vertex_idx) - set(idx))
        # print(len(vertex_idx), '--------------')

    if len(vertex_idx) == 0: # 没有倒凹直接返回
        return mesh, [], []

    vertex_idx_copy = vertex_idx.copy()


    faces_mask = np.any(np.isin(mesh.faces, vertex_idx), axis=1)
    selected_faces = mesh.faces[faces_mask]
    visible_mesh = mesh.submesh([~faces_mask], append=True)
    # visible_mesh.export('delRegion.stl')  # 保存可见 mesh 文件

    #根据连通性对检测区域进行分组
    index_ray_list = connected_components_on_vertices(mesh, vertex_idx)

    # delregion_mm = mn.meshFromFacesVerts(verts=visible_mesh.vertices, faces=visible_mesh.faces)

    new_idx = []
    for i, index in enumerate(index_ray_list):
        # pc = trimesh.PointCloud(mesh.vertices[index])
        # pc.export(rf'hole_{i}.ply')



        faces_mask = np.any(np.isin(mesh.faces, index), axis=1)
        selected_faces = mesh.faces[faces_mask]
        all_idx = list(set(np.unique(selected_faces.ravel())))
        hole_boundary_idx = list(set(np.unique(selected_faces.ravel())) - set(index))
        # pc = trimesh.PointCloud(mesh.vertices[hole_boundary_idx])
        # pc.export(rf'hole_boundary{i}.ply')

        # hole_point = mesh.vertices[index].reshape(-1, 3)
        hole_boundary_point = mesh.vertices[hole_boundary_idx].reshape(-1, 3)

        # boundary_tree = cKDTree(hole_boundary_point[:, :2])
        # _, hole_point_idx = boundary_tree.query(hole_point[:, :2])
        # hole_point[:, :2] = hole_boundary_point[:, :2][hole_point_idx]
        # print(len(index), len(hole_boundary_idx))
        try:
            convex_hull = trimesh.Trimesh(vertices=hole_boundary_point).convex_hull  #  一个点的情况， 报错跳过
        except:
            continue
        # convex_hull.export(rf'curve_mesh{i}.ply')

        # 1️⃣ 计算面中心
        face_centers = convex_hull.triangles_center  # shape (F, 3)
        # 2️⃣ 面法线
        face_normals = convex_hull.face_normals  # shape (F, 3), 默认外向

        # 3️⃣ mesh 重心（或中心点）
        mesh_center = mesh.centroid  # shape (3,)

        # 4️⃣ 计算每个面中心指向 mesh 中心的向量
        to_center = face_centers - mesh_center
        to_center.reshape(-1, 3)
        to_center[:, 2] = 0
        to_center_norm = to_center / np.linalg.norm(to_center, axis=1)[:, None]  # 单位向量

        # 5️⃣ 点积判断方向
        # dot > 0 → 面法线指向 mesh 中心
        # dot < 0 → 面法线远离 mesh 中心
        dot = np.einsum('ij,ij->i', face_normals, to_center_norm)

        # # 6️⃣ 分类
        # faces_pointing_in = np.where(dot > 0)[0]
        # faces_pointing_out = np.where(dot < 0)[0]

        # 6️⃣ 保留朝外面片（法线远离中心）
        faces_out_mask = dot > 0
        faces_out = convex_hull.faces[faces_out_mask]

        # 7️⃣ 构建新的 mesh
        convex_hull = trimesh.Trimesh(vertices=convex_hull.vertices, faces=faces_out, process=False)
        # convex_hull.export(rf'curve_mesh{i}.ply')
        # mm.saveMesh(delregion_mm, rf'curve_mesh{i}.ply')

        # convex_hull.export(rf'curve_mesh{i}.ply')
        # move_points = mesh.vertices[all_idx].copy()
        # closest_points, distance, face_index = trimesh.proximity.closest_point(convex_hull, move_points)
        # move_points[:,:2] = closest_points[:,:2]
        # mesh.vertices[all_idx] = move_points

        vertices = mesh.vertices[all_idx].reshape(-1, 3)
        # move_normals = mesh.vertex_normals[index].reshape(-1, 3)
        move_normals = vertices - mesh_center
        move_normals.reshape(-1, 3)
        move_normals[:, 2] = 0

        # 向量归一化（避免 0 向量）
        norm = np.linalg.norm(move_normals, axis=1, keepdims=True)
        norm[norm == 0] = 1e-8  # 避免除 0
        move_normals = move_normals / norm

        # origin 就是顶点本身
        ray_origins = vertices
        index_tri, index_ray, locations = convex_hull.ray.intersects_id(
            ray_origins=ray_origins,
            ray_directions=move_normals,
            multiple_hits=True,
            return_locations=True
        )

        N = len(vertices)
        last_hit = [None] * N  # 保存最终交点

        if len(index_ray) > 0:
            # index_ray 是交点对应的射线编号，例如 [0,0,0,1,1, ...]
            for r in range(N):
                mask = (index_ray == r)
                if np.any(mask):
                    # 多个交点 → 取最后一个
                    last_hit[r] = locations[mask][-1]  # (3,)
                else:
                    last_hit[r] = None

        final_intersections = np.array([
            p if p is not None else np.array([np.nan, np.nan, np.nan])
            for p in last_hit
        ])
        # print(final_intersections)
        # final_intersections 中 nan 表示无交点 → 保留原点
        valid = ~np.isnan(final_intersections).any(axis=1)
        # 替换原点坐标
        vertices[valid] = final_intersections[valid]
        mesh.vertices[all_idx] = vertices

        # pc  = trimesh.PointCloud(vertices[valid])
        # pc.export(rf'pc{i}.ply')
        # print(valid.sum(), len(all_idx))
        # print((~valid).sum(), len(all_idx))
        #
        # pc = trimesh.PointCloud(vertices[~valid])
        # pc.export(rf'pc{i}_.ply')
        # move_points = mesh.vertices[np.array(all_idx)[~valid]].copy()
        # closest_points, distance, face_index = trimesh.proximity.closest_point(convex_hull, move_points)
        # move_points[:,:2] = closest_points[:,:2]
        # mesh.vertices[np.array(all_idx)[~valid]] = move_points

        mesh_combined = trimesh.util.concatenate([visible_mesh, convex_hull])
        vertices = mesh.vertices[np.array(all_idx)[~valid]]
        ray_directions = np.tile(best_normal, (len(vertices), 1))  # shape (N,3)
        # origin 就是顶点本身
        ray_origins = vertices
        index_tri, index_ray, locations = mesh_combined.ray.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=True,
            return_locations=True
        )

        # 计算距离
        # locations 对应 index_ray
        hit_origins = ray_origins[index_ray]  # 对应击中的射线原点
        distances = np.linalg.norm(locations - hit_origins, axis=1)

        # 筛除距离接近0的
        eps = 1e-6  # 可根据需要调整
        valid_mask = distances > eps

        # 过滤后的结果
        index_ray_filtered = index_ray[valid_mask]

        new_idx = new_idx + list(np.array(all_idx)[~valid][index_ray_filtered])
        # if len(index_ray_filtered) > 0:
        #     move_points = vertices[index_ray_filtered]
        #     closest_points, distance, face_index = trimesh.proximity.closest_point(convex_hull, move_points)
        #     move_points = closest_points
        #     mesh.vertices[np.array(all_idx)[~valid][index_ray_filtered]] = move_points

        # pc = mm.PointCloud()
        # points_ = mesh.vertices[all_idx]
        # normals_ = mesh.copy().vertex_normals[all_idx]
        #
        # # 3️⃣ 循环添加点
        # for p, n in zip(points_, normals_):
        #     point_vec = mm.Vector3f(p[0], p[1], p[2])
        #     normal_vec = mm.Vector3f(n[0], n[1], n[2])
        #     pc.addPoint(point_vec, normal_vec)
        #
        # params = mm.PointsToMeshParameters()
        # params.voxelSize = 0.3
        # params.minWeight = 1.0
        # nefertiti_mesh = mm.pointsToMeshFusion(pc, params)
        # mm.saveMesh(nefertiti_mesh, rf"Mesh{i}.ply")

    # mesh.export('test.ply')
    vertex_idx_copy = list(set(vertex_idx_copy) - set(new_idx))

    mesh_mm = mn.meshFromFacesVerts(verts=mesh.vertices, faces=mesh.faces)
    new_vert = mm.VertBitSet()
    new_vert.resize(mesh_mm.topology.getValidVerts().size())
    for ancV in new_idx:
        new_vert.set(mm.VertId(ancV), True)

    mm.expand(mesh_mm.topology, new_vert, 1)
    new_face = mm.getIncidentFaces(mesh_mm.topology, new_vert)
    mm.delRegionKeepBd(mesh_mm, new_face, keepLoneHoles=True)

    # mm.saveMesh(mesh_mm, "delRegion.stl")
    trimesh_vertices = mn.getNumpyVerts(mesh_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(mesh_mm.topology)
    delRegion_mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(), process=False)

    components = delRegion_mesh.split(only_watertight=False)
    delRegion_mesh = max(components, key=lambda m: len(m.faces))
    mesh_mm = mn.meshFromFacesVerts(verts=delRegion_mesh.vertices, faces=delRegion_mesh.faces)
    # print()

    holes = mesh_mm.topology.findHoleRepresentiveEdges()
    hole_info = []
    for e in holes:
        dir_area = mm.holeDirArea(mesh_mm.topology, mesh_mm.points, e)
        area = dir_area.length()
        hole_info.append((e, area, dir_area))
    # 按面积降序排列
    hole_info.sort(key=lambda x: x[1], reverse=True)
    # newfacebit = mm.FaceBitSet()
    # newfacebit.resize(mesh_mm.topology.getValidFaces().size())
    # Edge_mean_Len = np.array(list(mm.edgeLengths(mesh_mm.topology, mesh_mm.points.vec))).mean()  # 平均长度
    # Edge_mean_Len = np.array(list(mm.edgeLengths(mesh_mm.topology, mesh_mm.points.vec))).max()  # 最大长度
    Edge_mean_Len = 0.5 * (np.array(list(mm.edgeLengths(mesh_mm.topology, mesh_mm.points.vec))).max()
                           - np.array(list(mm.edgeLengths(mesh_mm.topology, mesh_mm.points.vec))).min())  # 中值

    all_bitset =  mm.VertBitSet()
    all_bitset.resize(mesh_mm.topology.getValidVerts().size())
    for i in range(1, len(hole_info)):
        params = mm.FillHoleParams()
        edge_metric = mm.getEdgeLengthFillMetric(mesh_mm)
        params.metric = edge_metric
        # 创建一个 FaceBitSet 用来接收新面
        bitset = mm.FaceBitSet()
        params.outNewFaces = bitset  # 记录补洞时的新面
        e, _, _ = hole_info[i]
        mm.fillHole(mesh_mm, e, params)

        new_bitset = mm.getIncidentVerts(mesh_mm.topology, bitset)
        # mm.expand(mesh_mm.topology, new_bitset, 3)
        all_bitset |= new_bitset

        settings = mm.SubdivideSettings()
        settings.smoothMode = True
        settings.maxEdgeLen = Edge_mean_Len
        settings.maxEdgeSplits = 200
        settings.maxDeviationAfterFlip = 0.2
        settings.region = bitset

        # 根据细分后的新顶点进行可选曲率定位.
        newVertsBitSet = mm.VertBitSet()
        settings.newVerts = newVertsBitSet
        mm.subdivideMesh(mesh_mm, settings)

        # mm.saveMesh(mesh_mm, 'repair_mesh.stl')

        mm.expand(mesh_mm.topology, newVertsBitSet, 0)

        for _ in range(2):
            mm.positionVertsSmoothly(mesh_mm, newVertsBitSet,
                                     mm.LaplacianEdgeWeightsParam.Cotan,
                                     mm.VertexMass.NeiArea
                                     )

        all_bitset |= newVertsBitSet
    # mm.saveMesh(mesh_mm, "fill_hole.stl")
    # points_mm = mn.getNumpyVerts(mesh_mm)
    indices_mm = [int(v) for v in list(all_bitset)]
    vertex_idx_copy = list(set(vertex_idx_copy) - set(indices_mm))
    if idx is not None:
        vertex_idx_copy = list(vertex_idx_copy) + list(idx)

    if len(vertex_idx_copy) > 0:
        f_points_mm = mesh.vertices[vertex_idx_copy]
    # # print(indices_mm)
    # pc = trimesh.PointCloud(f_points_mm)
    # pc.export("test0.ply")

    mesh_mm.pack()


    trimesh_vertices = mn.getNumpyVerts(mesh_mm)  # 转为trimesh格式
    trimesh_faces = mn.getNumpyFaces(mesh_mm.topology)
    mesh = trimesh.Trimesh(vertices=trimesh_vertices.copy(), faces=trimesh_faces.copy(), process=False)


    # mesh = clean_mesh(mesh)

    # pc = trimesh.PointCloud(mesh.vertices[vertex_idx_copy])
    # pc.export("test1.ply")
    if len(vertex_idx_copy) > 0:
        map_tree = cKDTree(mesh.vertices)
        _, map_idx = map_tree.query(f_points_mm)
    else:
        map_idx = []

    # move_points = mesh.vertices.copy()
    # closest_points, distance, face_index = trimesh.proximity.closest_point(delRegion_mesh, move_points)
    # move_points = closest_points
    # mesh.vertices = move_points

    index_ray_filtered = get_invisible_idx(mesh, best_normal)
    _, boundary_idx = find_boundary_edges(mesh)
    vertex_layers = find_vertex_layers(mesh, boundary_idx)
    selected_indices = [v for v, layer in vertex_layers.items() if layer < 3]
    vertex_idx_ = list(set(index_ray_filtered) - set(selected_indices)- set(map_idx))

    # ---------- 滤除顶部点索引
    # boundary_centroid = np.mean(boundary_vertices, axis=0)
    raw_mesh = mesh.copy()
    boundary_centroid = mesh.centroid
    vectors = generate_xy_vectors_at_point(n=45)
    vectors = np.vstack([vectors, vectors[0]])
    # 找到需要最大扩大的点
    up_idx = get_big_index(vectors, mesh, boundary_centroid)
    # vertex_layers = find_vertex_layers(mesh, boundary_idx)
    # # 获取层级 < 5 的顶点索引
    # selected_indices = [v for v, layer in vertex_layers.items() if layer < 3]
    up_idx = list(set(up_idx) - set(selected_indices))

    faces_mask1 = np.all(np.isin(raw_mesh.faces, up_idx), axis=1)
    upper_faces = raw_mesh.faces[faces_mask1]
    up_idx = np.unique(upper_faces)
    old_to_new = {v: i for i, v in enumerate(up_idx)}
    remapped_faces = np.vectorize(old_to_new.get)(upper_faces)
    upper_mesh = trimesh.Trimesh(vertices=raw_mesh.vertices[up_idx], faces=remapped_faces, process=False)
    # upper_mesh.export('upper_mesh.ply')

    components = upper_mesh.split(only_watertight=False)  # 关闭 only_watertight 以保留非封闭的面片
    upper_mesh = max(components, key=lambda m: len(m.faces))

    up_idx = list(fill_hole_on_mesh(raw_mesh, upper_mesh))
    vertex_idx_ = list(set(vertex_idx_) - set(up_idx))
    # ---------- 滤除顶部点索引

    # pc = trimesh.PointCloud(mesh.vertices[vertex_idx_])
    # pc.export("test1_.ply")

    # mesh.export('test1.ply')
    return  mesh, vertex_idx_, map_idx


if __name__ == '__main__':
    mesh = trimesh.load_mesh(r"Z:\10-算法组\gbz\测试数据11170900\48\r_crown_top_mesh.ply")

    # mesh = clean_mesh(mesh)
    map_idx = None
    while True:
    # mesh, r_idx, map_idx = repair_mesh(mesh, np.array([0, 0, 1]), idx=map_idx)
    # print(len(r_idx))
        mesh, r_idx, map_idx = repair_mesh(mesh, np.array([0, 0, 1]), idx=None)
        print(len(r_idx))
        if len(r_idx) == 0:

            break

    # import os
    # root_path = r'Z:\10-算法组\gbz\测试数据11170900'
    #
    # for root, dirs, files in os.walk(root_path):
    #     if len(dirs) == 0:
    #
    #         if os.path.exists(os.path.join(root, "r_base_cut.ply")):
    #             print(root)
    #             mesh = trimesh.load_mesh(os.path.join(root, "r_base_cut.ply"))
    #             mesh = clean_mesh(mesh)
    #             map_idx = None
    #             while True:
    #                 mesh, r_idx, map_idx = repair_mesh(mesh, np.array([0, 0, 1]), idx=None)
    #                 print(len(r_idx))
    #                 if len(r_idx) == 0:
    #                     break
    #
    #             mesh.export(os.path.join(root, "repair_mesh1.ply"))

    # mesh_mm = mm.loadMesh(r"Z:\10-算法组\gbz\测试数据11170900\48\r_crown_top_mesh.ply")
    #
    # # 2) 准备 FindParams：upDirection 与 wallAngle（单位同文档，wallAngle 可为正/负）
    # #    upDirection: 可用 Vector3f(0,0,1) 表示 Z+ 为“上”
    # #    wallAngle: 0 => 严格垂直墙；正数扩大下方墙；负数缩小
    # find_params = mm.FixUndercuts.FindParams(mm.Vector3f(0.0, 0.0, 1.0), 0.0)
    #
    # # 3) 计算合适的 voxelSize（体素分辨率），通常按 bbox 大小的比例设定
    # bbox_diag = mesh_mm.computeBoundingBox().diagonal()
    # voxel_size = float(bbox_diag) * 5e-3  # 举例：bbox * 0.005 -> 精度/内存权衡
    #
    # # 4) bottomExtension：底部延伸最小值（用于在下方构建垂直墙体时延伸基底）
    # bottom_ext = float(bbox_diag) * 2e-2  # 举例：bbox * 0.02
    #
    # # 5) region：如果想修复整个 mesh，传 None；要固定局部则传 FaceBitSet（见下）
    # region = None  # 整个网格
    #
    # # 6) smooth：是否对 voxels 做一次高斯平滑（若存在薄壁建议 True）
    # smooth = True
    #
    #
    # # 7) 进度回调 cb(progress: float) -> bool。返回 False 可中断操作
    # def progress_cb(p: float) -> bool:
    #     # print(f"[FixUndercuts] progress: {p * 100:.1f}%")
    #     pass
    #     return True
    #
    #
    # # 8) 构造 FixParams（使用文档中的聚合构造器）
    # params = mm.FixUndercuts.FixParams(find_params, voxel_size, bottom_ext, region, smooth, progress_cb)
    #
    # # 9) 执行修复（在原 mesh 上原地修改；操作是 voxel-based，会重建网格）
    # mm.FixUndercuts.fix(mesh_mm, params)
    #
    # # 10) 保存结果
    # mm.saveMesh(mesh_mm, "input_fixed.stl")
    #
    # settings = mm.RemeshSettings()
    # settings.targetEdgeLen = 0.25
    # settings.useCurvature = True
    # settings.projectOnOriginalMesh = False
    # mm.remesh(mesh_mm, settings)  # 执行 remesh
    # mesh_mm.pack()
    # mm.saveMesh(mesh_mm, 'input_fixed1.stl')


    # mesh = trimesh.load_mesh(r"Z:\10-算法组\gbz\测试数据11170900\48\r_crown_top_mesh.ply")
    # vertex_idx = get_invisible_idx(mesh, np.array([0, 0, 1]))
    #
    # faces_mask = np.any(np.isin(mesh.faces, vertex_idx), axis=1)
    # visible_mesh = mesh.submesh([~faces_mask], append=True)
    # visible_mesh.export('delRegion.stl')  # 保存可见 mesh 文件
    # print(len(vertex_idx))
    #
    # mesh_idx = list(fill_hole_on_mesh(mesh, visible_mesh))
    # pc = trimesh.PointCloud(mesh.vertices[mesh_idx])
    # pc.export('delRegion.ply')
    # faces_mask = np.any(np.isin(mesh.faces, mesh_idx), axis=1)
    # visible_mesh = mesh.submesh([faces_mask], append=True)
    # visible_mesh.export('delRegion1.stl')
    # print(len(mesh_idx), len(mesh.vertices))
