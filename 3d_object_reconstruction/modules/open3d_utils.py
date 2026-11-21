from __future__ import annotations

from pathlib import Path

import numpy as np


def _open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is not installed. Run: pip install open3d") from exc
    return o3d


_PLY_TYPE_MAP = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _read_ply_vertex_data(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read scalar PLY vertex fields without requiring Open3D (ASCII or binary PLY)."""
    path = Path(path)
    with path.open("rb") as stream:
        header_lines: list[str] = []
        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise ValueError("PLY header is incomplete: missing end_header.")
            line = raw_line.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                header_bytes = stream.tell()
                break

    if not header_lines or header_lines[0] != "ply":
        raise ValueError("File does not have a PLY header.")

    file_format = None
    vertex_count = None
    current_element = None
    vertex_properties: list[tuple[str, str]] = []
    element_order: list[str] = []
    for line in header_lines[1:]:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "format" and len(tokens) >= 2:
            file_format = tokens[1]
        elif tokens[0] == "element" and len(tokens) == 3:
            current_element = tokens[1]
            element_order.append(current_element)
            if current_element == "vertex":
                vertex_count = int(tokens[2])
        elif tokens[0] == "property" and current_element == "vertex":
            if len(tokens) == 3 and tokens[1] in _PLY_TYPE_MAP:
                vertex_properties.append((tokens[2], tokens[1]))
            else:
                raise ValueError("PLY vertex list properties are not supported by the lightweight reader.")

    if vertex_count is None or vertex_count < 1:
        raise ValueError("PLY file has no vertices.")
    if not vertex_properties:
        raise ValueError("PLY file has no scalar vertex properties.")
    if element_order and element_order[0] != "vertex":
        raise ValueError("PLY vertex data is not the first element; use Open3D to read this file.")

    names = [name for name, _ in vertex_properties]
    if not all(axis in names for axis in ("x", "y", "z")):
        raise ValueError("PLY vertex data does not contain x, y, and z coordinates.")

    if file_format == "ascii":
        values = np.loadtxt(path, skiprows=len(header_lines), max_rows=vertex_count)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[0] != vertex_count or values.shape[1] < len(vertex_properties):
            raise ValueError("PLY vertex data is incomplete.")
        column_indices = {name: index for index, name in enumerate(names)}
        points = values[:, [column_indices[axis] for axis in ("x", "y", "z")]].astype(np.float64)
        color_names = ("red", "green", "blue")
        colors = None
        if all(name in column_indices for name in color_names):
            colors = values[:, [column_indices[name] for name in color_names]].astype(np.float64)
            if colors.size and colors.max() > 1.0:
                colors /= 255.0
        return points, colors

    endian = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(file_format or "")
    if endian is None:
        raise ValueError(f"Unsupported PLY format: {file_format}")

    dtype = np.dtype([
        (name, endian + _PLY_TYPE_MAP[type_name]) for name, type_name in vertex_properties
    ])
    records = np.fromfile(path, dtype=dtype, count=vertex_count, offset=header_bytes)
    if len(records) != vertex_count:
        raise ValueError("PLY vertex data is incomplete.")
    points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float64)
    colors = None
    if all(name in records.dtype.names for name in ("red", "green", "blue")):
        colors = np.column_stack((records["red"], records["green"], records["blue"])).astype(np.float64)
        if colors.size and colors.max() > 1.0:
            colors /= 255.0
    return points, colors


def _load_point_cloud_data(path: Path, max_points: int | None = None):
    """Load a PLY point cloud with a lightweight reader; use Open3D for other formats."""
    path = Path(path)
    if path.suffix.lower() == ".ply":
        try:
            points, colors = _read_ply_vertex_data(path)
        except Exception:
            # If the lightweight PLY reader does not support a variant, try Open3D.
            o3d = _open3d()
            pcd = o3d.io.read_point_cloud(str(path))
            if pcd.is_empty():
                raise ValueError(f"Point cloud is empty or unreadable: {path}")
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    else:
        o3d = _open3d()
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            raise ValueError(f"Point cloud is empty or unreadable: {path}")
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    if max_points is not None and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    return points, colors


def get_point_cloud_stats(point_cloud_path: Path) -> dict:
    points, colors = _load_point_cloud_data(point_cloud_path)
    if len(points) == 0:
        raise ValueError(f"Point cloud is empty or unreadable: {point_cloud_path}")
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    return {
        "point_count": int(len(points)),
        "has_colors": colors is not None,
        "bounds_min": [round(float(v), 5) for v in bounds_min],
        "bounds_max": [round(float(v), 5) for v in bounds_max],
        "approx_extent": [round(float(v), 5) for v in (bounds_max - bounds_min)],
    }

def create_poisson_mesh(point_cloud_path: Path, output_path: Path, depth: int = 8) -> tuple[Path, dict]:
    o3d = _open3d()
    point_cloud_path = Path(point_cloud_path)
    output_path = Path(output_path)
    pcd = o3d.io.read_point_cloud(str(point_cloud_path))
    if pcd.is_empty() or len(pcd.points) < 100:
        raise ValueError("Point cloud is empty or too small for surface reconstruction.")

    if len(pcd.points) > 2_000_000:
        pcd = pcd.voxel_down_sample(voxel_size=0.002)

    if len(pcd.points) > 50:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(k=30)
    except Exception:
        # Normal orientation may fail on sparse / fragmented clouds; continue with estimated normals.
        pass

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
    density_values = np.asarray(densities)
    if len(density_values):
        threshold = float(np.quantile(density_values, 0.02))
        vertices_to_remove = density_values < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), mesh):
        raise IOError(f"Open3D could not write mesh: {output_path}")
    stats = {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "poisson_depth": int(depth),
        "output": str(output_path),
    }
    return output_path, stats


def load_point_cloud_for_plot(path: Path, max_points: int = 50000):
    points, colors = _load_point_cloud_data(path, max_points=max_points)
    if len(points) == 0:
        return None
    return points[:, 0], points[:, 1], points[:, 2], colors

def load_mesh_for_plot(path: Path, max_faces: int = 100000):
    o3d = _open3d()
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        return None
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    if len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[indices]
    if colors is not None and len(colors) != len(vertices):
        colors = None
    return vertices, faces, colors


def export_mesh_to_glb(mesh_path: Path, output_path: Path) -> Path:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is not installed. Run: pip install trimesh") from exc
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError("The source mesh is empty or unreadable.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="glb")
    if not output_path.exists():
        raise IOError("GLB export did not create an output file.")
    return output_path
