from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Callable

LogCallback = Callable[[str, float | None], None]


class ColmapError(RuntimeError):
    pass


def resolve_colmap_executable(value: str | None) -> str:
    """Resolve COLMAP from a user path or PATH."""
    candidate = (value or "").strip().strip('"')
    if candidate:
        expanded = os.path.expandvars(os.path.expanduser(candidate))
        path = Path(expanded)
        if path.exists() and path.is_file():
            return str(path.resolve())
        found = shutil.which(candidate)
        if found:
            return found
        raise ColmapError(
            f"COLMAP executable not found: {candidate}\n"
            "Install COLMAP, add it to PATH, or enter the full path to colmap / COLMAP.bat in the sidebar."
        )

    found = shutil.which("colmap") or shutil.which("COLMAP")
    if found:
        return found
    raise ColmapError("COLMAP executable not found. Enter its path in the sidebar.")


def _run_command(
    executable: str,
    args: list[str],
    cwd: Path,
    callback: LogCallback,
    progress: float,
) -> list[str]:
    """Run one COLMAP command, stream its output, and return captured lines."""
    full_args = [str(executable), *args]
    command_label = " ".join(full_args)
    callback(f"\n$ {command_label}", progress)

    try:
        if os.name == "nt" and Path(executable).suffix.lower() in {".bat", ".cmd"}:
            command_line = subprocess.list2cmdline(full_args)
            process = subprocess.Popen(
                command_line,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=True,
            )
        else:
            process = subprocess.Popen(
                full_args,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
    except OSError as exc:
        raise ColmapError(f"Could not start COLMAP. Check the executable path. Details: {exc}") from exc

    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        callback(line, progress)
    return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(output_lines[-30:])
        raise ColmapError(
            f"COLMAP command failed with exit code {return_code}.\n"
            f"Command: {command_label}\n\nLast output lines:\n{tail}"
        )
    return output_lines


def _database_diagnostics(database_path: Path) -> dict:
    """Read lightweight feature/matching counts from COLMAP's SQLite database."""
    summary = {
        "database_exists": database_path.exists(),
        "images_in_database": 0,
        "images_with_keypoints": 0,
        "total_keypoints": 0,
        "matched_image_pairs": 0,
        "putative_matches": 0,
        "verified_image_pairs": 0,
        "geometric_inlier_matches": 0,
        "per_image_keypoints": [],
    }
    if not database_path.exists():
        return summary

    try:
        with sqlite3.connect(str(database_path)) as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "images" in tables:
                summary["images_in_database"] = connection.execute(
                    "SELECT COUNT(*) FROM images"
                ).fetchone()[0]
            if "keypoints" in tables:
                summary["images_with_keypoints"] = connection.execute(
                    "SELECT COUNT(*) FROM keypoints WHERE rows > 0"
                ).fetchone()[0]
                summary["total_keypoints"] = connection.execute(
                    "SELECT COALESCE(SUM(rows), 0) FROM keypoints"
                ).fetchone()[0]
            if "images" in tables and "keypoints" in tables:
                rows = connection.execute(
                    "SELECT images.name, COALESCE(keypoints.rows, 0) "
                    "FROM images LEFT JOIN keypoints USING(image_id) "
                    "ORDER BY keypoints.rows ASC LIMIT 8"
                ).fetchall()
                summary["per_image_keypoints"] = [
                    {"image": name, "keypoints": int(count)} for name, count in rows
                ]
            if "matches" in tables:
                summary["matched_image_pairs"] = connection.execute(
                    "SELECT COUNT(*) FROM matches WHERE rows > 0"
                ).fetchone()[0]
                summary["putative_matches"] = connection.execute(
                    "SELECT COALESCE(SUM(rows), 0) FROM matches"
                ).fetchone()[0]
            if "two_view_geometries" in tables:
                summary["verified_image_pairs"] = connection.execute(
                    "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"
                ).fetchone()[0]
                summary["geometric_inlier_matches"] = connection.execute(
                    "SELECT COALESCE(SUM(rows), 0) FROM two_view_geometries"
                ).fetchone()[0]
    except sqlite3.Error as exc:
        summary["database_read_error"] = str(exc)
    return summary


def _sparse_tree_listing(sparse_dir: Path) -> list[str]:
    if not sparse_dir.exists():
        return ["<sparse folder does not exist>"]
    entries: list[str] = []
    for path in sorted(sparse_dir.rglob("*")):
        if path.is_file():
            try:
                entries.append(str(path.relative_to(sparse_dir)))
            except ValueError:
                entries.append(path.name)
        if len(entries) >= 40:
            entries.append("...")
            break
    return entries or ["<sparse folder is empty>"]


def find_sparse_model(sparse_dir: Path) -> Path | None:
    """Find a COLMAP model in binary or text format, including nested folders."""
    if not sparse_dir.exists():
        return None

    required_sets = (
        {"cameras.bin", "images.bin", "points3D.bin"},
        {"cameras.txt", "images.txt", "points3D.txt"},
    )
    candidates: list[Path] = []
    possible_dirs = [sparse_dir, *(p for p in sparse_dir.rglob("*") if p.is_dir())]
    for folder in possible_dirs:
        names = {p.name for p in folder.iterdir() if p.is_file()}
        if any(required.issubset(names) for required in required_sets):
            candidates.append(folder)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda folder: sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),
    )


def _no_sparse_model_error(sparse_dir: Path, database_path: Path, image_count: int) -> ColmapError:
    diagnostics = _database_diagnostics(database_path)
    lines = [
        "COLMAP mapper finished but did not produce a recognizable sparse model.",
        "",
        "COLMAP database diagnostics:",
        f"- Images in database: {diagnostics['images_in_database']} (uploaded: {image_count})",
        f"- Images with keypoints: {diagnostics['images_with_keypoints']}",
        f"- Total SIFT keypoints: {diagnostics['total_keypoints']}",
        f"- Image pairs with putative matches: {diagnostics['matched_image_pairs']}",
        f"- Total putative matches: {diagnostics['putative_matches']}",
        f"- Geometrically verified image pairs: {diagnostics['verified_image_pairs']}",
        f"- Geometric inlier matches: {diagnostics['geometric_inlier_matches']}",
    ]
    if diagnostics.get("database_read_error"):
        lines.append(f"- Database read warning: {diagnostics['database_read_error']}")
    if diagnostics.get("per_image_keypoints"):
        counts = "; ".join(
            f"{item['image']}: {item['keypoints']}" for item in diagnostics["per_image_keypoints"]
        )
        lines.append(f"- Lowest per-image keypoint counts: {counts}")

    lines.extend(["", "Sparse folder contents:"])
    lines.extend(f"- {entry}" for entry in _sparse_tree_listing(sparse_dir))
    lines.extend(["", "How to interpret this:"])

    if diagnostics["images_in_database"] < image_count:
        lines.append(
            "- COLMAP indexed fewer images than the app supplied. Check the feature_extractor log for "
            "unsupported, unreadable, or skipped image files."
        )
    if image_count < 8:
        lines.append(
            f"- Only {image_count} image(s) were supplied. That is a very small set; try 20–50 sharp, "
            "overlapping photos taken around the object at multiple heights."
        )
    if diagnostics["images_with_keypoints"] < 2 or diagnostics["total_keypoints"] == 0:
        lines.append(
            "- Feature extraction found too few usable features. Use sharper, textured images and check "
            "that the object is in focus and sufficiently large in the frame."
        )
    if diagnostics["matched_image_pairs"] == 0 or diagnostics["putative_matches"] == 0:
        lines.append(
            "- No putative matches were stored. Neighboring views may not overlap enough, or the images "
            "may have too few repeatable visual features."
        )
    elif diagnostics["verified_image_pairs"] == 0 or diagnostics["geometric_inlier_matches"] == 0:
        lines.append(
            "- Features were detected and matched, but geometric verification found no usable image pairs. "
            "Capture closer neighboring viewpoints with 60–80% overlap; avoid reflections, plain surfaces, "
            "large viewpoint jumps, and moving backgrounds."
        )
    else:
        lines.append(
            "- Some image pairs passed geometric verification, but the mapper could not register a model. "
            "Check the mapper log above, use more connected views around the full object, and avoid changing zoom."
        )
    lines.append(
        "- A prior run summary is not evidence that this run succeeded: each new run rebuilds the database "
        "and sparse folder from the images currently saved in this project."
    )
    return ColmapError("\n".join(lines))


def _export_sparse_ply(
    executable: str,
    sparse_model: Path,
    output_path: Path,
    project_dir: Path,
    callback: LogCallback,
) -> tuple[Path | None, str | None]:
    try:
        _run_command(
            executable,
            [
                "model_converter",
                "--input_path", str(sparse_model),
                "--output_path", str(output_path),
                "--output_type", "PLY",
            ],
            project_dir,
            callback,
            0.46,
        )
        if output_path.exists() and output_path.stat().st_size > 0:
            callback(f"Sparse point cloud exported: {output_path}", 0.49)
            return output_path, None
        return None, "COLMAP model_converter finished but did not create sparse.ply."
    except ColmapError as exc:
        return None, str(exc)


def run_colmap_pipeline(
    project_dir: Path,
    image_dir: Path,
    colmap_executable: str,
    run_dense: bool = False,
    max_image_size: int = 2000,
    use_gpu_for_sift: bool = False,
    log_callback: LogCallback | None = None,
) -> dict:
    """Run COLMAP sparse SfM and, optionally, the CUDA-dependent dense MVS stage."""
    project_dir = Path(project_dir).resolve()
    image_dir = Path(image_dir).resolve()
    callback = log_callback or (lambda message, fraction=None: None)
    image_files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if len(image_files) < 2:
        raise ColmapError("At least two valid images are required.")

    database_path = project_dir / "database.db"
    sparse_dir = project_dir / "sparse"
    dense_dir = project_dir / "dense"
    output_dir = project_dir / "output"
    for path in (database_path, sparse_dir, dense_dir, output_dir):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    sparse_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    callback(
        f"Starting sparse reconstruction with {len(image_files)} input images. "
        f"SIFT/matching device: {'GPU' if use_gpu_for_sift else 'CPU'}.",
        0.02,
    )
    if len(image_files) < 8:
        callback(
            "Warning: fewer than 8 images were supplied. The pipeline will run, but 20–50 well-overlapped "
            "photos are a much better starting point for an object.",
            None,
        )

    gpu_flags = [] if use_gpu_for_sift else [
        "--SiftExtraction.use_gpu", "0", "--SiftMatching.use_gpu", "0"
    ]

    _run_command(
        colmap_executable,
        [
            "feature_extractor", "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "SIMPLE_RADIAL",
            *gpu_flags[:2],
        ],
        project_dir, callback, 0.08,
    )
    _run_command(
        colmap_executable,
        ["exhaustive_matcher", "--database_path", str(database_path), *gpu_flags[2:]],
        project_dir, callback, 0.20,
    )
    try:
        _run_command(
            colmap_executable,
            ["mapper", "--database_path", str(database_path), "--image_path", str(image_dir),
             "--output_path", str(sparse_dir)],
            project_dir, callback, 0.38,
        )
    except ColmapError as exc:
        diagnostics = _database_diagnostics(database_path)
        raise ColmapError(
            f"{exc}\n\nFeature/matching diagnostics after mapper failure:\n"
            f"{diagnostics}"
        ) from exc

    sparse_model = find_sparse_model(sparse_dir)
    if sparse_model is None:
        raise _no_sparse_model_error(sparse_dir, database_path, len(image_files))

    sparse_ply_path, sparse_export_error = _export_sparse_ply(
        colmap_executable,
        sparse_model,
        output_dir / "sparse.ply",
        project_dir,
        callback,
    )
    result = {
        "status": "sparse_only" if not run_dense else "sparse_completed_dense_pending",
        "project_directory": str(project_dir),
        "input_image_count": len(image_files),
        "sparse_model": str(sparse_model),
        "sparse_point_cloud": str(sparse_ply_path) if sparse_ply_path else None,
        "sparse_export_error": sparse_export_error,
        "dense_completed": False,
        "dense_error": None,
        "point_cloud": None,
        "mesh": None,
        "textured_model_directory": None,
    }

    if sparse_export_error:
        callback(f"Sparse model exists, but PLY export failed: {sparse_export_error}", None)

    if not run_dense:
        result["status"] = "sparse_completed"
        callback("Sparse reconstruction completed. Dense reconstruction was disabled; sparse.ply is available when exported.", 1.0)
        return result

    dense_dir.mkdir(parents=True, exist_ok=True)
    callback("Starting dense reconstruction. This stage generally needs a compatible CUDA-enabled COLMAP build/GPU.", 0.50)
    try:
        _run_command(
            colmap_executable,
            ["image_undistorter", "--image_path", str(image_dir), "--input_path", str(sparse_model),
             "--output_path", str(dense_dir), "--output_type", "COLMAP", "--max_image_size", str(max_image_size)],
            project_dir, callback, 0.55,
        )
        _run_command(
            colmap_executable,
            ["patch_match_stereo", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP",
             "--PatchMatchStereo.geom_consistency", "true"],
            project_dir, callback, 0.72,
        )
        fused_path = dense_dir / "fused.ply"
        _run_command(
            colmap_executable,
            ["stereo_fusion", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP",
             "--input_type", "geometric", "--output_path", str(fused_path)],
            project_dir, callback, 0.84,
        )
        if not fused_path.exists() or fused_path.stat().st_size == 0:
            raise ColmapError("COLMAP stereo_fusion completed but dense/fused.ply was not created.")
        result["point_cloud"] = str(fused_path)
    except ColmapError as exc:
        result["status"] = "sparse_completed_dense_failed"
        result["dense_error"] = str(exc)
        if "num_cuda_devices" in str(exc) or "CUDA" in str(exc).upper():
            callback(
                "Dense reconstruction stopped because COLMAP could not access a CUDA device. "
                "Your sparse model is still available in output/sparse.ply.",
                1.0,
            )
        else:
            callback(
                "Dense reconstruction failed, but the sparse model is still available in output/sparse.ply.",
                1.0,
            )
        return result

    mesh_path = dense_dir / "meshed-poisson.ply"
    try:
        _run_command(
            colmap_executable,
            ["poisson_mesher", "--input_path", str(fused_path), "--output_path", str(mesh_path)],
            project_dir, callback, 0.91,
        )
        if mesh_path.exists():
            result["mesh"] = str(mesh_path)
            textured_prefix = dense_dir / "textured"
            try:
                _run_command(
                    colmap_executable,
                    ["mesh_texturer", "--workspace_path", str(dense_dir), "--input_path", str(mesh_path),
                     "--output_path", str(textured_prefix)],
                    project_dir, callback, 0.97,
                )
                result["textured_model_directory"] = str(textured_prefix)
            except ColmapError as exc:
                callback(f"Texturing was skipped or failed; point cloud and mesh are still available. {exc}", None)
    except ColmapError as exc:
        callback(f"COLMAP meshing was skipped or failed; dense point cloud is still available. {exc}", None)

    result["dense_completed"] = True
    result["status"] = "dense_completed"
    callback("Dense reconstruction pipeline completed.", 1.0)
    return result
