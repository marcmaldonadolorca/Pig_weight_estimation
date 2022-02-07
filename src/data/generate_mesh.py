import open3d as o3d;
import glob
import numpy as np


filenames = glob.glob("data/processed/pcd/*.ply")

for i in filenames:
    pcd = o3d.io.read_point_cloud(i)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(1)


    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=15)

    densities = np.asarray(densities)
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals

    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh = mesh.compute_vertex_normals().paint_uniform_color([1, 0.706, 0])


    o3d.io.write_triangle_mesh('data/processed/ply/'+ i.split('/')[-1], mesh)


# Previous code to get the mesh with alpha shape
'''''''''''''''
    pcd = o3d.io.read_point_cloud(i)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 15)
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                              vertex_normals=np.asarray(mesh.vertex_normals))

    trimesh.repair.fill_holes(tri_mesh)
    tri_mesh.export('data/processed/ply/'+i.split('/')[-1])
'''''''''''''''
# Previous code to get the mesh with ball pivoting
'''''''''''''''
    pcd = o3d.io.read_point_cloud(i)
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location(pcd.get_center())

    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([pcd, rec_mesh])

    pcd = o3d.io.read_point_cloud(i)
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location(pcd.get_center())


    # Or to flip normals to make the point cloud outward, not mandatory
    pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

    # Surface reconstruction using Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = 15)
    mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
    vertices_to_remove = densities < np.quantile(densities, 0.04)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    o3d.io.write_triangle_mesh('data/processed/ply/'+i.split('/')[-1], mesh)


pcd = o3d.io.read_point_cloud(filenames[0])
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(100)
radii = [2, 2]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))

tri_mesh = trimesh.Trimesh(np.asarray(rec_mesh.vertices), np.asarray(rec_mesh.triangles),
                          vertex_normals=np.asarray(rec_mesh.vertex_normals))

trimesh.repair.fill_holes(tri_mesh)
tri_mesh.export('data/processed/ply/'+filenames[0].split('/')[-1])
#o3d.io.write_triangle_mesh('data/processed/ply/'+filenames[0].split('/')[-1], rec_mesh)
'''''''''''''''