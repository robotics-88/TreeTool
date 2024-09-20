import pclpy
import numpy as np
import treetool.seg_tree as seg_tree
import treetool.utils as utils
import treetool.tree_tool as tree_tool
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

PointCloud = pclpy.pcl.PointCloud.PointXYZ()

file_directory = r'data/utm.pcd'
pclpy.pcl.io.loadPCDFile(file_directory,PointCloud)
PointCloudV = seg_tree.voxelize(PointCloud.xyz,0.06)
# utils.open3dpaint(PointCloudV, reduce_for_vis = False  , voxel_size = 0.5)

PointCloudV.shape, PointCloud.xyz.shape


My_treetool = tree_tool.treetool(PointCloudV)

My_treetool.step_1_remove_floor()

#Obtained attributes:
#non_ground_cloud: All points in the point cloud that don't belong to the ground
#ground_cloud: All points in the point cloud that belong to the ground
# utils.open3dpaint([My_treetool.non_ground_cloud,My_treetool.ground_cloud],reduce_for_vis = True  , voxel_size = 0.1)

#Get point normals for filtering

#Obtained attributes:
#non_filtered_points: Same as non_ground_cloud
#non_filtered_normals: Normals of points in non_filtered_points
#filtered_points: Points that pass the normal filter
#filtered_normals: Normals of points that pass the normal filter
My_treetool.step_2_normal_filtering(verticality_threshold=0.04, curvature_threshold=0.06, search_radius=0.12)
# utils.open3dpaint([My_treetool.non_ground_cloud.xyz, My_treetool.non_filtered_points.xyz + My_treetool.non_filtered_normals * 0.1, My_treetool.non_filtered_points.xyz + My_treetool.non_filtered_normals * 0.2], reduce_for_vis = True , voxel_size = 0.1)

# utils.open3dpaint([My_treetool.filtered_points.xyz, My_treetool.filtered_points.xyz + My_treetool.filtered_normals * 0.05, My_treetool.filtered_points.xyz + My_treetool.filtered_normals * 0.1], reduce_for_vis = True , voxel_size = 0.1)

My_treetool.step_3_euclidean_clustering(tolerance=0.2, min_cluster_size=40, max_cluster_size=6000000)

#Obtained attributes:
#cluster_list: List of all clusters obtained with Euclidean Clustering

# utils.open3dpaint(My_treetool.cluster_list,reduce_for_vis = True  , voxel_size = 0.1)

#Group stem segments
My_treetool.step_4_group_stems(max_distance=0.4)

#Obtained attributes:
#complete_Stems: List of all complete stems obtained by joining clusters belonging to the same tree
            
# utils.open3dpaint(My_treetool.complete_Stems,reduce_for_vis = True  , voxel_size = 0.1)

My_treetool.step_5_get_ground_level_trees(lowstems_height=1, cutstems_height=10)

#Obtained attributes:
#low_stems: List of all stems truncated to the specified height

# utils.open3dpaint(My_treetool.low_stems,reduce_for_vis = True  , voxel_size = 0.1)


My_treetool.step_6_get_cylinder_tree_models(search_radius=0.1)

#Obtained attributes:
#finalstems: List of Dictionaries with two keys 'tree' which contains the points used to fit the cylinder model and 'model' which contains the cylinder model parameters
#visualization_cylinders: List of the pointclouds that represent the tree modeled with a cylinder

# utils.open3dpaint([i['tree'] for i in My_treetool.finalstems] + My_treetool.visualization_cylinders,reduce_for_vis = True  , voxel_size = 0.1)
  

My_treetool.step_7_ellipse_fit()

#Obtained attributes:
#Three new keys in our finalstems dictionaries:
#final_diameter: Final DBH of every tree
#cylinder_diameter: DBH obtained with cylinder fitting
#ellipse_diameter;DBH obtained with Ellipse fitting

My_treetool.save_results(save_location = 'results/myresults.csv')

PointCloud = pclpy.pcl.PointCloud.PointXYZ()
pclpy.pcl.io.loadPCDFile('data/utm.pcd',PointCloud)
PointCloudV = seg_tree.voxelize(PointCloud.xyz,0.05)
utils.open3dpaint(PointCloudV, reduce_for_vis = True  , voxel_size = 0.1)

My_treetool.set_point_cloud(PointCloudV)

My_treetool.full_process(verticality_threshold=0.04,
    curvature_threshold=0.06,
    tolerance=0.1,
    min_cluster_size=40,
    max_cluster_size=6000000,
    max_distance=0.4,
    lowstems_height=5,
    cutstems_height=5,
    search_radius=0.1)

cloud_match = [i['tree'] for i in My_treetool.finalstems]+[i for i in My_treetool.visualization_cylinders]
utils.open3dpaint(cloud_match+[PointCloudV], voxel_size = 0.1)

#####################################################
# #Get ground truth
# tree_data = pd.read_csv('data/TLS_Benchmarking_Plot_3_LHD.txt',sep = '\t',names = ['x','y','height','DBH'])
# Xcor,Ycor,diam = tree_data.iloc[0,[0,1,3]]
# cylinders_from_GT = [utils.makecylinder(model=[Xcor, Ycor, 0,0,0,1,diam/2],height=10,density=20)]
# TreeDict = [np.array([Xcor,Ycor,diam])]
# for i,rows in tree_data.iloc[1:].iterrows():
#     Xcor,Ycor,diam = rows.iloc[[0,1,3]]
#     if not np.any(np.isnan([Xcor,Ycor,diam])):
#         cylinders_from_GT.append(utils.makecylinder(model=[Xcor, Ycor, 0,0,0,1,diam/2],height=10,density=10))
#         TreeDict.append(np.array([Xcor,Ycor,diam]))
# cloud_of_cylinders_from_GT = [p for i in cylinders_from_GT for p in i]

# #DataBase
# #Found trees
# #Hungarian Algorithm assignment
# CostMat = np.ones([len(TreeDict),len(My_treetool.visualization_cylinders)])
# for X,datatree in enumerate(TreeDict):
#     for Y,foundtree in enumerate(My_treetool.finalstems):
#         CostMat[X,Y] = np.linalg.norm([datatree[0:2]-foundtree['model'][0:2]])

# dataindex, foundindex = linear_sum_assignment(CostMat,maximize=False)

# #Get metrics
# locationerror = []
# correctlocationerror = []
# diametererror = []
# diametererrorElipse = []
# diametererrorComb = []
# cloud_match = []
# for i,j in zip(dataindex, foundindex):
#     locationerror.append(np.linalg.norm((My_treetool.finalstems[j]['model'][0:2]-TreeDict[i][0:2])))
#     if locationerror[-1]<0.6:
#         if My_treetool.finalstems[j]['cylinder_diameter'] is not None:
#             diametererror.append(abs(My_treetool.finalstems[j]['cylinder_diameter']-TreeDict[i][2]))        
#             diametererrorElipse.append(abs(My_treetool.finalstems[j]['ellipse_diameter']-TreeDict[i][2]))        
#             mindi = max(My_treetool.finalstems[j]['cylinder_diameter'],My_treetool.finalstems[j]['ellipse_diameter'])
#             mendi = np.mean([My_treetool.finalstems[j]['cylinder_diameter'],My_treetool.finalstems[j]['ellipse_diameter']])
#             diametererrorComb.append(abs(mindi-TreeDict[i][2]))
#             correctlocationerror.append(np.linalg.norm((My_treetool.finalstems[j]['model'][0:2]-TreeDict[i][0:2])))
#             cloud_match.append(np.vstack([cylinders_from_GT[i],My_treetool.finalstems[j]['tree'],My_treetool.visualization_cylinders[j]]))


# n_ref = len(TreeDict)
# n_match = (len(diametererror))
# n_extr = len(locationerror) - n_match

# Completeness = n_match/n_ref
# Correctness = n_match/(n_extr+n_match)

# plt.figure(figsize = (20,6))
# plt.subplot(1,3,1)
# plt.hist(diametererror,50)
# plt.title('Cylinder DBH error')

# plt.subplot(1,3,2)
# plt.hist(diametererrorComb,50)
# plt.title('Final DBH error')

# plt.subplot(1,3,3)
# plt.hist(correctlocationerror,50)
# plt.title('Location error')

# print('Total number of trees in Ground Truth: ', n_ref)
# print('Total number of trees matched with Ground Truth: ', n_match)
# print('Total number of trees extra trees found: ', n_extr)

# print('Percentage of matched trees: ', round(Completeness*100), '%')
# print('Percentage of correctly matched trees: ', round(Correctness*100), '%')

# print('Cylinder DBH mean Error: ', np.mean(diametererror),)
# print('Ellipse DBH mean Error: ', np.mean(diametererrorElipse))
# print('Final DBH mean Error: ', np.mean(diametererrorComb))