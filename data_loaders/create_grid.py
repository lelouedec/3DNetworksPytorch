import open3d as opend
import sys
import numpy as np
import time


class Grid ():
    def __init__(self,points,colors,cell_size,radius,annotations):
        self.corners = []
        self.centers = []
        self.cells = []
        self.assigned = []
        self.colors = []
        self.grid_lines = []
        self.annotations = []
        self.create_grid(points,colors,cell_size,radius,annotations)
        # self.attribute_points( points)

    def create_grid(self,points,colors,cell_size,radius,annotations):

        max_width = np.amax(points[:, 0])
        min_width = np.amin(points[:, 0])

        min_height = np.amin(points[:, 1])
        max_height = np.amax(points[:, 1])

        max_depth = np.amax(points[:, 2])
        min_depth = np.amin(points[:, 2])
        print(max_width,min_width)

        counter = 0

        for i in range(int(min_width * cell_size), int(max_width * cell_size) + 1):
            for j in range(int(min_height * cell_size) - 1, int(max_height * cell_size) + 1):
                for k in range(int(min_depth * cell_size), int(max_depth * cell_size) + 2 ):
                    if (i < int(max_width * cell_size) + 1  and j  < int(max_height * cell_size) + 1 and k + 1 < int(max_depth * cell_size)):
                        self.grid_lines.append([counter, counter + 1])
                    if (i  < int(max_width * cell_size) +1 and j +1  < int(max_height * cell_size) and k + 1 < int(max_depth * cell_size)):
                        self.grid_lines.append([counter, counter + (int(max_height * cell_size) ) +  int(max_depth * cell_size) ])
                    if (i < int(max_width * cell_size) + 1 and j < int(max_height * cell_size) + 1 and k < int(max_depth * cell_size) + 2):
                        self.cells.append([i / cell_size, j / cell_size, k / cell_size, (i + 1) / cell_size, (j + 1) / cell_size,
                                           (k + int((max_depth - min_depth) * 2)) / cell_size])
                        self.corners.append(([i / cell_size, j / cell_size, k / cell_size]))
                        counter = counter + 1

        print(self.corners)
        print(self.cells)

        ########################################################################################################################################
        ####################Create group for each center of cell #############################################################################
        ######################################################################################################################################
        pcd2 = opend.PointCloud()
        pcd2.points = opend.Vector3dVector(points)
        pcd2.colors = opend.Vector3dVector(colors)
        # opend.draw_geometries([pcd2])
        pcd_tree = opend.KDTreeFlann(pcd2)
        counter = 0
        nb_pt = 0
        for i,c in enumerate(self.cells):
            center =  np.array([c[0] + (c[3]-c[0])/2, c[1] + (c[4]-c[1])/2 , c[2] + (c[5]-c[2])/2])
            [k, idx, _] = pcd_tree.search_radius_vector_3d(center,radius)
            if(np.asarray(idx).shape[0]>256):
                counter = counter + 1
                nb_pt = nb_pt + np.asarray(idx).shape[0]
                self.assigned.append( points[np.array(idx)] )
                self.colors.append(colors[np.array(idx)])
                self.annotations.append(annotations[np.array([idx])].squeeze(0))
                self.centers.append(center)


        ########################################################################################################################################
        ####################Fuse neigboring cells if their number of point is too low #########################################################
        ########################################################################################################################################
        pcd3 = opend.PointCloud()
        pcd3.points = opend.Vector3dVector(self.centers)
        pcd_tree = opend.KDTreeFlann(pcd3)
        deleted = []
        for j,pg in enumerate(self.assigned):
            center = self.centers[j]
            [k, idx, _] = pcd_tree.search_knn_vector_3d(center, 3)
            if(j not in deleted):
                if (pg.shape[0] <= 1024):
                    cell_values      = self.assigned[idx[0]]
                    next_cell_values = self.assigned[idx[1]]
                    next_next_cell_values = self.assigned[idx[2]]
                    if (cell_values.shape[0] + next_cell_values.shape[0] <= 8000):
                        self.assigned[j] = np.concatenate([self.assigned[j], next_cell_values], 0)
                        self.colors[j] = np.concatenate([self.colors[j], self.colors[idx[1]]], 0)
                        self.annotations[j] = np.concatenate([self.annotations[j], self.annotations[idx[1]]],0)
                        deleted.append(idx[1])
                        if (cell_values.shape[0] + next_cell_values.shape[0] <= 1024):
                            if (cell_values.shape[0] + next_cell_values.shape[0] + next_next_cell_values.shape[0] <= 8000):
                                self.assigned[j] = np.concatenate([self.assigned[j], next_next_cell_values], 0)
                                self.colors[j] = np.concatenate([self.colors[j], self.colors[idx[2]]], 0)
                                self.annotations[j] = np.concatenate([self.annotations[j], self.annotations[idx[2]]], 0)
                                deleted.append(idx[2])
        res_assigned = []
        res_annotations = []
        for j,p in enumerate(self.assigned):
           if(j not in deleted):
               res_assigned.append(self.assigned[j])
               res_annotations.append(self.annotations[j])

        self.assigned = res_assigned
        self.annotations = res_annotations

    def display(self):

        self.colors = []
        distribcr = np.random.randint(low=0, size=len(self.assigned), high=len(self.assigned))/len(self.assigned)
        distribcg = np.random.randint(low=0, size=len(self.assigned), high=len(self.assigned)) / len(self.assigned)
        distribcb = np.random.randint(low=0, size=len(self.assigned), high=len(self.assigned)) / len(self.assigned)
        for i,jk in enumerate(self.assigned):
            self.colors.append(np.array([[distribcr[i], distribcg[i], distribcb[i]] for k in range(jk.shape[0])]))
        opend.draw_geometries([pcd2])

    def attribute_points(self,points):
        for c in self.cells:
            cellule = []
            deleted = []
            for i, p in enumerate(points):
                # print(i,p)
                if(p[0]>=c[0] and p[0]<=c[3] and p[1]>=c[1] and p[1]<=c[4] and p[2]>=c[2] and p[0]<=c[5]):
                    cellule.append(p)
                    deleted.append(i)
            points = np.delete(points,deleted,0)
            self.assigned.append(cellule)






if __name__ == "__main__":

    # path = sys.argv[1]
    #
    #
    # pcd = opend.read_point_cloud(path)
    points = np.random.rand(20048,3)
    colors = np.random.rand(20048,3)
    # pcd2 = opend.PointCloud()
    # pcd2.points = opend.Vector3dVector(points)
    # pcd2.colors = opend.Vector3dVector(colors)
    # opend.draw_geometries([pcd2])

    # points = np.asarray(pcd.points)
    start_time = time.time()
    annotation = np.ones((points.shape[0],1))
    grid = Grid(points,colors, 0.05, 0.025,annotation)#points,colors,cell_size,radius,annotations

    print(time.time() - start_time)
    print(grid.assigned)
    pcd2 = opend.PointCloud()
    pcd2.points = opend.Vector3dVector(np.concatenate(grid.assigned)+np.array([1.0,0.0,0.0]))
    pcd2.colors = opend.Vector3dVector(np.concatenate(grid.colors))

    pcd3 = opend.PointCloud()
    pcd3.points = opend.Vector3dVector(np.array(grid.corners))

    line_set = opend.LineSet()
    line_set.points = opend.Vector3dVector(np.array(grid.corners))
    line_set.lines = opend.Vector2iVector(np.array(grid.grid_lines))
    opend.draw_geometries([pcd,pcd2,pcd3,line_set])
