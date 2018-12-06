import glob,os
import numpy as np
from vispy import visuals, app, scene
from vispy.util.transforms import perspective, translate, rotate
from random import uniform
import binascii
import struct
# from plyfile import PlyData, PlyElement
import numpy as np



def OpenOffFile(path):
    with open(path, "r") as fp:
         line = fp.readline()
         line = fp.readline()
         line = line.split(" ")
         nbOflines = int(line[0])
         line = fp.readline()
         cnt = 1
         points = np.empty((nbOflines,3))
         print(points.shape)
         while (cnt<=nbOflines):
             points[cnt-1] = (float(line.split(" ")[0]), float(line.split(" ")[1]), float(line.split(" ")[2]))
             line = fp.readline()
             cnt += 1
    return points

def OpenOBJFile(path):
    points = []
    with open(path, "r") as fp:
        # line = fp.readline()
        for line in fp :
            if (line.split()):
                if(line.split()[0] == 'v' or line.split()[0] == 'V'):
                    points.append((float(line.split()[1]), float(line.split()[2]), float(line.split()[3])))
                    # line = fp.readline()
    return np.asarray(points)

def OpenFLYFile(path):
    points = []
    with open(path, "r") as fp:
        # line = fp.readline()
        for line in fp :
            if (line.split()):
                points.append((float(line.split()[0]), float(line.split()[1]), float(line.split()[2])))
                    # line = fp.readline()
    return np.asarray(points)

def DisplayPointCloud(points,colors):
    #
    # Make a canvas and add simple view
    #
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    #create scatter object and fill in the data
    Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
    scatter = Scatter3D(parent=view.scene)
    scatter.set_gl_state('translucent', blend=True, depth_test=True)
    scatter.set_data(points,face_color=colors,  symbol='o',size=15)

    view.add(scatter)

    view.camera = 'arcball' #scene.TurntableCamera(elevation=30, azimuth=30, up='+y')
#'fly'  #dict_keys([None, \'base\', \'panzoom\', \'perspective\', \'turntable\', \'fly\', \'arcball\'])'


    # add a colored 3D axis for orientation
    axis = scene.visuals.XYZAxis(parent=view.scene)
    app.run()

def importbinaryply(path):
    plydata = PlyData.read(path)
    x = plydata.elements[0].data['x']
    y = plydata.elements[0].data['y']
    z = plydata.elements[0].data['z']
    red = plydata.elements[0].data['red']
    green = plydata.elements[0].data['green']
    blue = plydata.elements[0].data['blue']
    points = np.stack((x,y,z),axis=1)
    #print(points.shape)
    colors = np.stack((red/255,green/255,blue/255),axis=1)
    return points,colors


if __name__ == "__main__":

    #pointCloud = OpenOffFile("./bathtub_0090.off")
    # pointCloud = OpenOBJFile("./Strawberry.obj")
    # # pointCloud = OpenOBJFile("./Bberry.obj")
    # # pointCloud2 = OpenOBJFile("./leaves.obj")
    # #pointCloud = np.concatenate((pointCloud,pointCloud2),axis=0)
    # # pointCloud = OpenFLYFile("./sunflower-take0.ply")
    # # n = pointCloud.shape[0]
    # # pointCloud2 = OpenOBJFile("./Strawberry.obj")
    # # n2 = pointCloud2.shape[0]
    # # color2 = np.array([[1, 0.4, 0.6]] * pointCloud2.shape[0])
    # # color3 = np.array([[1, 0.4, 0]] * pointCloud.shape[0])
    # # colors = np.vstack((color1, color2, color3))
    # # print("dsfsfsdfs")
    # # cloud3 = np.copy(pointCloud)
    # # pointCloud3 = np.vstack((pointCloud*10,(pointCloud2*1000)+5000))
    # #
    # # pointCloud3 = np.vstack((pointCloud3,(cloud3*10)+5000))
    # # print(colors.shape)
    # # print(pointCloud3.shape)
    # # for i in range(0,10):
    # #     pointCloud3 = np.vstack((pointCloud3,(cloud3*10) +uniform(5000,60000) ))
    # #     colors = np.vstack((colors, color3))
    # #     pointCloud3 = np.vstack((pointCloud3,(pointCloud2*1000) +uniform(5000,60000) ))
    # #     colors = np.vstack((colors, color2))
    # pointCloud,colors = importbinaryply("./08_54_36_127/point_cloud.ply")
    # print(pointCloud.shape)
    # pointCloud = np.clip(pointCloud,0,20)
    # print(pointCloud.shape)
    # #colors = np.array([[1, 1, 1]] * pointCloud.shape[0])
    # print(colors.shape)
    # for i in range(0,pointCloud.shape[0]):
    #     if(pointCloud[i][1]<0):
    #         print(pointCloud[1][2])
    # DisplayPointCloud(pointCloud,colors)
    print("coucou")
