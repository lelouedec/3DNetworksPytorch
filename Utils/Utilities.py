import numpy as np
import numba



@jit(nopython=True)
def aligne_depth(depth_raw,rowCount,columnCount,depth_aligned,focal_length, principal_point, color_focal_length, color_principal_point,extrinsic):
    for depth_y in range(0,rowCount):
        for depth_x in range(0, columnCount ):
            ## depth to meters
            z =  depth_raw[depth_y][depth_x]   * 0.001
            if(z!=0 and z <= 3):
                ##### top left corner to rgb image
                ## map depth map to 3D point
                x_a = (depth_x - 0.5 - principal_point[0] )  /  focal_length[0]
                y_a = (depth_y - 0.5 - principal_point[1] ) /  focal_length[1]

                ## Rotate and translate 3D point into RGB point of view
                point1 = np.array([x_a*z,y_a*z,z, 1.0])
                point1 = np.dot(extrinsic,point1)
                point1 = point1[:3] / point1[3]

                #### mapping 3D depth points to RGB image  ###
                rgb_x1 = point1[0] / point1[2]
                rgb_y1 = point1[1] / point1[2]
                rgb_x_1 =  color_focal_length[0] * rgb_x1  + color_principal_point[0]
                rgb_y_1 =  color_focal_length[1] * rgb_y1  + color_principal_point[1]

                rgb_x_1 =  int(rgb_x_1 + 0.5)
                rgb_y_1 =  int(rgb_y_1 + 0.5 )


                ## Bottom right corner to rgb image
                ## map depth map to 3D point
                x_b = (depth_x + 0.5 - principal_point[0])  /  focal_length[0]
                y_b = (depth_y + 0.5 - principal_point[1] ) /  focal_length[1]

                ## Rotate and translate 3D point into RGB point of view
                point2 = np.array([x_b *z ,y_b *z ,z,1.0])
                point2 = np.dot(extrinsic,point2)
                point2 = point2[:3] / point2[3]

                #### mapping 3D depth points to RGB image  ###
                rgb_x2 = point2[0] /point2[2]
                rgb_y2 = point2[1] /point2[2]
                rgb_x_2 =  color_focal_length[0] * rgb_x2  + color_principal_point[0]
                rgb_y_2 =  color_focal_length[1] * rgb_y2  + color_principal_point[1]

                rgb_x_2 =  int(rgb_x_2 + 0.5)
                rgb_y_2 =  int(rgb_y_2 + 0.5)

                if(rgb_x_1 > 0 and rgb_y_1 > 0  and rgb_y_2 < len(depth_aligned) and  rgb_x_2 < len(depth_aligned[0]) ):
                    for a in range(rgb_y_1,rgb_y_2+1):
                        for b in range(rgb_x_1,rgb_x_2+1):
                            if(depth_aligned[a][b] != 0):
                                depth_aligned[a][b] = min(depth_aligned[a][b], depth_raw[depth_y][depth_x] )
                            else:
                                depth_aligned[a][b] = depth_raw[depth_y][depth_x]
    return depth_aligned
