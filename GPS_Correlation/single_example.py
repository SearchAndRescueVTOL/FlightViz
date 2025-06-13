import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import pandas as pd
import glob
def gps_to_ned(subject_gps, cam_gps):
    lat0, lon0, alt0 = cam_gps
    lat, lon, alt = subject_gps
    d_lat = (lat - lat0) * 111111
    d_lon = (lon - lon0) * 111111 * np.cos(np.radians(lat0))
    d_alt = alt0 - alt 
    return np.array([d_north := d_lat, d_east := d_lon, d_down := d_alt])

def project_to_image(point_cam, intrinsics):
    fx, fy, cx, cy = intrinsics
    print(point_cam)
    x, y, z = point_cam
    # y = -y # You can try inverting the final camera position coordinates 
    # x = -x
    # z = -z
    if z <= 0:
        return None 
    u = fx * x / z + cx
    v = fy * y / z + cy
    return int(u), int(v)

def draw_subjects_on_image(
    lat, lon, alt,
    camera_quat_wxyz,
    subject_gps_list,
    camera_intrinsics,
    image_path,
    save_path
):
    fx, fy, cx, cy, width, height = camera_intrinsics
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image = cv2.resize(image, (width, height))
    output = image.copy()

    w, x, y, z = camera_quat_wxyz
    quat = np.array([x, y, z, w])
    r_frd_to_ned = R.from_quat(quat)
    flip_rot = R.from_euler('xy', [0,0], degrees=True) # Play aroudn with the rotation of the quaternion here
    r_frd_to_ned = flip_rot * r_frd_to_ned
    ned_to_optical = np.array([ 
        [1, 0,  0],  
        [0, 0,  1],  
        [0, -1, 0]   
    ])
    r_cam_to_optical = ned_to_optical @ r_frd_to_ned.as_matrix()

    cam_gps = [lat, lon, alt]

    for subj_gps in subject_gps_list:
        ned = gps_to_ned(subj_gps, cam_gps)
        cam_point = r_cam_to_optical.T @ ned 

        pixel = project_to_image(cam_point, (fx, fy, cx, cy))
        print(pixel)
        if pixel is not None:
            u, v = pixel
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(output, (u, v), 40, (0, 0, 255), -1)
    print("saved image")
    cv2.imwrite(save_path, output)


if __name__ == "__main__":
    path_to_clicks = "06_06_flight2_clicks.csv"
    camera_points_path = "06_06_FLIGHT2_CAM_GPS.txt"
    RGBdir = 'C:/Users/tqpat/OneDrive/Documents/HARELAB/06_06_FLIGHT2/06_06_FLIGHT2_RGB/'
    save_path = "C:/Users/tqpat/OneDrive/Documents/HARELAB/06_06_FLIGHT2/projections/{number}.tiff"

    fx = 2704.30108
    fy = 2705.88235
    cx = 2012.0
    cy = 1518.0
    width = 4024
    height = 3036
    camera_intrinsics = (fx, fy, cx, cy, width, height)
    people = pd.read_csv(path_to_clicks, header=None)
    people_np = people.iloc[:,:2]
    people_np['alt'] = 20
    people_np = people_np.to_numpy()
    print(people_np)
    camera_points = pd.read_csv(camera_points_path, header=None)
    result = []
    temp = f'{RGBdir}/*.tiff'
    file_list= glob.glob(temp) 
    for file in file_list:
        result.append(file)

    
    for index,row in camera_points.iterrows():
        num, lat, lon, alt, _,_,dead_reck, timestamp, w, x, y, z, _, _, _, _ = (row.tolist())
        num = int(num)
        camera_quat_wxyz = (w, x, y, z)
        save_path_temp = save_path.format(number=str(num))
        draw_subjects_on_image(lat, lon, alt, 
            camera_quat_wxyz, 
            people_np,# The subject gps  
            camera_intrinsics, # a tuple of fx,fy,cx,cy,width,height
            result[num - 5],
            save_path=save_path_temp
        )

