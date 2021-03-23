import os
import sys

import numpy as np
import cv2

try:
    import queue
except ImportError:
    import Queue as queue

sys.path.append("../simulator/carla/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg")

import carla


def compute_data_buffer(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array

def compute_depth_from_buffer(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    R, G, B = array[:, :, 2], array[:, :, 1], array[:, :, 0]
    out = (R.astype(np.uint32) + G.astype(np.uint32)*256 + B.astype(np.uint32)*256*256).astype(np.float64)/(256.0*256.0*256.0 - 1.0)
    return out.astype(np.float32)
    

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        print('CarlaSyncMode.enter')
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        print('CarlaSyncMode.exit')
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on cv2 image.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """
        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """
        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """
        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """
        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords 
    
    @staticmethod
    def draw_bounding_boxes(image, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        for bbox in bounding_boxes:
            imgtype = 'uint8'
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            image = cv2.line(image.astype(imgtype), points[0], points[1], (0, 255, 0), 1)
            image = cv2.line(image.astype(imgtype), points[1], points[2], (0, 255, 0), 1)
            image = cv2.line(image.astype(imgtype), points[2], points[3], (0, 255, 0), 1)
            image = cv2.line(image.astype(imgtype), points[3], points[0], (0, 255, 0), 1)
            # top
            image = cv2.line(image.astype(imgtype), points[4], points[5], (0, 255, 0), 1)
            image = cv2.line(image.astype(imgtype), points[5], points[6], (0, 255, 0), 1)
            image = cv2.line(image.astype(imgtype), points[6], points[7], (0, 255, 0), 1)
            image = cv2.line(image.astype(imgtype), points[7], points[4], (0, 255, 0), 1)
            
            # base-top
            image = cv2.line(image.astype(imgtype), points[0], points[4], (0, 255, 0), 1)
            image = cv2.line(image.astype(imgtype), points[1], points[5], (0, 255, 0), 1)
            image = cv2.line(image.astype(imgtype), points[2], points[6], (0, 255, 0), 1)
            image = cv2.line(image.astype(imgtype), points[3], points[7], (0, 255, 0), 1)
        
        return image

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


def create_camera_sensors(world, vehicle, params, create_all_types=False):
    """
        creates rgb, semantic and depth cameras
        
        parameters
        ==========
        
            world: carla world object instance
            vehicle: actor to attach the sensors
            
            params: dict with
                - width: image camera width
                - height: image camera height
                - fov: camera field of view
                - x, y, z, roll, pitch, yaw: position of the sensor

            create_all_types: bool - create semantic segmentation and depth camera
        return
        ======
            
            [camera_rgb, camera_semseg, camera_depth] list with an rgb, semantic seg and depth camera object instance
        
    """
    blueprint_library = world.get_blueprint_library()
    transform = carla.Transform(carla.Location(x=params['x'], y=params['y'], z=params['z']),
                                carla.Rotation(pitch=params['pitch'], roll=params['roll'], yaw=params['yaw']))
    
    calibration = np.identity(3)
    calibration[0, 2] = params['width'] / 2.0
    calibration[1, 2] = params['height'] / 2.0
    calibration[0, 0] = calibration[1, 1] = params['width'] / (2.0 * np.tan(params['fov'] * np.pi / 360.0))
    
    output = []

    if create_all_types:
        sensors = ['sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'sensor.camera.depth']
    else:
        sensors = ['sensor.camera.rgb']

    for sensor_name in sensors:
        bp = blueprint_library.find(sensor_name)
        bp.set_attribute('image_size_x', str(params['width']))
        bp.set_attribute('image_size_y', str(params['height']))
        bp.set_attribute('fov', str(params['fov']))
        
        cam = world.spawn_actor(bp, transform, attach_to=vehicle)
        cam.calibration = calibration
        output.append(cam)
    
    return output


def create_lidar_sensor(world, vehicle, params):
    """
        create a lidar sensor
        
        parameters
        ==========
            
            world: carla world object instance
            vehicle: actor to attach the sensors
            
            params: dict with
                - channels: number of lines
                - range: maximum range
                - upper_fov: Angle in degrees of the upper most laser
                - lower_fov: Angle in degrees of the lower most laser
                - x, y, z, roll, pitch, yaw: position of the sensor
            
        return
        ======
            
            lidar_sensor: carla lidar sensor
    """
    blueprint_library = world.get_blueprint_library()
    transform = carla.Transform(carla.Location(x=params['x'], y=params['y'], z=params['z']),
                                carla.Rotation(pitch=params['pitch'], roll=params['roll'], yaw=params['yaw']))
    
    bp = blueprint_library.find('sensor.lidar.ray_cast')

    bp.set_attribute('channels', str(params['channels']))
    bp.set_attribute('rotation_frequency', str(params['rotation_frequency']))
    bp.set_attribute('points_per_second', str(params['points_per_second']))
    bp.set_attribute('range', str(params['range']))

    # bp.set_attribute('channels', str(params['channels']))
    # bp.set_attribute('points_per_second', str(params['points_per_second']))
    # bp.set_attribute('range', str(params['range']))
    bp.set_attribute('upper_fov', str(params['upper_fov']))
    bp.set_attribute('lower_fov', str(params['lower_fov']))

    return world.spawn_actor(bp, transform, attach_to=vehicle)


def create_gnss_sensor(world, vehicle, params):
    blueprint_library = world.get_blueprint_library()
    transform = carla.Transform(carla.Location(x=params['x'], y=params['y'], z=params['z']),
                                carla.Rotation(pitch=params['pitch'], roll=params['roll'], yaw=params['yaw']))
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    return world.spawn_actor(gnss_bp, transform, attach_to=vehicle)


def create_imu_sensor(world, vehicle, params):
    blueprint_library = world.get_blueprint_library()
    transform = carla.Transform(carla.Location(x=params['x'], y=params['y'], z=params['z']),
                                carla.Rotation(pitch=params['pitch'], roll=params['roll'], yaw=params['yaw']))
    return world.spawn_actor(blueprint_library.find('sensor.other.imu'), transform, attach_to=vehicle)


def sensor_factory(world, vehicle, sensor_list, create_all_camera_types=False):
    """
        creates and attach sensors to a vehicle

        parameters
        ==========

            sensor_list: list with sensor parameters
            create_all_camera_types: create all types of cameras if true (rgb, semantic segmentation and depth)

        return
        ======

            sensor_actors: list with all sensor actors attached to the vehicle
            sensor_labels: list with all sensor labels
    """
    sensor_actors = []
    sensor_labels = []

    for params in sensor_list:
        
        if params['sensor_type'] == 'camera':
            print(">>> Creating camera", params)
            sensor_actors += create_camera_sensors(world, vehicle, params, create_all_camera_types)
            
            if create_all_camera_types:
                sensor_labels += ['rgb_' + params['sensor_label'], 'semseg_' + params['sensor_label'], 'depth_' + params['sensor_label']]
            else:
                sensor_labels += ['rgb_' + params['sensor_label']]
        
        elif params['sensor_type'] == 'lidar':
            print(">>> Creating LIDAR", params)
            sensor_actors.append(create_lidar_sensor(world, vehicle, params))
            sensor_labels += ['lidar_' + params['sensor_label']]
        
        elif params['sensor_type'] == 'gnss':
            print(">>> Creating GNSS", params)
            sensor_actors.append(create_gnss_sensor(world, vehicle, params))
            sensor_labels += ['gnss_' + params['sensor_label']]
        
        elif params['sensor_type'] == 'imu':
            print(">>> Creating IMU", params)
            sensor_actors.append(create_imu_sensor(world, vehicle, params))
            sensor_labels += ['imu_' + params['sensor_label']]
    
    return sensor_actors, sensor_labels


## Pose and Transform

def lidar_measurement_to_np_array(lidar_measurement):
    data = list()
    for location in lidar_measurement:
        data.append([location.x, location.y, location.z])            
    return np.array(data).reshape((-1, 3))


def get_rotation_translation_matrix(x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def get_calibration_matrix(params):
    calibration = np.identity(3)
    calibration[0, 2] = params['width'] / 2.0
    calibration[1, 2] = params['height'] / 2.0
    calibration[0, 0] = calibration[1, 1] = params['width'] / (2.0 * np.tan(params['fov'] * np.pi / 360.0))
    return calibration



def get_bbox_vehicle_in_world(vehicle):

    bbox = ClientSideBoundingBoxes._create_bb_points(vehicle)
    
    vehicle 

    return None


def compute_bouding_boxes(vehicles, 
                          lidar_params, lidar_measurement,
                          camera_params, camera_image, camera_sensor):
    """
    - a vehicle's bouding box is constant
    - the pose of all vehicles are in world coordinate system
    """
    calibration_matrix = get_calibration_matrix(camera_params)

    ego_vehicle = vehicles[0]
    co_vehicle = vehicles[1]

    image = compute_data_buffer(camera_image)
    points = lidar_measurement_to_np_array(lidar_measurement)

    bbox_ego = ClientSideBoundingBoxes._create_bb_points(ego_vehicle)
    bbox_co = ClientSideBoundingBoxes._create_bb_points(co_vehicle)
    
    print('ego pose', ego_vehicle.get_transform())
    print('bb  pose', ego_vehicle.bounding_box.location)
    print('cam pose', camera_sensor.get_transform())
    print('diff', ego_vehicle.get_transform().location.y-camera_sensor.get_transform().location.y )

    bb_cords = bbox_ego

    # Convert bouding box to vehicle coordinate

    # bbox to vehicle
    bGv = get_rotation_translation_matrix(ego_vehicle.bounding_box.location.x,
                                          ego_vehicle.bounding_box.location.y,
                                          ego_vehicle.bounding_box.location.z)

    # vehicle to world
    vGw = get_rotation_translation_matrix(ego_vehicle.get_transform().location.x,
                                          ego_vehicle.get_transform().location.y,
                                          ego_vehicle.get_transform().location.y,
                                          ego_vehicle.get_transform().rotation.roll,
                                          ego_vehicle.get_transform().rotation.pitch,
                                          ego_vehicle.get_transform().rotation.yaw)
    # print(vGw - bGv)


    #print('co  pose', co_vehicle.get_transform(),)

    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord('q'):
        return False

    return True