import numpy as np
import tools.data_util as du
import tools.training_util as tu

import carla
import math
import time
import cv2

from tensorflow.keras.preprocessing.image import img_to_array


# class ControlUnit2:
#     def __init__(self, vehicle, csv_file, im_height, im_width):
#         print("<!> A Control Unit is created for:", vehicle.type_id)
#         self.vehicle = vehicle
#         self.data = du.load_intersection_data(csv_file)
#         self.im_height = im_height
#         self.im_width = im_width
#         self.npc_autopilot = False
#         print(">>>>", self.npc_autopilot)
#
#     def steer2throttle(self, steering, min_spd, max_spd):
#         speed_limit = max_spd
#         control = self.vehicle.get_control()
#         v = self.vehicle.get_velocity()
#         speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
#
#         if speed > speed_limit:
#             speed_limit = min_spd  # slow down
#             throttle = 0.0
#             brake = 1.0
#         else:
#             throttle = 1.0 - (steering ** 2) - (speed / speed_limit) ** 2
#             speed_limit = max_spd
#             brake = 0.0
#
#         # print("Current speed limit: ", speed_limit)
#         return throttle, brake
#
#     def check_enter_int(self, data, d=30):
#         # Get the current location of the vehicle
#         location = self.vehicle.get_location()
#         loc_x = location.x
#         loc_y = location.y
#
#         # Initialize variable that indicates whether the car is at the intersection or not
#         ent_int = False
#
#         # check if vehicle is entering an intersection
#         for x, y in data:
#             if x - d <= loc_x <= x + d and y + d >= loc_y >= y - d:
#                 ent_int = True
#                 break
#
#         return ent_int
#
#     def passing_trafficlight(self):
#         if self.vehicle.is_at_traffic_light():
#             traffic_light = self.vehicle.get_traffic_light()
#             if traffic_light.get_state() == carla.TrafficLightState.Red:
#                 # world.hud.notification("Traffic light changed! Good to go!")
#                 traffic_light.set_state(carla.TrafficLightState.Green)
#
#     def level_zero(self):
#         self.passing_trafficlight()
#         # self.vehicle.set_autopilot(True)
#
#     def level_one(self, steer, min_spd=2, max_spd=5):
#         # self.passing_trafficlight()
#         throttle, brake = self.steer2throttle(steer, min_spd, max_spd)
#         self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
#
#     def auto_switch(self, steer, steer_limit=0.4, min_spd=2, max_spd=5):
#         cond0 = self.vehicle.is_at_traffic_light()
#         ent_int = self.check_enter_int(self.data)
#
#         if ent_int:
#             if not self.npc_autopilot:
#                 print("Switched to Level 0")
#                 self.npc_autopilot = True
#                 self.vehicle.set_autopilot(self.npc_autopilot)
#             self.level_zero()
#
#         else:
#             if self.npc_autopilot:
#                 print("Switched to Level 1")
#                 self.npc_autopilot = False
#                 self.vehicle.set_autopilot(self.npc_autopilot)  # Turn off Carla built-in autopilot
#
#             self.level_one(steer, min_spd, max_spd)


class ControlUnit:
    def __init__(self, vehicle, model, carla_int_csv, gps_int_csv, im_height, im_width):
        print("<!> A Control Unit is created for:", vehicle.type_id)
        self.vehicle = vehicle
        self.model = model
        self.carla_intersections = du.load_intersection_data(carla_int_csv)
        self.gps_intersections = du.load_intersection_data(gps_int_csv)
        self.im_height = im_height
        self.im_width = im_width
        self.image = None
        self.steer = 0
        self.throttle = None
        self.brake = None
        self.level = None
        self.ent_int = None
        self.lidar_detection = None
        self.npc_autopilot = False
        self.v = self.vehicle.get_velocity()
        self.speed = math.sqrt(self.v.x ** 2 + self.v.y ** 2 + self.v.z ** 2)

    def build_image(self, image):
        # method 1
        # img = np.asarray(image.raw_data)
        # img = img.reshape((self.im_height, self.im_width, 4))
        # img = img_to_array(img)
        # img = img.astype(np.uint8)
        # img = img[:, :, :3]

        start_time = time.time()
        # method 2
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.im_height, self.im_width, 4))
        array = array[:, :, :3]
        self.image = array
        process_time = time.time() - start_time
        # print("Build image took: {} (s)".format(process_time))

    def predict_steering(self, image):
        self.build_image(image)

        start_time = time.time()
        main_array = self.image[:, :, ::-1]
        img = tu.preprocess1(main_array)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

        steer = self.model.predict(img)
        self.steer = float(steer[0][0])
        # print("Prediction took: {} (s)".format(time.time() - start_time))

    def check_enter_carla_int(self, carla_intersection):
        # Get the current location of the vehicle
        location = self.vehicle.get_location()
        loc_x, loc_y = location.x, location.y

        # Initialize variable that indicates whether the car is at the intersection or not
        ent_int = False

        # check if vehicle is entering an intersection
        for num, x, y, d in carla_intersection:
            if x - d <= loc_x <= x + d and y + d >= loc_y >= y - d:
                ent_int = True
                break

        return ent_int

    def check_enter_gps_int(self, gps_intersection, gnss_data):
        # Initialize variable that indicates whether the car is at the intersection or not
        current_data = gnss_data[-1]
        current_latitude = current_data[1]
        current_longitude = current_data[2]
        ent_int = False

        # check if vehicle is entering an intersection
        for tag, lat, lon, dis in gps_intersection:
            lat_dis = self.km2latdeg(dis)
            lon_dis = self.km2londeg(dis, current_latitude)
            if lat - lat_dis <= current_latitude <= lat + lat_dis and lon + lon_dis >= current_longitude >= lon - lon_dis:
                ent_int = True
                break

        return ent_int

    def steering_to_throttle(self, steering, min_spd, max_spd):
        speed_limit = max_spd
        control = self.vehicle.get_control()
        self.v = self.vehicle.get_velocity()
        self.speed = math.sqrt(self.v.x ** 2 + self.v.y ** 2 + self.v.z ** 2)

        if self.speed > speed_limit:
            speed_limit = min_spd  # slow down
            throttle = 0.0
            brake = 1.0
        else:
            throttle = 1.0 - (steering ** 2) - (self.speed / speed_limit) ** 2
            speed_limit = max_spd
            brake = 0.0

        # print("Current speed limit: ", speed_limit)

        return throttle, brake

    def passing_trafficlight(self):
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                # world.hud.notification("Traffic light changed! Good to go!")
                traffic_light.set_state(carla.TrafficLightState.Green)

    def show_opencv_window(self, pp1=False):
        font = cv2.FONT_HERSHEY_DUPLEX  # font
        org = (50, 50)                  # org
        font_scale = 0.6                # fontScale
        color = (0, 0, 0)
        thickness = 1                   # Line thickness of 2 px

        if self.level == 0:
            displayed_text = "Level {}-{}: Human Driving".format(self.level, np.round(self.speed, 2))
        else:
            displayed_text = "Level {}-{}: Autonomous Driving".format(self.level, np.round(self.speed, 2))

        if pp1:
            self.image = tu.preprocess1(self.image)

        text_image = cv2.putText(np.ascontiguousarray(self.image), displayed_text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imshow("front_cam", text_image)
        cv2.waitKey(1)

    def level_zero(self):
        self.passing_trafficlight()
        self.v = self.vehicle.get_velocity()
        self.speed = math.sqrt(self.v.x ** 2 + self.v.y ** 2 + self.v.z ** 2)

    def level_one(self, image, min_spd, max_spd, camera=False):
        self.predict_steering(image)
        self.throttle, self.brake = self.steering_to_throttle(self.steer, min_spd, max_spd)
        self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle, steer=self.steer, brake=self.brake))

        if camera:
            self.show_opencv_window()

    def auto_switch_02(self, image, min_spd, max_spd, gnss_data, camera=False):
        # self.ent_int = self.check_enter_carla_int(self.carla_intersections)
        self.ent_int = self.check_enter_gps_int(self.gps_intersections, gnss_data)

        if self.ent_int:
            if not self.npc_autopilot:
                print("State: Human Driving")
                self.npc_autopilot = True
                self.vehicle.set_autopilot(self.npc_autopilot)

            self.level = 0
            self.level_zero()
            self.build_image(image)

        else:
            if self.npc_autopilot:
                print("State: Autonomous Driving")
                self.npc_autopilot = False
                self.vehicle.set_autopilot(self.npc_autopilot)  # Turn off Carla built-in autopilot

            self.level = 1
            self.level_one(image, min_spd, max_spd)

        if camera:
            self.show_opencv_window()

    def auto_switch_03(self, image, min_spd, max_spd, lidar_data, gnss_data, camera=False):
        # self.ent_int = self.check_enter_carla_int(self.carla_intersections)
        self.ent_int = self.check_enter_gps_int(self.gps_intersections, gnss_data)
        self.lidar_detection = any(lidar_data[-5:-1])

        if self.ent_int:
            if not self.npc_autopilot:
                print("State: Human Driving")
                self.npc_autopilot = True
                self.vehicle.set_autopilot(self.npc_autopilot)

            self.level = 0
            self.level_zero()
            self.build_image(image)

        else:
            if self.npc_autopilot:
                print("State: Autonomous Driving")
                self.npc_autopilot = False
                self.vehicle.set_autopilot(self.npc_autopilot)  # Turn off Carla built-in autopilot

            if self.lidar_detection:
                print("State: Stop due to object detected")
                self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
                time.sleep(3)
            else:
                self.level = 1
                self.level_one(image, min_spd, max_spd)

        if camera:
            self.show_opencv_window()

    @staticmethod
    def latdeg2km(lat):
        km = lat * 110.574
        return km

    @staticmethod
    def londeg2km(lat_coordinate, lon):
        factor = 111.320 * math.cos(lat_coordinate)
        km = lon * factor
        return km

    @staticmethod
    def km2latdeg(km):
        lat = km / 110.574
        return lat

    @staticmethod
    def km2londeg(km, lat_coordinate):
        factor = 111.320 * math.cos(lat_coordinate)
        lon = km / factor
        return lon
