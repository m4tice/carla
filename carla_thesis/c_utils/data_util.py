import os
import math
import csv
import cv2
import glob
import time
import random
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tools import training_util as tu
from scipy.stats.kde import gaussian_kde
from shutil import copyfile


np.random.seed(2)


# -- Error Print -------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# -- Tensorflow version -------
def tensorflow_version():
    eprint("Tensorflow version: ",  tf.__version__)


# == LOAD CSV =====
# -- Load town map -----
def load_map(csv_map):
    print("Calling: loading map")
    map_data_df = pd.read_csv(csv_map, names=['x', 'y'])
    waypoints = map_data_df[['x', 'y']].values

    return waypoints


# -- Load intersections coordinates -----
def load_intersection_data(csv_coors):
    data_df = pd.read_csv(csv_coors, names=['number', 'x', 'y', 'r'])
    coors = data_df[['number', 'x', 'y', 'r']].values

    return coors


# -- Load gnss data -----
def load_gnss_data(gnss_csv):
    data_df = pd.read_csv(gnss_csv, names=['tag', 'lat', 'lon', 'alt'])
    gnss_data = data_df[['tag', 'lat', 'lon', 'alt']].values

    return gnss_data


# -- Load imu data -----
def load_imu_data(imu_csv):
    data_df = pd.read_csv(imu_csv, names=['tag', 'a0', 'a1', 'a2', 'g0', 'g1', 'g2', 'compass'])
    imu_data = data_df[['tag', 'a0', 'a1', 'a2', 'g0', 'g1', 'g2', 'compass']].values

    return imu_data


# -- Load spawn points -----
def load_spawn_points_data(csv_pts):
    data_df = pd.read_csv(csv_pts, names=['x', 'y', 'z', 'pitch', 'yaw', 'roll'])
    pts = data_df[['x', 'y', 'z', 'pitch', 'yaw', 'roll']].values

    return pts


# -- Load recorded data -----
def load_camera_data(csv_file):
    data_df = pd.read_csv(csv_file,
                          names=['center', 'loc_x', 'loc_y', 'atTrafficLight', 'left_lane', 'right_lane', 'throttles',
                                 'braking', 'steering'])
    y = data_df[['center', 'loc_x', 'loc_y', 'atTrafficLight', 'left_lane', 'right_lane', 'throttles', 'braking',
                 'steering']].values

    return y


def load_lidar_data(csv_file):
    data_df = pd.read_csv(csv_file, names=['x', 'y', 'z', 'intensity', 'loc_x', 'loc_y', 'theta',
                                           'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'])
    data = data_df[['x', 'y', 'z', 'intensity', 'loc_x', 'loc_y', 'theta',
                    'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']].values

    return data


# == EXPORT CSV =====
# -- Export vehicle data ---
def export_csv(csv_file, features):
    print("Calling: exporting vehicle data (writing into csv file!)")

    if os.path.isfile(csv_file):
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for img_name, loc_x, loc_y, atTrafficLight, left_lane, right_lane, throttle, brake, steer in features:
                writer.writerow([img_name, loc_x, loc_y, atTrafficLight, left_lane, right_lane, throttle, brake, steer])
    else:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for img_name, loc_x, loc_y, atTrafficLight, left_lane, right_lane, throttle, brake, steer in features:
                writer.writerow([img_name, loc_x, loc_y, atTrafficLight, left_lane, right_lane, throttle, brake, steer])


def export_gnss(gnss_csv, gnss_data):
    print("Calling: exporting GNSS data (writing into csv file!)")
    if os.path.isfile(gnss_csv):
        with open(gnss_csv, 'a', newline='') as file:
            writer = csv.writer(file)
            for tag, lat, lon, alt in gnss_data:
                writer.writerow([tag, lat, lon, alt])
    else:
        with open(gnss_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            for tag, lat, lon, alt in gnss_data:
                writer.writerow([tag, lat, lon, alt])


def export_imu(imu_csv, imu_data):
    print("Calling: exporting IMU data (writing into csv file!)")
    if os.path.isfile(imu_csv):
        with open(imu_csv, 'a', newline='') as file:
            writer = csv.writer(file)
            for tag, a0, a1, a2, g0, g1, g2, compass in imu_data:
                writer.writerow([tag, a0, a1, a2, g0, g1, g2, compass])
    else:
        with open(imu_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            for tag, a0, a1, a2, g0, g1, g2, compass in imu_data:
                writer.writerow([tag, a0, a1, a2, g0, g1, g2, compass])


def latdeg2km(lat):
    km = lat * 110.574
    return km


def londeg2km(lat_coordinate, lon):
    factor = 111.320 * math.cos(lat_coordinate)
    km = lon * factor
    return km


def km2latdeg(km):
    lat = km / 110.574
    return lat


def km2londeg(km, lat_coordinate):
    factor = 111.320 * math.cos(lat_coordinate)
    lon = km / factor
    return lon


def check_enter_intersection_data(data, loc_x, loc_y):
    # Initialize variable that indicates whether the car is at the intersection or not
    ent_int = False

    # check if vehicle is entering an intersection
    for num, x, y, d in data:
        if x - d <= loc_x <= x + d and y + d >= loc_y >= y - d:
            ent_int = True
            break

    return ent_int


# == DATA VISUALIZING =====

def plot_rectangle(ax, c, alpha=1.0):
    d = c[3]
    ax.scatter(c[1], c[2], marker='+', color='yellow')
    ax.plot([c[1] - d, c[1] + d, c[1] + d, c[1] - d, c[1] - d], [c[2] + d, c[2] + d, c[2] - d, c[2] - d, c[2] + d], color='red', alpha=alpha)
    ax.text(c[1] - d, c[2] + 1.05 * d, c[0], fontsize=6, color='red', alpha=alpha)


def plot_gps_rectangle(ax, c, alpha=1.0):
    km = c[3]
    d_lat = km2latdeg(km)
    d_lon = km2londeg(km, c[1])
    ax.scatter(c[1], c[2], marker='+', color='yellow')
    ax.plot([c[1]-d_lat, c[1]+d_lat, c[1]+d_lat, c[1]-d_lat, c[1]-d_lat], [c[2]+d_lon, c[2]+d_lon, c[2]-d_lon, c[2]-d_lon, c[2]+d_lon], color='red', alpha=alpha)
    ax.text(c[1]-d_lat, c[2] + 1.05 * d_lon, int(c[0]), fontsize=6, color='red', alpha=alpha)


def plot_spawn_points(fig, ax, pts):
    spx = pts[:, 0]
    spy = pts[:, 1]

    ax.scatter(spx, spy, color='black', marker='+')

    for i, p in enumerate(pts):
        ax.text(p[0], p[1]+1, i, color='black')


def display_sample(path, ax, i, sample, eit, raw=True):
    name = sample[0].split("/")
    name = name[1]
    frame = mpimg.imread(os.path.join(path, name))
    if not raw:
        frame = tu.preprocess1(frame)
        # frame = tu.crop_img(frame)
        # frame = tu.resize_img(frame)

    ax.imshow(frame)
    ax.text(1.5, 5,  "ATL: {}".format(sample[3]), fontsize=6, color='white')
    ax.text(1.5, 10, "LLM: {}".format(sample[4]), fontsize=6, color='white')
    ax.text(1.5, 15, "RLM: {}".format(sample[5]), fontsize=6, color='white')
    ax.text(1.5, 20, "BRK: {}".format(sample[7]), fontsize=6, color='white')
    ax.text(1.5, 25, "FRM: {}".format(i+1), fontsize=6, color='white')
    ax.text(1.5, 30, "EIT: {}".format(eit), fontsize=6, color='white')


def play_folder_cond(csv_file, csv_coors, path, raw=True, tf=0.01, d=18):
    data = load_camera_data(csv_file)
    coors = load_intersection_data(csv_coors)

    try:
        fig, ax = plt.subplots()
        for i, sample in enumerate(data):
            entint = check_enter_intersection_data(coors, sample[1], sample[2])

            # for num, x, y, d in coors:
            #     if x - d <= sample[1] <= x + d and y + d >= sample[2] >= y - d:
            #         entint = True
            #         break

            if not entint:
                display_sample(path, ax, i, sample, entint, raw)
                plt.pause(tf)
                ax.cla()

    except Exception as e:
        print("Error:", e)


def play_back(csv_file, path, pp1=False):
    data = load_camera_data(csv_file)
    data = data[:, 0]
    data = sorted(data)

    for img in data:
        img = img.split("/")
        img = img[-1]
        img = tu.load_carla_image(img, path)
        if pp1:
            img = tu.preprocess1(img)

        cv2.imshow("front camera", img)
        cv2.waitKey(25)


def play_brake(csv_file, path, tf=1):
    data = load_camera_data(csv_file)
    fig, ax = plt.subplots()

    for i, sample in enumerate(data):
        ax.cla()
        if sample[7] > 0.1:
            name = sample[0].split("/")
            image = mpimg.imread(os.path.join(path, name[1]))
            ax.imshow(image)
            plt.pause(tf)


def gps_tracking(csv_file, csv_map, csv_coor, alpha=0.5, tf=0.1):
    y = load_camera_data(csv_file)
    coors = load_intersection_data(csv_coor)
    waypoints = load_map(csv_map)

    data_df = pd.read_csv(csv_file,
                          names=['center', 'loc_x', 'loc_y', 'atTrafficLight', 'left_lane', 'right_lane', 'throttles',
                                 'braking', 'steering'])
    veh_coors = data_df[['loc_x', 'loc_y', 'braking', 'steering']].values
    print("- Data amount:", len(veh_coors))
    vxs = veh_coors[:, 0]
    vys = veh_coors[:, 1]

    try:
        fig1, ax1 = plt.subplots()
        ax1.axis('equal')

        for i, sample in enumerate(y):
            # map
            offset = 20
            mxs = waypoints[:, 0]
            mys = waypoints[:, 1]

            ax1.grid(True)
            ax1.axis([min(mxs) - offset, max(mxs) + offset, min(mys) - offset, max(mys) + offset])
            ax1.set_facecolor('gray')
            ax1.scatter(mxs, mys, color='white', marker='.', alpha=alpha)
            ax1.scatter(vxs, vys, c=[[28 / 255, 134 / 255, 255 / 255]], marker='.')

            ent_int = check_enter_intersection_data(coors, sample[1], sample[2])

            # coors
            for item in coors:
                plot_rectangle(ax1, item, alpha)

            # for num, x, y, d in coors:
            #     if x - d <= sample[1] <= x + d and y + d >= sample[2] >= y - d:
            #         ent_int = True
            #         break

            # cond = ent_int
            cond = sample[7] == 1

            if cond:
                ax1.plot(sample[1], sample[2], marker='o', markersize=3, color="red", alpha=alpha)
            else:
                ax1.plot(sample[1], sample[2], marker='o', markersize=3, color="blue", alpha=alpha)

            plt.pause(tf)
            ax1.cla()

    except Exception as e:
        print("Error:", e)


def plot_intersection_coors(csv_map, csv_coors):
    # map
    offset = 20
    waypoints = load_map(csv_map)
    mxs = waypoints[:, 0]
    mys = waypoints[:, 1]

    fig1, ax1 = plt.subplots()
    ax1.axis('equal')
    ax1.grid(True)
    ax1.axis([min(mxs) - offset, max(mxs) + offset, min(mys) - offset, max(mys) + offset])
    ax1.set_facecolor('gray')
    ax1.scatter(mxs, mys, color='white', marker='.')

    # coors
    coors = load_intersection_data(csv_coors)
    for item in coors:
        plot_rectangle(ax1, item)


# -- Plot map ----------
def map_plot(csv_map, csv_pts):
    print("Calling: showing map")
    offset = 20

    waypoints = load_map(csv_map)
    mxs = waypoints[:, 0]
    mys = waypoints[:, 1]

    pts = load_spawn_points_data(csv_pts)

    fig1, ax1 = plt.subplots()
    ax1.axis('equal')
    ax1.grid(True)
    ax1.axis([min(mxs) - offset, max(mxs) + offset, min(mys) - offset, max(mys) + offset])
    ax1.set_facecolor('gray')
    ax1.scatter(mxs, mys, color='white', marker='.')

    plot_spawn_points(fig1, ax1, pts)

    return ax1


# -- Plot tracking -----
def tracking_plot(csv_map, csv_file, csv_coors, point=None, title=None, grid=True, stop_plot=False, angle=None, roundlimit=3):
    print("Calling: showing tracking")
    offset = 20

    waypoints = load_map(csv_map)
    mxs = waypoints[:, 0]
    mys = waypoints[:, 1]

    data_df = pd.read_csv(csv_file,
                          names=['center', 'loc_x', 'loc_y', 'atTrafficLight', 'left_lane', 'right_lane', 'throttles',
                                 'braking', 'steering'])
    veh_coors = data_df[['loc_x', 'loc_y', 'braking', 'steering']].values
    print("- Data amount:", len(veh_coors))
    vxs = veh_coors[:, 0]
    vys = veh_coors[:, 1]

    fig1, ax1 = plt.subplots()
    ax1.axis('equal')
    ax1.set_title(title)
    ax1.grid(True)
    ax1.axis([min(mxs) - offset, max(mxs) + offset, min(mys) - offset, max(mys) + offset])

    if not grid:
        ax1.set_xticks([])
        ax1.set_yticks([])

    ax1.set_facecolor('gray')
    ax1.scatter(mxs, mys, color='white', marker='o')
    ax1.scatter(vxs, vys, c=[[28 / 255, 134 / 255, 255 / 255]], marker='.')

    if stop_plot:
        coors = load_intersection_data(csv_coors)
        for item in coors:
            plot_rectangle(ax1, item, alpha=1)

        stop_points = []
        for item in veh_coors:
            if item[2] >= 0.4:
                stop_points.append(item)

        stop_points = np.asarray(stop_points)

        if len(stop_points) > 0:
            sxs = stop_points[:, 0]
            sys = stop_points[:, 1]
            ax1.scatter(sxs, sys, color='red', marker='.')

    ax1.scatter(vxs[0], vys[0], color='black', marker='x')
    ax1.scatter(vxs[-1], vys[-1], color='black', marker='o')
    # ax1.set_xlim([-210, -130])
    # ax1.set_ylim([75, 115])

    if point is not None:
        ax1.scatter(vxs[point], vys[point], color='green', marker='x')

    if angle is not None:
        x_steer = []
        y_steer = []
        for x, y, b, s in veh_coors:
            if np.round(s, roundlimit) == angle:
                x_steer.append(x)
                y_steer.append(y)

        if len(x_steer) > 0:
            print("Steering list length: ", len(x_steer))
            ax1.scatter(x_steer, y_steer, color="purple", marker="o")
        else:
            print("Cannot find specified steering angle.")


# -- Heatmap for data points
def heatmap(csv_map, csv_file, title=None, cmapname="jet", alpha=0.3):
    offset = 20

    # Map coordinates
    way_points = load_map(csv_map)
    mxs = way_points[:, 0]
    mys = way_points[:, 1]

    # Vehicles coordinates
    data_df = pd.read_csv(csv_file,
                          names=['center', 'loc_x', 'loc_y', 'atTrafficLight', 'left_lane', 'right_lane', 'throttles',
                                 'braking', 'steering'])
    veh_coors = data_df[['loc_x', 'loc_y']].values
    vxs = veh_coors[:, 0]
    vys = veh_coors[:, 1]
    x = np.asarray(vxs)
    y = np.asarray(vys)

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[min(x):max(x):len(x) ** 0.5 * 1j, min(y):max(y):len(y) ** 0.5 * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    fig, ax = plt.subplots()
    # ax.axis('equal')
    ax.set_title(title)
    ax.set_facecolor('gray')
    # ax.grid(True)

    # Graph limits
    ax.axis([min(mxs) - offset, max(mxs) + offset, min(mys) - offset, max(mys) + offset])
    # ax.set_xlim(min(mxs), max(mxs))
    # ax.set_ylim(min(mys), max(mys))

    ax.scatter(mxs, mys, color='white', marker='o', alpha=1)
    # ax.scatter(vxs, vys, c=[[28 / 255, 134 / 255, 255 / 255]], marker='.', alpha=1)

    # Heatmap types
    ax.hist2d(x, y, bins=(400, 400), cmap=plt.get_cmap(cmapname), alpha=alpha)
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=alpha)
    ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=alpha)


# -- Histogram plot -----
def histplot(csv_file, nob, title=None, fs=15):
    data_df = pd.read_csv(csv_file,
                          names=['center', 'loc_x', 'loc_y', 'atTrafficLight', 'left_lane', 'right_lane', 'throttles',
                                 'braking', 'steering'])
    y = data_df['steering'].values
    y = np.ndarray.tolist(y)
    print(type(y[0]))

    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=fs)
    ax.set_xlabel("steering values", fontsize=fs)
    ax.set_ylabel("value distribution", fontsize=fs)
    tempy, tempx, _ = ax.hist(y, bins=nob, color='dimgrey')
    ax.set_ylim([0, max(tempy)+10])


def clean_data(csv_file, csv_coors, new_file):
    print("Calling: clean data")
    y = load_camera_data(csv_file)
    original_len = len(y)
    coors = load_intersection_data(csv_coors)
    temp = []
    rm = []

    try:
        if os.path.isfile(new_file):
            print("- Removing existed file")
            os.remove(new_file)
        for i, sample in enumerate(y):
            entint = check_enter_intersection_data(coors, sample[1], sample[2])

            # for x, y, d in coors:
            #     if x - d <= sample[1] <= x + d and y + d >= sample[2] >= y - d:
            #         entint = True
            #         break

            if not entint:
                temp.append(sample)

        for i, sample in enumerate(temp):
            cond1 = sample[7] > 0.2
            cond2 = sample[6] == 0 and sample[7] == 0 and sample[8] == 0
            cond3 = math.isclose(sample[6], 0.7, abs_tol=0)

            if cond1 or cond2 or cond3:
                rm.append(i)

        rm.reverse()

        for item in rm:
            temp.pop(item)

        np.random.shuffle(temp)
        print("- Before: {} - After: {}".format(original_len, len(temp)))
        export_csv(new_file, temp)

    except Exception as e:
        print("- Error:", e)


def histdisplay(y, ax, tempy=1000, nob=1001, title=None, fs=15):
    ax.set_title(title, fontsize=fs)
    ax.set_xlabel("steering values", fontsize=fs)
    ax.set_ylabel("value distribution", fontsize=fs)
    ax.hist(y, bins=nob, color='dimgrey')  # y, x, _ =
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 10000 + 50])
    name = "D:/TUAN/Workspace/Data/carla/average_data/"+str(title)+".png"
    plt.pause(2)
    # plt.savefig(name, format='png')
    ax.cla()


def average_display(csv_file, tempy=1000, nob=1001):
    y = load_camera_data(csv_file)
    print(type(y))
    feature = y[:, 0]
    label = y[:, 8]

    fig, ax = plt.subplots()

    for j in range(100):
        generals = []
        for k in range(8000):
            batch_size = 40
            angles = np.empty(batch_size)
            i = 0
            for index in np.random.permutation(len(label)):
                steering = label[index]

                # translate
                if np.random.rand() < 0.5:
                    trans_x = 100 * (np.random.rand() - 0.5)
                    trans_y = 10 * (np.random.rand() - 0.5)
                    steering += trans_x * 0.002

                angles[i] = float(steering)

                i += 1
                if i == batch_size:
                    break

            generals.extend(angles)
            # histdisplay(angles, ax, nob=10)

        generals = np.asarray(generals)
        print(generals.shape)

        histdisplay(generals, ax, tempy=tempy, nob=nob, title=j)


def size_estimation(csv_file, path):
    print("Calling: size estimation")
    data = load_camera_data(csv_file)
    data = data[:, 0]
    data = [os.path.join(path, item) for item in data]

    sumkb = 0
    for item in data:
        kb = math.ceil(os.path.getsize(item) / 1000)
        sumkb = sumkb + kb

    print("- Size in KB:", sumkb)
    print("- Size in MB:", math.ceil(sumkb / 1024))
    print("- Size in GB:", round(sumkb / 1024 / 1024, 1))


def balance_data(csv_file, new_file, roundlimit=3, sample_limit=10):
    print("Calling: balance data")
    d = load_camera_data(csv_file)
    d = np.ndarray.tolist(d)

    for i in range(5):
        random.shuffle(d)

    new_list = []

    step = 1 / pow(10, roundlimit)
    dr = np.arange(-0.4, 0.5 + step, step)
    print("- Value amount: ", len(dr))

    if os.path.isfile(new_file):
        print("- Removing existed file")
        os.remove(new_file)

    for i in range(len(dr)):
        count = 0
        # if dr[i] % 0.01 == 0:
        base = np.around(dr[i], roundlimit)
        # print(base)

        for item in d:
            steer = round(float(item[-1]), roundlimit)
            if steer == base:
                new_list.append([item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8]])
                # new_list.append([item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], steer])
                count += 1
                if count == sample_limit:
                    break

    random.shuffle(new_list)
    new_list = np.asarray(new_list)

    export_csv(new_file, new_list)

    return new_list


# == DATA ERROR CHECKING =====

def redundant_img_check(img_dir, csv_file):
    if os.path.isdir(img_dir):
        print("Calling: redundant image check")
        temp = os.listdir(img_dir)
        img_list = [[item, False] for item in temp]

        data = load_camera_data(csv_file)
        temp = data[:, 0]
        data = [item.split("/") for item in temp]
        data = [item[1] for item in data]

        for item in img_list:
            # print(item, "==" * 50)
            for sample in data:
                if item[0] == sample:
                    # print(item[0], sample)
                    item[1] = True
                    break

        rm_list = []
        for item in img_list:
            if not item[1]:
                rm_list.append(item[0])
                os.remove(img_dir + "/" + item[0])

        if len(rm_list) > 0:
            print("- Removed {} item(s)".format(len(rm_list)))
            print("- Removed:", rm_list)

    else:
        print("- Directory not found!")


def existence_check(img_dir, csv_file):
    print("Calling: existent check")
    data = load_camera_data(csv_file)
    temp = data[:, 0]
    data = [item.split("/") for item in temp]
    data = [item[1] for item in data]

    pms = []
    for item in data:
        pm = os.path.isfile(os.path.join(img_dir, item))
        pms.append(pm)

    return all(pms)


# - Count labels
def count_line_csv(csv_file):
    with open(csv_file, 'r', newline='') as file:
        file_object = csv.reader(file)
        row_count = sum(1 for row in file_object)  # fileObject is your csv.reader
        return row_count


# - Count images
def images_count(dir):
    count = len([img for img in os.listdir(dir)])
    return count


# - Check for balance between features and labels
def data_balance_check(dir, csvfile):
    print("Calling: data balance checking")
    img_num = images_count(dir)
    line_num = count_line_csv(csvfile)
    print("-  Count: Features: {} - Labels: {}".format(img_num, line_num))

    if img_num == line_num and img_num != 0 and line_num != 0:
        print('- Result: Balanced data.')
        return True
    else:
        print('- Result: Unbalanced data.')
        return False


# -- Check error images
def error_image_check(path):
    print("Calling: corrupted images checking")
    imgs = os.listdir(path)
    corrupted = []
    for item in imgs:
        try:
            mpimg.imread(os.path.join(path, item))
        except:
            corrupted.append(os.path.join(path, item))

    if len(corrupted) > 0:
        print(' result: {} corrupted file(s) found.'.format(len(corrupted)))
        for item in corrupted:
            os.remove(item)
    else:
        print(' result: No corrupted file found.')


def final_process(dir, csv_file):
    error_image_check(dir)
    data_balance_check(dir, csv_file)


def overall_check(img_dir, csv_file):
    print("Calling: Data checking")
    redundant_img_check(img_dir, csv_file)
    print(existence_check(img_dir, csv_file))
    data_balance_check(img_dir, csv_file)


def copy_data(csv_file, src_dir, des_dir):
    print("Calling: copy images")
    data = load_camera_data(csv_file)
    temp = data[:, 0]
    data = [item.split("/") for item in temp]
    data = [item[1] for item in data]

    print("- Copying {} item(s)".format(len(data)))
    for num, item in enumerate(data):
        src = os.path.join(src_dir, item)
        des = os.path.join(des_dir, item)

        if not os.path.isfile(des):
            copyfile(src, des)

    print("- Copy completed.")


def test_preprocess2(csv_file, path, pp1=False, rx=100, fs=25):
    data = load_camera_data(csv_file)
    choice = random.randint(0, len(data)-1)
    print(choice)
    sample = data[choice]

    original_image_name = sample[0]
    original_image_name = os.path.join(path, original_image_name)
    original_steer = sample[-1]

    if os.path.isfile(original_image_name):
        original_image = tu.load_carla_image(original_image_name)
        modified_image, modified_steer = tu.preprocess2(original_image_name, path, original_steer, range_x=rx)

        if pp1:
            original_image = tu.preprocess1(original_image)
            modified_image = tu.preprocess1(modified_image)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(original_image)
        ax[0].set_title("ORIGINAL\nsteering = %.8f" % original_steer, fontsize=fs)
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)

        ax[1].imshow(modified_image)
        ax[1].set_title("MODIFIED\nsteering = %.8f" % modified_steer, fontsize=fs)
        ax[1].axes.xaxis.set_visible(False)
        ax[1].axes.yaxis.set_visible(False)

        return abs(original_steer - modified_steer)

    else:
        print("ERROR: Image does not exist")


def test_sample(amount, src_csv, des_csv, src, des):
    data = load_camera_data(src_csv)
    np.random.shuffle(data)

    temp = data[:amount]
    export_csv(des_csv, temp)

    if os.path.isfile(des_csv):
        copy_data(des_csv, src, des)
    else:
        print("ERROR: File not found")
