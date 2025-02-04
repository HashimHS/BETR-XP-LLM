"""Interface to Ai2thor simulator."""

from interfaces.base_world_interface import BaseWorldInterface
from ai2thor.controller import Controller
import numpy as np
from PIL import Image
import open3d as o3d

import torch
torch.set_grad_enabled(False)
torch.manual_seed(0)

from reflect.main.scene_graph import SceneGraph as BaseSceneGraph
from reflect.main.scene_graph import Node as GraphNode
from reflect.main.scene_graph import Edge as GraphEdge
from reflect.main.action_primitives import *
from reflect.main.get_local_sg import get_2d_bbox_from_3d_pcd
from reflect.main.utils import *

import cv2
import copy

DIRECTIONS = {
    'w' : "MoveAhead",
    'a' : "MoveLeft",
    's' : "MoveBack",
    'd' : "MoveRight"
}
ROTATIONS = {
    'r' : 'RotateRight',
    'l' : 'RotateLeft'
}


def gen_node(obj, event, obj_held_prev=False):
    name = obj['objectId']
    object_id = obj['objectId']
    total_points = torch.tensor(np.array([]))
    pcd_obj = torch.tensor(np.array([]))
    height, width, channel = event.frame.shape
    camera_space_xyz = depth_frame_to_camera_space_xyz(
            depth_frame=torch.as_tensor(event.depth_frame.copy()), mask=None, fov=event.metadata['fov'])
    x = event.metadata['agent']['position']['x']
    y = event.metadata['agent']['position']['y']
    z = event.metadata['agent']['position']['z']

    if not event.metadata['agent']['isStanding']:
        y = y - 0.22

    world_points = camera_space_xyz_to_world_xyz(
        camera_space_xyzs=camera_space_xyz,
        camera_world_xyz=torch.as_tensor([x, y, z]),
        rotation=event.metadata['agent']['rotation']['y'],
        horizon=event.metadata['agent']['cameraHorizon'],
    ).reshape(channel, height, width).permute(1, 2, 0)

    sinkbasin_pts = None

    if object_id.split("|")[0] in ["Window", "Floor", "Wall", "Ceiling", "Cabinet"]:
        return None

    # register this box in 3D
    mask = event.instance_masks[object_id].reshape(height, width)
    obj_points = torch.as_tensor(world_points[mask])
    
    if len(obj_points) < 700:
        return None
    
    depth_obj = event.depth_frame[mask]
    # obj_colors = torch.as_tensor(world_colors[mask])

    # downsample point cloud
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
    voxel_down_pcd = obj_pcd.voxel_down_sample(voxel_size=0.01)

    # denoise point cloud
    if "Pan" == object_id.split("|")[0] or "EggCracked" == object_id.split("|")[0] or "Bowl" == object_id.split("|")[0] or "Pot" == object_id.split("|")[0]:
        _, ind = voxel_down_pcd.remove_radius_outlier(nb_points=30, radius=0.03)
        inlier = voxel_down_pcd.select_by_index(ind)
        pcd_obj = torch.tensor(np.array(inlier.points))
    elif "CounterTop" == object_id.split("|")[0]:
        _, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.1)
        inlier = voxel_down_pcd.select_by_index(ind)
        pcd_obj = torch.tensor(np.array(inlier.points))
    elif "SinkBasin" in object_id:
        sinkbasin_pts = torch.tensor(np.array(voxel_down_pcd.points))
    else:
        pcd_obj = torch.tensor(np.array(voxel_down_pcd.points))
        # assert points[object_id].shape == colors[object_id].shape
    #==============================================================

    if is_receptacle(object_id, event):
        if is_moving(object_id, event) or is_picked_up(object_id, event) or obj_held_prev == object_id:
            total_points = pcd_obj
        else:
            total_points = torch.unique(torch.cat((total_points, pcd_obj), 0), dim=0)
    else:
        total_points = pcd_obj

    if object_id.split("|")[0] == "Sink" and sinkbasin_pts is not None:
        total_points = torch.unique(torch.cat((total_points, sinkbasin_pts), 0), dim=0)

    boxes3d_pts = o3d.utility.Vector3dVector(total_points)
    box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(boxes3d_pts)

    # Generate local scene graph
    total_points_dict = {}
    total_points_dict[object_id] = total_points
    bbox = get_2d_bbox_from_3d_pcd(event, object_id, total_points_dict)
    if name is not None and bbox is not None:
        node = GraphNode(name, 
                    object_id=object_id, 
                    pos3d=box.get_center(), 
                    corner_pts=np.array(box.get_box_points()), 
                    bbox2d=bbox, 
                    pcd=total_points,
                    depth=depth_obj)
        return node
    return None

class SceneGraph(BaseSceneGraph):
        
    def add_edges(self, node, new_node):
        target_object = new_node.name
        relative_object = node.name
        relation = None

        if self.object_position_known[target_object] and self.object_position_known[relative_object]:
            if abs(self.object_positions[target_object][0] - self.object_positions[relative_object][0]) < 0.01 and \
                abs(self.object_positions[target_object][1] - self.object_positions[relative_object][1]) < 0.01 and \
                abs(self.object_positions[target_object][2] - \
                    BaseWorldInterface.CUBE_SIZE - self.object_positions[relative_object][2]) < 0.01:
                relation = 'on'

            elif abs(self.object_positions[target_object][0] - self.object_positions[relative_object][0]) < 0.01 and \
                abs(self.object_positions[target_object][1] - self.object_positions[relative_object][1]) < 0.01 and \
                abs(self.object_positions[target_object][2] - self.object_positions[relative_object][2]) < 0.03:
                relation = 'in'

        elif isinstance(relative_object, np.ndarray):
            if self.object_position_known[target_object]:
                if self.calc_distance(target_object, relative_object) < 0.01:
                   relation = 'at'
                   
        if relation is not None:
            self.edges[(new_node.name, node.name)] = GraphEdge(new_node, node, relation)


class WorldInterface(BaseWorldInterface):

    def __init__(self, scene='FloorPlan16', movable_objects=[], graspable_objects=[], known_objects=[], gridSize=0.25, root_folder_path=''):
        self.root_folder_path = root_folder_path
        video_path = os.path.join(root_folder_path, 'video.avi')
        self.video_color = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 4, (960, 960))
        self.gridSize = gridSize

        self.grid = np.mgrid[-5:5.1:gridSize, -5:5.1:gridSize].transpose(1,2,0)
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=1.0,
            scene=scene,
            gridSize=self.gridSize,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            rotateStepDegrees=90,
            width=960,
            height=960,
            fieldOfView=60,
        )
        # self.controller.step(action="SetHandSphereRadius", radius=0.1)

        self.graspable_objects = graspable_objects
        self.movable_objects = movable_objects
        self.scene_graph = SceneGraph(event=self.controller.last_event, task=None)
        self.scene_graph_nodes = [node.name for node in self.scene_graph.total_nodes]
        self.scene_graph_file = 'scene_graph.txt'
        self.text_graph = ''
        self.hierarchical_summary_file = 'hierarchical_summary.txt'

        self.grasped_object = None
        self.manipulation_target = None
        self.object_dict = {}
        self.object_positions = {}
        self.object_position_known = {}
        self.scene_graph.object_position_known = self.object_position_known
        self.scene_graph.object_positions = self.object_positions
        self.object_upright = {}
        self.object_opened = {}
        self.object_unlocked = {}
        self.held_prev = []
        self.scene_changes = []
        self.error_message = ''
        self.failed_behavior = ''

        for obj in known_objects:
            object_id = self.get_id(obj)
            self.object_dict[obj] = object_id
            self.object_position_known[object_id] = True
        
        for obj in self.controller.last_event.metadata["objects"]:
            self.update_scene_graph(obj, self.controller.last_event)
        self.update_scene_graph_file() 
           
    def get_updated_image(self , file_path=None):
        """ Returns the current image of the last event"""
        if file_path is None:
            file_path=self.root_folder_path
        image = self.controller.last_event.cv2img
        file_path = os.path.join(file_path, 'updated_image.png')
        cv2.imwrite(file_path, image)
        return [file_path]
    
    def update_scene_graph_file(self, file_path=None):
        """
        Reads the scene graph from a file and returns its content as text.
        """
        self.text_graph = ""
        for edge in self.scene_graph.edges.keys():
            edge_text = f"{self.scene_graph.edges[edge]}"
            self.text_graph += edge_text + "\n"

        if file_path is None:
            file_path=self.root_folder_path
        try:
            file_path = os.path.join(file_path, self.scene_graph_file)        
            with open(file_path, 'w') as f:
                f.write(self.text_graph)
            return self.scene_graph_file
        except FileNotFoundError:
            print(f"[ERROR] Scene graph file '{self.scene_graph_file}' not found.")
            return None
        
    def update_hierarchical_summary_file(self, file_path=None):
        """
        Reads the hierarchical summary from a file and returns its content as text.
        """
        if self.scene_changes == []:
            return

        Timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        observation = ""
        for edge in self.scene_changes:
            observation += f" {edge},"
        self.text_summary = f"Timestamp: {Timestamp} | Observation:{observation}\n"
        self.scene_changes = []
        
        if file_path is None:
            file_path=self.root_folder_path
        try:
            file_path = os.path.join(file_path, self.hierarchical_summary_file)        
            with open(file_path, 'a') as f:
                f.write(self.text_summary)
            return self.hierarchical_summary_file
        except FileNotFoundError:
            print(f"[ERROR] Hierarchical summary file '{self.hierarchical_summary_file}' not found.")
            return None

    def get_feedback(self):
        event = self.controller.last_event
        self.robot_position = self.controller.last_event.metadata['agent']['position']
        self.robot_orientation = self.controller.last_event.metadata['agent']['rotation']

        self.color_frame = self.controller.last_event.cv2img
        self.video_color.write(self.color_frame)
        self.depth_frame = self.controller.last_event.depth_frame

        pre_edges = copy.deepcopy(self.scene_graph.edges)
        pre_nodes = self.scene_graph_nodes
        for obj in event.metadata['objects']:
            if obj['pickupable']:
                if obj['objectId'] not in self.graspable_objects:
                    self.graspable_objects.append(obj['objectId'])
                if obj['isPickedUp']:
                    self.grasped_object = obj['objectId']
                    self.scene_graph.edges[(obj['objectId'],"robot_gripper")] = GraphEdge(obj['objectId'], "robot_gripper", edge_type="in")
                    if obj['objectId'] not in self.scene_graph_nodes:
                        self.scene_graph_nodes.append(obj['objectId'])
                    if obj['objectId'] not in self.held_prev:
                        self.held_prev.append(obj['objectId'])
            if obj['moveable']:
                if obj['objectId'] not in self.movable_objects:
                    self.movable_objects.append(obj['objectId'])                
            if obj['toggleable']:
                self.object_unlocked[obj['objectId']] = obj['isToggled']
            if obj['openable']:
                self.object_opened[obj['objectId']] = obj['isOpen']
            # if obj['rotation']['x'] < 0.1 and obj['rotation']['z'] < 0.1:
            #     self.object_upright[obj['objectId']] = True
            # else:
            #     self.object_upright[obj['objectId']] = False
            self.update_scene_graph(obj, event)         

        for edge in self.scene_graph.edges.keys():
            if edge not in pre_edges.keys() or self.scene_graph.edges[edge].edge_type != pre_edges[edge].edge_type:
                if edge[0] in pre_nodes or edge[1] in pre_nodes or edge[0] in self.held_prev:
                    self.scene_changes.append(self.scene_graph.edges[edge])

        self.update_scene_graph_file()
        self.update_hierarchical_summary_file()

        return True
    
    def update_scene_graph(self, obj, event):
        if obj['objectId'] not in self.scene_graph_nodes:
            if obj['visible']:
                self.scene_graph_nodes.append(obj['objectId'])
                self.object_position_known[obj['objectId']] = True
                self.object_positions[obj['objectId']] = self.dict_to_pos(obj['position'])
                node = gen_node(obj, event, obj['objectId'] in self.held_prev) # Reflects Scene Graph
                # node = GraphNode(obj['name'], object_id=obj['objectId']) # BETR-XP-LLM Scene Graph
                self.scene_graph.add_node_wo_edge(node)
                if node is not None:
                    self.scene_graph.add_node(node)

            elif obj['objectId'] in self.object_position_known.keys():
                if self.object_position_known[obj['objectId']] == True:
                    # self.scene_graph_nodes.append(obj['objectId'])
                    self.object_positions[obj['objectId']] = self.dict_to_pos(obj['position'])
            else:
                self.object_position_known[obj['objectId']] = False

        elif obj['visible']:
            if not self.object_position_known[obj['objectId']]:
                self.object_position_known[obj['objectId']] = True
                self.object_positions[obj['objectId']] = self.dict_to_pos(obj['position'])

            if self.calc_distance3d(obj['objectId'], self.dict_to_pos(obj['position'])) > 0.01 and not obj["isPickedUp"]: # Object moved
                self.object_positions[obj['objectId']] = self.dict_to_pos(obj['position'])

                remove_list = []
                for edge in self.scene_graph.edges.keys():
                    if obj['objectId'] in edge:
                        remove_list.append(edge)
                for edge in remove_list:
                    self.scene_graph.edges.pop(edge)

                node = gen_node(obj, event, obj['objectId'] in self.held_prev) # Reflects Scene Graph
                # node = GraphNode(obj['name'], object_id=obj['objectId']) # BETR-XP-LLM Scene Graph
                if node is not None:
                    self.scene_graph.add_node(node)
                    
        # elif self.object_position_known[obj['objectId']] == False:
        #     self.scene_graph_nodes.remove(obj['objectId'])
        #     for edge in self.scene_graph.edges.keys():
        #         if obj['objectId'] in edge:
        #             self.scene_graph.edges.pop(edge)

    def calc_distance3d(self, target_object, position):
        """ Calculates the distance between target object and given position """
        return np.linalg.norm(self.object_positions[target_object] - position)

    def get_id(self, obj_name):
        """ Get the object id from the object name """
        for obj in self.controller.last_event.metadata['objects']:
            if obj_name in obj['name']:
                return obj['objectId']
        return None

    def get_name(self, obj_id):
        """ Get the object name from the object id """
        for obj in self.controller.last_event.metadata['objects']:
            if obj['objectId'] == obj_id:
                return obj['name']
        return None

    def get_obj(self, obj_id):
        """ Get the object from the object id """
        return next(obj for obj in self.controller.last_event.metadata['objects'] if obj["objectId"] == obj_id)

    def get_position(self, target_object):
        """ Get the position of an object """
        for obj in self.controller.last_event.metadata['objects']:
            if obj['objectId'] == target_object:
                return obj['position']

    def is_near_robot(self, target_object, distance=1):
        """ Checks if object is within reach """
        self.robot_position = self.controller.last_event.metadata['agent']['position']
        self.object_positions[target_object] = self.dict_to_pos(self.get_position(target_object))
        print("diff: ", self.calc_distance(target_object, self.dict_to_pos(self.robot_position)))
        print("distance threshold: ", distance)
        # self.object_position_known[target_object] = True
        if self.object_position_known[target_object] and \
            self.calc_distance(target_object, self.dict_to_pos(self.robot_position)) < distance:
            return True
        else:
            return False

    def object_at(self, target_object, relation, relative_object):
        """ Checks if object is at a specific relation to another object """
        if (target_object, relative_object) in self.scene_graph.edges.keys():
            return relation == self.scene_graph.edges[(target_object, relative_object)].edge_type
        else:
            return False

    def move(self, direction, magnitude=0.25):
        """ Move one step in the specified direction """
        return self.controller.step(action=DIRECTIONS[direction], moveMagnitude=magnitude)

    def rotate(self, rotation, degrees=90):
        """ Rotate the robot in the specified direction """
        return self.controller.step(action=ROTATIONS[rotation], degrees=degrees)
    
    def move_armbase(self, height=0.75):
        e = self.controller.step(
            action="MoveArmBase",
            y=height,
            speed=1,
            returnToStart=True,
            fixedDeltaTime=0.02
        )
        if e.metadata['lastActionSuccess']:
            print(e.metadata['errorMessage'])

    def move_linear(self, position, orientation=None):
        """ Move the arm end-effector to a specific location """
        e = self.controller.step(action="MoveArm",
                                    position=position,
                                    coordinateSpace="world",
                                    restrictMovement=False,
                                    speed=1,
                                    returnToStart=False,
                                    fixedDeltaTime=0.02
                                )
        if e.metadata['lastActionSuccess']:
            print(e.metadata['errorMessage'])
        else:
            print(e.metadata['actionReturn'])

    def move_cfree(self, position, orientation=None):
        """ Move the arm end-effector to a specific location along a collision-free path """
        return self.controller.step(action="MoveArm",
                                    position=position,
                                    coordinateSpace="world",
                                    restrictMovement=True,
                                    speed=1,
                                    returnToStart=False,
                                    fixedDeltaTime=0.02
                                )

    def teleport_to(self, target_object):
        """ Navigate to a specific object """
        position = self.get_position(target_object)
        return self.controller.step(action="Teleport", position=position)

    def pick_up(self, target_object):
        """ Pick up an object """
        self.controller.step(action='PickupObject', objectId=target_object, forceAction=True, manualInteract=False)

    def drop(self):
        """ Drop the object held by the robot """
        return self.controller.step(action='ReleaseObject')

    def place_obj(self, target_object, position):
        """ Place an object at a specific location """
        if type(position) == str:
            position = self.get_position(position)
        else:
            position = self.pos_to_dict(position)
        if self.grasped_object == target_object:
            return self.controller.step(action='PlaceObjectAtPoint', objectId=target_object, position=position)

    def put_on(self, target_object, receptacle):
        """ Put an object on another object """
        placing_position = self.controller.step(action="GetSpawnCoordinatesAboveReceptacle", objectId=receptacle).metadata['actionReturn']
        return self.place_obj(target_object, placing_position)

    def put_in(self, target_object, receptacle):
        """ Put an object in another object """

        receptacle_obj = self.get_obj(receptacle)
        target_obj = self.get_obj(target_object)
        target_obj_type = target_obj['objectType']
        # src_obj = self.controller.last_event.metadata['arm']['heldObjects'][0] if len(self.controller.last_event.metadata['arm']['heldObjects']) > 0 else None
        src_obj = self.get_obj(self.grasped_object)

        if len(receptacle_obj['receptacleObjectIds']) > 0:
            print("[ERROR] Receptacle is already occupied")
            return None

        print(f"[INFO] Execute action: Putting {target_object} in {receptacle}")

        if src_obj is None:
            print("The robot is not holding anything")
        elif src_obj['objectType'] != target_obj_type:
            print(f"The robot is not holding {target_obj_type}")
        else:
            print("The robot is holding:", src_obj['objectId'], src_obj['objectType'])

        # thor-specific, put in sink sometimes does not work as expected
        if target_obj_type == 'Sink':
            target_obj_type = 'SinkBasin'

        receptacle_pos = receptacle_obj['position']

        # if navigation is required
        if not receptacle_obj['visible'] and receptacle_obj['objectType'] not in ['Floor', 'Wall', 'Ceiling']:
            self.navigate_to_obj(receptacle)

        # look at object
        robot_pos = self.controller.last_event.metadata['agent']['position']
        self.look_at(target_pos=receptacle_pos)

        # can only put one object in microwave
        if target_obj_type == 'Microwave' and len(receptacle_obj['receptacleObjectIds']) > 0:
            print("Microwave already contains an object: ", receptacle_obj['receptacleObjectIds'])
            e = self.controller.last_event
            return

        if target_obj_type == 'Toaster' and receptacle_obj['isToggled']:
            place_obj_in_small_receptacle(receptacle_pos)
        else:
            if src_obj:
                e = self.controller.step(
                    action="PutObject",
                    objectId=receptacle,
                    forceAction=False,
                    placeStationary=True
                )
                if e.metadata['lastActionSuccess']:
                    self.controller.step(action="Done")
                    time.sleep(1)
                else:
                    print("thor put_obj did not work, try place obj in small recetacle primitive")
                    if target_obj_type not in ["CoffeeMachine", "Microwave"]:
                        place_obj_in_small_receptacle(self, receptacle_pos)

        return self.controller.step(action="Done")

    def toggle_on(self, target_object):
        """ Toggle an object on """
        return self.controller.step(action='ToggleObjectOn', objectId=target_object)

    def toggle_off(self, target_object):
        """ Toggle an object off """
        return self.controller.step(action='ToggleObjectOff', objectId=target_object)
    
    def is_toggled(self, target_object):
        """ Check if an object is toggled on """
        return self.get_obj(target_object)['isToggled']

    def open_obj(self, target_object):
        """ Open an object """
        return self.controller.step(action='OpenObject', objectId=target_object)
   
    def close_obj(self, target_object):
        """ Open an object """
        return self.controller.step(action='CloseObject', objectId=target_object)
     
    def is_opened(self, target_object):
        """ Check if an object is opened """
        return self.get_obj(target_object)['isOpen']

    def close_obj(self, target_object):
        """ Close an object """
        return self.controller.step(action='CloseObject', objectId=target_object)

    def fill_obj(self, target_object, liquid):
        """ Fill an object with liquid """
        return self.controller.step(action='FillObjectWithLiquid', objectId=target_object, receptacleObjectId=liquid)

    def crack_obj(self, target_object):
        """ Crack an object """
        return self.controller.step(action='BreakObject', objectId=target_object)

    def slice_obj(self, target_object):
        """ Slice an object """
        return self.controller.step(action='SliceObject', objectId=target_object)

    def pos_to_dict(self, pos):
        return {'x': pos[0], 'y': pos[1], 'z': pos[2]}

    def dict_to_pos(self, pos):
        return np.array([pos['x'], pos['y'], pos['z']])

    @staticmethod
    def get_2d_reachable_points(reachable_positions):
        reachable_points = []
        for p in reachable_positions:
            reachable_points.append([p['x'], p['z']])
        reachable_points = np.array(reachable_points)
        return reachable_points

    def navigate_to_obj(self, target_object, counter=0):
        print("[INFO] Execute action: Navigate to", target_object)
        target_obj = self.get_obj(target_object)
        self.nav_actions = {}

        # BFS search for poth
        reachable_positions = self.controller.step(action="GetReachablePositions").metadata['actionReturn']
        reachable_points = self.get_2d_reachable_points(reachable_positions)
        closest_pos = closest_position(target_obj["position"], reachable_positions)
        robot_pos = self.controller.last_event.metadata['agent']['position']
        target_pos_val = [closest_pos['x'], closest_pos['z']]
        # print("robot_pos, target_pos, closest_to_target_pos: ", robot_pos, target_pos_val, closest_pos)
        # calculate grid_index from grid_value
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                if [round(robot_pos['x'], 2), round(robot_pos['z'], 2)] == [self.grid[row, col, 0], self.grid[row, col, 1]]:
                    robot_x = row
                    robot_y = col
                if [round(target_pos_val[0], 2), round(target_pos_val[1], 2)] == [self.grid[row, col, 0], self.grid[row, col, 1]]:
                    target_x = row
                    target_y = col
        robot_pos = [robot_x, robot_y]
        target_pos = [target_x, target_y]
        print("*** start, goal: ", robot_x, robot_y, target_pos)
        path = findPath(self.grid, x=robot_x, y=robot_y, target_pos=target_pos, reachable_points=reachable_points)
        # print("path: ", path)

        if path is None:
            print("[ERROR] No valid path is found from robot to target object")
            self.controller.step(action="Done")
            return

        for p in path:
            x = self.grid[p.x,p.y][0]
            z = self.grid[p.x,p.y][1]
            y = 0.9
            # print("p: ", p, x, z)
            e = self.controller.step(
                action="Teleport",
                position=dict(x=x, y=y, z=z),
                forceAction=True,
                # horizon=30,
                standing=True
            )
            self.controller.step(action="Done")
            self.get_feedback()
            # self.grasped_object = self.controller.last_event.metadata['arm']['heldObjects'][0]['objectId'] if len(self.controller.last_event.metadata['arm']['heldObjects']) > 0 else None

        self.look_at(target_pos=target_obj["position"])
        return self.controller.step(action="Done")

    def look_at(self, target_pos, center_to_camera_disp=0.6):
        robot_pos = self.controller.last_event.metadata['agent']['position']
        robot_y = robot_pos['y'] + center_to_camera_disp
        yaw = np.arctan2(target_pos['x']-robot_pos['x'], target_pos['z']-robot_pos['z'])
        yaw = math.degrees(yaw)

        tilt = -np.arctan2(target_pos['y']-robot_y, np.sqrt((target_pos['z']-robot_pos['z'])**2 + (target_pos['x']-robot_pos['x'])**2))
        tilt = np.round(np.math.degrees(tilt),1)
        org_tilt = self.controller.last_event.metadata['agent']['cameraHorizon']
        final_tilt = tilt - org_tilt
        if tilt > 60:
            final_tilt = 60
        if tilt < -30:
            final_tilt = -30
        final_tilt = np.round(final_tilt, 1)

        e = self.controller.step(action="Teleport", **robot_pos, rotation=dict(x=0, y=yaw, z=0), forceAction=True)
        self.controller.step(action="Done")
        # print("tilt degree: ", final_tilt)
        if final_tilt > 0:
            e = self.controller.step(
                action="LookDown",
                degrees=final_tilt
            )
        elif final_tilt < 0:
            e = self.controller.step(
                action="LookUp",
                degrees=-final_tilt
            )
        return self.controller.step(action="Done")

    def place_obj_in_small_receptacle(self, place_location):
        print("[INFO] Running primitive to place object in small receptacle")
        robot_pos = self.controller.last_event.metadata['agent']['position']
        tilt = self.controller.last_event.metadata['agent']['cameraHorizon']
        dist = np.sqrt((robot_pos['x'] - place_location['x'])**2 + (robot_pos['z'] - place_location['z'])**2)
        # print("tilt, dist: ", tilt, dist)
        tilt = np.round(tilt, 1)
        dist = np.round(dist, 1) - 0.4
        # Look straight (tilt = 0)
        if tilt > 0:
            e = self.controller.step(
                action="LookUp",
                degrees=tilt
            )
        else:
            e = self.controller.step(
                action="LookDown",
                degrees=tilt
            )
        # print("Look: ", e)
        self.controller.step(action="Done")

        # Move object over receptacle
        e = self.controller.step(
            action="MoveHeldObjectAhead",
            moveMagnitude=dist,
            forceVisible=False
        )
        self.controller.step(action='Done')
        # print("move object: ", e)

        # Drop object
        e = self.controller.step(
            action="DropHandObject",
            forceAction=False
        )
        self.controller.step(action='Done')
        # print("drop object: ", e)

        # Look at the receptacle again
        if tilt > 0:
            e = self.controller.step(
                action="LookDown",
                degrees=tilt
            )
        else:
            e = self.controller.step(
                action="LookUp",
                degrees=tilt
            )
        # print("Look: ", e)
        return self.controller.step(action="Done")

    def place_obj_on_large_receptacle(self, src_obj, target_obj_id, thresh=0.8):
        print("[INFO] Running primitive to place object on large receptacle")
        if target_obj_id is None:
            print("target object id is not specified.")
        else:
            print("target object id:", target_obj_id)
        target_obj = self.get_obj(target_obj_id)
        target_obj_type = target_obj['objectType']
        if target_obj_type in NAME_MAP:
            target_obj_type_in_sim = NAME_MAP[target_obj_type]
        src_obj_type, src_obj_id = src_obj['objectType'], src_obj['objectId']
        src_obj_type_in_sim = src_obj_type
        if src_obj_type in NAME_MAP:
            src_obj_type_in_sim = NAME_MAP[src_obj_type]

        target_objs = []
        if target_obj_id is None:
            if "-" not in target_obj_type:
                robot_pos = self.controller.last_event.metadata['agent']['position']
                for obj in self.controller.last_event.metadata["objects"]:
                    if obj["objectType"] == target_obj_type:
                        temp_obj_pos = obj['position']
                        dist = np.sqrt((robot_pos['x'] - temp_obj_pos['x'])**2 + (robot_pos['z'] - temp_obj_pos['z'])**2)
                        tup = (dist, obj)
                        target_objs.append(tup)
                target_objs = sorted(target_objs, key=lambda d: d[0])
            else:
                for obj_unity_name, v in self.unity_name_map.items():
                    if v == target_obj_type:
                        target_obj = next(obj for obj in self.controller.last_event.metadata["objects"] if obj["name"] == obj_unity_name)
                        target_objs.append((0, target_obj))
        else: # target_obj_id is specified
            target_objs.append((0, target_obj))

        # check if target object is in view
        found_obj = False
        for dist, target_obj in target_objs:
            target_obj_id = target_obj['objectId']
            e = self.controller.step(
                action="GetSpawnCoordinatesAboveReceptacle",
                objectId=target_obj_id,
                anywhere=False
            )
            # print("spawnPoints: ", e)
            if e.metadata['lastActionSuccess'] and len(e.metadata['actionReturn']) > 0:
                found_obj = True
                print("receptacle found in current view")
                break

        # navigate to the closest target object
        if not found_obj:
            for i in range(len(target_objs)):
                _, target_obj = target_objs[i]
                print("[INFO] Navigate to the closest target object:", target_obj['objectId'])
                self.navigate_to_obj(target_obj['objectId'])
                e = self.controller.step(
                    action="GetSpawnCoordinatesAboveReceptacle",
                    objectId=target_obj['objectId'],
                    anywhere=True
                )
                # print("GetSpawnCoordinatesAboveReceptacle: ", e)
                if e.metadata['actionReturn'] is not None:
                    break

        print("chosen counterTop:", target_obj_id)
        self.controller.step(action="Done")
        time.sleep(1)
        place_locations = e.metadata['actionReturn']
        # print("total potential place points: ", len(place_locations))

        # find valid locations on the receptacle to put object
        placed = False
        visible = False
        robot_pos = self.controller.last_event.metadata['agent']['position']

        counter = 0
        while (not placed or not visible):
            counter += 1
            # if too many trials, just drop the object
            if counter > 200:
                e = self.controller.step(
                    action="DropHandObject",
                    forceAction=False
                )
                print("DropHandObject: ", e)
                break
            visible = False
            placed = False
            place_location = np.random.choice(place_locations)
            # place point should be close enough to robot
            dist = np.sqrt((robot_pos['x'] - place_location['x'])**2 + (robot_pos['z'] - place_location['z'])**2)
            if dist > thresh:
                continue
            e = self.controller.step(
                action="PlaceObjectAtPoint",
                objectId=src_obj_id,
                position=place_location
            )
            # print("PlaceObjectAtPoint: ", e)
            self.controller.step(action="Done")
            if e.metadata['lastActionSuccess']:
                placed = True
            if src_obj['visible']:
                visible = True

        self.look_at(place_location)
        return self.controller.step(action="Done")

    def run_program(self, programs):
        """ Run an action """
        try:
            for program, args in programs:
                if args == []:
                    event = program()
                else:
                    event = program(*args)
                if not event.metadata['lastActionSuccess']:            
                    return False
            return True
        except Exception as e:
            self.error_message = str(e)
            return False
