from vis_nav_game import Player, Action
import pygame
import cv2
import math
import threading
import queue
import numpy as np
import hashlib
import time
from graph import Graph
from graph import Node


class KeyboardPlayerPyGame(Player):

    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.counter = 0
        self.graph = Graph()
        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.counter = 0
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]

        return self.last_act

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])
        # concat_img = cv2.imread('test_image.png')

        w, h = concat_img.shape[:2]

        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0),
                              (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)),
                              (h, int(w/2)), color, 2)
        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset),
                    font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset,
                    w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(
            w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset,
                    int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def dijkstra(self, start, end):
        """
        params: 
            start: the starting node
            end: the ending node
        return: the shortest path from start to end represented as a list of images and actions 
        """

    def pre_navigation(self):
        # TODO: 1. reconnect graph if we have not already done so in the exploration phase
        # TODO: 2. run Dijsktra's algorithm to find the shortest path to the target

        pass

    def add_node(self, fpv):  # add current fpv to the graph as a node
        if self.graph.get_root() is None:
            # set the first node as root
            self.graph.set_root(Node(fpv, self.last_act))
            self.graph.set_current(self.graph.get_root())
        elif self.last_act == Action.IDLE:
            return  # do nothing, no new image added
        else:
            node = Node(fpv, self.last_act)
            self.graph.get_current().add_edge(node)
            self.graph.set_current(node)

    def connect_nodes(self, fpv):  # connect the spatially adjacent nodes in graph
        if self.graph.get_root() or self.graph.get_current() is None:
            return

        # TODO 1. run vpr to find the spatially adjacent nodes
        # TODO 2. connect the spatially adjacent nodes in graph by adding an edge, can do it offline or online while exploring

    def see(self, fpv):
        self.fpv = fpv

        # add the current fpv to the graph as a node
        self.add_node(fpv)
        # connect the spatially adjacent nodes in graph every 5 frames
        if self.counter % 5 == 0:
            self.connect_nodes(fpv)

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            # (height,width,Number of colors) -> (width, height)
            shape = opencv_image.shape[1::-1]
            pygame_image = pygame.image.frombuffer(
                opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
