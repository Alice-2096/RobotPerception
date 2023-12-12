from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import math
import threading
import queue
import numpy as np
from config import update_config
from vpr_new import generate_histogram, train_vocab, best_match


class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        super(KeyboardPlayerPyGame, self).__init__()
        self.kmeans = None
        self.path = None  # list of actions to navigate to the target image
        self.current_index = 0  # index of the current pos while walking the path

        # each element is a {histogram, image, keypress} pair
        self.live_image_list = []
        self.counter = 0

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.path = None
        self.current_index = 0
        self.kmeans = None
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def pre_exploration(self):
        
        if self.kmeans is not None:
            return  # already trained

        print('pre exploration phase ...')
        # Update YAML file with K matrix
        K = self.get_camera_intrinsic_matrix()
        # print(type(K))
        # if K is not None:
        #     update_config('test.yaml', K)

        # build visual vocab
        kmeans = train_vocab(200)
        self.kmeans = kmeans
        print('done updating kmeans, now entering pre navigation phase ...')
        

    def pre_navigation(self):
        
        print('pre navigation phase begins...')
        # find the best match from the live image list
        hist_list = [item['hist'] for item in self.live_image_list]
        target_images = self.get_target_images()
        kmeans = self.kmeans
        if hist_list is None or len(hist_list) <= 0:
            print('No live images found.')
            return
        print("found live images: ", len(hist_list))
        best_match_indices = best_match(target_images, kmeans, hist_list)
        print('Best match indices: ', best_match_indices)
        for i in best_match_indices:
            image = self.live_image_list[i]['image']
            # show image in another window
            cv2.imshow(f'Best Match #{i}', image)
            cv2.waitKey(1)

        # user enter the index of the desired target image to navigate to
        target_index = int(input('Enter the index of the target image: '))
        # output the keys to navigate to the target image from index 0 to target_index, skip IDLE
        self.path = []
        
        for i in range(target_index):
            key = self.live_image_list[i]['key']
            if key != Action.IDLE:
                self.path.append(key)
        print('Path length to target image: ', len(self.path))

    def act(self):
        # if in NAV phase, follow the pre-determined path
        
        if self.path is not None and len(self.path) > 0:
            if self.current_index >= len(self.path):
                print('Reached target image!')
                self.path = None
                self.current_index = 0
            else:
                act = self.path[self.current_index]
                self.current_index += 1
                return act

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

    def build_hist(self, fpv):
        print('process captured images ...')
        if self.kmeans is not None:
            hist = generate_histogram(fpv, self.kmeans)
            self.live_image_list.append(
                {'hist': hist, 'image': fpv, 'key': self.last_act})

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv
        self.counter += 1

        # build histogram in every frame
        self.build_hist(fpv)
        # !alternatively, build histogram every K frames
        # k = 5
        # if self.counter % k == 0:
        #     self.build_hist(fpv)

        # state = self.get_state()
        # if state is not None:
        #     if state == Phase.EXPLORATION:  # in the NAV phase, we generate histogram for each captured image
        #         # ! for some reason this does not run at all in the exploration phase
        #         print('EXPLOREATION!')
        #         pass

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
