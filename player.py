from vis_nav_game import Player, Action
import pygame
import cv2
import math
import threading
import queue
import numpy as np
from config import update_config

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        super(KeyboardPlayerPyGame, self).__init__()
        
        self.arrow_pos = np.array([300, 300])
        self.arrow_angle = 90  # Angle in degrees, 90 means pointing upwards
        self.arrow_trail = []  # List to store the positions of the arrow

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
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
        # Update YAML file with K matrix 
        K = self.get_camera_intrinsic_matrix() 
        print(type(K))
        if K is not None:
            update_config('test.yaml', K)
        # Initialize SLAM 

    
    def show_additional_window(self):
        """
        Display an additional window with a green arrow.
        The direction of the arrow is based on the last action.
        """
        # Creating a black canvas
        canvas = np.zeros((600, 600, 3), dtype=np.uint8)

        # Drawing the trail
        for i in range(1, len(self.arrow_trail)):
            cv2.line(canvas, tuple(self.arrow_trail[i-1]), tuple(self.arrow_trail[i]), (0, 255, 0), 2)
        
               
        # Calculate the tip of the arrow based on position and angle
        tip_x = self.arrow_pos[0] + 10 * math.cos(math.radians(self.arrow_angle))
        tip_y = self.arrow_pos[1] - 10 * math.sin(math.radians(self.arrow_angle))

        # Drawing the arrow
        cv2.arrowedLine(canvas, tuple(self.arrow_pos), (int(tip_x), int(tip_y)), (0, 255, 0), 5)

        # Displaying the canvas
        cv2.imshow('Additional Window', canvas)
        cv2.waitKey(1)


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

        move_step = 5
        rotate_step = 2.5
          
        if self.last_act & Action.FORWARD:
            self.arrow_pos[0] += move_step * math.cos(math.radians(self.arrow_angle)) 
            self.arrow_pos[1] -= move_step * math.sin(math.radians(self.arrow_angle))
        elif self.last_act & Action.BACKWARD:
            self.arrow_pos[0] -= move_step * math.cos(math.radians(self.arrow_angle)) 
            self.arrow_pos[1] -= move_step * math.sin(math.radians(-self.arrow_angle))
  


        if self.last_act & Action.LEFT:
            self.arrow_angle += rotate_step
        elif self.last_act & Action.RIGHT:
            self.arrow_angle -= rotate_step
        
        #self.arrow_angle %= 360  # Keep the angle within 0 to 359 degrees

        # Add the new position to the trail
        self.arrow_trail.append(self.arrow_pos.copy())

        self.show_additional_window()
        return self.last_act
    
    def match_features(self, img1, img2):
        """Match features between two images."""
        orb = cv2.ORB_create()

        # Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # Create a brute force matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort them in ascending order of distance
        matches = sorted(matches, key=lambda x: x.distance)

        return len(matches)

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)
        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)



        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        targets = self.get_target_images()
        
        state = self.get_state() 
        # print(state)
        print(self.get_camera_intrinsic_matrix())

        target_names = ["Front", "Left", "Back", "Right"]            
      
        for index, target in enumerate(targets):
            match_score = self.match_features(fpv, target)

            # Check if the match score meets the threshold
            if match_score > 147:  # Using a threshold of 148                
                # Display the match score for each target on the FPV with corresponding names
                cv2.putText(self.fpv, f'{target_names[index]} View, Score: {match_score}', 
                    (40, 40 ),  # Adjust position based on target index
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
