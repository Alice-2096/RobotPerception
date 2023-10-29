from vis_nav_game import Player, Action
import pygame
import cv2
import numpy as np
import pypangolin as pangolin
import OpenGL.GL as gl

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.fst_image = None  # Initialize previous image to None
        self.initPhase = True 
        self.map = None 
        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.fst_image = None

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
        rgb = convert_opencv_img_to_pygame(fpv) #current pic 
        if self.fst_image is None:
            self.fst_image = fpv
        elif self.initPhase:  
            self.build_map_init(self.fst_image, fpv)
            self.initPhase = False 
    
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

   
    # match features between the first two frames, estimate camera pose, and triangulate the inlier points to obtain the initial map
    def build_map_init(self, image1, image2): 
        # ORB feature detection
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Detect keypoints and compute descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Match features 
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        """
        1. Estimate the Essential Matrix (E) and the Fundamental Matrix (F) from the feature matches
        2. Decomposed E into [R|t] using SVD to retrieve camera pose 
        3. Triangulate the inlier points to obtain the initial map
        4. Refine the 3D points and camera poses to minimize the reprojection error using bundle adjustment
        5. Place the optimized points triangulated from frame 1 and frame 2 to the local map
        """ 
        # Applying RANSAC to matchings and Compute the Essential Matrix
        K = self.get_camera_intrinsic_matrix() 
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=3.0)

        # Recover pose (R, t) from Essential Matrix using SVD decomposition
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K)

        # Use the mask to filter out outliers, the rest is inliers 
        src_pts = src_pts[mask.ravel() == 1]
        dst_pts = dst_pts[mask.ravel() == 1]

        # print("src_pts size: ", src_pts.shape)
        print("dst_pts size: ", dst_pts.shape)
        
        # print("R: ", R)
        # print("t: ", t)
        
        # Triangulate for inlier points --  calculates the 3D coordinates of the inliear feature points in the world coordinate system 
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))
        points_3d = cv2.triangulatePoints(P1, P2, src_pts.T, dst_pts.T).T 
        # Assuming points_3d is (352, 1, 4)
        points_3d = points_3d.reshape(-1, 4)  # Reshape to (352, 4)

        # Extract the X, Y, and Z coordinates
        x_coordinates = points_3d[:, 0]
        y_coordinates = points_3d[:, 1]
        z_coordinates = points_3d[:, 2]

        # Combine them into an Nx3 array
        points_3d = np.column_stack((x_coordinates, y_coordinates, z_coordinates))


        print("points_3d size: ", points_3d.shape)
        # Refine the 3D points and camera poses to minimize the reprojection error using bundle adjustment
        reproj_thresh = 1.0 #! what value should it be? 
        reproj_error, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, dst_pts, K, None, flags=cv2.SOLVEPNP_P3P,
        iterationsCount=1000, reprojectionError=reproj_thresh) 
        R_refined, _ = cv2.Rodrigues(rvec) 
        
        # Place the optimized points triangulated from frame 1 and frame 2 to the local map
        # Initialize Pangolin and create a window
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        handler = pangolin.Handler3D(scam)

        # Create a 3D view for the local map
        d_cam = pangolin.CreateDisplay()
        d_cam.SetBounds(pangolin.Attach(0),
                        pangolin.Attach(1), 
                        pangolin.Attach(0),
                        pangolin.Attach(1), 640/480)
        d_cam.SetHandler(handler)

        # Create a 3D axis for reference
        d_axis = pangolin.CreateDisplay()
        d_axis.SetBounds(pangolin.Attach(0),
                        pangolin.Attach(1), 
                        pangolin.Attach(0),
                        pangolin.Attach(1), 640/480)

        # Set up the OpenGL viewport for the local map
        # Clear the view
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Activate the 3D view for the local map
        d_cam.Activate()

        # ! buggy -- Load the camera pose <- what does it do? 
        # pangolin.glDrawFrustum(K, 640, 480, 0.1, 1000, pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000), R_refined, -R_refined.T @ tvec)

        # Draw the refined 3D points
        gl.glColor3f(0.0, 1.0, 0.0)  # Set color to green
        gl.glPointSize(2)  # Adjust point size as needed
        gl.glBegin(gl.GL_POINTS)
        for point in points_3d:
            gl.glVertex3d(point[0], point[1], point[2])
        gl.glEnd()

        self.map = gl # Save the map for updates 

        # Activate the 3D axis display
        d_axis.Activate()

        pangolin.FinishFrame()
        
    """
    for every new frame, 
    1. estimate camera pose based on matching features with the previous frame
    2. create new 3D points using PnP and triangulation
    3. optimize the map using bundle adjustment for every 100 frames 
    """ 
    # TODO called every frame to update the map with new points 
    def map_new_points(self, current_image):
        pass
    
    # TODO called every 100 frames to optimize the map using bundle adjustment
    def optimize_map(self):
        pass

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
