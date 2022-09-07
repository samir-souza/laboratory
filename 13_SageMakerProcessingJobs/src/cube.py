from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import os
import cv2
import sys
import time
import argparse
import numpy as np

# The cube class
class Cube:

    # Constructor for the cube class
    def __init__(self, window_size=(400,400)):
        self.window_size = window_size
        self.rotate_y = 0.0
        self.rotate_x = 0.0
        self.rotate_z = 0.0

    # Initialize
    def init(self):
        # Set background to black
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(self.window_size[0])/float(self.window_size[1]), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)


    # The display function
    def display(self):
        # Reset the matrix
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glTranslatef(0.0,0.0,-6.0)

        glRotatef(self.rotate_x,1.0,0.0,0.0)
        glRotatef(self.rotate_y,0.0,1.0,0.0)
        glRotatef(self.rotate_z,0.0,0.0,1.0)

        glBegin(GL_QUADS)

        glColor3f(0.0,1.0,0.0)
        glVertex3f( 1.0, 1.0,-1.0)
        glVertex3f(-1.0, 1.0,-1.0)
        glVertex3f(-1.0, 1.0, 1.0)
        glVertex3f( 1.0, 1.0, 1.0)

        glColor3f(1.0,0.0,0.0)
        glVertex3f( 1.0,-1.0, 1.0)
        glVertex3f(-1.0,-1.0, 1.0)
        glVertex3f(-1.0,-1.0,-1.0)
        glVertex3f( 1.0,-1.0,-1.0)

        glColor3f(0.0,1.0,0.0)
        glVertex3f( 1.0, 1.0, 1.0)
        glVertex3f(-1.0, 1.0, 1.0)
        glVertex3f(-1.0,-1.0, 1.0)
        glVertex3f( 1.0,-1.0, 1.0)

        glColor3f(1.0,1.0,0.0)
        glVertex3f( 1.0,-1.0,-1.0)
        glVertex3f(-1.0,-1.0,-1.0)
        glVertex3f(-1.0, 1.0,-1.0)
        glVertex3f( 1.0, 1.0,-1.0)

        glColor3f(0.0,0.0,1.0)
        glVertex3f(-1.0, 1.0, 1.0)
        glVertex3f(-1.0, 1.0,-1.0)
        glVertex3f(-1.0,-1.0,-1.0)
        glVertex3f(-1.0,-1.0, 1.0)

        glColor3f(1.0,0.0,1.0)
        glVertex3f( 1.0, 1.0,-1.0)
        glVertex3f( 1.0, 1.0, 1.0)
        glVertex3f( 1.0,-1.0, 1.0)
        glVertex3f( 1.0,-1.0,-1.0)

        glEnd()
        glutSwapBuffers()

    def step(self, key, step=5):
        # Rotate cube
        if key == 0:
            self.rotate_y += step
        if key == 1:
            self.rotate_y -= step
        if key == 2:
            self.rotate_x += step
        if key == 3:
            self.rotate_x -= step
        glutPostRedisplay()

def generate_images(num_mosaics, image_size, mosaic_tiles):
    glutInit(sys.argv)

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(image_size, image_size)
    glutInitWindowPosition(100, 100)

    print("Creating window...")
    windowId = glutCreateWindow(b"OpenGL Cube")
    glutHideWindow() # faster
    
    cube = Cube(window_size=(image_size, image_size))
    cube.init()    
    s = time.time()
    print(f"Start generating images at {s}")   
    mosaic_size=mosaic_tiles*image_size
    images_per_mosaic = mosaic_tiles ** 2
    for num_mosaic in range(num_mosaics):
        print(f"Creating mosaic {num_mosaic}/{num_mosaics}")
        mosaic = np.zeros((mosaic_size, mosaic_size, 4), dtype=np.uint8)
        for num_image in range(images_per_mosaic):
            row = num_image // mosaic_tiles
            col = num_image % mosaic_tiles
            p_row = row*image_size
            p_col = col*image_size   

            cube.display()
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            data = glReadPixels(0, 0, image_size, image_size, GL_RGBA, GL_UNSIGNED_BYTE)
            image = np.frombuffer(data, dtype=np.uint8).reshape(image_size,image_size,4)
    
            mosaic[p_col:p_col+image_size, p_row:p_row+image_size] = image[:]
            cube.step(np.random.randint(0,4), np.random.randint(5,50))
        cv2.imwrite(f"/opt/ml/processing/output/mosaic_{num_mosaic:015d}.png", mosaic)

    print(f"Elapsed time to generate {num_mosaics * (mosaic_tiles ** 2)} images: {time.time()-s}")
    print(f"Destroying window {windowId}")    
    glutDestroyWindow(windowId)
    
if __name__=='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-mosaics', type=int, default=10)
    parser.add_argument('--mosaic-tiles', type=int, default=100)
    parser.add_argument('--image-size', type=int, default=400)

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    if args.mosaic_tiles * args.image_size > 65500:
        raise Exception( f"Invalid combination mosaic_tiles {args.mosaic_tiles} x image_size {args.image_size}. It needs to be at most 65500")

    generate_images(args.num_mosaics, args.image_size, args.mosaic_tiles)
