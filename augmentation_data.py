import Augmentor
import os


def rotate_img(furniture):
    furniture.rotate(probability=1,
                     max_left_rotation=25,
                     max_right_rotation=25)
    furniture.process()


def zoom_img(furniture):
    furniture.zoom(probability=1,
                   min_factor=1.2,
                   max_factor=1.7)
    furniture.process()


def flip_img(furniture):
    furniture.flip_left_right(probability=1)
    furniture.process()


def skew_img(furniture):
        furniture.skew(probability=1,
                       magnitude=0.7)
        furniture.process()


for root, dirs, files in os.walk("./furniture/"):
    for directory in dirs:
        if directory != "output":
            furniture = Augmentor.Pipeline("./furniture/" + directory)
            rotate_img(furniture)
            zoom_img(furniture)
            flip_img(furniture)
            skew_img(furniture)
