# ELEC4544 Final Project - IASD

Intelligent Automatic Speed Detection program with YOLOV8n CNN model (Ultralytics).

This is the place where the primary source code will be located and any helpful and relevant documents, useful links etc., will all be stored here at an appropriate time. 

To execute the code using command-line arguments, please locate the helpful ArgParsers.xlsx file to easily modify argument parsers as a stirng to input in the console.

**PRIMARY LIBRARIES USED:** 

*Numpy*<br />
*ultralytics*<br />
*matplotlib*<br />
*cv2*<br />

*Imported libraries used (dependencies):*<br /> 

from ultralytics import YOLO<br />
from ultralytics.utils.checks import check_imshow, check_requirements<br />
from ultralytics.utils.plotting import Annotator, colors<br />
import cv2<br />
import argparse<br />
from collections import defaultdict<br />
from shapely.geometry import LineString, Point, Polygon<br />
from time import time<br />
import numpy as np<br />







