"""
This file contains base code to be implemented in the DWA cost function.

Currently contains the code for 
    creating a distance grid given an occupancy grid
        with values (0, 100, -1) which is the information that we will get from turtlebot
    distance to goal
    angle alignment to goal

NOTE: other items to be included in cost function will be pulled/produced via determining
different possible routes.
"""


import numpy as np
import math
from collections import deque
from typing import List, Tuple, Set

def getNeighborCoords(xcoordinate, ycoordinate, grid):
    """
    returns all valid neighbors for a given coordinate. Important to
    remember that a given map for turtlebot is a non-rectangular shape
    within a rectangular grid. the actual shape of the map will be denoted
    by unknown in the occupancy grid

    Args: 
        - int1: the x coordinate of a given point
        - int2: the y coordinate of a given point
        - list[][]: the occupancy grid of the map

    Returns:
        - list[(int,int)]: all valid neighbors for a given coordinate
    """
    #list of all horizontal and vertical neighbors
    """
    chebyshev distance
    neighbors = [(xcoordinate+1, ycoordinate+1),
            (xcoordinate+1, ycoordinate-1),
            (xcoordinate-1, ycoordinate+1),
            (xcoordinate-1, ycoordinate-1)]"""
    #manhattan distance
    neighbors = [
    (xcoordinate + 1, ycoordinate),
    (xcoordinate - 1, ycoordinate),
    (xcoordinate, ycoordinate + 1),
    (xcoordinate, ycoordinate - 1),
    ]
    #gathering all valid neighbors. This is not grid based because
    #the maps provided by tbot will likely need to be processed as a non-rectangular shape
    #eg. a circle shape in a square grid
    validNeighbors = []
    UNKNOWN = -1
    width, height = grid.shape

    for nx, ny in neighbors:
        if 0 <= nx < width and 0 <= ny < height:
            if grid[nx, ny] != UNKNOWN:
                validNeighbors.append((nx, ny))

    return validNeighbors

def initializeGrid(grid, WIDTH, HEIGHT, OCCUPIED):
    """
    initializes all necessary datastructures for the creation of distance grid
    including a 
        - list to keep track of checked points
        - a list to keep track of coordinates and their values
        - a deque (queue) to be used for BFS
    
    Args:
        - list[(int, int, int)]: grid given by turtlebot
        - int1: width of grid
        - int2: height of grid
        - int3: value grid items are set to when they are considered occupied

    Returns:
        - set[(int, int)]: a list to hold all visited coordinates
        - list[(int, int, int)]: a list to hold all final values
        - deque[(int, int, int)]: a deque used for BFS
    """
    checkedCoords = set()
    finalValues = []
    queuedCoords = deque()
    for xcoordinate in range(WIDTH):
        for ycoordinate in range(HEIGHT):
            if grid[xcoordinate, ycoordinate] == OCCUPIED:
                checkedCoords.add((xcoordinate, ycoordinate))
                finalValues.append((xcoordinate, ycoordinate, 0))
                queuedCoords.append((xcoordinate, ycoordinate, 0))
    return checkedCoords, finalValues, queuedCoords

def BFS_distanceCalculation(checkedCoords, finalValues, queuedCoords, grid):
    """
    applies BFS with all obstacles as start points in order to find the shortest distance
    to each spot in the grid
    
    Args:
        - list[(int, int)]: a list to hold all visited coordinates
        - list[(int, int, int)]: a list to hold all final values
        - deque[(int, int, int)]: a deque used for BFS
        - list[][]: the occupancy grid of the map

    Returns:
        - list[(int, int, int)]: a list of all coordinates and their value
    """
    while queuedCoords:
        curCoord = queuedCoords.popleft()
        neighborCoords = getNeighborCoords(curCoord[0], curCoord[1], grid)
        for coords in neighborCoords:
            if coords not in checkedCoords:
                queuedCoords.append((coords[0], coords[1], curCoord[2]+1))
                finalValues.append((coords[0], coords[1], curCoord[2]+1))
                checkedCoords.add(coords)
    return finalValues

def newGrid(finalValues, WIDTH, HEIGHT):
    """
    applies all values in finalValues to a new grid
    
    Args:
        - list[(int, int, int)]: a list of coordinates and values to apply to a distance grid
        - int1: width of grid
        - int2: height of grid

    Returns:
        - list[][]: distance grid
    """
    newGrid = np.zeros((WIDTH, HEIGHT), dtype=int)
    for coordinate in finalValues:
        newGrid[coordinate[0], coordinate[1]] = coordinate[2]
    return newGrid

def getDistanceGrid(grid, WIDTH, HEIGHT, OCCUPIED):
    """
    creates a distance grid based on an occupancy grid filled with
    OCCUPIED=100, UNKNOWN=-1 (referenced in getNeighborCoords()), and EMPTY=0

    Args:
        - list[int][int]: a grid representing the map object given by turtlebot
        - int1: the width of the grid
        - int2: the height of the grid

    Returns:
        - list[int][int]: a grid representing the distance grid based on input grid
    """
    #start the process by adding all start points (obstacles)
    checkedCoords, finalValues, queuedCoords = initializeGrid(grid, WIDTH, HEIGHT, OCCUPIED)
    
    #BFS
    finalValues = BFS_distanceCalculation(checkedCoords, finalValues, queuedCoords, grid)

    #transfer all calculations to a new grid
    return newGrid(finalValues, WIDTH, HEIGHT)
    

def testGetDistanceGrid():
    """
    tests getDistanceGrid() on a prebuilt grid and prints output to a file
    
    Args:

    Returns:
    """
    WIDTH = 100
    HEIGHT = 100
    # empty 100x100 grid
    grid = np.zeros((WIDTH, HEIGHT), dtype=int)

    OCCUPIED = 100
    UNKNOWN = -1

    # borders
    grid[0, :] = UNKNOWN
    grid[-1, :] = UNKNOWN
    grid[:, 0] = UNKNOWN
    grid[:, -1] = UNKNOWN

    # adding in obstacles
    grid[20:40, 20:25] = OCCUPIED# vertical block
    grid[60:65, 10:40] = OCCUPIED# horizontal block
    grid[30:50, 60:70] = OCCUPIED# square block
    grid[70:90, 80:85] = OCCUPIED# vertical block
    grid[45:55, 45:55] = OCCUPIED# center block
    distGrid = getDistanceGrid(grid, WIDTH, HEIGHT, OCCUPIED)

    #write the distance grid messily to a file
    f = open("output.txt", "w")
    for x in range(100):
        for y in range(100):
            f.write(str(distGrid[x][y]))
            f.write(",")
        f.write("\n")
    f.close()

def getDistanceToGoal(currentPos, goal):
    """
    finds euclidean distance to goal
    
    Args:
        - Tuple(int, int)1 : current location
        - Tuple(int, int)2 : goal location

    Returns:
    """
    return math.dist(currentPos, goal)

def getAngleAlignment(posx, posy, theta, goalx, goaly):
    """
    finds the angle error towards the goal
    
    Args:
        - int1: current x position of robot
        - int2: current y position of robot
        - int3: current angle of direction of robot
        - int4: goal x position
        - int5: goal y position

    Returns:
        - int: radians of angle error towards goal
    """
    dx = goalx - posx
    dy = goaly - posy
    goalAngle = math.atan2(dy, dx)
    angleErr = goalAngle - theta
    #get radians
    angleErr = (angleErr + math.pi) % (2 * math.pi) - math.pi
    return angleErr

def generateSizeContainingPoints(occupiedPoints):
    smallestX = largestX = occupiedPoints[0][0]
    smallestY = largestY = occupiedPoints[0][1]

    for x, y in occupiedPoints[1:]:
        smallestX = min(smallestX, x)
        largestX = max(largestX, x)
        smallestY = min(smallestY, y)
        largestY = max(largestY, y)

    return (largestX - smallestX, largestY - smallestY)

def generateRadiusWithPoints(occupiedPoints):
    longestDistance = max(occupiedPoints[0][0], occupiedPoints[0][1])
    for point in occupiedPoints[1:]:
        if point[0] > longestDistance:
            longestDistance = point[0]
        if point[1] > longestDistance:
            longestDistance = point[1]
    return longestDistance

def placeObstacles(grid, occupiedPoints):
    width, height = grid.shape
    for x, y in occupiedPoints:
        if 0 <= x < width and 0 <= y < height:
            grid[x, y] = 100
    return grid

def placeUnknown(grid, radius):
    width = grid.shape[0]
    height = grid.shape[1]
    center = (width / 2, height / 2)
    for x in range(width):
        for y in range(height):
            if getDistanceToGoal((x,y), center) > radius:
                grid[x][y] = -1
    return grid

def generateOccupancyGrid(occupiedPoints, radius=None):
    size = None
    if radius == None:
        radius = generateRadiusWithPoints(occupiedPoints)
    size = (radius*2,radius*2)
    WIDTH = size[0]
    HEIGHT = size[1]
    grid = np.zeros((WIDTH, HEIGHT), dtype=int)
    grid = placeObstacles(grid, occupiedPoints)
    grid = placeUnknown(grid, radius)
    return grid

def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    cells = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy
    x, y = x0, y0

    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return cells

def get_path(points):
    complete_path = []
    for i in range(len(points)-1):
        complete_path.extend(bresenham(points[i][0], points[i][1], points[i+1][0], points[i+1][1])[1:])
    return complete_path

def get_path_cost(path, distanceGrid):
    total_cost = 0
    for point in path:
        x_pos = point[0] + int(distanceGrid.shape[0] / 2)
        y_pos = point[1] + int(distanceGrid.shape[1] / 2)
        total_cost+= distanceGrid[x_pos][y_pos]
    return total_cost

def get_costs_for_all_paths(all_paths, occupiedPoints, radius=None):
    grid = generateOccupancyGrid(occupiedPoints, radius=None)
    grid = getDistanceGrid(grid,grid.shape[0],grid.shape[1],100)
    all_costs = []
    for path in all_paths:
        all_costs.append(get_path_cost(path, grid))
    return all_costs
