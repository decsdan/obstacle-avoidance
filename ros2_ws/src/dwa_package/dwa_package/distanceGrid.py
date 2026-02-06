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
    height, width = grid.shape

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
    newGrid = np.full((WIDTH, HEIGHT), -1, dtype=int)
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
    if not occupiedPoints:
        return (0, 0)
    smallestX = largestX = occupiedPoints[0][0]
    smallestY = largestY = occupiedPoints[0][1]

    for x, y in occupiedPoints[1:]:
        smallestX = min(smallestX, x)
        largestX = max(largestX, x)
        smallestY = min(smallestY, y)
        largestY = max(largestY, y)

    return (largestX - smallestX, largestY - smallestY)

def generateRadiusWithPoints(occupiedPoints):
    if not occupiedPoints:
        return 0
    cx = sum(x for x, _ in occupiedPoints) / len(occupiedPoints)
    cy = sum(y for _, y in occupiedPoints) / len(occupiedPoints)
    return max(math.hypot(x - cx, y - cy) for x, y in occupiedPoints)

"""np is row order so arr[row][col]"""

def print_grid_to_file(grid, filename="gridPrint.txt"):
    with open(filename, "w") as f:
        for row in grid:
            f.write(" ".join(str(cell) for cell in row) + "\n")

def place_obstacles(grid, occupiedPoints):
    for point in occupiedPoints:
        #grid[row][col]
        grid[point[1], point[0]] = 100

def clean_points(points):
    transformedPoints = []
    for point in points:
        transformedX = int(point[0]*10) + 80
        transformedY = int(point[1]*10) + 80
        if transformedX < 0 or transformedX > 160 or transformedY < 0 or transformedY > 160:
            continue
        transformedPoints.append((transformedX, transformedY))
    return transformedPoints

def generate_obstacle_grid(occupiedPoints, size = (0,0)):
    WIDTH = 161
    HEIGHT = 161
    obstacleGrid = np.zeros((HEIGHT, WIDTH), dtype=int)
    place_obstacles(obstacleGrid, occupiedPoints)
    return obstacleGrid

def distance_from_obstacles(obstacleGrid: np.ndarray) -> np.ndarray:
    HEIGHT, WIDTH = obstacleGrid.shape
    # setting up the queue and a grid of -1 for unvisited sspots
    dist_grid = -np.ones_like(obstacleGrid, dtype=int)
    queue = deque()
    
    # find all spots where there's an obstacle and add it to queue
    obstacle_coords = np.argwhere(obstacleGrid == 100)
    for y, x in obstacle_coords:
        dist_grid[y, x] = -2
        queue.append((y, x))
    
    # manahttan distance neighbors
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    #this is the actual bfs part
    while queue:
        y, x = queue.popleft()
        current_dist = dist_grid[y, x]
        
        #check neighbors
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < HEIGHT and 0 <= nx < WIDTH and dist_grid[ny, nx] == -1:
                dist_grid[ny, nx] = current_dist + 1
                queue.append((ny, nx))
    
    return dist_grid

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

def get_path_given_points(trajectory):
    finalPath = []

    for i in range(len(trajectory) - 1):
        x0, y0 = trajectory[i]
        x1, y1 = trajectory[i + 1]
        line_points = bresenham(x0, y0, x1, y1)
        finalPath.extend(line_points[1:])

    # this removes the duplicates but keeps the order
    # there would normally be duplicates from the start and end points
    finalPath = list(dict.fromkeys(finalPath))
    return finalPath

def get_path_cost(path, distanceGrid):
    totalCost = 0
    for x,y in path:
        if distanceGrid[y, x] <= 0:
            cost = math.inf
        else:
            cost = math.exp(-1 * distanceGrid[y, x])
        totalCost += cost
    return totalCost

def normalize_path_costs(allCosts):
    # these are all of the non infinite values
    finiteCosts = [cost for cost in allCosts if math.isfinite(cost)]

    minVal = min(finiteCosts)
    maxVal = max(finiteCosts)

    normalizedCosts = [
        (cost - minVal) / (maxVal - minVal) if math.isfinite(cost) else cost
        for cost in allCosts
    ]
    return normalizedCosts

def get_all_path_costs(allPaths, occupiedPoints):
    occupiedPoints = clean_points(occupiedPoints)
    obstacleGrid = generate_obstacle_grid(occupiedPoints)
    distanceGrid = distance_from_obstacles(obstacleGrid)
    allCosts = []
    for path in allPaths:
        allCosts.append(get_path_cost(path, distanceGrid))
    normalizedCosts = normalize_path_costs(allCosts)
    return normalizedCosts