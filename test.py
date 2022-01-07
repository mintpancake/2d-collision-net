import numpy as np
from shapely.geometry import Polygon
p1 = Polygon([(0, 0), (1, 1), (1, 0)])
p2 = Polygon([(0, 1), (1, 0), (1, 1)])
p3 = Polygon([(100, 101), (101, 100), (101, 101)])
x = list(zip(*(p1.intersection(p2)).exterior.coords.xy))
y = p1.intersection(p2)
z = p1.intersects(p3)
print(z)
