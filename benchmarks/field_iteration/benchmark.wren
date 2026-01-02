class Point {
  x=(value) { _x = value }
  y=(value) { _y = value }

  x { _x }
  y { _y }

  construct new(x, y) {
    _x = x
    _y = y
  }
}

var point = Point.new(0, 0)

for (i in 1..100000) {
  point.x = i
  point.y = point.y + i
}

System.print(point.x)
System.print(point.y)

