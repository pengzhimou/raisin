


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

class PointBuilder:
    def __init__(self):
        self.x = None
        self.y = None

    def with_x(self, x):
        self.x = x
        return self

    def with_y(self, y):
        self.y = y
        return self

    def build(self):
        return Point(self.x, self.y)

# Create a point using the builder
point = (
    PointBuilder()
    .with_x(1)
    .with_y(2)
    .build()
)

# Print the point
print(point)  # Output: (1, 2)