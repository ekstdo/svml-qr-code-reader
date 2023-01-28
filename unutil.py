class Blob:
    def __init__(self, point, bounds, count):
        self.point = point
        self.bounds = bounds
        self.count = count

    def minx(self):
        return self.bounds[0]

    def maxx(self):
        return self.bounds[1]

    def miny(self):
        return self.bounds[2]

    def maxy(self):
        return self.bounds[3]


def flood_fill(img, x, y, from_, to, limit):  # useless, weil zu langsam D:
    stack = [(x, y)]
    leny, lenx = img.shape
    minx = maxx = x
    miny = maxy = y
    count = 0
    while len(stack) > 0:
        ax, ay = stack.pop()
        if img[ay, ax] == from_:
            img[ay, ax] = to
            count += 1
            if ax < minx:
                minx = ax
            elif ax > maxx:
                maxx = ax
            if ay < miny:
                miny = ay
            elif ay > maxy:
                maxy = ay
            if ay > 0:
                stack.append((x, y - 1))
            if ax > 0:
                stack.append((x - 1, y))
            if ax < lenx - 1:
                stack.append((x + 1, y))
            if ay < leny - 1:
                stack.append((x, y + 1))
    return Blob((x, y), [minx, maxx, miny, maxy], count)


