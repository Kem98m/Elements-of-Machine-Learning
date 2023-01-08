from types import SimpleNamespace as SnS
import numpy as np
from matplotlib import pyplot as plt

small_ = np.sqrt(np.finfo(float).eps)


def vector(u, v):
    return np.array((u, v), dtype=float)


def point(u, v):
    return vector(u, v)


def segment(p, q):
    return np.array((p, q))


def line(w, b):
    nw = np.linalg.norm(w)
    assert nw > 0, 'w cannot be zero'
    w, b = w / nw, b / nw
    return SnS(w=vector(w[0], w[1]), b=b)


def polygon(points):
    return np.array([vector(p[0], p[1]) for p in points])


def data_space():
    return polygon([point(0, 0), point(1, 0), point(1, 1), point(0, 1)])


def left_normal(u):
    return vector(-u[1], u[0])


def line_through(p, q):
    w = left_normal(q - p)
    b = -np.dot(w, p)
    return line(w=w, b=b)


def intersect_lines(c, d):
    p = None
    if np.abs(np.dot(c.w, d.w)) < 1:
        a = np.array((c.w, d.w))
        b = -np.array((c.b, d.b))
        p = np.linalg.lstsq(a, b, rcond=None)[0]
    return p


def point_distance(p, q):
    return np.linalg.norm(p - q)


def point_line_signed_distance(p, c):
    return np.dot(c.w, p) + c.b


def point_line_distance(p, c):
    return np.abs(point_line_signed_distance(p, c))


def segment_length(s):
    return point_distance(s[0], s[1])


def segment_unit_vector(s):
    n = segment_length(s)
    assert n > 0., 'zero-length segment'
    v = (s[1] - s[0]) / n
    return v, n


def extend_segment(s, by=small_):
    v = segment_unit_vector(s)[0]
    return segment(s[0] - by * v, s[1] + by * v)


def point_is_in_segment(p, s):
    v_unit, nv = segment_unit_vector(s)
    a = p - s[0]
    na = np.linalg.norm(a)
    if na == 0.:
        return True
    a_unit = a / na
    return 0. <= np.dot(a, v_unit) <= nv and np.dot(a_unit, v_unit) > 1. - small_


def intersect_line_and_segment(c, s):
    p, internal, endpoint = None, False, None
    if segment_length(s) > 0.:
        t = extend_segment(s)
        d = [point_line_signed_distance(q, c) for q in t]
        if d[0] * d[1] < 0.:
            p = intersect_lines(c, line_through(s[0], s[1]))
            if p is not None:
                internal = point_is_in_segment(p, s)
                d = [point_distance(p, q) for q in s]
                endpoint = 0 if d[0] < small_ else 1 if d[1] < small_ else None
    return SnS(point=p, internal=internal, endpoint=endpoint)


def centroid(points):
    if points.ndim < 2:
        return points
    else:
        return np.mean(points, axis=0)


def cut_polygon(poly, boundary):
    partition = SnS(segment=None, left=None, right=None)
    if poly is not None:
        n = len(poly)
        if n > 2:
            candidates = []
            for k, p in enumerate(poly):
                q = poly[(k + 1) % n]
                side = segment(p, q)
                i = intersect_line_and_segment(boundary, side)
                if i.point is not None:
                    candidates.append(SnS(point=i.point, side=k,
                                          internal=i.internal, endpoint=i.endpoint))
            if len(candidates) > 1:
                if len(candidates) > 2:
                    intersections = []
                    for k, this in enumerate(candidates):
                        if this.internal and this.endpoint is None:
                            intersections.append(this)
                        elif this.internal and this.endpoint == 1:
                            after = candidates[(k + 1) % len(candidates)]
                            if after.endpoint != 0:
                                intersections.append(this)
                        else:
                            before = candidates[(k - 1) % len(candidates)]
                            if this.endpoint == 0 and before.endpoint == 1:
                                if this.internal and not before.internal:
                                    intersection = this
                                elif before.internal and not this.internal:
                                    intersection = before
                                else:
                                    vertex = poly[this.side]
                                    this_distance = point_distance(this.point, vertex)
                                    before_distance = point_distance(before.point, vertex)
                                    if this.internal and before.internal:
                                        intersection = this if this_distance > before_distance else before
                                    else:
                                        intersection = this if this_distance < before_distance else before
                                intersections.append(intersection)
                    assert len(intersections) == 2, 'could not disambiguate intersections'
                else:
                    intersections = candidates

                s = [intersections[0].point, intersections[1].point]
                loop_a = list(range(intersections[1].side + 1, n)) + \
                    list(range(intersections[0].side + 1))
                loop_b = list(range(intersections[0].side + 1, intersections[1].side + 1))
                poly_a = np.array([poly[k] for k in loop_a] + s)
                poly_b = np.array([poly[k] for k in loop_b] + [s[1], s[0]])
                centroid_a = centroid(poly_a)
                d = point_line_signed_distance(centroid_a, boundary)
                if d > 0.:
                    right, left = poly_a, poly_b
                    s = segment(s[0], s[1])
                else:
                    right, left = poly_b, poly_a
                    s = segment(s[1], s[0])
                partition = SnS(segment=s, left=left, right=right)
            if partition.segment is None:
                d = point_line_signed_distance(centroid(poly), boundary)
                if d > 0.:
                    partition.right = poly
                else:
                    partition.left = poly
    return partition


def unpack_coordinates(p):
    u = list(p[:, 0]) + [p[0, 0]]
    v = list(p[:, 1]) + [p[0, 1]]
    return u, v


def plot_polygon(p, boundary_color=None, fill_color=None, alpha=1.):
    if p is not None:
        u, v = unpack_coordinates(p)
        if boundary_color is not None:
            plt.plot(u, v, color=boundary_color)
        if fill_color is not None:
            plt.fill(u, v, fill_color, alpha=alpha)


def plot_segment(s, color='k'):
    if s is not None:
        u, v = [s[0][0], s[1][0]], [s[0][1], s[1][1]]
        plt.plot(u, v, color=color)
