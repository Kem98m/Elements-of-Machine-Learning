import numpy as np
from types import SimpleNamespace as SnS
import geometry as geo

s_min, d_max = 1, 100
small = 1.e-8


def node(boundary=None, p=None, left=None, right=None):
    return SnS(boundary=boundary, p=p, left=left, right=right)


def distribution(ys, n_labels):
    p = [np.count_nonzero(ys == c) for c in range(n_labels)]
    return np.array(p, dtype=float) / np.max([np.sum(p), 1.])


def error_rate(ys, n_labels):
    return (1. - np.amax(distribution(ys, n_labels))) / (1. - 1. / n_labels)


def gini_index(ys, n_labels):
    return (1. - np.sum(np.square(distribution(ys, n_labels)))) / (1. - 1. / n_labels)


def clip(x):
    return x if x > small else 0.


def impurity(ys, n_labels):
    return clip(gini_index(ys, n_labels))


def data_point(sample):
    return sample[0]


def data_points(samples):
    return np.array([data_point(sample) for sample in samples])


def value(sample):
    return sample[1]


def values(samples):
    return np.array([value(sample) for sample in samples])


def is_leaf(tau):
    return tau.left is None and tau.right is None


def split(x, tau):
    signed_distance = geo.point_line_signed_distance(x, tau.boundary)
    return tau.left if signed_distance < 0 else tau.right


def ok_to_split(ys, depth, n_labels):
    return impurity(ys, n_labels) > 0. and len(ys) > s_min and depth < d_max


def majority_label(ys, n_labels):
    d = distribution(ys, n_labels)
    label = np.argmax(d)
    return label, d[label]


def impurity_change(i_samples, y_left, y_right, y_samples, n_labels):
    i_left, i_right = impurity(y_left, n_labels), impurity(y_right, n_labels)
    if i_left > i_right:
        y_left, y_right = y_right, y_left
        i_left, i_right = i_right, i_left
    n_left, n_right, n_samples = len(y_left), len(y_right), len(y_samples)
    delta = i_samples - (n_left * i_left + n_right * i_right) / n_samples
    return delta


def pick(items, conditions):
    return [item for item, condition in zip(items, conditions) if condition]


def hyperplane(j, t):
    w = geo.vector(1, 0) if j == 0 else geo.vector(0, 1)
    b = - t
    return geo.line(w, b)


def train_tree(samples, depth, n_labels, find_split):
    if ok_to_split(values(samples), depth, n_labels):
        left, right, boundary = find_split(samples, n_labels)
        if len(left) == 0 or len(right) == 0:
            return node(p=distribution(values(samples), n_labels))
        left_subtree = train_tree(left, depth + 1, n_labels, find_split)
        right_subtree = train_tree(right, depth + 1, n_labels, find_split)
        return node(boundary=boundary, left=left_subtree, right=right_subtree)
    else:
        return node(p=distribution(values(samples), n_labels))


def plot_decision_regions(t, depth, region, colors):
    if depth == 0:
        geo.plot_polygon(region, boundary_color='k')
    if is_leaf(t):
        label = np.argmax(t.p)
        color = colors[label % len(colors)]
        geo.plot_polygon(region, fill_color=color)
    else:
        cut = geo.cut_polygon(region, t.boundary)
        geo.plot_segment(cut.segment)
        plot_decision_regions(t.left, depth + 1, cut.left, colors)
        plot_decision_regions(t.right, depth + 1, cut.right, colors)


def predict(x, tau, summary):
    return summary(tau.p) if is_leaf(tau) else predict(x, split(x, tau), summary)
