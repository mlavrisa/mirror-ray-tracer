import numpy as np
from matplotlib.colors import hsv_to_rgb

from ray_tracer import (
    BoundingBox,
    CircularSlice,
    ConicSection,
    Ellipsoid,
    Hyperboloid,
    Paraboloid,
    Plane,
    Rays,
    RectangularSlice,
    Simulation,
)


def on_axis_newt():
    f = 3.0
    a = 1 / (4 * f)
    r = 0.5
    h = r**2 * a

    s = 0.015  # sensor size

    primary_slice = CircularSlice(r, np.array([0.0, 0.0, 0.0]))
    primary = Paraboloid(f, np.array([0.0, 0.0, 0.0]), primary_slice)
    detector_slice = RectangularSlice(-s, s, -s, s)
    detector = Plane(np.array([0.0, 0.0, f]), np.array([0.0, 0.0, -1.0]), detector_slice)

    sim = Simulation(BoundingBox(np.array([-r, -r, -1.0]), np.array([r, r, max(h, f) * 1.5])), [primary, detector])

    mirror_scatter = primary_slice.scatter(0.0149, concentric=True)

    colours = hsv_to_rgb(
        np.stack(
            (
                (np.arctan2(mirror_scatter[:, 1], mirror_scatter[:, 0]) / np.pi * 0.5) % 1.0,
                np.linalg.norm(mirror_scatter[:, :2], axis=1) / primary_slice.r,
                np.linalg.norm(mirror_scatter[:, :2], axis=1) / primary_slice.r * 0.8 + 0.2,
            ),
            axis=1,
        )
    )

    off_center_angle = np.arctan(1 / 235.0)
    incid_roots = np.concatenate((mirror_scatter, np.full((mirror_scatter.shape[0], 1), 5.0)), axis=1)
    incid_rays = Rays(incid_roots, np.array([0, 0, -1.0]))
    _, off_ax_roots = primary.intersect(incid_rays)  # This does not modify incident rays, only reflect does
    off_ax_rays = Rays(off_ax_roots, np.array([off_center_angle, 0, -1.0]))
    sources = [incid_rays, off_ax_rays]
    sim.trace(sources)
    sim.render(len(sources), detector, np.array([0.0, 1.0, 0.0]), colours, True)


def ritchey_chretien():

    f = 8.0  # effective focal length
    b = 1.1  # secondary real focal length
    d = 1.0  # distance between primary and secondary
    m = (f - b) / d  # magnification

    bf = b - d  # distance from primary vertex to focal plane
    f1 = -d * f / (bf + d - f)  # primary focal length

    r1 = 0.5  # primary radius
    r2 = 0.1  # secondary radius

    k1 = -1 - 2 * b / (d * m**3)  # primary conic constant
    k2 = -1 - 2 * (m * (2 * m - 1) + b / d) / (m - 1) ** 3  # secondary conic constant

    h = 2.0

    s = 0.045  # sensor size

    primary_slice = CircularSlice(r1, np.array([0.0, 0.0, 0.0]))
    primary = ConicSection(k1, f1, np.array([0.0, 0.0, 0.0]), primary_slice)
    secondary_slice = CircularSlice(r2, np.array([0.0, 0.0, d]))
    secondary = ConicSection(k2, f1 - d, np.array([0.0, 0.0, d]), secondary_slice)
    detector_slice = RectangularSlice(-s, s, -s, s)
    detector = Plane(np.array([0.0, 0.0, 0.154]), np.array([0.0, 0.0, -1.0]), detector_slice)

    sim = Simulation(
        BoundingBox(np.array([-r1, -r1, -1.0]), np.array([r1, r1, max(h, b) * 1.5])),
        [primary, secondary, detector],
        use_gpu=False,
    )

    mirror_scatter = primary_slice.scatter(0.0149, concentric=True)

    colours = hsv_to_rgb(
        np.stack(
            (
                (np.arctan2(mirror_scatter[:, 1], mirror_scatter[:, 0]) / np.pi * 0.5) % 1.0,
                np.linalg.norm(mirror_scatter[:, :2], axis=1) / primary_slice.r,
                np.linalg.norm(mirror_scatter[:, :2], axis=1) / primary_slice.r * 0.8 + 0.2,
            ),
            axis=1,
        )
    )

    off_center_angle = np.arctan(1 / 235.0)
    incid_roots = np.concatenate((mirror_scatter, np.full((mirror_scatter.shape[0], 1), 5.0)), axis=1)
    incid_rays = Rays(incid_roots, np.array([0, 0, -1.0]))
    _, off_ax_roots = primary.intersect(incid_rays)  # This does not modify incident rays, only reflect does
    off_ax_rays = Rays(off_ax_roots, np.array([off_center_angle, 0, -1.0]))
    sources = [incid_rays, off_ax_rays]
    sim.trace(sources)
    sim.render(len(sources), detector, np.array([0.0, 1.0, 0.0]), colours, use_opengl=True)


if __name__ == "__main__":
    on_axis_newt()
