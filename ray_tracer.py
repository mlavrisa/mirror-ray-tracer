import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from rendering import VisualizerApp

has_gpu = False

try:
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise ValueError
    has_gpu = True
    print("Warning: GPU implementation is experimental, and currently slow")
except ImportError:
    cp = None
    print("No cupy installation found, using numpy")
except ValueError:
    cp = None
    print("No GPU found, using numpy")

# TODO: Implement hexagonal array, and optical axis of conic section mirrors not parallel to z axis
# TODO: Fix cupy implementation, currently too much overhead shifting arrays between cpu and gpu
# Much later: add full interface for modifying the simulation parameters in real time


def to_cpu(array):
    if cp is None:
        return array
    return array.get() if isinstance(array, cp.ndarray) else array


def to_gpu(array):
    if cp is None:
        return array
    return cp.asarray(array) if isinstance(array, np.ndarray) else array


class Slice:
    use_gpu = False

    def to_gpu(self):
        raise NotImplementedError("Slice subclasses must override this method")

    def to_cpu(self):
        raise NotImplementedError("Slice subclasses must override this method")

    def in_bounds(self, xyz: np.ndarray):
        raise NotImplementedError("Slice subclasses must override this method")

    def scatter(self, distance: float):
        raise NotImplementedError("Slice subclasses must override this method")


class CircularSlice(Slice):
    def __init__(self, radius: float, center: np.ndarray):
        self.r = radius
        self.c = center
        self.use_gpu = False

    def to_gpu(self):
        self.use_gpu = True
        self.c = to_gpu(self.c)

    def to_cpu(self):
        self.use_gpu = False
        self.c = to_cpu(self.c)

    def in_bounds(self, xyz: np.ndarray):
        """
        Returns a boolean array indicating which points of `xyz` are within the bounds of the circular slice.

        :param xyz: 3D coordinates of points to test
        :return: Boolean array of same shape as `xyz` indicating whether each point is within the bounds
        """
        xp = cp if self.use_gpu and has_gpu else np
        return xp.square(xyz[..., :2] - self.c[None, None, :2]).sum(axis=-1) <= xp.square(self.r)

    def scatter(self, distance: float, concentric=False):
        """
        Returns an array of points in the circular slice.

        If `concentric` is `True`, the points are arranged in a concentric circle pattern. Otherwise, a hexagonal
        array is used.

        :param distance: The distance between the points
        :param concentric: Whether to use a concentric pattern or a hexagonal array
        :return: An array of points in the circular slice
        """
        pts = np.zeros((1, 2))
        lgth = int(self.r / (distance * np.sqrt(3) / 2.0))
        if concentric:
            for idx in range(0, lgth, 1):
                num_pts = np.around(2.0 * np.pi * (idx + 1)).astype(int)
                delta = 2.0 * np.pi / num_pts
                rad = distance * (idx + 1) * np.sqrt(3) / 2.0
                angles = np.arange(num_pts) * delta + (idx % 2) * delta * 0.5
                pts = np.concatenate((pts, rad * np.stack((np.cos(angles), np.sin(angles)), axis=1)), axis=0)
                # pts = rad * np.stack((np.cos(angles), np.sin(angles)), axis=1)
                pts += self.c[:2]
            return pts
        else:
            raise NotImplementedError("Hexagonal array not yet implemented")


class RectangularSlice(Slice):
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.use_gpu = False  # not actually needed, but here for consistency

    def to_gpu(self):
        self.use_gpu = True

    def to_cpu(self):
        self.use_gpu = False

    def in_bounds(self, xyz: np.ndarray):
        return (
            (self.xmin < xyz[..., 0])
            & (xyz[..., 0] < self.xmax)
            & (self.ymin < xyz[..., 1])
            & (xyz[..., 1] < self.ymax)
        )


class CompoundSlice(Slice):
    def __init__(self, add_slices: list[Slice], subtract_slices: list[Slice]):
        self.add_slices: list[Slice] = add_slices
        self.subtract_slices: list[Slice] = subtract_slices
        self.use_gpu = False

    def to_gpu(self):
        self.use_gpu = True
        for s in self.add_slices:
            s.to_gpu()
        for s in self.subtract_slices:
            s.to_gpu()

    def to_cpu(self):
        self.use_gpu = False
        for s in self.add_slices:
            s.to_cpu()
        for s in self.subtract_slices:
            s.to_cpu()

    def in_bounds(self, xyz: np.ndarray):
        # must hit any of the add slices and not any of the subtract slices
        # useful for e.g. a mirror with a central hole
        xp = cp if self.use_gpu and has_gpu else np
        return xp.any(xp.stack([s.in_bounds(xyz) for s in self.add_slices], axis=1), axis=1) & ~xp.any(
            xp.stack([s.in_bounds(xyz) for s in self.subtract_slices], axis=1), axis=1
        )

    def scatter(self, distance: float, concentric=False):
        if concentric:
            assert self.add_slices[0] is CircularSlice
            return self.add_slices[0].scatter(distance, concentric=True)
        else:
            raise NotImplementedError("Hexagonal array not yet implemented")


class Mirror:
    use_gpu = False

    def to_gpu(self):
        raise NotImplementedError("Mirror subclasses must override this method")

    def to_cpu(self):
        raise NotImplementedError("Mirror subclasses must override this method")

    def normal(self, xyz: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Mirror subclasses must override this method")

    def intersect(self, rays: "Rays") -> np.ndarray:
        raise NotImplementedError("Mirror subclasses must override this method")


class Paraboloid(Mirror):
    def __init__(self, focal_length: float, vertex: np.ndarray, slice: Slice):
        self.f = focal_length
        self.a = 1 / (4 * self.f)
        self.v = vertex
        self.slice = slice
        self.use_gpu = False

    def to_gpu(self):
        self.use_gpu = True
        self.v = to_gpu(self.v)
        self.slice.to_gpu()

    def to_cpu(self):
        self.use_gpu = False
        self.v = to_cpu(self.v)
        self.slice.to_cpu()

    def intersect(self, rays: "Rays"):
        """
        Calculates the intersection point of the rays with the paraboloid
        This function takes a bundle of rays as input and returns the intersection point of the rays
        with the paraboloid.
        The intersection point is calculated by solving the quadratic equation of the ray and the paraboloid.
        The specific solution depends on whether the paraboloid is convex or concave.
        If the ray does not intersect with the paraboloid, the function returns nan.
        :param rays: A bundle of rays
        :return: The intersection point of the rays with the paraboloid
        """
        xp = cp if self.use_gpu and has_gpu else np
        root = rays.root
        direction = rays.direction
        # transform coordinates to be centered on vertex
        p = root - self.v[None, :]
        xy = p[:, :2]
        z = p[:, 2]
        dxy = direction[:, :2]
        dz = direction[:, 2]
        # symmetric about the z axis, so x and y treated together
        # z = a r^2 -- where r^2 = x^2 + y^2, and x,y,z at parameter t = p + t*d
        a = xp.sum(dxy * dxy, axis=1) * self.a
        b = 2.0 * xp.sum(dxy * xy, axis=1) * self.a - dz
        c = xp.sum(xy * xy, axis=1) * self.a - z
        # one intersection - used when direction is parallel to z axis
        one_intn = xp.zeros((root.shape[0], 2, 1))
        one_intn[...] = (-c / b)[:, None, None]
        two_intn = xp.zeros((root.shape[0], 2, 1))
        discrm = b * b - 4 * a * c
        # clip the discriminant to 0 to avoid complex roots
        two_intn[:, 0, 0] = (-b - xp.sqrt(discrm.clip(min=0))) / (2 * a)
        two_intn[:, 1, 0] = (-b + xp.sqrt(discrm.clip(min=0))) / (2 * a)
        t = xp.where(a[:, None, None] != 0, two_intn, one_intn)
        intn = direction[:, None] * t + p[:, None]  # Intersection point in transformed coordinates

        # setting all coordinates to infinity where the discriminant is negative should ensure it misses the slice
        intn[discrm < 0, :] = xp.inf

        # Note that if direction is parallel to z, t will have only one solution, but we will return a duplicate so
        # that the dimensions match
        all_xyz = intn + self.v[None, None]
        hits = self.slice.in_bounds(all_xyz)

        # if the ray would hit the paraboloid at negative t but is already rooted, ignore it (with a small tolerance)
        behind = xp.logical_and(rays.rooted[:, None, None], t < 1e-6)
        t[behind] = xp.inf

        # Now choose the closest valid intersection
        min_t = xp.min(xp.where(hits[..., None], t, xp.inf), axis=1)
        result = direction * min_t + root
        no_hits = ~xp.any(hits & ~behind[..., 0], axis=1)
        result[no_hits, :] = xp.nan
        return min_t.ravel(), result

    def normal(self, xyz):
        """
        Calculates the normal vector at a given point on the paraboloid.

        :param xyz: The point at which to calculate the normal vector
        :return: The normal vector at the given point
        """
        # gradient of the level set of the paraboloid is always perpendicular to surface
        # level set of paraboloid is f(x, y, z) = 0 = z - a(x^2 + y^2)
        # intuition: moving along tangent plane gives a change of 0 in f, thus the gradient of f must be perpendicular
        # alternative: changing value of f is steepest along the direction perpendicular, each level set is a nested
        # paraboloid inside or outside of the one defined here.
        xp = cp if self.use_gpu and has_gpu else np
        p = xyz - self.v
        grad_f = -2 * self.a * p
        grad_f[:, 2] = 1.0
        grad_f /= xp.linalg.norm(grad_f, axis=1, keepdims=True)  # This will divide by nan if rays missed mirror
        return grad_f


class Ellipsoid(Mirror):
    def __init__(self, near_focus: float, far_focus: float, solution: float, vertex: np.ndarray, slice: Slice):
        self.f1 = near_focus
        self.f2 = far_focus
        assert np.sign(near_focus) == np.sign(far_focus) and abs(near_focus) < abs(far_focus)
        self.solution = -np.sign(near_focus)  # 1 for convex, -1 for concave
        self.a = (far_focus + near_focus) / 2  # along optical axis, equivalent now to z axis
        self.b = np.sqrt(near_focus * far_focus)
        self.c = np.copy(vertex)
        self.c[2] += self.a
        self.v = vertex
        self.slice = slice
        self.use_gpu = False

    def to_gpu(self):
        self.use_gpu = True
        self.v = to_gpu(self.v)
        self.c = to_gpu(self.c)
        self.slice.to_gpu()

    def to_cpu(self):
        self.use_gpu = False
        self.v = to_cpu(self.v)
        self.c = to_cpu(self.c)
        self.slice.to_cpu()

    def intersect(self, rays: "Rays"):
        """
        Calculates the intersection point of the rays with the ellipsoid
        This function takes a bundle of rays as input and returns the intersection point of the rays
        with the ellipsoid.
        The intersection point is calculated by solving the quadratic equation of the ray and the ellipsoid.
        The specific solution depends on whether the ellipsoid is convex or concave.
        If the ray does not intersect with the ellipsoid, the function returns xp.nan.
        :param rays: A point that the ray passes through
        :return: The intersection point of the ray with the ellipsoid
        """
        xp = cp if self.use_gpu and has_gpu else np
        root = rays.root
        direction = rays.direction
        # transform coordinates to be centered on ellipsoid's center
        # This makes it easier to select the convex or concave case (simply filter z > 0 or z < 0)
        p = root - self.c[None, :]
        xy = p[:, :2]
        z = p[:, 2]
        dxy = direction[:, :2]
        dz = direction[:, 2]
        # symmetric about the z axis, so x and y treated together
        # 1 = z^2/b^2 + r^2/a^2 -- where r^2 = x^2 + y^2, and x,y,z at parameter t = p + t*d
        a = xp.sum(dxy * dxy, axis=1) / self.b**2 + xp.square(dz) / self.a**2
        b = 2.0 * (xp.sum(dxy * xy, axis=1) / self.b**2 + dz * z / self.a**2)
        c = xp.sum(xy * xy, axis=1) / self.b**2 + xp.square(z) / self.a**2 - 1.0
        # single intersection case is not needed since ellipsoid is a closed surface, always 2 intersections (may be
        # degenerate)
        t = xp.zeros((root.shape[0], 2, 1))
        discrm = b * b - 4 * a * c
        # clip the discriminant to 0 to avoid complex roots
        t[:, 0, 0] = (-b - xp.sqrt(discrm.clip(min=0))) / (2 * a)
        t[:, 1, 0] = (-b + xp.sqrt(discrm.clip(min=0))) / (2 * a)
        intn = direction[:, None] * t + p[:, None]  # Intersection point in transformed coordinates

        # setting all coordinates to infinity where the discriminant is negative should ensure it misses the slice
        intn[discrm < 0, :] = xp.inf
        # if the sign of the solution doesn't match the convexity of the ellipsoid, ensure it misses the slice
        correct_sign = xp.sign(intn[..., 2]) == self.solution
        intn[~correct_sign, :] = xp.inf

        all_xyz = intn + self.c[None, None]
        hits = self.slice.in_bounds(all_xyz)

        # if the ray would hit the paraboloid at negative t but is already rooted, ignore it (with a small tolerance)
        behind = xp.logical_and(rays.rooted[:, None, None], t < 1e-6)
        t[behind] = xp.inf

        # Now choose the closest valid intersection
        min_t = xp.min(xp.where(hits[..., None], t, xp.inf), axis=1)
        result = direction * min_t + root
        no_hits = ~xp.any(hits & ~behind[..., 0], axis=1)
        result[no_hits, :] = xp.nan
        return min_t.ravel(), result

    def normal(self, xyz):
        """
        Calculates the normal vector at a given point on the ellipsoid.

        :param xyz: The point at which to calculate the normal vector
        :return: The normal vector at the given point
        """
        # gradient of the level set of the ellipsoid is always perpendicular to surface
        # level set of ellipsoid is f(x, y, z) = 1 = z^2/a^2 + r^2/b^2
        # intuition: moving along tangent plane gives a change of 0 in f, thus the gradient of f must be perpendicular
        # alternative: changing value of f is steepest along the direction perpendicular, each level set is a nested
        # ellipsoid inside or outside of the one defined here.
        xp = cp if self.use_gpu and has_gpu else np
        p = xyz - self.c  # broadcasting happens because last dimension matches
        scale = xp.array([self.b**2, self.b**2, self.a**2])
        grad_f = self.solution * 2 * p / scale  # direction of normal vector doesn't *really* matter, but nice to have
        grad_f /= xp.linalg.norm(grad_f, axis=1, keepdims=True)
        return grad_f


class Hyperboloid(Mirror):
    def __init__(self, front_focus: float, rear_focus: float, vertex: np.ndarray, slice: Slice):
        self.f1 = front_focus
        self.f2 = rear_focus
        # front focus always positive, rear focus always negative
        # either being 0 is a terrible way to define a line segment, unsupported
        assert np.sign(front_focus) > 0 and np.sign(rear_focus) < 0
        # if front_focus is shorter than rear focus, use +ve z solution, otherwise -ve
        # solution is in a way opposite to ellipsoid
        a = (front_focus + rear_focus) / 2
        self.solution = -np.sign(a)
        assert self.solution != 0.0  # a terrible way to define a plane, unsupported
        self.a = abs(a)
        self.b = np.sqrt(0.25 * (front_focus - rear_focus) ** 2 - a**2)
        self.c = np.copy(vertex)
        self.c[2] += a
        self.v = vertex
        self.slice = slice
        self.use_gpu = False

    def to_gpu(self):
        self.use_gpu = True
        self.v = to_gpu(self.v)
        self.c = to_gpu(self.c)
        self.slice.to_gpu()

    def to_cpu(self):
        self.use_gpu = False
        self.v = to_cpu(self.v)
        self.c = to_cpu(self.c)
        self.slice.to_cpu()

    def intersect(self, rays: "Rays"):
        """
        Calculates the intersection point of the rays with the hyperboloid
        This function takes a bundle of rays as input and returns the intersection point of the rays
        with the hyperboloid.
        The intersection point is calculated by solving the quadratic equation of the ray and the hyperboloid.
        The specific solution depends on whether the hyperboloid is convex or concave.
        If the ray does not intersect with the hyperboloid, the function returns np.nan.
        :param rays: A point that the ray passes through
        :return: The intersection point of the ray with the hyperboloid
        """
        xp = cp if self.use_gpu and has_gpu else np
        root = rays.root
        direction = rays.direction
        # transform coordinates to be centered on hyperboloid's center
        # This makes it easier to select the convex or concave case (simply filter z > 0 or z < 0)
        p = root - self.c[None, :]
        xy = p[:, :2]
        z = p[:, 2]
        dxy = direction[:, :2]
        dz = direction[:, 2]
        # symmetric about the z axis, so x and y treated together
        # 0 = 1 - z^2/a^2 + r^2/b^2 -- where r^2 = x^2 + y^2, and x,y,z at parameter t => p + t*d
        a = xp.sum(dxy * dxy, axis=1) / self.b**2 - xp.square(dz) / self.a**2
        b = 2.0 * (xp.sum(dxy * xy, axis=1) / self.b**2 - dz * z / self.a**2)
        c = xp.sum(xy * xy, axis=1) / self.b**2 - xp.square(z) / self.a**2 + 1.0
        # single intersection case is not needed since hyperboloid always has 2 intersections (may be degenerate)
        t = xp.zeros((root.shape[0], 2, 1))
        discrm = b * b - 4 * a * c
        # clip the discriminant to 0 to avoid complex roots
        t[:, 0, 0] = (-b - xp.sqrt(discrm.clip(min=0))) / (2 * a)
        t[:, 1, 0] = (-b + xp.sqrt(discrm.clip(min=0))) / (2 * a)
        intn = direction[:, None] * t + p[:, None]  # Intersection point in transformed coordinates

        # setting all coordinates to infinity where the discriminant is negative should ensure it misses the slice
        intn[discrm < 0, :] = xp.inf
        # if the sign of the solution doesn't match the convexity of the hyperboloid, ensure it misses the slice
        correct_sign = xp.sign(intn[..., 2]) == self.solution
        intn[~correct_sign, :] = xp.inf

        all_xyz = intn + self.c[None, None]
        hits = self.slice.in_bounds(all_xyz)

        # if the ray would hit the paraboloid at negative t but is already rooted, ignore it (with a small tolerance)
        behind = xp.logical_and(rays.rooted[:, None, None], t < 1e-6)
        t[behind] = xp.inf

        # Now choose the closest valid intersection
        min_t = xp.min(xp.where(hits[..., None], t, xp.inf), axis=1)
        result = direction * min_t + root
        no_hits = ~xp.any(hits & ~behind[..., 0], axis=1)
        result[no_hits, :] = xp.nan
        return min_t.ravel(), result

    def normal(self, xyz):
        """
        Calculates the normal vector at a given point on the hyperboloid.

        :param xyz: The point at which to calculate the normal vector
        :return: The normal vector at the given point
        """
        # gradient of the level set of the hyperboloid is always perpendicular to surface
        # level set of hyperboloid is f(x, y, z) = -1 = -z^2/a^2 + r^2/b^2
        # intuition: moving along tangent plane gives a change of 0 in f, thus the gradient of f must be perpendicular
        # alternative: changing value of f is steepest along the direction perpendicular, each level set is a nested
        # hyperboloid inside or outside of the one defined here.
        xp = cp if self.use_gpu and has_gpu else np
        p = xyz - self.c  # broadcasting happens because last dimension matches
        scale = xp.array([self.b**2, self.b**2, -self.a**2])  # note -ve sign on z
        grad_f = self.solution * 2 * p / scale  # direction of normal vector doesn't *really* matter, but nice to have
        grad_f /= xp.linalg.norm(grad_f, axis=1, keepdims=True)
        return grad_f


class ConicSection(Mirror):
    def __init__(self, K: np.ndarray, near_focus: float, vertex: np.ndarray, slice: Slice):
        K = abs(K)
        e = np.sqrt(K)  # forget about oblate ellipsoids for now
        if K == 1.0:
            self.mirror = Paraboloid(near_focus, vertex, slice)
        elif K < 1.0:
            # we assume that the focal length provided is the one that's nearer to the vertex
            # a = c + f and a = c / e, so:
            c = near_focus / (1.0 / e - 1.0)
            f2 = c - near_focus  # mirrored about the center
            self.mirror = Ellipsoid(near_focus, f2, vertex, slice)
        else:
            # a = c - f and a = c / e, so:
            c = near_focus / (1.0 - 1.0 / e)
            if near_focus > 0:
                # we need to find the further focal length
                pos = near_focus
                neg = -c - near_focus  # mirrored about the center
            else:
                neg = near_focus
                pos = -c - near_focus  # mirrored about the center
            self.mirror = Hyperboloid(pos, neg, vertex, slice)
        self.use_gpu = False

    def to_gpu(self):
        self.use_gpu = True
        self.mirror.to_gpu()

    def to_cpu(self):
        self.use_gpu = False
        self.mirror.to_cpu()

    def intersect(self, rays: "Rays"):
        return self.mirror.intersect(rays)

    def normal(self, xyz):
        return self.mirror.normal(xyz)


class Plane(Mirror):
    # Can be used for boundary conditions, sensor, or a planar mirror
    def __init__(self, vertex: np.ndarray, normal: np.ndarray, slice: Slice):
        self.v = vertex
        self.n = normal / np.linalg.norm(normal)
        self.slice = slice
        self.use_gpu = False

    def to_gpu(self):
        self.use_gpu = True
        self.v = to_gpu(self.v)
        self.n = to_gpu(self.n)
        self.slice.to_gpu()

    def to_cpu(self):
        self.use_gpu = False
        self.v = to_cpu(self.v)
        self.n = to_cpu(self.n)
        self.slice.to_cpu()

    def intersect(self, rays: "Rays"):
        """
        Calculates the intersection point of the rays with the plane
        This function takes a point and a unit vector as input and returns the intersection point of the ray
        with the plane. The ray is defined by the point and the direction of the unit vector.
        If the ray does not intersect with the plane, the function returns np.nan.
        :param xyz: The point that the ray passes through
        :param direction: The unit vector that points in the direction of the ray
        :return: The intersection point of the ray with the plane
        """
        xp = cp if self.use_gpu and has_gpu else np
        xyz = rays.root
        direction = rays.direction
        t = -(xyz - self.v[None]) @ self.n / (direction @ self.n)
        new_xyz = direction * t[:, None] + xyz
        hits = self.slice.in_bounds(new_xyz[:, None])
        no_hit = xp.full_like(xyz, xp.nan)
        t[~hits.ravel()] = xp.inf
        return t, xp.where(hits, new_xyz, no_hit)

    def normal(self, xyz: np.ndarray):
        """
        Returns the normal vector of the plane.

        :param xyz: The point at which to calculate the normal vector
        :return: The normal vector of the plane
        """
        xp = cp if self.use_gpu and has_gpu else np
        return xp.repeat(self.n[None], xyz.shape[0], axis=0)


class Rays:
    def __init__(self, root: np.ndarray, direction: np.ndarray, rooted: np.ndarray = None):
        """
        Initializes the Rays object with the specified root, direction, and optionally rooted status.

        This constructor sets up the initial state of the rays, including their starting positions,
        normalized directions, and various attributes to track their interactions with objects.

        :param root: The starting positions of the rays, as a 2D numpy array with shape (n, 3).
        :param direction: The direction vectors of the rays, as a 2D numpy array with shape (n, 3)
                        or a 1D numpy array to be broadcasted to all roots.
        :param rooted: Optional boolean array indicating whether each ray is initially rooted.
                    If not provided, all rays are initialized as unrooted.
        """
        self.root = root
        if direction.ndim == 1:
            direction = np.repeat(direction[None], root.shape[0], axis=0)
        self.direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
        self.terminus = np.full_like(root, np.nan)
        if rooted is None:
            self.rooted = np.full(root.shape[0], False)
        else:
            self.rooted = rooted
        self.terminates = np.full_like(self.rooted, False)
        self.min_t = np.full_like(self.rooted, np.inf, dtype=float)
        self.blocked_terminus = np.full_like(self.root, np.nan)
        self.use_gpu = False

    def to_gpu(self):
        self.use_gpu = True
        self.root = to_gpu(self.root)
        self.direction = to_gpu(self.direction)
        self.terminus = to_gpu(self.terminus)
        self.rooted = to_gpu(self.rooted)
        self.terminates = to_gpu(self.terminates)
        self.min_t = to_gpu(self.min_t)
        self.blocked_terminus = to_gpu(self.blocked_terminus)

    def to_cpu(self):
        self.use_gpu = False
        self.root = to_cpu(self.root)
        self.direction = to_cpu(self.direction)
        self.terminus = to_cpu(self.terminus)
        self.rooted = to_cpu(self.rooted)
        self.terminates = to_cpu(self.terminates)
        self.min_t = to_cpu(self.min_t)
        self.blocked_terminus = to_cpu(self.blocked_terminus)

    def reflect(self, mirror: Mirror):
        """
        Reflects the rays off the given mirror and returns the resulting reflected rays.

        This function calculates the intersection of the rays with the given mirror, determines if any rays are
        blocked, and updates their terminus and termination status. It then calculates the new direction for the rays
        based on the mirror's normal at the intersection points and returns the reflected rays.

        :param mirror: The mirror object that the rays will reflect off.
        :return: A new Rays object representing the reflected rays.
        """
        assert mirror.use_gpu == self.use_gpu
        xp = cp if self.use_gpu and has_gpu else np
        mirror_t, new_root = mirror.intersect(self)

        # figure out if any of the rays were blocked
        is_blocked = mirror_t > self.min_t
        self.terminus[is_blocked, :] = self.blocked_terminus[is_blocked, :]
        self.terminates = ~xp.any(xp.isnan(self.terminus), axis=1)

        # Only set the terminus for rays that haven't already terminated, usually the ones that were blocked
        new_root[self.terminates, :] = xp.nan

        self.terminus[~self.terminates, :] = new_root[~self.terminates, :]
        self.terminates = ~xp.any(xp.isnan(new_root), axis=1) | self.terminates
        normals = mirror.normal(new_root)
        new_direction = self.direction - 2.0 * xp.sum(normals * self.direction, axis=1, keepdims=True) * normals
        return Rays(new_root, new_direction, self.terminates)

    def test_block(self, object: Mirror):
        """
        Tests if the rays are blocked by an object (mirror) and if this is the earliest the ray currently intersects
        with anything, updates the blocked terminus and minimum intersection parameter.

        This function computes the intersection of the rays with the given object, checks if the intersection occurs
        before any previously found intersections, and updates the blocked terminus and minimum intersection time
        accordingly.

        :param object: The mirror object to test against the rays.
        """
        assert object.use_gpu == self.use_gpu
        xp = cp if self.use_gpu and has_gpu else np
        t, root = object.intersect(self)
        less_than = t < self.min_t
        self.blocked_terminus[less_than, :] = root[less_than, :]
        self.min_t = xp.minimum(self.min_t, t)


class BoundingBox:
    def __init__(self, min: np.ndarray, max: np.ndarray):
        self.min = min
        self.max = max
        self.use_gpu = False

    def to_gpu(self):
        self.use_gpu = True
        self.min = to_gpu(self.min)
        self.max = to_gpu(self.max)

    def to_cpu(self):
        self.use_gpu = False
        self.min = to_cpu(self.min)
        self.max = to_cpu(self.max)

    def bound_rays(self, rays: Rays):
        """
        Binds the given rays to the bounding box by finding the intersections with the box and updating the root and
        terminus of the rays accordingly.

        The bounding box is described by its minimum and maximum coordinates. The function computes the intersection of
        the rays with the box by finding the intersection of the ray lines with the box's edges.

        The intersection is described by a parameter t, which is the parameter of the ray equation such that t = 0
        is the root of the ray and t = 1 is one unit along the direction of the ray.

        The function then computes the "away" and "towards" matrices, which describe whether the ray is heading away
        from a face or towards it. The away matrix is used to find the root of the ray within the box, and the towards
        matrix is used to find the terminus of the ray within the box.

        The function then updates the root and terminus of the rays by taking the maximum of the possible roots and the
        minimum of the possible termini.

        :param rays: The rays to be bounded.
        """
        assert rays.use_gpu == self.use_gpu
        xp = cp if self.use_gpu and has_gpu else np
        bounds = xp.concatenate((xp.diag(self.min), xp.diag(self.max)), axis=0)[None]
        norms = xp.concatenate((xp.eye(3), -xp.eye(3)), axis=0)

        # parameter t for intersection, t = 0 is the root, t = 1 is 1 unit along direction
        denom = xp.einsum("ik,jk->ij", rays.direction, norms)
        t = -xp.einsum("ijk,jk->ij", rays.root[:, None] - bounds, norms) / denom
        t[denom == 0] = xp.nan

        # away: when true, rooting is possible. If parallel, always false
        # towards: opposite, but still false when parallel - for termination
        away = xp.einsum("jk,ik->ij", norms, rays.direction) > 0
        towards = xp.einsum("jk,ik->ij", norms, rays.direction) < 0

        possible_roots = xp.max(xp.where(away, t, -xp.inf), axis=1)
        possible_termini = xp.min(xp.where(towards, t, xp.inf), axis=1)
        new_roots = rays.root + rays.direction * possible_roots[:, None]
        new_termini = rays.root + rays.direction * possible_termini[:, None]
        rays.root = xp.where(rays.rooted[:, None], rays.root, new_roots)
        rays.terminus = xp.where(rays.terminates[:, None], rays.terminus, new_termini)
        # rays must always terminate
        rays.rooted[...] = True
        rays.terminates[...] = True


class Simulation:
    def __init__(self, bounding_box: BoundingBox, objects: list[Mirror], use_gpu: bool = True):
        self.bounding_box = bounding_box
        self.objects = objects
        self.finished_rays = []
        self.use_gpu = use_gpu

    def trace(self, sources: list[Rays]):
        propagate = sources

        # transfer to gpu if necessary
        if self.use_gpu:
            self.bounding_box.to_gpu()
            for source in propagate:
                source.to_gpu()
            for obj in self.objects:
                obj.to_gpu()

        for idx, obj in enumerate(self.objects):
            for jdx, source in enumerate(propagate):
                if idx < len(self.objects) - 1:
                    for blocker in self.objects[idx + 1 :]:
                        source.test_block(blocker)
                new_rays = source.reflect(obj)
                if self.use_gpu:
                    new_rays.to_gpu()
                propagate[jdx] = new_rays
                self.bounding_box.bound_rays(source)
                self.finished_rays.append(source)

        for source in propagate:
            self.bounding_box.bound_rays(source)
            self.finished_rays.append(source)

        # transfer back to cpu if necessary
        if self.use_gpu:
            self.bounding_box.to_cpu()
            for obj in self.objects:
                obj.to_cpu()
            for source in self.finished_rays:
                source.to_cpu()

    def render(
        self,
        num_sources: int,
        detector_plane: Plane,
        detector_vertical: np.ndarray,
        c: np.ndarray = None,
        use_opengl: bool = False,
        render_final: bool = False,
    ):
        vis_app = None
        if use_opengl:
            n_obj = len(self.objects)
            n_render = (n_obj + (1 if render_final else 0)) * num_sources
            vis_app = VisualizerApp.from_bundles(self.finished_rays[:n_render], c)
            vis_app.start()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            if render_final:
                bundles = self.finished_rays
            else:
                bundles = self.finished_rays[:-num_sources]
            for bundle in bundles[::-1]:
                lines = np.stack((bundle.root, bundle.terminus), axis=1)
                # TODO: this will not work if the bundles are not all the same size
                line_collection = Line3DCollection(lines, colors=c, linewidths=1)
                ax.add_collection(line_collection)
            ax.set_xlim(self.bounding_box.min[0], self.bounding_box.max[0])
            ax.set_ylim(self.bounding_box.min[1], self.bounding_box.max[1])
            ax.set_zlim(self.bounding_box.min[2], self.bounding_box.max[2])
            # set axes equal
            ax.set_aspect("equal")
            plt.show()

        # project the roots of the rays onto the detector plane and plot
        vertical = detector_vertical - detector_vertical @ detector_plane.n * detector_plane.n
        vertical /= np.linalg.norm(vertical)
        horizontal = np.cross(detector_plane.n, vertical)

        _, ax = plt.subplots(1, 1)
        bundle = self.finished_rays[0]
        v = bundle.root @ vertical
        h = bundle.root @ horizontal
        ax.scatter(h, v, s=1.5, c=c)
        ax.set_aspect("equal")
        ax.set_xlim(self.bounding_box.min[0], self.bounding_box.max[0])
        ax.set_ylim(self.bounding_box.min[1], self.bounding_box.max[1])
        plt.show()

        _, ax = plt.subplots(1, 1)

        for bundle in self.finished_rays[-num_sources:]:
            v = bundle.root @ vertical
            h = bundle.root @ horizontal
            ax.scatter(h, v, s=1.5, c=c)
        ax.set_aspect("equal")
        ax.set_xlim(detector_plane.slice.xmin, detector_plane.slice.xmax)
        ax.set_ylim(detector_plane.slice.ymin, detector_plane.slice.ymax)
        plt.show()

        # keep the opengl visualization alive until the user closes it intentionally
        try:
            while vis_app is not None and vis_app.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            vis_app.stop()
