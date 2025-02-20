# Mirror Ray Tracer

An in-progress project enabling fast ray tracing of custom reflective optics, primarily for simulating telescopes. Currently dependencies include [numpy](https://numpy.org/), [matplotlib](https://matplotlib.org/), [modernGL](https://moderngl.readthedocs.io/en/5.8.2/) and [moderngl-window](https://moderngl-window.readthedocs.io/en/3.1.1/), and supporting compatibility with [CuPy](https://cupy.dev/).

## Usage

To use this library, clone the files into a subfolder of your project directory.

For examples of basic usage, check out examples.py, which shows a newtonian telescope and the familiar conical off-axis coma, and a Ritchey Chretien telescope (without a field flattener) with the same incoming rays showing field curvature and astigmatism.

Here is a basic description of what it looks like to set up a simulation without going into too many fussy details:

```python
    # define the shape of the mirror as seen from above
    primary_slice = CircularSlice(radius, center)
    # define the mirror, passing in the slice
    primary = Paraboloid(focal_length, vertex_pos, primary_slice)
    # a detector for now is just a planar mirror
    detector_slice = RectangularSlice(-s, s, -s, s)
    detector = Plane(anchor_point, normal_vec, detector_slice)

    # define a bounding box with ndarrays specifying the min and max xyz
    bbox = BoundingBox(minima_array, maxima_array)
    # create the simulation environment: the bounding box and objects
    sim = Simulation(bbox, [primary, detector])

    # define a list of sources of rays
    # slices have a "scatter" method which helps with this
    # you can pass a single direction or an array the same size as anchor_points
    incid_rays = Rays(anchor_points, init_direction)
    sources = [incid_rays]

    # perform ray tracing
    sim.trace(sources)
    # render the results - this treats the last optical element as a detector
    # you can give the rays a custom colour palette to help with analysis
    sim.render(len(sources), detector, detector_up_direction, ray_colours)
```

## Features

Conic sections and planes can be used as mirrors. Conic sections can be defined in two ways: using their focal points, or using a focus and a conic constant.

Mirrors can be placed anywhere in 3D space, but aside from planes, are currently restricted to having their optical axes along the z axis.

Rendering the ray paths in 3D can be done with Matplotlib or ModernGL - the latter is much more capable in the face of a large number of rays, even with antialiasing and depth sorting turned on, whereas matplotlib starts to struggle above a few thousand.

![Matplotlib's attempt](/img/moderngl_focal_points_sparse.png)

Any number of sources, rays and mirrors can be placed into a scene, dependent only on how long you wish to wait for results, and your available memory :) Tracing is currently _pretty_ fast for a few sources with high ray density, but this is something I want to accelerate substantially, so that simulations can be updated in real time.

The roots of the incident rays are also plotted as they look upon striking the primary mirror, as well as the way they look upon striking the detector (i.e. the image they would produce).

![Off Axis Coma from a Newt](/img/off_axis_coma.png)
![Off Axis Performance from an RCT](/img/rc_off_axis_field_curvature_astig.png)

## Roadmap

Some of these may never happen, but these are currently on my own wish list, in roughly the order I think I would like to get to them.

- More fine-grained control of initial scattering of light rays, perhaps a better interface
- Add the K,R method of defining conic section mirrors
- A concatenate method to the ray class
- Correct handling of multiple ray sources with different numbers of rays
- Optimized GPU acceleration of ray tracing
- A full Qt based interface to set up and test various predefined telescope designs and custom optics, allowing real-time modifications and ray-tracing of the design
- An obstacle class, which only serves to block light, useful for baffles, veins, and sensors
- A sensor class, with the ability to simulate images obtained with that sensor
- Monte-Carlo simulation of light rays to mimic shot noise
- Adding a parameter to specify the optical axis of the conic section classes
- Automatic optimization of design parameters to improve image quality while retaining desired optical
- Adding to PyPI and conda forge
- Nebulous light sources, where initial direction is sampled from a distribution
- Eventually, refractive optics and different wavelengths of rays (would be especially nice for field flatteners, corrector plates, etc.)
- Diffraction caused by edges
- Adding sky glow
- A catalogue of star fields and deep sky objects to choose to simulate

### Gallery

These next 3 are from the Ritchey Chretien simulation, as viewed by modernGL. Upon really zooming in, you can see that the off axis image to the left is past its focal point, whereas the on axis image to the right is just short of it, indicating substantial field curvature, which would normally be corrected by refractive optics. ModernGL handles this with ease.
![](/img/moderngl_focal_points_distant.png)
![](/img/moderngl_focal_points.png)
![](/img/moderngl_rays.png)

Here you can see the off-axis image produced by a newtonian, you can even see the little conical shape where it intersects with the detector.
![](/img/off_axis_coma_3d.png)
