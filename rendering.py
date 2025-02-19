import queue
import threading
import time
from typing import TYPE_CHECKING

import moderngl
import moderngl_window as mglw
import numpy as np
import pyrr
from matplotlib.colors import hsv_to_rgb

if TYPE_CHECKING:
    from ray_tracer import Rays


class LineSegmentVisualizer(mglw.WindowConfig):
    data_queue = queue.Queue()
    gl_version = (3, 3)
    title = "Line Segment Visualizer"
    window_size = (1280, 720)
    resizable = True
    samples = 8

    # Class attributes for passing data
    start_positions = None
    end_positions = None
    colors = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Use stored data or defaults
        self.start_positions = (
            LineSegmentVisualizer.start_positions
            if LineSegmentVisualizer.start_positions is not None
            else np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]], dtype=np.float32)
        )
        self.end_positions = (
            LineSegmentVisualizer.end_positions
            if LineSegmentVisualizer.end_positions is not None
            else np.array([[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)
        )
        self.colors = (
            LineSegmentVisualizer.colors
            if LineSegmentVisualizer.colors is not None
            else np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        )

        # Create shaders
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 transform;
                in vec3 in_position;
                in vec3 in_color;
                out vec3 color;
                void main() {
                    gl_Position = transform * vec4(in_position, 1.0);
                    color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(color, 1.0);
                }
            """,
        )

        self.rotation = pyrr.matrix44.create_identity(dtype="f4")  # Start with identity matrix
        self.rotation_quat = pyrr.quaternion.create()  # Store rotation as a quaternion
        self.mouse_down = False  # Track if the mouse is pressed
        self.middle_mouse_down = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_sensitivity = 0.008
        self.fov_logit = 0.0
        self.zoom_sensitivity = 0.5
        self.camera_position = pyrr.Vector3([0.0, 0.0, 8.0])  # Start at (0,0,2)
        self.center_position = pyrr.Vector3([0.0, 0.0, 0.0])
        self.pan_sensitivity = 0.013

        self.dirty = True
        self.offscreen_fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture(self.wnd.size, 4)])

        # Create vertex buffer
        self.update_vertex_data()

    def update_vertex_data(self):
        """Update the GPU buffer with new line segment data while handling resizing properly"""
        vertices = np.hstack((self.start_positions, self.colors, self.end_positions, self.colors))
        vertex_bytes = vertices.astype("f4").tobytes()
        new_size = len(vertex_bytes)

        # If buffer exists and the size hasn't changed, reuse it
        if hasattr(self, "vbo") and self.vbo is not None:
            if new_size == self.vbo.size:
                self.vbo.write(vertex_bytes)  # Reuse the existing buffer
            else:
                self.vbo.release()  # Release old buffer if size changed
                self.vbo = self.ctx.buffer(vertex_bytes)  # Allocate a new buffer
        else:
            self.vbo = self.ctx.buffer(vertex_bytes)  # Create first-time buffer

        # Always recreate VAO to ensure it uses the correct buffer
        if hasattr(self, "vao") and self.vao is not None:
            self.vao.release()  # Release old VAO before replacing it

        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f 3f", "in_position", "in_color")],
        )

        self.ctx.enable(moderngl.DEPTH_TEST)

        self.dirty = True

    def load_data(self, start_positions, end_positions, colors):
        """Update the displayed lines dynamically"""
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.colors = colors
        self.update_vertex_data()

    def on_resize(self, width, height):
        self.dirty = True
        if hasattr(self, "offscreen_fbo") and self.offscreen_fbo is not None:
            self.offscreen_fbo.release()  # Free GPU memory, avoiding memory leaks

        self.offscreen_fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture((width, height), 4)])
        return super().on_resize(width, height)

    def on_render(self, time, frametime):
        """Render loop"""

        # Check if there's new data in the queue
        try:
            start_positions, end_positions, colors = self.data_queue.get_nowait()
            self.start_positions = start_positions
            self.end_positions = end_positions
            self.colors = colors
            self.update_vertex_data()
        except queue.Empty:
            pass  # No new data, continue rendering

        if not self.dirty:
            self.ctx.copy_framebuffer(self.ctx.screen, self.offscreen_fbo)
            return

        # Ensure aspect ratio is correct
        self.aspect_ratio = self.wnd.size[0] / self.wnd.size[1]

        # Perspective projection (Zoom is controlled by FOV)
        fov = 90.0 / (1.0 + np.exp(-self.fov_logit))
        projection = pyrr.matrix44.create_perspective_projection(fov, self.aspect_ratio, 0.1, 100.0, dtype="f4")

        # View matrix (Camera position controls panning)
        view = pyrr.matrix44.create_look_at(
            self.camera_position,  # Camera position
            self.camera_position + pyrr.Vector3([0.0, 0.0, -1.0]),  # Looking forward
            pyrr.Vector3([0.0, 1.0, 0.0]),  # Up direction
            dtype="f4",
        )

        translation = pyrr.matrix44.create_from_translation(self.center_position, dtype="f4")

        # Combine projection, view, and rotation
        transform = pyrr.matrix44.multiply(translation, self.rotation)
        transform = pyrr.matrix44.multiply(transform, view)
        transform = pyrr.matrix44.multiply(transform, projection)

        self.prog["transform"].write(transform)

        self.ctx.clear(0.2, 0.2, 0.2)
        self.vao.render(moderngl.LINES)
        self.ctx.copy_framebuffer(self.offscreen_fbo, self.ctx.screen)

        self.dirty = False

    def on_mouse_press_event(self, x, y, button):
        """Detect when the mouse is pressed"""
        if button == 1:  # Left mouse button
            self.mouse_down = True
        elif button == 3:  # Middle button → Pan
            self.middle_mouse_down = True

    def on_mouse_release_event(self, x, y, button):
        """Detect when the mouse is released"""
        if button == 1:  # Left mouse button
            self.mouse_down = False
        elif button == 3:  # Middle button → Pan
            self.middle_mouse_down = False

    def on_mouse_drag_event(self, x, y, dx, dy):
        """Update rotation quaternion when mouse moves"""
        scale = 1.0 / (1.0 + np.exp(-self.fov_logit))

        if self.mouse_down:
            axis = pyrr.vector3.create(dy, dx, 0.0)  # Mouse movement determines rotation axis
            angle = -self.mouse_sensitivity * pyrr.vector.length(axis) * scale  # Scale by sensitivity
            axis = pyrr.vector.normalize(axis)  # Normalize the axis

            rotation_delta = pyrr.quaternion.create_from_axis_rotation(axis, angle)  # Create rotation quaternion
            self.rotation_quat = pyrr.quaternion.cross(self.rotation_quat, rotation_delta)  # Apply rotation
            self.rotation = pyrr.matrix44.create_from_quaternion(self.rotation_quat, dtype="f4")  # Convert to matrix
            self.dirty = True

        if self.middle_mouse_down:  # Middle button → Pan
            right_vector = pyrr.vector3.create(self.rotation[0][0], self.rotation[1][0], self.rotation[2][0])
            up_vector = pyrr.vector3.create(self.rotation[0][1], self.rotation[1][1], self.rotation[2][1])
            # right_vector = pyrr.vector3.create(1.0, 0.0, 0.0)
            # up_vector = pyrr.vector3.create(0.0, 1.0, 0.0)

            self.center_position += right_vector * dx * self.pan_sensitivity * scale
            self.center_position -= up_vector * dy * self.pan_sensitivity * scale
            self.dirty = True

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.fov_logit = np.clip(self.fov_logit - y_offset * self.zoom_sensitivity, -12.0, 4.0)
        self.dirty = True


class VisualizerApp:
    """Runs the visualization in a separate thread"""

    def __init__(self, start_positions: np.ndarray, end_positions: np.ndarray, colors: np.ndarray = None):
        assert start_positions.shape == end_positions.shape

        self.start_positions = start_positions.astype(np.float32)
        self.end_positions = end_positions.astype(np.float32)
        if colors is None:
            self.colors = np.ones_like(self.start_positions)  # all white rays
        else:
            assert colors.shape == start_positions.shape
            self.colors = colors.astype(np.float32)

        self.thread = None
        self.running = False

    @classmethod
    def from_bundles(cls, ray_bundles: list["Rays"], colors: np.ndarray = None):
        start_positions = np.concatenate([bundle.root for bundle in ray_bundles], axis=0)
        end_positions = np.concatenate([bundle.terminus for bundle in ray_bundles], axis=0)
        if colors is None:
            colors = np.ones_like(start_positions)  # all white rays
        else:
            # TODO: this will fail if the bundles are not all the same size.
            colors = np.concatenate([colors for _ in range(len(ray_bundles))])
        return cls(start_positions, end_positions, colors)

    def start(self):
        """Start the visualization in a separate thread"""
        if self.thread and self.thread.is_alive():
            print("Visualization already running")
            return

        # Pass data via class attributes
        LineSegmentVisualizer.start_positions = self.start_positions
        LineSegmentVisualizer.end_positions = self.end_positions
        LineSegmentVisualizer.colors = self.colors

        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

        # Wait for the window to initialize
        time.sleep(1)

    def _run(self):
        """Launch Moderngl Window"""
        try:
            mglw.run_window_config(LineSegmentVisualizer)
        except Exception as e:
            print(f"Error in visualization thread: {e}")
        finally:
            self.running = False

    def update_data(self, start_positions, end_positions, colors):
        """Send new data to the visualization thread"""
        if LineSegmentVisualizer.data_queue is not None:
            LineSegmentVisualizer.data_queue.put((start_positions, end_positions, colors))
        print("Data update sent to renderer")

    def stop(self):
        """Stop visualization thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


if __name__ == "__main__":
    # Example data
    start_positions = np.array([[np.nan, -0.8, 0.0], [0.8, -0.8, 0.0], [-0.2, 0.2, 0.0]], dtype=np.float32)
    end_positions = np.array([[-0.8, 0.8, 0.0], [0.8, 0.8, 0.0], [-0.2, -0.2, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    # Create and start visualization
    app = VisualizerApp(start_positions, end_positions, colors)
    app.start()

    # Update data dynamically (after 3 seconds)
    time.sleep(3)
    N = 1000000
    new_start_positions = np.random.randn(N, 3).astype(np.float32)
    new_end_positions = np.random.randn(N, 3).astype(np.float32)
    new_colors = hsv_to_rgb(
        np.stack(
            (
                np.random.rand(N),
                np.random.rand(N) * 0.3 + 0.7,
                np.ones(N),
            ),
            axis=1,
        )
    ).astype(np.float32)

    app.update_data(new_start_positions, new_end_positions, new_colors)

    # Keep running
    try:
        while app.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        app.stop()
