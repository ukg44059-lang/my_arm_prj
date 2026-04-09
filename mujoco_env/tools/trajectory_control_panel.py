"""
Trajectory Recording Control Panel
Interactive control panel using matplotlib
Similar implementation to JointAnglePlotter
"""

import matplotlib
matplotlib.use('TkAgg')  # TkAgg backend for interactive windows

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import threading


class TrajectoryControlPanel:
    """Trajectory Recording Control Panel"""

    def __init__(self, command_queue):
        """
        Initialize trajectory control panel

        Args:
            command_queue: Queue for passing commands
        """
        self.command_queue = command_queue
        self.recording = False
        self.exported = False

        # Create figure window
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.fig.suptitle('End-Effector Trajectory Recorder', fontsize=14, fontweight='bold')

        # Remove axes
        self.ax.axis('off')

        # Create status text
        self.status_text = self.ax.text(0.5, 0.75, 'Status: Idle',
                                        ha='center', va='center',
                                        fontsize=14, fontweight='bold',
                                        color='gray',
                                        transform=self.ax.transAxes)

        # Create data count text
        self.data_count_text = self.ax.text(0.5, 0.60, 'Data Points: 0',
                                           ha='center', va='center',
                                           fontsize=11, color='#666666',
                                           transform=self.ax.transAxes)

        # Create buttons
        # Start button (green)
        ax_start = plt.axes([0.15, 0.40, 0.7, 0.14])
        self.btn_start = Button(ax_start, '▶ Start Recording',
                               color='#4CAF50', hovercolor='#45a049')
        self.btn_start.on_clicked(self.on_start_record)

        # Stop button (orange)
        ax_stop = plt.axes([0.15, 0.22, 0.7, 0.14])
        self.btn_stop = Button(ax_stop, '⏹ Stop Recording',
                              color='#FF9800', hovercolor='#e68900')
        self.btn_stop.on_clicked(self.on_stop_record)

        # Export button (blue)
        ax_export = plt.axes([0.15, 0.04, 0.7, 0.14])
        self.btn_export = Button(ax_export, '💾 Export Trajectory',
                                color='#2196F3', hovercolor='#0b7dda')
        self.btn_export.on_clicked(self.on_export)

        # Enable interactive mode
        plt.ion()
        self.fig.canvas.draw()
        plt.show(block=False)

        # Try to bring window to front
        try:
            self.fig.canvas.manager.window.lift()  # Raise window
            self.fig.canvas.manager.window.attributes('-topmost', True)  # Set as topmost
        except:
            pass

        print("✓ Trajectory recording control panel opened")

    def on_start_record(self, event):
        """Start recording trajectory"""
        self.command_queue.put("start_record")
        self.recording = True
        self.status_text.set_text('Status: Recording 🔴')
        self.status_text.set_color('red')
        self.fig.canvas.draw_idle()
        print("▶ Start recording trajectory")

    def on_stop_record(self, event):
        """Stop recording trajectory"""
        self.command_queue.put("stop_record")
        self.recording = False
        self.status_text.set_text('Status: Stopped ⏸')
        self.status_text.set_color('orange')
        self.fig.canvas.draw_idle()
        print("⏹ Stop recording trajectory")

    def on_export(self, event):
        """Export trajectory to file"""
        self.command_queue.put("export")
        self.exported = True
        self.status_text.set_text('Status: Exported ✓')
        self.status_text.set_color('green')
        self.fig.canvas.draw_idle()
        print("💾 Trajectory exported")

    def update_data_count(self, count):
        """Update the displayed data point count"""
        self.data_count_text.set_text(f'Data Points: {count}')
        self.fig.canvas.draw_idle()

    def is_alive(self):
        """Check if window is still alive"""
        try:
            return plt.fignum_exists(self.fig.number)
        except:
            return False

    def bring_to_front(self):
        """Bring window to front"""
        try:
            self.fig.canvas.manager.window.lift()
        except:
            pass
