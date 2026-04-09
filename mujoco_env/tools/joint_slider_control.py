#!/usr/bin/env python3
import sys
import threading
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QCheckBox,
    QGridLayout,
    QMessageBox,
)


JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

# xian wei yi jing zuo hao le
JOINT_LIMITS = {
    "joint1": (-3.14, 3.14),
    "joint2": (-2.5, 2.5),
    "joint3": (-2.5, 2.5),
    "joint4": (-3.14, 3.14),
    "joint5": (-2.5, 2.5),
    "joint6": (-3.14, 3.14),
    "gripper": (0.0, 1.0),
}

DEFAULT_VALUES = {
    "joint1": 0.0,
    "joint2": 0.3,
    "joint3": -0.8,
    "joint4": 0.0,
    "joint5": -0.6,
    "joint6": 0.0,
    "gripper": 0.0,
}


class JointTargetPublisher(Node):
    def __init__(self) -> None:
        super().__init__("joint_slider_control")
        self.publisher = self.create_publisher(JointState, "/joint_target", 10)
        self.get_logger().info("Joint slider publisher started -> /joint_target")

    def publish_joint_target(self, positions: List[float]) -> None:
        if len(positions) != len(JOINT_NAMES):
            self.get_logger().error(
                f"Invalid positions length: {len(positions)}, expected {len(JOINT_NAMES)}"
            )
            return

        msg = JointState()
        msg.name = JOINT_NAMES
        msg.position = positions
        self.publisher.publish(msg)
        self.get_logger().info(f"Published joint target: {positions}")


class JointSliderUI(QWidget):
    def __init__(self, ros_node: JointTargetPublisher) -> None:
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle("Joint Slider Control")
        self.resize(720, 420)

        # 每个 slider 用整数表示，显示时再缩放成浮点
        self.slider_scale = 1000

        self.sliders = {}
        self.value_labels = {}

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout()

        title = QLabel("机械臂关节滑条控制")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        root.addWidget(title)

        grid = QGridLayout()

        for row, joint_name in enumerate(JOINT_NAMES):
            min_val, max_val = JOINT_LIMITS[joint_name]
            default_val = DEFAULT_VALUES[joint_name]

            name_label = QLabel(joint_name)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * self.slider_scale))
            slider.setMaximum(int(max_val * self.slider_scale))
            slider.setValue(int(default_val * self.slider_scale))
            slider.setSingleStep(1)

            value_label = QLabel(f"{default_val:.3f}")

            slider.valueChanged.connect(
                lambda value, j=joint_name: self._on_slider_changed(j, value)
            )

            self.sliders[joint_name] = slider
            self.value_labels[joint_name] = value_label

            grid.addWidget(name_label, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value_label, row, 2)

        root.addLayout(grid)

        self.realtime_checkbox = QCheckBox("实时发送")
        root.addWidget(self.realtime_checkbox)

        button_row = QHBoxLayout()

        send_button = QPushButton("发送当前值")
        send_button.clicked.connect(self.send_current_values)
        button_row.addWidget(send_button)

        reset_button = QPushButton("重置默认姿态")
        reset_button.clicked.connect(self.reset_to_default)
        button_row.addWidget(reset_button)

        root.addLayout(button_row)

        self.setLayout(root)

    def _on_slider_changed(self, joint_name: str, raw_value: int) -> None:
        value = raw_value / self.slider_scale
        self.value_labels[joint_name].setText(f"{value:.3f}")

        if self.realtime_checkbox.isChecked():
            self.send_current_values()

    def get_current_positions(self) -> List[float]:
        positions = []
        for joint_name in JOINT_NAMES:
            raw_value = self.sliders[joint_name].value()
            positions.append(raw_value / self.slider_scale)
        return positions

    def send_current_values(self) -> None:
        try:
            positions = self.get_current_positions()
            self.ros_node.publish_joint_target(positions)
        except Exception as e:
            QMessageBox.critical(self, "发送失败", str(e))

    def reset_to_default(self) -> None:
        for joint_name in JOINT_NAMES:
            self.sliders[joint_name].setValue(
                int(DEFAULT_VALUES[joint_name] * self.slider_scale)
            )

        if self.realtime_checkbox.isChecked():
            self.send_current_values()


def main() -> None:
    rclpy.init()

    ros_node = JointTargetPublisher()

    ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_spin_thread.start()

    app = QApplication(sys.argv)
    window = JointSliderUI(ros_node)
    window.show()

    exit_code = app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()