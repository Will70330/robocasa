import argparse
import json
import time
from collections import OrderedDict

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from termcolor import colored
from pynput.keyboard import Key, Listener

import robocasa.macros as macros
from robocasa.scripts.collect_demos import collect_human_trajectory
from robocasa.wrappers.enclosing_wall_render_wrapper import (
    EnclosingWallRenderWrapper,
    install_enclosing_wall_hotkeys,
)


# Cameras to cycle through per robot flag (press 'c' during teleop)
# Robot-XML cameras get "robot0_" prefix; base-XML cameras get "mobilebase0_" prefix.
ROBOT_CAMERAS = {
    "default":     ["free", "robot0_frontview"],
    "tidybot_yam": ["free", "mobilebase0_base1", "mobilebase0_base2", "robot0_eye_in_hand"],
    "tidybot":     ["free", "mobilebase0_base1", "mobilebase0_base2", "robot0_eye_in_hand"],
}

# Inline gripper actuators for robots whose gripper is embedded in the robot XML
# (not managed by the composite controller / GRIP controller).
# Maps robot flag → (actuator_name_in_xml, open_ctrl, close_ctrl)
INLINE_GRIPPER_CFG = {
    "tidybot_yam": ("gripper", 0.041, 0.0),   # left_finger position actuator; open=0.041, close=0.0
}


class InlineGripperWrapper:
    """
    Thin env wrapper that directly drives an inline (robot-XML-embedded) gripper
    actuator from the teleop device's grasp state.

    The composite controller never touches this actuator (NullGripper dof=0), so
    setting sim.data.ctrl[idx] after each step persists to the next physics step.
    """

    def __init__(self, env, robot_idn, actuator_base_name, open_ctrl, close_ctrl):
        self.env = env
        self._open = open_ctrl
        self._close = close_ctrl
        self._act_id = None
        self._device = None
        self._robot_idn = robot_idn

        full_name = f"robot{robot_idn}_{actuator_base_name}"
        try:
            base = env
            while hasattr(base, "env"):
                base = base.env
            self._act_id = base.sim.model.actuator_name2id(full_name)
            print(f"[gripper] Inline actuator '{full_name}' found at ctrl idx {self._act_id}")
        except Exception as e:
            print(f"[gripper] Warning: could not find actuator '{full_name}': {e}")

    def register_device(self, device):
        self._device = device

    def _apply_gripper(self):
        if self._act_id is None or self._device is None:
            return
        # Device state is only available after start_control() is called
        if not hasattr(self._device, "grasp_states") or not hasattr(self._device, "active_robot"):
            return
        ri = self._device.active_robot
        ai = self._device.active_arm_indices[ri]
        is_grasping = self._device.grasp_states[ri][ai]
        base = self.env
        while hasattr(base, "env"):
            base = base.env
        base.sim.data.ctrl[self._act_id] = self._close if is_grasping else self._open

    def step(self, action):
        result = self.env.step(action)
        self._apply_gripper()
        return result

    def reset(self):
        result = self.env.reset()
        self._apply_gripper()
        return result

    def __getattr__(self, name):
        return getattr(self.env, name)


def install_camera_cycle_hotkeys(env, camera_names):
    """Install a 'c' hotkey that cycles through `camera_names` on the env's viewer.

    The switch is applied immediately in the pynput background thread — no pending
    flag needed because MjviewerRenderer.update() re-applies camera_id every frame.

    Press 'c' to advance to the next camera in the list.
    """
    if not camera_names:
        return

    camera_names = list(camera_names)
    state = {"index": 0}

    # Walk through wrappers once at install time to find the base env.
    # (The viewer is not yet created here, so we resolve it lazily in the callback.)
    def _base_env():
        base = env
        while hasattr(base, "env"):
            base = base.env
        return base

    def _on_press(key):
        if not (hasattr(key, "char") and key.char == "c"):
            return
        state["index"] = (state["index"] + 1) % len(camera_names)
        cam_name = camera_names[state["index"]]
        base = _base_env()
        try:
            if cam_name == "free":
                base.viewer.set_camera(-1)  # -1 = free camera mode; mouse controls active
                print("\n[camera] → free (left-drag: rotate  right-drag: pan  scroll: zoom)")
            else:
                cam_id = base.sim.model.camera_name2id(cam_name)
                base.viewer.set_camera(cam_id)
                print(f"\n[camera] → {cam_name}")
        except Exception as e:
            print(f"\n[camera] Could not switch to '{cam_name}': {e}")

    env._camera_cycle_listener = Listener(on_press=_on_press)
    env._camera_cycle_listener.start()
    print(f"[camera] Press 'c' to cycle cameras: {camera_names}")


def choose_option(
    options, option_name, show_keys=False, default=None, default_message=None
):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("Here is a list of {}s:\n".format(option_name))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1,
                default_message,
            )
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    # Return the chosen environment name
    return choice


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="task (choose among 365 tasks)")
    parser.add_argument(
        "--layout", type=int, help="kitchen layout (choose number 1-60)"
    )
    parser.add_argument("--style", type=int, help="kitchen style (choose number 1-60)")
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse"],
        help="Teleop device (default: keyboard)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="default",
        choices=["default", "tidybot_yam", "tidybot"],
        help="Robot to use: default (PandaOmron), tidybot_yam (TidybotYam w/ YAM arm), tidybot (TidybotKinova w/ Gen3 arm + 2F85 gripper)",
    )
    args = parser.parse_args()

    robot_configs = {
        "default":      {"name": "PandaOmron",    "camera": None},
        "tidybot_yam":  {"name": "TidybotYam",    "camera": None},   # YAM 6-DOF arm on TidyBot2 base
        "tidybot":      {"name": "TidybotKinova", "camera": None},   # Kinova Gen3 on TidyBot2 base
    }
    robot_name   = robot_configs[args.robot]["name"]
    render_cam   = robot_configs[args.robot]["camera"]

    tasks = OrderedDict(
        [
            ("PickPlaceCounterToCabinet", "pick and place from counter to cabinet"),
            ("PickPlaceCounterToSink", "pick and place from counter to sink"),
            ("PickPlaceMicrowaveToCounter", "pick and place from microwave to counter"),
            ("PickPlaceStoveToCounter", "pick and place from stove to counter"),
            ("OpenSingleDoor", "open cabinet or microwave door"),
            ("CloseDrawer", "close drawer"),
            ("TurnOnMicrowave", "turn on microwave"),
            ("TurnOnSinkFaucet", "turn on sink faucet"),
            ("TurnOnStove", "turn on stove"),
            ("ArrangeVegetables", "arrange vegetables on a cutting board"),
            ("MicrowaveThawing", "place frozen food in microwave for thawing"),
            ("RestockPantry", "restock cans in pantry"),
            ("PreSoakPan", "prepare pan for washing"),
            ("PrepareCoffee", "make coffee"),
        ]
    )

    if args.task is None:
        args.task = choose_option(
            tasks, "task", default="PickPlaceCounterToCabinet", show_keys=True
        )

    # Create argument configuration
    config = {
        "env_name": args.task,
        "robots": robot_name,
        "controller_configs": load_composite_controller_config(robot=robot_name),
        "layout_ids": args.layout,
        "style_ids": args.style,
    }

    args.renderer = "mjviewer"

    print(colored(f"Initializing environment...", "yellow"))
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=render_cam,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=args.renderer,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)
    env = EnclosingWallRenderWrapper(env, alpha=0.1, enabled=False)
    install_enclosing_wall_hotkeys(env)
    install_camera_cycle_hotkeys(env, ROBOT_CAMERAS[args.robot])

    # Wrap env for inline gripper control (robots whose gripper is embedded in robot XML)
    gripper_wrapper = None
    if args.robot in INLINE_GRIPPER_CFG:
        act_name, open_pos, close_pos = INLINE_GRIPPER_CFG[args.robot]
        gripper_wrapper = InlineGripperWrapper(env, 0, act_name, open_pos, close_pos)
        env = gripper_wrapper

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # initialize device
    device = args.device
    if device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)
    elif device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=4.0,
            rot_sensitivity=4.0,
            vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )
    else:
        raise ValueError

    # Register the device with the inline gripper wrapper (if active)
    if gripper_wrapper is not None:
        gripper_wrapper.register_device(device)

    # collect demonstrations
    while True:
        ep_directory, discard_traj = collect_human_trajectory(
            env,
            device,
            "right",
            "single-arm-opposed",
            mirror_actions=True,
            render=(args.renderer != "mjviewer"),
            max_fr=30,
        )
        print()
