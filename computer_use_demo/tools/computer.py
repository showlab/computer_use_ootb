import subprocess
import platform
import pyautogui
import asyncio
import base64
import os
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4
import io
from PIL import Image

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import run

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]


class Resolution(TypedDict):
    width: int
    height: int


MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    Adapted for Windows using 'pyautogui'.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()

        # Get screen width and height using Windows command
        self.width, self.height = self.get_screen_size()
        self.display_num = None

        # Path to cliclick
        self.cliclick = "cliclick"
        self.key_conversion = {"Page_Down": "pagedown", "Page_Up": "pageup", "Super_L": "win"}

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )

            if action == "mouse_move":
                pyautogui.moveTo(x, y)
                return ToolResult(output=f"Moved mouse to ({x}, {y})")
            elif action == "left_click_drag":
                current_x, current_y = pyautogui.position()
                pyautogui.dragTo(x, y, duration=0.5)  # Adjust duration as needed
                return ToolResult(output=f"Dragged mouse from ({current_x}, {current_y}) to ({x}, {y})")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                # Handle key combinations
                keys = text.split('+')
                for key in keys:
                    key = self.key_conversion.get(key.strip(), key.strip())
                    pyautogui.keyDown(key)  # Press down each key
                for key in reversed(keys):
                    key = self.key_conversion.get(key.strip(), key.strip())
                    pyautogui.keyUp(key)    # Release each key in reverse order
                return ToolResult(output=f"Pressed keys: {text}")
            
            elif action == "type":
                pyautogui.typewrite(text, interval=TYPING_DELAY_MS / 1000)  # Convert ms to seconds
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(output=text, base64_image=screenshot_base64)

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                x, y = pyautogui.position()
                x, y = self.scale_coordinates(ScalingSource.COMPUTER, x, y)
                return ToolResult(output=f"X={x},Y={y}")
            else:
                if action == "left_click":
                    pyautogui.click()
                elif action == "right_click":
                    pyautogui.rightClick()
                elif action == "middle_click":
                    pyautogui.middleClick()
                elif action == "double_click":
                    pyautogui.doubleClick()
                return ToolResult(output=f"Performed {action}")

        raise ToolError(f"Invalid action: {action}")

    async def screenshot(self) -> ToolResult:
        screenshot = pyautogui.screenshot()
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Resize the image if it's too large
        if len(img_byte_arr) > 5 * 1024 * 1024:
            img_byte_arr = self.resize_image(img_byte_arr)

        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        return ToolResult(output="Screenshot taken", base64_image=base64_image)

    async def shell(self, command: str, take_screenshot=True) -> ToolResult:
        """Run a shell command and return the output, error, and optionally a screenshot."""
        _, stdout, stderr = await run(command)
        base64_image = None

        if take_screenshot:
            # delay to let things settle before taking a screenshot
            await asyncio.sleep(self._screenshot_delay)
            base64_image = (await self.screenshot()).base64_image

        return ToolResult(output=stdout, error=stderr, base64_image=base64_image)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None
        for dimension in MAX_SCALING_TARGETS.values():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                break
        if target_dimension is None:
            return x, y
        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)

    def get_screen_size(self):
        if platform.system() == "Darwin":  # macOS
            try:
                output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"]).decode('utf-8')
                for line in output.split('\n'):
                    if "Resolution" in line:
                        resolution = line.split(':')[1].strip()
                        width, height = map(lambda x: int(x.split()[0]), resolution.split(' x '))
                        return width, height
            except Exception as e:
                print(f"Error getting screen size: {e}")
                return 1920, 1080  # Default fallback resolution
        elif platform.system() == "Windows":
            # Keep existing Windows code
            cmd = "wmic path Win32_VideoController get CurrentHorizontalResolution,CurrentVerticalResolution"
        else:  # Linux or other OS
            cmd = "xrandr | grep '*' | awk '{print $1}'"

        try:
            output = subprocess.check_output(cmd, shell=True).decode()
            
            if platform.system() == "Windows":
                lines = output.strip().split('\n')[1:]  # Skip the header
                width, height = map(int, lines[0].split())
            else:
                resolution = output.strip().split()[0]
                width, height = map(int, resolution.split('x'))

            return width, height

        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            return 1920, 1080  # Default fallback resolution
        
    
    def get_mouse_position(self):
        # TODO: enhance this func
        from AppKit import NSEvent
        from Quartz import CGEventSourceCreate, kCGEventSourceStateCombinedSessionState

        loc = NSEvent.mouseLocation()
        # Adjust for different coordinate system
        return int(loc.x), int(self.height - loc.y)

    def ma