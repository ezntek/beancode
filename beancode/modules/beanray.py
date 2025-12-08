import pyray as p
from beancode.bean_ast import BCArray
from beancode.bean_ffi import *
from beancode.error import BCError


def _color(arr: BCValue) -> p.Color:
    color = arr.get_array().get_flat()

    if len(color) < 4:
        raise BCError(f"expected 4 args for a raylib Color, got {len(color)}")

    return p.Color(
        color[0].get_integer(),
        color[1].get_integer(),
        color[2].get_integer(),
        color[3].get_integer(),
    )


def _rectangle(rect: BCValue) -> p.Rectangle:
    rec = rect.get_array().get_flat()

    if len(rec) < 4:
        raise BCError(f"expected 4 args for a raylib Rectangle, got {len(rec)}")

    return p.Rectangle(
        rec[0].get_integer(),
        rec[1].get_integer(),
        rec[2].get_integer(),
        rec[3].get_integer(),
    )


def init_window(args: BCArgsList):
    width = args["width"].get_integer()
    height = args["height"].get_integer()
    title = args["title"].get_string()

    p.init_window(width, height, title)


def close_window(_: BCArgsList):
    p.close_window()


def set_target_fps(args: BCArgsList):
    fps = args["fps"].get_integer()

    p.set_target_fps(fps)


def window_should_close(_: BCArgsList) -> BCValue:
    return BCValue.new_boolean(p.window_should_close())


def begin_drawing(_: BCArgsList):
    p.begin_drawing()


def end_drawing(_: BCArgsList):
    p.end_drawing()


def clear_background(args: BCArgsList):
    color = _color(args["color"])

    p.clear_background(color)


def draw_fps(args: BCArgsList):
    x = args["x"].get_integer()
    y = args["y"].get_integer()

    p.draw_fps(x, y)


def draw_rectangle(args: BCArgsList):
    x = args["x"].get_integer()
    y = args["y"].get_integer()
    width = args["width"].get_integer()
    height = args["height"].get_integer()
    color = _color(args["color"])

    p.draw_rectangle(x, y, width, height, color)


def draw_rectangle_rec(args: BCArgsList):
    rect = _rectangle(args["rect"])
    color = _color(args["color"])

    p.draw_rectangle_rec(rect, color)


def draw_text(args: BCArgsList):
    text = args["text"].get_string()
    x = args["x"].get_integer()
    y = args["y"].get_integer()
    size = args["size"].get_integer()
    color = _color(args["color"])

    p.draw_text(text, x, y, size, color)


_COLOR_T = BCArrayType.new_flat(BCPrimitiveType.INTEGER, (1, 4))
_RECT_T = BCArrayType.new_flat(BCPrimitiveType.INTEGER, (1, 4))


def _color_to_bc_array(color: p.Color) -> BCValue:
    flat = [
        BCValue.new_integer(color[0]),
        BCValue.new_integer(color[1]),
        BCValue.new_integer(color[2]),
        BCValue.new_integer(color[3]),
    ]  # type: ignore
    return BCValue.new_array(BCArray.new_flat(_COLOR_T, flat))


constants = [
    BCConstant("BEIGE", _color_to_bc_array(p.BEIGE)),
    BCConstant("BLACK", _color_to_bc_array(p.BLACK)),
    BCConstant("BLANK", _color_to_bc_array(p.BLANK)),
    BCConstant("BLUE", _color_to_bc_array(p.BLUE)),
    BCConstant("BROWN", _color_to_bc_array(p.BROWN)),
    BCConstant("DARKBLUE", _color_to_bc_array(p.DARKBLUE)),
    BCConstant("DARKBROWN", _color_to_bc_array(p.DARKBROWN)),
    BCConstant("DARKGREEN", _color_to_bc_array(p.DARKGREEN)),
    BCConstant("DARKGRAY", _color_to_bc_array(p.DARKGRAY)),
    BCConstant("DARKPURPLE", _color_to_bc_array(p.DARKPURPLE)),
    BCConstant("GOLD", _color_to_bc_array(p.GOLD)),
    BCConstant("GRAY", _color_to_bc_array(p.GRAY)),
    BCConstant("GREEN", _color_to_bc_array(p.GREEN)),
    BCConstant("LIGHTGRAY", _color_to_bc_array(p.LIGHTGRAY)),
    BCConstant("LIME", _color_to_bc_array(p.LIME)),
    BCConstant("MAGENTA", _color_to_bc_array(p.MAGENTA)),
    BCConstant("MAROON", _color_to_bc_array(p.MAROON)),
    BCConstant("ORANGE", _color_to_bc_array(p.ORANGE)),
    BCConstant("PINK", _color_to_bc_array(p.PINK)),
    BCConstant("PURPLE", _color_to_bc_array(p.PURPLE)),
    BCConstant("RAYWHITE", _color_to_bc_array(p.RAYWHITE)),
    BCConstant("RED", _color_to_bc_array(p.RED)),
    BCConstant("SKYBLUE", _color_to_bc_array(p.SKYBLUE)),
    BCConstant("VIOLET", _color_to_bc_array(p.VIOLET)),
    BCConstant("WHITE", _color_to_bc_array(p.WHITE)),
    BCConstant("YELLOW", _color_to_bc_array(p.YELLOW)),
]

procs = [
    BCProcedure(
        "InitWindow",
        {
            "width": BCPrimitiveType.INTEGER,
            "height": BCPrimitiveType.INTEGER,
            "title": BCPrimitiveType.STRING,
        },
        init_window,
    ),
    BCProcedure("CloseWindow", {}, close_window),
    BCProcedure("SetTargetFPS", {"fps": BCPrimitiveType.INTEGER}, set_target_fps),
    BCProcedure("BeginDrawing", {}, begin_drawing),
    BCProcedure("EndDrawing", {}, end_drawing),
    BCProcedure("ClearBackground", {"color": _COLOR_T}, clear_background),
    BCProcedure(
        "DrawFPS",
        {"x": BCPrimitiveType.INTEGER, "y": BCPrimitiveType.INTEGER},
        draw_fps,
    ),
    BCProcedure(
        "DrawRectangle",
        {
            "x": BCPrimitiveType.INTEGER,
            "y": BCPrimitiveType.INTEGER,
            "width": BCPrimitiveType.INTEGER,
            "height": BCPrimitiveType.INTEGER,
            "color": _COLOR_T,
        },
        draw_rectangle,
    ),
    BCProcedure(
        "DrawRectangleRec", {"rect": _RECT_T, "color": _COLOR_T}, draw_rectangle_rec
    ),
    BCProcedure(
        "DrawText",
        {
            "text": BCPrimitiveType.STRING,
            "x": BCPrimitiveType.INTEGER,
            "y": BCPrimitiveType.INTEGER,
            "size": BCPrimitiveType.INTEGER,
            "color": _COLOR_T,
        },
        draw_text,
    ),
]

funcs = [
    BCFunction("WindowShouldClose", {}, BCPrimitiveType.INTEGER, window_should_close),
]

EXPORTS: Exports = {
    "constants": constants,
    "variables": [],
    "procs": procs,
    "funcs": funcs,
}
